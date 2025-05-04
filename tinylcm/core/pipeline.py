from typing import Any, Dict, List, Optional, Union, Tuple, Callable, Set
import numpy as np
import time
import uuid
import threading
from collections import deque

from tinylcm.core.data_structures import FeatureSample, AdaptationEvent, AdaptiveState
from tinylcm.core.base import AdaptiveComponent, AdaptiveClassifier
from tinylcm.core.feature_extractors.base import BaseFeatureExtractor, NullFeatureExtractor
from tinylcm.core.handlers.hybrid import HybridHandler
from tinylcm.core.handlers import AdaptiveHandler
from tinylcm.core.classifiers.knn import LightweightKNN
from tinylcm.core.state_manager import StateManager
from tinylcm.core.adaptation_tracker import AdaptationTracker
from tinylcm.core.data_logger.logger import DataLogger

# Import drift detection components
from tinylcm.core.drift_detection.base import DriftDetector, AutonomousDriftDetector
try:
    from tinylcm.core.quarantine.buffer import QuarantineBuffer, QuarantineStatus, QuarantinedSample
except ImportError:
    QuarantineBuffer = None
    QuarantineStatus = None
    QuarantinedSample = None

try:
    from tinylcm.core.heuristics.adapter import HeuristicAdapter, AdaptationAction
except ImportError:
    HeuristicAdapter = None
    AdaptationAction = None

from tinylcm.utils.logging import setup_logger
from tinylcm.constants import DEFAULT_CONFIG, AdaptationStrategy

logger = setup_logger(__name__)


class AdaptivePipeline:
    """Main pipeline for adaptive learning in TinyLCM.
    
    This class integrates feature extraction, classification, and adaptation
    into a unified pipeline. It provides the main interface for interacting
    with the adaptive learning system.
    """
    
    def __init__(
        self,
        feature_extractor: Optional[BaseFeatureExtractor] = None,
        classifier: Optional[AdaptiveClassifier] = None,
        handler: Optional[AdaptiveHandler] = None,
        state_manager: Optional[StateManager] = None,
        adaptation_tracker: Optional[AdaptationTracker] = None,
        data_logger: Optional[DataLogger] = None,
        config: Optional[Dict[str, Any]] = None,
        autonomous_monitors: Optional[List[AutonomousDriftDetector]] = None,
        quarantine_buffer: Optional["QuarantineBuffer"] = None,
        heuristic_adapter: Optional["HeuristicAdapter"] = None
    ):
        """Initialize the adaptive pipeline.
        
        Args:
            feature_extractor: Component for extracting features from input data
            classifier: Component for classifying extracted features
            handler: Component for managing adaptation of the classifier
            state_manager: Component for saving and loading system state
            adaptation_tracker: Component for tracking adaptation events
            data_logger: Component for logging data and samples
            config: Configuration dictionary (uses DEFAULT_CONFIG if None)
            autonomous_monitors: List of autonomous drift detectors that don't require labels
            quarantine_buffer: Buffer for storing samples flagged by autonomous detectors
            heuristic_adapter: Component for heuristic adaptation of quarantined samples
        """
        # Load configuration
        self.config = config or DEFAULT_CONFIG.get("adaptive", {})
        
        # Set up feature extractor
        self.feature_extractor = feature_extractor or NullFeatureExtractor()
        
        # Set up classifier
        if classifier is None:
            logger.debug("Creating default LightweightKNN classifier")
            self.classifier = LightweightKNN(
                k=self.config.get("k_neighbors", 5),
                distance_metric=self.config.get("distance_metric", "euclidean"),
                max_samples=self.config.get("max_training_samples", 100),
                use_numpy=self.config.get("use_numpy", True)
            )
        else:
            self.classifier = classifier
        
        # Set up handler
        if handler is None:
            logger.debug("Creating default HybridHandler")
            self.handler = HybridHandler(
                classifier=self.classifier,
                max_samples=self.config.get("max_training_samples", 100),
                batch_size=self.config.get("batch_size", 30),
                baseline_accuracy=self.config.get("baseline_accuracy", 0.9),
                cusum_threshold=self.config.get("cusum_threshold", 5.0),
                cusum_delta=self.config.get("cusum_delta", 0.25),
                enable_condensing=self.config.get("enable_condensing", False),
                condensing_method=self.config.get("condensing_method", "class_balanced"),
                use_numpy=self.config.get("use_numpy", True),
                metrics_callback=self._on_metrics_update,
                adaptation_callback=self._on_adaptation_event
            )
        else:
            self.handler = handler
            
            # Register callbacks if not already set
            if not hasattr(self.handler, "metrics_callback") or self.handler.metrics_callback is None:
                self.handler.metrics_callback = self._on_metrics_update
            
            if not hasattr(self.handler, "adaptation_callback") or self.handler.adaptation_callback is None:
                self.handler.adaptation_callback = self._on_adaptation_event
        
        # Set up state manager
        if state_manager is None:
            logger.debug("Creating default StateManager")
            self.state_manager = StateManager(
                storage_dir=self.config.get("state_storage_dir", "./adaptive_states"),
                max_states=self.config.get("max_saved_states", 10)
            )
        else:
            self.state_manager = state_manager
        
        # Set up adaptation tracker
        if adaptation_tracker is None:
            logger.debug("Creating default AdaptationTracker")
            self.adaptation_tracker = AdaptationTracker(
                log_dir=self.config.get("adaptation_logs_dir", "./adaptation_logs"),
                max_events_in_memory=self.config.get("max_events_in_memory", 100)
            )
            
            # Start a new session
            self.adaptation_tracker.start_session({
                "config": self.config,
                "feature_extractor_type": type(self.feature_extractor).__name__,
                "classifier_type": type(self.classifier).__name__,
                "handler_type": type(self.handler).__name__
            })
        else:
            self.adaptation_tracker = adaptation_tracker
        
        # Set up data logger
        self.data_logger = data_logger
        
        # Set up autonomous detection systems
        self.enable_autonomous_detection = self.config.get("enable_autonomous_detection", True)
        self.autonomous_monitors = autonomous_monitors or []
        
        # Set up quarantine buffer
        self.enable_quarantine = self.config.get("enable_quarantine", True) and QuarantineBuffer is not None
        self.quarantine_buffer = quarantine_buffer
        self.quarantine_check_interval = self.config.get("quarantine_check_interval", 50)
        self.samples_since_quarantine_check = 0
        
        # Set up heuristic adapter
        self.enable_heuristic_adaptation = self.config.get("enable_heuristic_adaptation", False) and HeuristicAdapter is not None
        self.heuristic_adapter = heuristic_adapter
        self.heuristic_confidence_threshold = self.config.get("heuristic_confidence_threshold", 0.7)
        
        # External validation settings
        self.external_validation = self.config.get("external_validation", False)
        self.validation_timeout_seconds = self.config.get("validation_timeout_seconds", 86400)  # 24 hours default
        
        # Inform heuristic adapter of known classes if available
        if self.heuristic_adapter and hasattr(self.classifier, "classes_"):
            self.heuristic_adapter.update_known_classes(set(self.classifier.classes_))
        
        # Initialize additional state
        self.n_samples_processed = 0
        self.n_samples_adapted = 0
        self.n_samples_quarantined = 0
        self.n_autonomous_drift_detected = 0
        self.adaptation_enabled = True
        
        # Track available labels for each sample (for evaluation)
        self.available_labels: Dict[str, Set[Any]] = {
            "true": set(),      # Ground truth labels seen
            "predicted": set()  # Model predictions seen
        }
        
        # User-defined callbacks
        self.prediction_callbacks = []
        self.adaptation_callbacks = []
        self.drift_callbacks = []
        
        # Set up periodic quarantine processing thread if needed
        self._stop_event = threading.Event()
        self._quarantine_thread = None
        
        if self.enable_quarantine and self.enable_heuristic_adaptation and not self.external_validation:
            self._start_quarantine_processing_thread()
        
        logger.debug(
            f"Initialized AdaptivePipeline with "
            f"extractor={type(self.feature_extractor).__name__}, "
            f"classifier={type(self.classifier).__name__}, "
            f"handler={type(self.handler).__name__}, "
            f"autonomous_monitors={len(self.autonomous_monitors)}, "
            f"quarantine={self.enable_quarantine}, "
            f"heuristic={self.enable_heuristic_adaptation}"
        )
    
    def process(
        self,
        input_data: Any,
        label: Optional[Any] = None,
        timestamp: Optional[float] = None,
        sample_id: Optional[str] = None,
        metadata: Optional[Dict[str, Any]] = None,
        extract_features: bool = True
    ) -> Dict[str, Any]:
        """Process input data through the adaptive pipeline.
        
        This is the main method for interaction with the pipeline. It:
        1. Extracts features from the input data
        2. Makes a prediction using the classifier
        3. Updates the classifier using the adaptation handler
        4. Updates autonomous drift detectors if enabled
        5. Logs data and events as needed
        6. Returns the prediction results
        
        IMPORTANT: This method executes synchronously and may block the calling thread
        while performing feature extraction and prediction. The execution time depends
        on the complexity of the feature extractor and classifier, as well as the
        hardware capabilities of the device. For resource-constrained devices, consider
        running this method in a separate thread if low-latency responses are required.
        
        While the I/O operations (logging, state saving) are performed asynchronously
        in background threads, the core ML operations (feature extraction, prediction,
        adaptation) are performed synchronously to ensure consistency and correctness.
        
        Args:
            input_data: Input data to process (raw data or pre-extracted features)
            label: Optional ground truth label for the sample
            timestamp: Optional timestamp for the sample (default: current time)
            sample_id: Optional unique ID for the sample (default: generated UUID)
            metadata: Optional metadata for the sample
            extract_features: Whether to extract features or use input_data as features
            
        Returns:
            Dictionary with prediction results
        """
        # Update sample counter
        self.n_samples_processed += 1
        
        # Process timestamp
        if timestamp is None:
            timestamp = time.time()
        
        # Process sample ID
        if sample_id is None:
            sample_id = str(uuid.uuid4())
        
        # Extract features if needed
        if extract_features:
            features = self.feature_extractor.extract_features(input_data)
        else:
            features = input_data
        
        # Make prediction
        predictions = self.classifier.predict(np.array([features]))
        prediction = predictions[0]
        
        # Update available labels
        if label is not None:
            self.available_labels["true"].add(label)
        self.available_labels["predicted"].add(prediction)
        
        # Get prediction probabilities if available
        probas = None
        try:
            probas = self.classifier.predict_proba(np.array([features]))
            confidence = np.max(probas[0]) if probas is not None else None
        except:
            confidence = None
        
        # Create feature sample
        sample = FeatureSample(
            features=features,
            label=label,
            prediction=prediction,
            timestamp=timestamp,
            sample_id=sample_id,
            metadata=metadata or {}
        )
        
        # Log sample to data logger if available
        if self.data_logger is not None:
            self.data_logger.log_sample(
                input_data, prediction, 
                label=label, 
                metadata={
                    "sample_id": sample_id,
                    "timestamp": timestamp,
                    "confidence": confidence,
                    **(metadata or {})
                }
            )
        
        # Prepare result
        result = {
            "prediction": prediction,
            "confidence": confidence,
            "probabilities": probas[0] if probas is not None else None,
            "sample_id": sample_id,
            "timestamp": timestamp,
            "adaptation_occurred": False,
            "drift_detected": False,
            "autonomous_drift_detected": False,
            "quarantined": False
        }
        
        # Create record for autonomous monitoring
        autonomous_record = {
            "features": features,
            "prediction": prediction,
            "confidence": confidence,
            "probabilities": probas[0].tolist() if probas is not None else None,
            "sample_id": sample_id,
            "timestamp": timestamp,
            "label": label,  # This may be None, which is expected for autonomous detection
            "metadata": metadata or {}
        }
        
        # Update autonomous monitors if enabled
        autonomous_drift_detected = False
        drift_info = None
        
        if self.enable_autonomous_detection and self.autonomous_monitors:
            for monitor in self.autonomous_monitors:
                try:
                    # Update the monitor with the record
                    drift_detected, monitor_drift_info = monitor.update(autonomous_record)
                    
                    if drift_detected:
                        autonomous_drift_detected = True
                        drift_info = {
                            "detector": type(monitor).__name__,
                            "sample_id": sample_id,
                            "timestamp": timestamp,
                            "metric": monitor_drift_info.get("metric", "unknown"),
                            "current_value": monitor_drift_info.get("current_value", 0.0),
                            "threshold": monitor_drift_info.get("threshold", 0.0),
                            "details": monitor_drift_info
                        }
                        # Don't break here - we want to update all monitors
                except Exception as e:
                    logger.warning(f"Error updating autonomous monitor {type(monitor).__name__}: {str(e)}")
            
            result["autonomous_drift_detected"] = autonomous_drift_detected
            if autonomous_drift_detected:
                self.n_autonomous_drift_detected += 1
        
        # Add to quarantine if drift detected and quarantine is enabled
        if autonomous_drift_detected and self.enable_quarantine and self.quarantine_buffer:
            # Check if sample should be quarantined based on strategy
            if self.quarantine_buffer.should_quarantine(autonomous_record, drift_info):
                try:
                    # Add to quarantine buffer
                    self.quarantine_buffer.add_sample(
                        features=features,
                        prediction=prediction,
                        drift_info=drift_info,
                        confidence=confidence,
                        raw_data=input_data if hasattr(input_data, "__len__") and len(input_data) < 1000 else None,
                        metadata=metadata,
                        sample_id=sample_id
                    )
                    
                    # Update counter and result
                    self.n_samples_quarantined += 1
                    result["quarantined"] = True
                    logger.debug(f"Quarantined sample {sample_id} due to {drift_info['detector']}")
                except Exception as e:
                    logger.warning(f"Error adding sample to quarantine: {str(e)}")
        
        # Increment samples since quarantine check
        if self.enable_quarantine and self.heuristic_adapter:
            self.samples_since_quarantine_check += 1
            self.heuristic_adapter.track_sample()
            
            # Periodically check quarantine
            if self.samples_since_quarantine_check >= self.quarantine_check_interval:
                self._check_quarantine()
        
        # Notify prediction callbacks
        for callback in self.prediction_callbacks:
            try:
                callback(sample, result)
            except Exception as e:
                logger.warning(f"Error in prediction callback: {str(e)}")
        
        # If we have a label, process adaptation via the label-based handler
        if label is not None and self.adaptation_enabled:
            # Update handler with sample
            adaptation_event = self.handler.process_sample(sample)
            
            # Update result with adaptation information
            if adaptation_event is not None:
                self.n_samples_adapted += 1
                result["adaptation_occurred"] = True
                result["drift_detected"] = adaptation_event.drift_detected
                result["adaptation_event_type"] = adaptation_event.event_type
                
                # Notify adaptation callbacks
                for callback in self.adaptation_callbacks:
                    try:
                        callback(adaptation_event, result)
                    except Exception as e:
                        logger.warning(f"Error in adaptation callback: {str(e)}")
        
        # Notify drift callbacks if autonomous drift was detected
        if autonomous_drift_detected and self.drift_callbacks:
            for callback in self.drift_callbacks:
                try:
                    callback(drift_info, result)
                except Exception as e:
                    logger.warning(f"Error in drift callback: {str(e)}")
        
        return result
    
    def predict(
        self,
        input_data: Any,
        extract_features: bool = True
    ) -> Any:
        """Make a prediction without adaptation.
        
        This is a simplified interface for making predictions without
        adaptation or tracking.
        
        Args:
            input_data: Input data to process (raw data or pre-extracted features)
            extract_features: Whether to extract features or use input_data as features
            
        Returns:
            Prediction for the input data
        """
        # Extract features if needed
        if extract_features:
            features = self.feature_extractor.extract_features(input_data)
        else:
            features = input_data
        
        # Make prediction
        predictions = self.classifier.predict(np.array([features]))
        return predictions[0]
    
    def predict_batch(
        self,
        batch_data: List[Any],
        extract_features: bool = True
    ) -> List[Any]:
        """Make predictions for a batch of inputs without adaptation.
        
        Args:
            batch_data: List of input data to process
            extract_features: Whether to extract features or use input_data as features
            
        Returns:
            List of predictions for the input batch
        """
        # Extract features if needed
        if extract_features:
            features = []
            for item in batch_data:
                features.append(self.feature_extractor.extract_features(item))
        else:
            features = batch_data
        
        # Make predictions
        return self.classifier.predict(np.array(features))
    
    def save_state(
        self,
        state_id: Optional[str] = None,
        metadata: Optional[Dict[str, Any]] = None
    ) -> str:
        """Save the current state of the adaptive pipeline.
        
        Args:
            state_id: Optional ID for the state
            metadata: Optional metadata to include with the state
            
        Returns:
            ID of the saved state
        """
        # Get serializable samples from the handler
        samples = []
        
        if hasattr(self.handler, "training_samples"):
            for sample in self.handler.training_samples:
                samples.append(sample.to_dict())
        
        # Get autonomous detector states
        autonomous_states = []
        if self.autonomous_monitors:
            for i, monitor in enumerate(self.autonomous_monitors):
                try:
                    state = monitor.get_state()
                    # Add monitor type to state
                    state["type"] = type(monitor).__name__
                    state["index"] = i
                    autonomous_states.append(state)
                except Exception as e:
                    logger.warning(f"Failed to save state for monitor {type(monitor).__name__}: {str(e)}")
        
        # Add quarantine state info
        quarantine_info = {}
        if self.enable_quarantine and self.quarantine_buffer:
            try:
                # Get stats instead of full state to avoid excessive size
                quarantine_info = self.quarantine_buffer.get_statistics()
                # Add session ID for potential full state restore
                quarantine_info["session_id"] = self.quarantine_buffer.session_id
            except Exception as e:
                logger.warning(f"Failed to save quarantine state: {str(e)}")
        
        # Add heuristic adapter state info
        heuristic_info = {}
        if self.enable_heuristic_adaptation and self.heuristic_adapter:
            try:
                # Get created classes and other key info
                heuristic_info = {
                    "created_classes": list(self.heuristic_adapter.created_classes),
                    "known_classes": list(self.heuristic_adapter.known_classes),
                    "total_adaptations": self.heuristic_adapter.total_adaptations,
                    "total_samples_processed": self.heuristic_adapter.total_samples_processed,
                    "new_class_counter": self.heuristic_adapter.new_class_counter
                }
            except Exception as e:
                logger.warning(f"Failed to save heuristic adapter state: {str(e)}")
        
        # Add default metadata
        if metadata is None:
            metadata = {}
        
        metadata.update({
            "timestamp": time.time(),
            "n_samples_processed": self.n_samples_processed,
            "n_samples_adapted": self.n_samples_adapted,
            "n_samples_quarantined": self.n_samples_quarantined,
            "n_autonomous_drift_detected": self.n_autonomous_drift_detected,
            "feature_extractor_type": type(self.feature_extractor).__name__,
            "classifier_type": type(self.classifier).__name__,
            "handler_type": type(self.handler).__name__,
            "autonomous_monitors": [type(m).__name__ for m in self.autonomous_monitors],
            "autonomous_states": autonomous_states,
            "quarantine_info": quarantine_info,
            "heuristic_info": heuristic_info,
            "enable_autonomous_detection": self.enable_autonomous_detection,
            "enable_quarantine": self.enable_quarantine,
            "enable_heuristic_adaptation": self.enable_heuristic_adaptation,
            "available_labels": {
                "true": list(self.available_labels["true"]),
                "predicted": list(self.available_labels["predicted"])
            }
        })
        
        # Save state using state manager
        return self.state_manager.save_state(
            classifier=self.classifier,
            handler=self.handler,
            extractor=self.feature_extractor,
            samples=samples,
            metadata=metadata,
            state_id=state_id
        )
    
    def load_state(self, state_id: str) -> Dict[str, Any]:
        """Load a saved state into the adaptive pipeline.
        
        Args:
            state_id: ID of the state to load
            
        Returns:
            Metadata from the loaded state
        """
        # Load state using state manager
        metadata = self.state_manager.load_state(
            state_id=state_id,
            classifier=self.classifier,
            handler=self.handler,
            extractor=self.feature_extractor
        )
        
        # Update sample counts from metadata
        self.n_samples_processed = metadata.get("n_samples_processed", self.n_samples_processed)
        self.n_samples_adapted = metadata.get("n_samples_adapted", self.n_samples_adapted)
        self.n_samples_quarantined = metadata.get("n_samples_quarantined", self.n_samples_quarantined)
        self.n_autonomous_drift_detected = metadata.get("n_autonomous_drift_detected", self.n_autonomous_drift_detected)
        
        # Restore autonomous detector states if available
        autonomous_states = metadata.get("autonomous_states", [])
        if autonomous_states and self.autonomous_monitors:
            # Create a map of monitor types to instances for easier matching
            monitor_map = {type(m).__name__: m for m in self.autonomous_monitors}
            
            for state in autonomous_states:
                monitor_type = state.get("type")
                if monitor_type in monitor_map:
                    try:
                        # Remove metadata fields that are not part of the actual state
                        state_copy = state.copy()
                        state_copy.pop("type", None)
                        state_copy.pop("index", None)
                        
                        # Restore state
                        monitor_map[monitor_type].set_state(state_copy)
                        logger.debug(f"Restored state for monitor {monitor_type}")
                    except Exception as e:
                        logger.warning(f"Failed to restore state for monitor {monitor_type}: {str(e)}")
        
        # Restore available labels if available
        if "available_labels" in metadata:
            try:
                for key in ["true", "predicted"]:
                    if key in metadata["available_labels"]:
                        self.available_labels[key] = set(metadata["available_labels"][key])
            except Exception as e:
                logger.warning(f"Failed to restore available labels: {str(e)}")
        
        # If we have a heuristic adapter, restore its key state
        if self.enable_heuristic_adaptation and self.heuristic_adapter:
            heuristic_info = metadata.get("heuristic_info", {})
            if heuristic_info:
                try:
                    # Restore created classes if available
                    if "created_classes" in heuristic_info:
                        self.heuristic_adapter.created_classes = set(heuristic_info["created_classes"])
                    
                    # Restore known classes if available
                    if "known_classes" in heuristic_info:
                        self.heuristic_adapter.known_classes = set(heuristic_info["known_classes"])
                    
                    # Restore counters
                    if "total_adaptations" in heuristic_info:
                        self.heuristic_adapter.total_adaptations = heuristic_info["total_adaptations"]
                    
                    if "total_samples_processed" in heuristic_info:
                        self.heuristic_adapter.total_samples_processed = heuristic_info["total_samples_processed"]
                    
                    if "new_class_counter" in heuristic_info:
                        self.heuristic_adapter.new_class_counter = heuristic_info["new_class_counter"]
                    
                    logger.debug(
                        f"Restored heuristic adapter state with "
                        f"{len(self.heuristic_adapter.created_classes)} created classes, "
                        f"{len(self.heuristic_adapter.known_classes)} known classes"
                    )
                except Exception as e:
                    logger.warning(f"Failed to restore heuristic adapter state: {str(e)}")
        
        # Note: We don't restore the full quarantine buffer state here to avoid excessive memory usage
        # The main quarantine state is persisted independently by the quarantine buffer itself
        # If needed, you can implement a full restore from the buffer's persistence files
        
        logger.info(f"Loaded adaptive state: {state_id}")
        
        return metadata
    
    def enable_adaptation(self) -> None:
        """Enable adaptation of the classifier."""
        self.adaptation_enabled = True
        if hasattr(self.handler, "enable_adaptation"):
            self.handler.enable_adaptation()
        logger.debug("Adaptation enabled")
    
    def disable_adaptation(self) -> None:
        """Disable adaptation of the classifier."""
        self.adaptation_enabled = False
        if hasattr(self.handler, "disable_adaptation"):
            self.handler.disable_adaptation()
        logger.debug("Adaptation disabled")
    
    def cleanup(self) -> None:
        """Clean up resources and stop background threads."""
        try:
            # Signal threads to stop
            self._stop_event.set()
            
            # Wait for quarantine thread to finish
            if self._quarantine_thread and self._quarantine_thread.is_alive():
                self._quarantine_thread.join(timeout=2.0)
                logger.debug("Stopped quarantine processing thread")
            
            # Close quarantine buffer if available
            if self.enable_quarantine and self.quarantine_buffer:
                try:
                    self.quarantine_buffer.close()
                    logger.debug("Closed quarantine buffer")
                except Exception as e:
                    logger.warning(f"Error closing quarantine buffer: {str(e)}")
            
            # Save final state if configured
            if self.config.get("save_state_on_cleanup", True):
                try:
                    state_id = self.save_state(
                        state_id=f"cleanup_{int(time.time())}",
                        metadata={"cleanup": True}
                    )
                    logger.info(f"Saved final state during cleanup: {state_id}")
                except Exception as e:
                    logger.warning(f"Error saving final state during cleanup: {str(e)}")
            
            logger.info("AdaptivePipeline cleanup complete")
        except Exception as e:
            logger.error(f"Error during cleanup: {str(e)}")
    
    def __del__(self) -> None:
        """Destructor to ensure cleanup."""
        try:
            self.cleanup()
        except:
            pass
    
    def register_prediction_callback(self, callback: Callable[[FeatureSample, Dict[str, Any]], None]) -> None:
        """Register a callback to be notified after each prediction.
        
        Args:
            callback: Function to call after predictions
        """
        self.prediction_callbacks.append(callback)
    
    def register_adaptation_callback(self, callback: Callable[[AdaptationEvent, Dict[str, Any]], None]) -> None:
        """Register a callback to be notified after adaptation events.
        
        Args:
            callback: Function to call after adaptation events
        """
        self.adaptation_callbacks.append(callback)
    
    def _on_metrics_update(self, metrics: Dict[str, Any]) -> None:
        """Internal callback for metrics updates.
        
        Args:
            metrics: Dictionary of metrics
        """
        # Log metrics to adaptation tracker
        self.adaptation_tracker.log_metrics(metrics)
    
    def _on_adaptation_event(self, event: AdaptationEvent) -> None:
        """Internal callback for adaptation events.
        
        Args:
            event: Adaptation event
        """
        # Log event to adaptation tracker
        self.adaptation_tracker.log_adaptation_event(event)
    
    def _check_quarantine(self) -> None:
        """Check quarantine buffer for samples to process.
        
        This method is called periodically to check if there are enough quarantined
        samples to attempt heuristic adaptation.
        """
        if not self.enable_quarantine or not self.quarantine_buffer or not self.heuristic_adapter:
            return
        
        try:
            # Reset counter
            self.samples_since_quarantine_check = 0
            
            # Get heuristic adapter statistics before potential adaptation
            self.heuristic_adapter.store_pre_adaptation_metrics()
            
            # Apply pseudo-labels
            adapted_samples = self.heuristic_adapter.apply_pseudo_labels(
                min_confidence=self.heuristic_confidence_threshold
            )
            
            if not adapted_samples:
                return
            
            logger.info(f"Applied heuristic adaptation to {len(adapted_samples)} samples")
            
            # Create adaptation event and log it
            adaptation_event = self.heuristic_adapter.create_adaptation_event(adapted_samples)
            
            # Log the event
            self._on_adaptation_event(adaptation_event)
            
            # Get samples for adaptation
            adaptation_samples = self.heuristic_adapter.get_samples_for_adaptation(
                min_confidence=self.heuristic_confidence_threshold
            )
            
            # Apply adaptations if we have samples and adaptation is enabled
            if adaptation_samples and self.adaptation_enabled:
                for fs in adaptation_samples:
                    # Only use samples with a label (pseudo or validated)
                    if fs.label is not None:
                        # Update the classifier
                        self.handler.add_sample(fs)
                
                # Update statistics
                self.n_samples_adapted += len(adaptation_samples)
                
                # Update metrics for heuristic evaluation
                self.heuristic_adapter.update_metrics(self.handler.get_metrics())
                
                # Check if rollback is needed
                if (self.heuristic_adapter.rollback_needed() and 
                    self.config.get("enable_rollback", True)):
                    logger.warning("Heuristic adaptation had negative impact, rolling back")
                    # Implement rollback mechanism here (e.g., load previous state)
                    if hasattr(self.handler, "rollback"):
                        self.handler.rollback()
            
            # Update heuristic adapter with known classes if classifier was modified
            if hasattr(self.classifier, "classes_"):
                self.heuristic_adapter.update_known_classes(set(self.classifier.classes_))
            
        except Exception as e:
            logger.error(f"Error checking quarantine: {str(e)}")
    
    def _start_quarantine_processing_thread(self) -> None:
        """Start a background thread for processing quarantine buffer.
        
        This is used for autonomous edge adaptation when external validation
        is not available.
        """
        if self._quarantine_thread is not None:
            return
        
        # Create and start thread
        self._quarantine_thread = threading.Thread(
            target=self._quarantine_processing_loop,
            daemon=True
        )
        self._quarantine_thread.start()
        logger.debug("Started quarantine processing thread")
    
    def _quarantine_processing_loop(self) -> None:
        """Background thread loop for processing quarantine buffer."""
        while not self._stop_event.is_set():
            try:
                # Wait for a while before checking
                time.sleep(10)
                
                # Skip if adaptation is disabled
                if not self.adaptation_enabled:
                    continue
                
                # Check if we should process quarantine
                if self.enable_quarantine and self.enable_heuristic_adaptation:
                    self._check_quarantine()
            
            except Exception as e:
                logger.error(f"Error in quarantine processing loop: {str(e)}")
    
    def register_drift_callback(self, callback: Callable[[Dict[str, Any], Dict[str, Any]], None]) -> None:
        """Register a callback to be notified of autonomous drift detection.
        
        Args:
            callback: Function to call when drift is detected by autonomous detectors
        """
        self.drift_callbacks.append(callback)
    
    def process_validation_results(self, validation_results: List[Dict[str, Any]]) -> int:
        """Process external validation results for quarantined samples.
        
        This method is used when external validation is enabled, to incorporate
        validated labels from an external source (server, human, etc.).
        
        Args:
            validation_results: List of validation results
            
        Returns:
            Number of successfully processed validation results
        """
        if not self.enable_quarantine or not self.quarantine_buffer:
            return 0
        
        # Process results in quarantine buffer
        count = self.quarantine_buffer.process_validation_results(validation_results)
        
        # If validation results have been processed, check for adaptation
        if count > 0 and self.adaptation_enabled:
            # Get validated samples
            validated_samples = self.quarantine_buffer.get_validated_samples()
            
            adapted_count = 0
            for sample in validated_samples:
                # Skip samples already added to model
                if sample.status == QuarantineStatus.ADDED_TO_MODEL:
                    continue
                
                # Create feature sample and add to handler
                fs = sample.create_feature_sample()
                
                if fs.label is not None:
                    self.handler.add_sample(fs)
                    adapted_count += 1
                    
                    # Update sample status
                    self.quarantine_buffer.update_sample_status(
                        sample_id=sample.sample_id,
                        status=QuarantineStatus.ADDED_TO_MODEL
                    )
            
            if adapted_count > 0:
                logger.info(f"Adapted model with {adapted_count} validated samples")
                
                # Create adaptation event
                event = AdaptationEvent(
                    event_type="external_validation",
                    timestamp=time.time(),
                    samples_added=adapted_count,
                    samples_removed=0,
                    drift_detected=True,
                    metadata={
                        "validated_samples": adapted_count,
                        "external_validation": True
                    }
                )
                
                # Log the event
                self._on_adaptation_event(event)
                
                # Update sample count
                self.n_samples_adapted += adapted_count
        
        return count
    
    def get_quarantine_statistics(self) -> Dict[str, Any]:
        """Get statistics about the quarantine buffer.
        
        Returns:
            Dictionary of quarantine statistics
        """
        if not self.enable_quarantine or not self.quarantine_buffer:
            return {"enabled": False}
        
        # Get basic buffer stats
        buffer_stats = self.quarantine_buffer.get_statistics()
        
        # Get heuristic stats if available
        heuristic_stats = {}
        if self.enable_heuristic_adaptation and self.heuristic_adapter:
            heuristic_stats = self.heuristic_adapter.get_statistics()
        
        return {
            "enabled": True,
            "n_samples_quarantined": self.n_samples_quarantined,
            "n_autonomous_drift_detected": self.n_autonomous_drift_detected,
            "quarantine_rate": self.n_samples_quarantined / self.n_samples_processed if self.n_samples_processed > 0 else 0,
            "buffer": buffer_stats,
            "heuristic": heuristic_stats,
            "external_validation": self.external_validation
        }
    
    def get_statistics(self) -> Dict[str, Any]:
        """Get statistics about the adaptive pipeline.
        
        Returns:
            Dictionary of statistics
        """
        # Basic stats
        stats = {
            "n_samples_processed": self.n_samples_processed,
            "n_samples_adapted": self.n_samples_adapted,
            "adaptation_rate": self.n_samples_adapted / self.n_samples_processed if self.n_samples_processed > 0 else 0,
            "adaptation_enabled": self.adaptation_enabled,
            "feature_extractor_type": type(self.feature_extractor).__name__,
            "classifier_type": type(self.classifier).__name__,
            "handler_type": type(self.handler).__name__,
            "autonomous_detection_enabled": self.enable_autonomous_detection,
            "autonomous_monitors_count": len(self.autonomous_monitors),
            "quarantine_enabled": self.enable_quarantine,
            "heuristic_adaptation_enabled": self.enable_heuristic_adaptation,
            "n_autonomous_drift_detected": self.n_autonomous_drift_detected,
            "n_samples_quarantined": self.n_samples_quarantined,
            "known_classes": {"true": list(self.available_labels["true"]), 
                             "predicted": list(self.available_labels["predicted"])}
        }
        
        # Add handler-specific stats
        if hasattr(self.handler, "n_adaptations"):
            stats["n_adaptations"] = self.handler.n_adaptations
        
        if hasattr(self.handler, "training_samples"):
            stats["n_training_samples"] = len(self.handler.training_samples)
        
        # Add drift statistics if available
        if hasattr(self.adaptation_tracker, "get_drift_statistics"):
            drift_stats = self.adaptation_tracker.get_drift_statistics()
            stats["drift_statistics"] = drift_stats
        
        # Add quarantine statistics if enabled
        if self.enable_quarantine:
            quarantine_stats = self.get_quarantine_statistics()
            stats["quarantine"] = quarantine_stats
        
        return stats