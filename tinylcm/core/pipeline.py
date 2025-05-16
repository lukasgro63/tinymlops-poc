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
from tinylcm.core.state_manager import AdaptiveStateManager
from tinylcm.core.adaptation_tracker import AdaptationTracker
from tinylcm.core.data_logger.logger import DataLogger
from tinylcm.core.operational_monitor.monitor import OperationalMonitor

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

try:
    from tinylcm.client.sync_client import SyncClient
except ImportError:
    SyncClient = None

from tinylcm.utils.logging import setup_logger
from tinylcm.constants import DEFAULT_CONFIG, AdaptationStrategy

logger = setup_logger(__name__)


class InferencePipeline:
    """Base pipeline for label-free inference and monitoring in TinyLCM.
    
    This class provides the fundamental infrastructure for:
    1. Feature extraction
    2. Classification
    3. Monitoring operational aspects
    4. Autonomous drift detection
    
    It provides a streamlined interface for model inference with operational
    monitoring, but without adaptation capabilities.
    """
    
    def __init__(
        self,
        feature_extractor: Optional[BaseFeatureExtractor] = None,
        classifier: Optional[AdaptiveClassifier] = None,
        operational_monitor: Optional[OperationalMonitor] = None,
        data_logger: Optional[DataLogger] = None,
        config: Optional[Dict[str, Any]] = None,
        autonomous_monitors: Optional[List[AutonomousDriftDetector]] = None
    ):
        """Initialize the inference pipeline.
        
        Args:
            feature_extractor: Component for extracting features from input data
            classifier: Component for classifying extracted features
            operational_monitor: Component for monitoring operational aspects
            data_logger: Component for logging data and samples
            config: Configuration dictionary (uses DEFAULT_CONFIG if None)
            autonomous_monitors: List of autonomous drift detectors that don't require labels
        """
        # Load configuration
        self.config = config or DEFAULT_CONFIG.get("inference", {})
        
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
        
        # Set up operational monitor
        if operational_monitor is None:
            logger.debug("Creating default OperationalMonitor")
            self.operational_monitor = OperationalMonitor(
                storage_dir=self.config.get("monitoring_dir", "./monitoring"),
                memory_window_size=self.config.get("monitoring_window_size", 1000),
                collect_system_metrics=self.config.get("collect_system_metrics", True)
            )
        else:
            self.operational_monitor = operational_monitor
        
        # Set up data logger
        self.data_logger = data_logger
        
        # Set up autonomous detection systems
        self.enable_autonomous_detection = self.config.get("enable_autonomous_detection", True)
        self.autonomous_monitors = autonomous_monitors or []
        self.autonomous_monitor_warmup_samples = self.config.get("autonomous_monitor_warmup_samples", 50)
        
        # Initialize additional state
        self.n_samples_processed = 0
        self.n_autonomous_drift_detected = 0
        
        # Track available labels for each sample (for evaluation)
        self.available_labels: Dict[str, Set[Any]] = {
            "true": set(),      # Ground truth labels seen
            "predicted": set()  # Model predictions seen
        }
        
        # User-defined callbacks
        self.prediction_callbacks = []
        self.drift_callbacks = []
        
        logger.debug(
            f"Initialized InferencePipeline with "
            f"extractor={type(self.feature_extractor).__name__}, "
            f"classifier={type(self.classifier).__name__}, "
            f"autonomous_monitors={len(self.autonomous_monitors)}"
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
        """Process input data through the inference pipeline.
        
        This is the main method for interaction with the pipeline. It:
        1. Extracts features from the input data
        2. Makes a prediction using the classifier
        3. Updates autonomous drift detectors if enabled
        4. Logs data and events as needed
        5. Returns the prediction results
        
        IMPORTANT: This method executes synchronously and may block the calling thread
        while performing feature extraction and prediction. The execution time depends
        on the complexity of the feature extractor and classifier, as well as the
        hardware capabilities of the device. For resource-constrained devices, consider
        running this method in a separate thread if low-latency responses are required.
        
        While the I/O operations (logging, monitoring) are performed asynchronously
        in background threads, the core ML operations (feature extraction, prediction)
        are performed synchronously to ensure consistency and correctness.
        
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
        # Start tracking inference time
        start_time = time.time()
        
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
            
        # Flatten feature tensor if it has more than 2 dimensions
        # This handles multi-dimensional feature tensors from CNNs
        if isinstance(features, np.ndarray) and len(features.shape) > 2:
            logger.info(f"Flattening feature tensor from shape {features.shape}")
            features = features.flatten()
            logger.info(f"Flattened shape: {features.shape}")
        
        # Make prediction
        predictions = self.classifier.predict(np.array([features]))
        prediction = predictions[0]
        
        # Calculate inference time
        latency_ms = (time.time() - start_time) * 1000
        
        # Update available labels
        if label is not None:
            self.available_labels["true"].add(label)
        self.available_labels["predicted"].add(prediction)
        
        # Get prediction probabilities if available
        probas = None
        try:
            logger.info("PIPELINE DEBUG: About to call predict_proba")
            probas = self.classifier.predict_proba(np.array([features]))
            raw_probas = probas[0] if probas is not None else None
            confidence = np.max(raw_probas) if raw_probas is not None else None
            logger.info(f"PIPELINE DEBUG: Raw probas from KNN: {raw_probas}")
            logger.info(f"PIPELINE DEBUG: Calculated confidence: {confidence}")
        except Exception as e:
            logger.error(f"PIPELINE DEBUG: Exception in predict_proba: {str(e)}")
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
        
        # Prepare metadata for operational monitor
        op_metadata = metadata or {}
        
        # If we're using the KNN classifier, include its debug information
        if isinstance(self.classifier, LightweightKNN) and hasattr(self.classifier, '_thread_local'):
            try:
                # Add KNN debug information if available (only the structured data)
                if hasattr(self.classifier._thread_local, 'neighbors_debug'):
                    op_metadata['knn_neighbors'] = self.classifier._thread_local.neighbors_debug
            except Exception as e:
                logger.warning(f"Error retrieving KNN debug info: {str(e)}")
        
        # Track operation in the operational monitor
        self.operational_monitor.track_inference(
            input_id=sample_id,
            prediction=prediction,
            confidence=confidence,
            latency_ms=latency_ms,
            metadata=op_metadata,
            timestamp=timestamp,
            features=features  # Include features for drift detection
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
            "latency_ms": latency_ms,
            "autonomous_drift_detected": False
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
            "metadata": metadata or {},
            "classifier": self.classifier  # Include direct reference to classifier for KNNDistanceMonitor
        }
        
        # Add KNN distances to the record if they exist
        if isinstance(self.classifier, LightweightKNN) and hasattr(self.classifier, '_last_distances'):
            # Include distances directly in record for easier access by drift detectors
            autonomous_record['_knn_distances'] = self.classifier._last_distances
            
        # Add KNN neighbor labels for NeighborDiversityDriftDetector if available
        if isinstance(self.classifier, LightweightKNN) and hasattr(self.classifier, 'get_last_neighbor_labels'):
            try:
                neighbor_labels = self.classifier.get_last_neighbor_labels()
                if neighbor_labels:
                    autonomous_record['_knn_neighbor_labels'] = neighbor_labels
            except Exception as e:
                logger.warning(f"Error getting KNN neighbor labels: {str(e)}")
        
        # Update autonomous monitors if enabled and warmup period is complete
        autonomous_drift_detected = False
        drift_info = None

        if self.enable_autonomous_detection and self.autonomous_monitors and self.n_samples_processed > self.autonomous_monitor_warmup_samples:
            for monitor_idx, monitor in enumerate(self.autonomous_monitors):
                try:
                    # Update the monitor with the record
                    drift_detected, monitor_drift_info = monitor.update(autonomous_record)

                    if drift_detected:
                        autonomous_drift_detected = True

                        # Create comprehensive drift info
                        drift_info = {
                            "detector_id": monitor_idx,
                            "detector_type": type(monitor).__name__,
                            "detector": type(monitor).__name__,  # For backwards compatibility
                            "sample_id": sample_id,
                            "timestamp": timestamp,
                            "metric": monitor_drift_info.get("metric", "unknown"),
                            "current_value": monitor_drift_info.get("current_value", 0.0),
                            "threshold": monitor_drift_info.get("threshold", 0.0),
                            "details": monitor_drift_info,
                            "drift_detected": True
                        }

                        # Instead of just collecting drift info, we'll handle the callback through the detector's
                        # _notify_callbacks method to respect its built-in cooldown mechanism

                        # However, the detector.update() method should already handle this correctly
                        # by calling _notify_callbacks internally when drift is detected.
                        # We won't need to do anything extra here.

                        # Don't break here - we want to update all monitors
                except Exception as e:
                    logger.warning(f"Error updating autonomous monitor {type(monitor).__name__}: {str(e)}")

            result["autonomous_drift_detected"] = autonomous_drift_detected
            if autonomous_drift_detected:
                result["drift_info"] = drift_info
                self.n_autonomous_drift_detected += 1
        
        # Notify prediction callbacks
        for callback in self.prediction_callbacks:
            try:
                callback(sample, result)
            except Exception as e:
                logger.warning(f"Error in prediction callback: {str(e)}")
        
        # Notify drift callbacks if autonomous drift was detected
        # We won't call drift callbacks directly here because we now rely on the
        # detector's internal _notify_callbacks method to handle drift notifications
        # with proper cooldown management. The callbacks should have already been
        # notified during the calls to monitor.update() above if drift was detected
        # and the detector wasn't in cooldown period.
        
        return result
    
    def predict(
        self,
        input_data: Any,
        extract_features: bool = True
    ) -> Any:
        """Make a prediction without monitoring.
        
        This is a simplified interface for making predictions without
        monitoring or tracking.
        
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
            
        # Flatten feature tensor if it has more than 2 dimensions
        # This handles multi-dimensional feature tensors from CNNs
        if isinstance(features, np.ndarray) and len(features.shape) > 2:
            logger.debug(f"Flattening feature tensor from shape {features.shape}")
            features = features.flatten()
        
        # Make prediction
        predictions = self.classifier.predict(np.array([features]))
        return predictions[0]
    
    def predict_batch(
        self,
        batch_data: List[Any],
        extract_features: bool = True
    ) -> List[Any]:
        """Make predictions for a batch of inputs without monitoring.
        
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
                feature = self.feature_extractor.extract_features(item)
                
                # Flatten feature tensor if it has more than 2 dimensions
                if isinstance(feature, np.ndarray) and len(feature.shape) > 2:
                    logger.debug(f"Flattening feature tensor from shape {feature.shape}")
                    feature = feature.flatten()
                    
                features.append(feature)
        else:
            features = []
            for item in batch_data:
                feature = item
                
                # Flatten feature tensor if it has more than 2 dimensions
                if isinstance(feature, np.ndarray) and len(feature.shape) > 2:
                    logger.debug(f"Flattening feature tensor from shape {feature.shape}")
                    feature = feature.flatten()
                    
                features.append(feature)
        
        # Make predictions
        return self.classifier.predict(np.array(features))
    
    def check_autonomous_drifts(self) -> List[Dict[str, Any]]:
        """Check autonomous drift detectors for drift.

        This method can be called periodically to explicitly check for drift
        in all autonomous monitors, regardless of the incoming samples.

        Returns:
            List of drift detection results, one for each monitor
        """
        if not self.enable_autonomous_detection or not self.autonomous_monitors:
            logger.info("Autonomous drift detection is disabled or no monitors configured")
            return []

        drift_results = []
        for i, monitor in enumerate(self.autonomous_monitors):
            try:
                # Check monitor for drift
                drift_detected, drift_info = monitor.check_for_drift()

                # Create result record
                result = {
                    "detector_id": i,
                    "detector_type": type(monitor).__name__,
                    "drift_detected": drift_detected,
                    "timestamp": time.time()
                }

                # Add drift info if available
                if drift_info:
                    result.update(drift_info)

                # Handle drift if detected
                if drift_detected:
                    self.n_autonomous_drift_detected += 1

                    # To respect the detector's built-in cooldown mechanism,
                    # instead of directly calling our callbacks, we'll temporarily
                    # register our callbacks with the detector and use its
                    # _notify_callbacks method which respects the cooldown period

                    # First, backup the detector's original callbacks
                    original_callbacks = monitor.callbacks.copy() if hasattr(monitor, "callbacks") else []

                    try:
                        # Temporarily register our pipeline callbacks with the detector
                        if hasattr(monitor, "callbacks"):
                            # Convert our callbacks to the detector's format
                            adapted_callbacks = []
                            for callback in self.drift_callbacks:
                                # Create a wrapper to handle the different signatures
                                def create_adapter(cb):
                                    def adapter(drift_info_arg):
                                        cb(drift_info_arg, {})
                                    return adapter

                                adapted_callbacks.append(create_adapter(callback))

                            # Replace detector's callbacks temporarily
                            monitor.callbacks = adapted_callbacks

                            # Call detector's notify method which respects cooldown
                            # The method returns True if callbacks were notified, False if in cooldown
                            callbacks_notified = monitor._notify_callbacks(result)
                            if not callbacks_notified:
                                # Cooldown information should already be in the result dictionary
                                # from the _notify_callbacks method, but let's ensure it's properly set
                                samples_since = result.get("samples_since_last_drift", "unknown")
                                cooldown_period = result.get("drift_cooldown_period", "unknown")

                                logger.debug(f"Detector {result.get('detector_type', 'unknown')} is in cooldown period "
                                             f"({samples_since}/{cooldown_period} samples since last drift), "
                                             f"drift will be ignored")

                                # Modify result to reflect that we're in cooldown
                                result["drift_detected"] = False
                                result["in_cooldown_period"] = True
                        else:
                            # Fallback if detector doesn't have callbacks attribute
                            for callback in self.drift_callbacks:
                                try:
                                    callback(result, {})
                                except Exception as e:
                                    logger.warning(f"Error in drift callback: {str(e)}")
                    finally:
                        # Restore original callbacks
                        if hasattr(monitor, "callbacks"):
                            monitor.callbacks = original_callbacks

                drift_results.append(result)

            except Exception as e:
                logger.error(f"Error checking drift in monitor {type(monitor).__name__}: {str(e)}")
                drift_results.append({
                    "detector_id": i,
                    "detector_type": type(monitor).__name__,
                    "drift_detected": False,
                    "error": str(e),
                    "timestamp": time.time()
                })

        return drift_results
    
    def register_prediction_callback(self, callback: Callable[[FeatureSample, Dict[str, Any]], None]) -> None:
        """Register a callback to be notified after each prediction.
        
        Args:
            callback: Function to call after predictions
        """
        self.prediction_callbacks.append(callback)
    
    def register_drift_callback(self, callback: Callable[[Dict[str, Any], Dict[str, Any]], None]) -> None:
        """Register a callback to be notified of autonomous drift detection.

        Args:
            callback: Function to call when drift is detected by autonomous detectors
        """
        self.drift_callbacks.append(callback)

        # Also register a wrapped version of this callback with each autonomous monitor
        # to ensure proper cooldown mechanism is used
        if self.autonomous_monitors:
            for monitor in self.autonomous_monitors:
                if hasattr(monitor, "register_callback"):
                    # We need to adapt our callback to the monitor's expected format
                    def adapted_callback(drift_info):
                        callback(drift_info, {})

                    monitor.register_callback(adapted_callback)
    
    def get_metrics(self) -> Dict[str, Any]:
        """Get current operational metrics.
        
        Returns:
            Dictionary of operational metrics
        """
        return self.operational_monitor.get_current_metrics()
    
    def get_statistics(self) -> Dict[str, Any]:
        """Get statistics about the inference pipeline.
        
        Returns:
            Dictionary of statistics
        """
        # Basic stats
        stats = {
            "n_samples_processed": self.n_samples_processed,
            "feature_extractor_type": type(self.feature_extractor).__name__,
            "classifier_type": type(self.classifier).__name__,
            "autonomous_detection_enabled": self.enable_autonomous_detection,
            "autonomous_monitors_count": len(self.autonomous_monitors),
            "n_autonomous_drift_detected": self.n_autonomous_drift_detected,
            "known_classes": {"true": list(self.available_labels["true"]), 
                             "predicted": list(self.available_labels["predicted"])}
        }
        
        # Add operational metrics
        operational_metrics = self.operational_monitor.get_current_metrics()
        stats["operational_metrics"] = operational_metrics
        
        # Add system metrics if available
        system_metrics = self.operational_monitor.get_system_metrics()
        if system_metrics:
            stats["system_metrics"] = system_metrics
        
        return stats
    
    def cleanup(self) -> None:
        """Clean up resources and stop background threads."""
        try:
            # Clean up operational monitor
            if hasattr(self.operational_monitor, "close"):
                self.operational_monitor.close()
                logger.debug("Closed operational monitor")
                
            logger.info("InferencePipeline cleanup complete")
        except Exception as e:
            logger.error(f"Error during cleanup: {str(e)}")
    
    def __del__(self) -> None:
        """Destructor to ensure cleanup."""
        try:
            self.cleanup()
        except:
            pass


class AdaptivePipeline(InferencePipeline):
    """Main pipeline for adaptive learning in TinyLCM.
    
    This class extends the InferencePipeline with adaptive learning capabilities.
    It integrates feature extraction, classification, monitoring, and adaptation
    into a unified pipeline, providing the complete TinyLCM experience including:
    
    1. Autonomous drift detection (from InferencePipeline)
    2. Quarantine of potentially drifted samples
    3. On-device heuristic adaptation
    4. Server-assisted validation and adaptation
    5. State management for versioning and rollbacks
    6. Comprehensive adaptation tracking
    
    The pipeline supports different operational modes:
    - Pure monitoring (inherited from InferencePipeline)
    - Autonomous heuristic adaptation
    - Server-assisted adaptation
    - Hybrid approach
    """
    
    def __init__(
        self,
        feature_extractor: Optional[BaseFeatureExtractor] = None,
        classifier: Optional[AdaptiveClassifier] = None,
        handler: Optional[AdaptiveHandler] = None,
        operational_monitor: Optional[OperationalMonitor] = None,
        state_manager: Optional[AdaptiveStateManager] = None,
        adaptation_tracker: Optional[AdaptationTracker] = None,
        data_logger: Optional[DataLogger] = None,
        config: Optional[Dict[str, Any]] = None,
        autonomous_monitors: Optional[List[AutonomousDriftDetector]] = None,
        quarantine_buffer: Optional["QuarantineBuffer"] = None,
        heuristic_adapter: Optional["HeuristicAdapter"] = None,
        sync_client: Optional["SyncClient"] = None
    ):
        """Initialize the adaptive pipeline.
        
        Args:
            feature_extractor: Component for extracting features from input data
            classifier: Component for classifying extracted features
            handler: Component for managing adaptation of the classifier
            operational_monitor: Component for monitoring operational aspects
            state_manager: Component for saving and loading system state
            adaptation_tracker: Component for tracking adaptation events
            data_logger: Component for logging data and samples
            config: Configuration dictionary (uses DEFAULT_CONFIG if None)
            autonomous_monitors: List of autonomous drift detectors that don't require labels
            quarantine_buffer: Buffer for storing samples flagged by autonomous detectors
            heuristic_adapter: Component for heuristic adaptation of quarantined samples
            sync_client: Optional client for synchronizing with a TinySphere server
        """
        # Load configuration
        _config = config or DEFAULT_CONFIG.get("adaptive", {})
        
        # Initialize the base class
        super().__init__(
            feature_extractor=feature_extractor,
            classifier=classifier,
            operational_monitor=operational_monitor,
            data_logger=data_logger,
            config=_config,
            autonomous_monitors=autonomous_monitors
        )
        
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
        if state_manager is None and self.config.get("enable_state_management", True):
            logger.debug("Creating default AdaptiveStateManager")
            self.state_manager = AdaptiveStateManager(
                storage_dir=self.config.get("state_storage_dir", "./adaptive_states"),
                max_snapshots=self.config.get("max_snapshots", 10),
                adaptation_tracker=adaptation_tracker
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
        
        # Set up quarantine buffer
        self.enable_quarantine = self.config.get("enable_quarantine", True) and QuarantineBuffer is not None
        self.quarantine_buffer = quarantine_buffer
        self.quarantine_check_interval = self.config.get("quarantine_check_interval", 50)
        self.samples_since_quarantine_check = 0
        
        if self.enable_quarantine and self.quarantine_buffer is None:
            logger.debug("Creating default QuarantineBuffer")
            self.quarantine_buffer = QuarantineBuffer(
                max_size=self.config.get("quarantine_max_size", 100),
                max_age=self.config.get("quarantine_max_age", 86400)  # 1 day in seconds
            )
        
        # Set up heuristic adapter
        self.enable_heuristic_adaptation = self.config.get("enable_heuristic_adaptation", False) and HeuristicAdapter is not None
        self.heuristic_adapter = heuristic_adapter
        self.heuristic_confidence_threshold = self.config.get("heuristic_confidence_threshold", 0.7)
        
        if self.enable_heuristic_adaptation and self.heuristic_adapter is None and self.enable_quarantine:
            logger.debug("Creating default HeuristicAdapter")
            self.heuristic_adapter = HeuristicAdapter(
                quarantine_buffer=self.quarantine_buffer,
                min_cluster_size=self.config.get("heuristic_min_cluster_size", 5),
                variance_threshold=self.config.get("heuristic_variance_threshold", 0.1),
                k_representatives=self.config.get("heuristic_k_representatives", 3),
                use_numpy=self.config.get("use_numpy", True)
            )
        
        # Set up sync client
        self.enable_server_sync = self.config.get("enable_server_sync", False) and SyncClient is not None
        self.sync_client = sync_client
        self.sync_interval = self.config.get("sync_interval", 3600)  # Default to 1 hour
        self.last_sync_time = 0
        
        # External validation settings
        self.external_validation = self.config.get("external_validation", False)
        self.validation_timeout_seconds = self.config.get("validation_timeout_seconds", 86400)  # 24 hours default
        
        # Inform heuristic adapter of known classes if available
        if self.heuristic_adapter and hasattr(self.classifier, "classes_"):
            self.heuristic_adapter.update_known_classes(set(self.classifier.classes_))
        
        # Initialize additional state
        self.n_samples_adapted = 0
        self.n_samples_quarantined = 0
        self.adaptation_enabled = True
        
        # User-defined callbacks
        self.adaptation_callbacks = []
        
        # Set up periodic quarantine processing thread if needed
        self._stop_event = threading.Event()
        self._quarantine_thread = None
        
        if self.enable_quarantine and self.enable_heuristic_adaptation and not self.external_validation:
            self._start_quarantine_processing_thread()
        
        logger.debug(
            f"Initialized AdaptivePipeline with "
            f"handler={type(self.handler).__name__}, "
            f"quarantine={self.enable_quarantine}, "
            f"heuristic={self.enable_heuristic_adaptation}, "
            f"server_sync={self.enable_server_sync}"
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
        
        This method extends the base InferencePipeline process method to include
        adaptation capabilities. It:
        
        1. Calls the base class process method for inference and monitoring
        2. Handles quarantining of samples if drift is detected
        3. Updates the adaptive handler with labeled samples
        4. Manages adaptation tracking
        
        Args:
            input_data: Input data to process (raw data or pre-extracted features)
            label: Optional ground truth label for the sample
            timestamp: Optional timestamp for the sample (default: current time)
            sample_id: Optional unique ID for the sample (default: generated UUID)
            metadata: Optional metadata for the sample
            extract_features: Whether to extract features or use input_data as features
            
        Returns:
            Dictionary with prediction and adaptation results
        """
        # First, use the base class process method for inference and monitoring
        result = super().process(
            input_data=input_data,
            label=label,
            timestamp=timestamp,
            sample_id=sample_id,
            metadata=metadata,
            extract_features=extract_features
        )
        
        # Extract features if we need them for further processing
        if extract_features:
            features = self.feature_extractor.extract_features(input_data)
        else:
            features = input_data
        
        # Get predictions and confidence if we need to reuse them
        prediction = result["prediction"]
        confidence = result.get("confidence")
        
        # Create feature sample if we don't already have it
        # (we might have created it in the base class)
        sample = FeatureSample(
            features=features,
            label=label,
            prediction=prediction,
            timestamp=timestamp or time.time(),
            sample_id=sample_id or str(uuid.uuid4()),
            metadata=metadata or {}
        )
        
        # Add adaptation-specific fields to result
        result.update({
            "adaptation_occurred": False,
            "drift_detected": False,
            "quarantined": False
        })
        
        # Handle autonomous drift detection and quarantine
        autonomous_drift_detected = result.get("autonomous_drift_detected", False)
        drift_info = result.get("drift_info")
        
        # Add to quarantine if drift detected and quarantine is enabled
        if autonomous_drift_detected and self.enable_quarantine and self.quarantine_buffer:
            # Check if sample should be quarantined based on strategy
            if self.quarantine_buffer.should_quarantine(drift_info, sample):
                try:
                    # Add to quarantine buffer
                    self.quarantine_buffer.add_sample(
                        features=features,
                        prediction=prediction,
                        drift_info=drift_info,
                        confidence=confidence,
                        raw_data=input_data if hasattr(input_data, "__len__") and len(input_data) < 1000 else None,
                        metadata=metadata,
                        sample_id=sample.sample_id
                    )
                    
                    # Log the quarantine event
                    if self.adaptation_tracker:
                        self.adaptation_tracker.log_quarantine(
                            sample_id=sample.sample_id,
                            reason=f"Autonomous drift detected by {drift_info.get('detector', 'unknown')}",
                            prediction=prediction,
                            confidence=confidence,
                            detector_id=drift_info.get("detector")
                        )
                    
                    # Update counter and result
                    self.n_samples_quarantined += 1
                    result["quarantined"] = True
                    logger.debug(f"Quarantined sample {sample.sample_id} due to {drift_info.get('detector', 'unknown')}")
                except Exception as e:
                    logger.warning(f"Error adding sample to quarantine: {str(e)}")
        
        # Increment samples since quarantine check
        if self.enable_quarantine and self.heuristic_adapter:
            self.samples_since_quarantine_check += 1
            
            # Periodically check quarantine
            if self.samples_since_quarantine_check >= self.quarantine_check_interval:
                self.run_heuristic_adaptation_cycle()
        
        # Check if server sync is needed
        if self.enable_server_sync and self.sync_client:
            current_time = time.time()
            if current_time - self.last_sync_time >= self.sync_interval:
                self.trigger_server_sync()
        
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
                
                # Create a snapshot if state manager is available and drift was detected
                if self.state_manager and adaptation_event.drift_detected:
                    try:
                        snapshot_id = self.create_snapshot(f"Adaptation triggered by labeled sample: {adaptation_event.event_type}")
                        if snapshot_id:
                            logger.debug(f"Created snapshot {snapshot_id} after adaptation")
                    except Exception as e:
                        logger.warning(f"Error creating snapshot after adaptation: {str(e)}")
                
                # Notify adaptation callbacks
                for callback in self.adaptation_callbacks:
                    try:
                        callback(adaptation_event, result)
                    except Exception as e:
                        logger.warning(f"Error in adaptation callback: {str(e)}")
        
        return result
    
    def provide_feedback(
        self, 
        features: np.ndarray,
        label: Any,
        is_validated_label: bool = False,
        sample_id: Optional[str] = None,
        timestamp: Optional[float] = None,
        confidence: Optional[float] = None,
        metadata: Optional[Dict[str, Any]] = None
    ) -> Optional[AdaptationEvent]:
        """Provide feedback for a sample to update the adaptive models.
        
        This method allows external sources (like server validation or user input)
        to provide ground truth labels for samples, enabling adaptation.
        
        Args:
            features: Feature vector for the sample
            label: Ground truth label for the sample
            is_validated_label: Whether this label comes from a validated source
            sample_id: Optional sample ID (generated if not provided)
            timestamp: Optional timestamp (current time if not provided)
            confidence: Optional confidence score for the label
            metadata: Optional metadata for the sample
            
        Returns:
            AdaptationEvent if adaptation occurred, None otherwise
        """
        if not self.adaptation_enabled:
            logger.info("Adaptation is disabled, feedback ignored")
            return None
        
        # Process timestamp and sample_id
        current_timestamp = timestamp or time.time()
        current_sample_id = sample_id or str(uuid.uuid4())
        
        # Create feature sample
        sample = FeatureSample(
            features=features,
            label=label,
            prediction=None,  # We don't have a prediction here
            timestamp=current_timestamp,
            sample_id=current_sample_id,
            metadata=metadata or {}
        )
        
        # Get prediction for tracking purposes
        try:
            prediction = self.classifier.predict(np.array([features]))[0]
            sample.prediction = prediction
            
            # Prepare operational metadata with KNN debug information
            op_metadata = metadata or {}
            if isinstance(self.classifier, LightweightKNN) and hasattr(self.classifier, '_thread_local'):
                try:
                    # Add KNN debug information if available (only the structured data)
                    if hasattr(self.classifier._thread_local, 'neighbors_debug'):
                        op_metadata['knn_neighbors'] = self.classifier._thread_local.neighbors_debug
                except Exception as e:
                    logger.warning(f"Error retrieving KNN debug info: {str(e)}")
                    
            # Track in operational monitor for feedback samples
            self.operational_monitor.track_inference(
                input_id=current_sample_id,
                prediction=prediction,
                confidence=confidence,
                latency_ms=None,  # We don't have latency for feedback samples
                metadata=op_metadata,
                timestamp=current_timestamp,
                features=features  # Include features for drift detection
            )
        except Exception as e:
            logger.warning(f"Error getting prediction for feedback sample: {str(e)}")
        
        # Create a snapshot before adaptation if state manager is available
        # (especially important for validated labels that could trigger major adaptation)
        snapshot_id = None
        if self.state_manager and is_validated_label:
            try:
                snapshot_id = self.create_snapshot(f"Pre-feedback adaptation snapshot")
                if snapshot_id:
                    logger.debug(f"Created snapshot {snapshot_id} before processing validated feedback")
            except Exception as e:
                logger.warning(f"Error creating snapshot before feedback: {str(e)}")
        
        # Process the sample through the handler
        adaptation_event = self.handler.provide_feedback(
            features=features,
            label=label,
            is_validated_label=is_validated_label,
            sample_id=sample.sample_id,
            timestamp=sample.timestamp
        )
        
        # Track adaptation if it occurred
        if adaptation_event is not None:
            self.n_samples_adapted += 1
            
            # Log in the adaptation tracker
            if self.adaptation_tracker:
                if is_validated_label:
                    self.adaptation_tracker.log_server_validation(
                        sample_id=sample.sample_id,
                        heuristic_label=None,  # No heuristic label in this case
                        validated_label=label,
                        is_correct=(sample.prediction == label) if sample.prediction is not None else False,
                        confidence=confidence
                    )
                
                self.adaptation_tracker.log_adaptation_event(adaptation_event)
            
            # Notify adaptation callbacks
            for callback in self.adaptation_callbacks:
                try:
                    callback(adaptation_event, {
                        "sample_id": sample.sample_id,
                        "is_validated_label": is_validated_label,
                        "pre_snapshot_id": snapshot_id
                    })
                except Exception as e:
                    logger.warning(f"Error in adaptation callback: {str(e)}")
        
        return adaptation_event
    
    def run_heuristic_adaptation_cycle(self) -> Dict[str, Any]:
        """Run a complete heuristic adaptation cycle.
        
        This method:
        1. Processes the quarantine buffer to find potential drift patterns
        2. Creates snapshots before applying heuristic adaptations
        3. Applies pseudo-labels to quarantined samples
        4. Adapts the classifier based on these pseudo-labels
        5. Tracks all adaptation events
        
        Returns:
            Dictionary with adaptation results
        """
        if not self.enable_quarantine or not self.quarantine_buffer or not self.heuristic_adapter:
            logger.warning("Cannot run heuristic adaptation cycle: requirements not met")
            return {
                "success": False,
                "reason": "Quarantine or heuristic adapter not available"
            }
        
        if not self.adaptation_enabled:
            logger.info("Adaptation is disabled, heuristic adaptation cycle skipped")
            return {
                "success": False,
                "reason": "Adaptation is disabled"
            }
        
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
                logger.debug("No samples adapted by heuristic")
                return {
                    "success": True,
                    "samples_adapted": 0,
                    "reason": "No eligible samples found"
                }
            
            logger.info(f"Applied heuristic adaptation to {len(adapted_samples)} samples")
            
            # Create adaptation event and log it
            adaptation_event = self.heuristic_adapter.create_adaptation_event(adapted_samples)
            
            # Group samples by new label for better tracking
            grouped_samples = {}
            for sample in adapted_samples:
                label = sample.label
                if label not in grouped_samples:
                    grouped_samples[label] = []
                grouped_samples[label].append(sample.sample_id)
            
            # Create a snapshot for each group
            snapshot_ids = {}
            if self.state_manager:
                for label, sample_ids in grouped_samples.items():
                    try:
                        snapshot_id = self.create_snapshot(
                            f"Pre-heuristic adaptation snapshot for label {label}"
                        )
                        if snapshot_id:
                            snapshot_ids[label] = snapshot_id
                            logger.debug(f"Created snapshot {snapshot_id} for heuristic label {label}")
                    except Exception as e:
                        logger.warning(f"Error creating snapshot for heuristic label {label}: {str(e)}")
            
            # Log heuristic adaptations
            if self.adaptation_tracker:
                for label, sample_ids in grouped_samples.items():
                    # Get original labels from adapted samples for tracking
                    original_labels = {}
                    for sample in adapted_samples:
                        if sample.sample_id in sample_ids:
                            # Use prediction as original label (before adaptation)
                            original_labels[sample.sample_id] = sample.prediction
                    
                    # Determine most common original label
                    original_label = None
                    if original_labels:
                        label_counts = {}
                        for orig_label in original_labels.values():
                            if orig_label is not None:
                                label_counts[orig_label] = label_counts.get(orig_label, 0) + 1
                        if label_counts:
                            original_label = max(label_counts.items(), key=lambda x: x[1])[0]
                    
                    self.adaptation_tracker.log_heuristic_adaptation(
                        samples=sample_ids,
                        original_label=original_label,
                        new_label=label,
                        pre_snapshot_id=snapshot_ids.get(label),
                        cluster_size=len(sample_ids)
                    )
            
            # Log the overall adaptation event
            if self.adaptation_tracker:
                self.adaptation_tracker.log_adaptation_event(adaptation_event)
            
            # Get samples for adaptation
            adaptation_samples = self.heuristic_adapter.get_samples_for_adaptation(
                min_confidence=self.heuristic_confidence_threshold
            )
            
            # Apply adaptations if we have samples
            adapted_count = 0
            if adaptation_samples:
                for fs in adaptation_samples:
                    # Only use samples with a label (pseudo or validated)
                    if fs.label is not None:
                        # Update the classifier
                        self.handler.add_sample(fs)
                        adapted_count += 1
                
                # Update statistics
                self.n_samples_adapted += adapted_count
                
                # Update metrics for heuristic evaluation
                self.heuristic_adapter.update_metrics(self.handler.get_metrics())
                
                # Check if rollback is needed
                if (self.heuristic_adapter.rollback_needed() and 
                    self.config.get("enable_rollback", True) and
                    self.state_manager):
                    
                    logger.warning("Heuristic adaptation had negative impact, rolling back")
                    
                    # Find the most recent snapshot
                    snapshots = list(snapshot_ids.values())
                    if snapshots:
                        # Sort by time (assuming snapshot IDs contain timestamps or are sequential)
                        snapshots.sort()
                        latest_snapshot = snapshots[0]
                        
                        # Load the snapshot
                        self.load_snapshot(latest_snapshot, reason="Negative impact rollback")
                        
                        # Log the rollback
                        if self.adaptation_tracker:
                            self.adaptation_tracker.log_rollback(
                                snapshot_id=latest_snapshot,
                                reason="Negative impact detected in heuristic adaptation",
                                samples_affected=sum(grouped_samples.values(), []),
                                triggered_by="heuristic"
                            )
            
            # Update heuristic adapter with known classes if classifier was modified
            if hasattr(self.classifier, "classes_"):
                self.heuristic_adapter.update_known_classes(set(self.classifier.classes_))
            
            return {
                "success": True,
                "samples_labeled": len(adapted_samples),
                "samples_adapted": adapted_count,
                "event_type": adaptation_event.event_type,
                "snapshot_ids": snapshot_ids
            }
            
        except Exception as e:
            logger.error(f"Error in heuristic adaptation cycle: {str(e)}")
            return {
                "success": False,
                "error": str(e)
            }
    
    def trigger_server_sync(self) -> Dict[str, Any]:
        """Manually trigger synchronization with the TinySphere server.
        
        This method:
        1. Synchronizes quarantined samples with the server
        2. Gets validation results for previously sent samples
        3. Applies validation results to update the classifier
        4. Reports drift events to the server
        
        This method is automatically called periodically based on the sync_interval,
        but can also be called manually to force a synchronization.
        
        Returns:
            Dictionary with synchronization results
        """
        if not self.enable_server_sync or not self.sync_client:
            logger.warning("Server synchronization is disabled or sync client not available")
            return {
                "success": False,
                "reason": "Server sync not enabled or client not available"
            }
        
        logger.info("Triggering server synchronization")
        self.last_sync_time = time.time()
        
        try:
            # Step 1: Sync quarantine buffer
            quarantine_sync_result = self.sync_client.sync_quarantine() if self.enable_quarantine else None
            
            # Step 2: Sync drift events
            drift_sync_result = self.sync_client.sync_drift_events() if self.autonomous_monitors else None
            
            # Step 3: Process validation results
            validation_results = []
            if quarantine_sync_result and self.enable_quarantine and self.quarantine_buffer:
                validation_count = quarantine_sync_result.get("validation_results_received", 0)
                if validation_count > 0:
                    validation_results = self.quarantine_buffer.get_validated_samples()
                    
                    # Apply validations to the classifier
                    if validation_results and self.adaptation_enabled:
                        for sample in validation_results:
                            if sample.label is not None:
                                self.provide_feedback(
                                    features=sample.features,
                                    label=sample.label,
                                    is_validated_label=True,
                                    sample_id=sample.sample_id,
                                    timestamp=sample.timestamp
                                )
            
            return {
                "success": True,
                "quarantine_sync": quarantine_sync_result,
                "drift_sync": drift_sync_result,
                "validation_results_processed": len(validation_results),
                "sync_time": self.last_sync_time
            }
            
        except Exception as e:
            logger.error(f"Error during server synchronization: {str(e)}")
            return {
                "success": False,
                "error": str(e)
            }
    
    def create_snapshot(self, reason: str, metadata: Optional[Dict[str, Any]] = None) -> Optional[str]:
        """Create a snapshot of the current state.
        
        This method creates a versioned snapshot of the current state that can be used
        for rollback if needed. It captures the state of the classifier, handler,
        and other adaptable components.
        
        Args:
            reason: Reason for creating this snapshot
            metadata: Additional metadata to include
            
        Returns:
            Snapshot ID if successful, None otherwise
        """
        if not self.state_manager:
            logger.warning("Cannot create snapshot: state manager not available")
            return None
        
        try:
            # Get component states
            component_states = {}
            
            # Get classifier state if available
            if hasattr(self.classifier, "get_state"):
                component_states["classifier"] = self.classifier.get_state()
            
            # Get handler state if available
            if hasattr(self.handler, "get_state"):
                component_states["handler"] = self.handler.get_state()
                
            # Get autonomous detector states if available
            if self.autonomous_monitors:
                detector_states = []
                for detector in self.autonomous_monitors:
                    if hasattr(detector, "get_state"):
                        detector_states.append({
                            "type": type(detector).__name__,
                            "state": detector.get_state()
                        })
                component_states["detectors"] = detector_states
            
            # Add default metadata
            if metadata is None:
                metadata = {}
                
            metadata.update({
                "timestamp": time.time(),
                "n_samples_processed": self.n_samples_processed,
                "n_samples_adapted": self.n_samples_adapted,
                "n_samples_quarantined": self.n_samples_quarantined,
                "n_autonomous_drift_detected": self.n_autonomous_drift_detected
            })
            
            # Create the snapshot
            snapshot_id = self.state_manager.create_snapshot(reason, metadata)
            if snapshot_id:
                logger.info(f"Created snapshot {snapshot_id}: {reason}")
                return snapshot_id
            else:
                logger.warning(f"Failed to create snapshot: {reason}")
                return None
                
        except Exception as e:
            logger.error(f"Error creating snapshot: {str(e)}")
            return None
    
    def load_snapshot(self, snapshot_id: str, reason: str = "Manual rollback") -> bool:
        """Load a snapshot and restore the state from it.
        
        Args:
            snapshot_id: ID of the snapshot to load
            reason: Reason for loading this snapshot
            
        Returns:
            True if successful, False otherwise
        """
        if not self.state_manager:
            logger.warning("Cannot load snapshot: state manager not available")
            return False
        
        try:
            # Load the snapshot
            component_states = self.state_manager.load_snapshot(snapshot_id, reason=reason)
            if not component_states:
                logger.warning(f"Failed to load snapshot {snapshot_id}")
                return False
            
            # Restore classifier state if available
            if "classifier" in component_states and hasattr(self.classifier, "set_state"):
                self.classifier.set_state(component_states["classifier"])
                
            # Restore handler state if available
            if "handler" in component_states and hasattr(self.handler, "set_state"):
                self.handler.set_state(component_states["handler"])
                
            # Restore detector states if available
            if "detectors" in component_states and self.autonomous_monitors:
                detector_map = {type(d).__name__: d for d in self.autonomous_monitors}
                for detector_state in component_states["detectors"]:
                    detector_type = detector_state.get("type")
                    state = detector_state.get("state")
                    if detector_type in detector_map and hasattr(detector_map[detector_type], "set_state"):
                        detector_map[detector_type].set_state(state)
            
            # Restore metadata if available
            if "metadata" in component_states:
                metadata = component_states["metadata"]
                self.n_samples_processed = metadata.get("n_samples_processed", self.n_samples_processed)
                self.n_samples_adapted = metadata.get("n_samples_adapted", self.n_samples_adapted)
                self.n_samples_quarantined = metadata.get("n_samples_quarantined", self.n_samples_quarantined)
                self.n_autonomous_drift_detected = metadata.get("n_autonomous_drift_detected", self.n_autonomous_drift_detected)
            
            # Log the rollback
            if self.adaptation_tracker:
                self.adaptation_tracker.log_rollback(
                    snapshot_id=snapshot_id,
                    reason=reason,
                    triggered_by="manual"
                )
                
            logger.info(f"Successfully loaded snapshot {snapshot_id}: {reason}")
            return True
            
        except Exception as e:
            logger.error(f"Error loading snapshot {snapshot_id}: {str(e)}")
            return False
    
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
        # Get base class statistics
        stats = super().get_statistics()
        
        # Add adaptive pipeline statistics
        stats.update({
            "n_samples_adapted": self.n_samples_adapted,
            "adaptation_rate": self.n_samples_adapted / self.n_samples_processed if self.n_samples_processed > 0 else 0,
            "adaptation_enabled": self.adaptation_enabled,
            "handler_type": type(self.handler).__name__,
            "quarantine_enabled": self.enable_quarantine,
            "heuristic_adaptation_enabled": self.enable_heuristic_adaptation,
            "n_samples_quarantined": self.n_samples_quarantined,
            "server_sync_enabled": self.enable_server_sync,
            "last_server_sync": self.last_sync_time
        })
        
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
        
        # Add server sync statistics if enabled
        if self.enable_server_sync and self.sync_client:
            try:
                sync_status = self.sync_client.get_sync_status()
                stats["server_sync"] = sync_status
            except Exception as e:
                logger.warning(f"Error getting sync status: {str(e)}")
                stats["server_sync"] = {"error": str(e)}
        
        return stats
    
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
        if self.adaptation_tracker:
            self.adaptation_tracker.log_metrics(metrics)
    
    def _on_adaptation_event(self, event: AdaptationEvent) -> None:
        """Internal callback for adaptation events.
        
        Args:
            event: Adaptation event
        """
        # Log event to adaptation tracker
        if self.adaptation_tracker:
            self.adaptation_tracker.log_adaptation_event(event)
    
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
                    self.run_heuristic_adaptation_cycle()
            
            except Exception as e:
                logger.error(f"Error in quarantine processing loop: {str(e)}")
    
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
                    if hasattr(self.quarantine_buffer, "close"):
                        self.quarantine_buffer.close()
                        logger.debug("Closed quarantine buffer")
                except Exception as e:
                    logger.warning(f"Error closing quarantine buffer: {str(e)}")
            
            # Close sync client if available
            if self.enable_server_sync and self.sync_client:
                try:
                    if hasattr(self.sync_client, "close"):
                        self.sync_client.close()
                        logger.debug("Closed sync client")
                except Exception as e:
                    logger.warning(f"Error closing sync client: {str(e)}")
            
            # Clean up adaptation tracker
            if self.adaptation_tracker:
                try:
                    if hasattr(self.adaptation_tracker, "join"):
                        self.adaptation_tracker.join()
                        logger.debug("Joined adaptation tracker")
                except Exception as e:
                    logger.warning(f"Error joining adaptation tracker: {str(e)}")
            
            # Clean up state manager
            if self.state_manager:
                try:
                    if hasattr(self.state_manager, "cleanup"):
                        self.state_manager.cleanup()
                        logger.debug("Cleaned up state manager")
                except Exception as e:
                    logger.warning(f"Error cleaning up state manager: {str(e)}")
            
            # Save final state if configured
            if self.state_manager and self.config.get("save_state_on_cleanup", True):
                try:
                    snapshot_id = self.create_snapshot(
                        reason="Final state on cleanup",
                        metadata={"cleanup": True, "final": True}
                    )
                    if snapshot_id:
                        logger.info(f"Saved final state during cleanup: {snapshot_id}")
                except Exception as e:
                    logger.warning(f"Error saving final state during cleanup: {str(e)}")
            
            # Call base class cleanup
            super().cleanup()
            
            logger.info("AdaptivePipeline cleanup complete")
        except Exception as e:
            logger.error(f"Error during cleanup: {str(e)}")