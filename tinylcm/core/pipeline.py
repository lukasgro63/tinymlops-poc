from typing import Any, Dict, List, Optional, Union, Tuple, Callable
import numpy as np
import time
import uuid
from collections import deque

from tinylcm.core.data_structures import FeatureSample, AdaptationEvent, AdaptiveState
from tinylcm.core.base import AdaptiveComponent, FeatureExtractor, AdaptiveClassifier, AdaptiveHandler
from tinylcm.core.feature_extractors.base import NullFeatureExtractor
from tinylcm.core.handlers.hybrid import HybridHandler
from tinylcm.core.classifiers.knn import LightweightKNN
from tinylcm.core.state_manager import StateManager
from tinylcm.core.adaptation_tracker import AdaptationTracker
from tinylcm.core.data_logger.logger import DataLogger
from tinylcm.utils.logging import setup_logger
from tinylcm.constants import DEFAULT_CONFIG

logger = setup_logger(__name__)


class AdaptivePipeline:
    """Main pipeline for adaptive learning in TinyLCM.
    
    This class integrates feature extraction, classification, and adaptation
    into a unified pipeline. It provides the main interface for interacting
    with the adaptive learning system.
    """
    
    def __init__(
        self,
        feature_extractor: Optional[FeatureExtractor] = None,
        classifier: Optional[AdaptiveClassifier] = None,
        handler: Optional[AdaptiveHandler] = None,
        state_manager: Optional[StateManager] = None,
        adaptation_tracker: Optional[AdaptationTracker] = None,
        data_logger: Optional[DataLogger] = None,
        config: Optional[Dict[str, Any]] = None
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
        
        # Initialize additional state
        self.n_samples_processed = 0
        self.n_samples_adapted = 0
        self.adaptation_enabled = True
        
        # User-defined callbacks
        self.prediction_callbacks = []
        self.adaptation_callbacks = []
        
        logger.debug(
            f"Initialized AdaptivePipeline with "
            f"extractor={type(self.feature_extractor).__name__}, "
            f"classifier={type(self.classifier).__name__}, "
            f"handler={type(self.handler).__name__}"
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
        4. Logs data and events as needed
        5. Returns the prediction results
        
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
            "drift_detected": False
        }
        
        # Notify prediction callbacks
        for callback in self.prediction_callbacks:
            try:
                callback(sample, result)
            except Exception as e:
                logger.warning(f"Error in prediction callback: {str(e)}")
        
        # If we have a label, process adaptation
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
        
        # Add default metadata
        if metadata is None:
            metadata = {}
        
        metadata.update({
            "timestamp": time.time(),
            "n_samples_processed": self.n_samples_processed,
            "n_samples_adapted": self.n_samples_adapted,
            "feature_extractor_type": type(self.feature_extractor).__name__,
            "classifier_type": type(self.classifier).__name__,
            "handler_type": type(self.handler).__name__
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
            "handler_type": type(self.handler).__name__
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
        
        return stats