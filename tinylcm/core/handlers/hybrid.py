from typing import Any, Dict, List, Optional, Tuple, Callable
import numpy as np
from collections import deque
import time
import copy

from tinylcm.core.data_structures import FeatureSample, AdaptationEvent
from tinylcm.core.classifiers.base import BaseAdaptiveClassifier
from tinylcm.core.handlers.base import BaseAdaptiveHandler
from tinylcm.core.drift_detection.cusum import AccuracyCUSUM
from tinylcm.utils.logging import setup_logger

logger = setup_logger(__name__)


class HybridHandler(BaseAdaptiveHandler):
    """Unified handler for both heuristic and validated feedback.
    
    The HybridHandler processes both potential labels from on-device heuristics
    and validated labels from external sources. It combines:
    
    1. Passive Update Logic: Adds misclassified samples to the training set
       and updates the classifier incrementally.
       
    2. Active Update Logic: For validated labels only, uses CUSUM to detect
       significant accuracy drops, then filters out pre-drift samples and
       retrains the classifier.
       
    This approach allows the device to adapt autonomously using heuristics while
    still benefiting from external validation when available.
    """
    
    def __init__(
        self,
        classifier: BaseAdaptiveClassifier,
        max_samples: int = 100,
        batch_size: int = 30,
        baseline_accuracy: float = 0.9,
        cusum_threshold: float = 5.0,
        cusum_delta: float = 0.1,
        use_numpy: bool = True,
        metrics_callback: Optional[Callable[[Dict[str, Any]], None]] = None,
        adaptation_callback: Optional[Callable[[AdaptationEvent], None]] = None
    ):
        """Initialize the hybrid handler.
        
        Args:
            classifier: The adaptive classifier to manage
            max_samples: Maximum number of training samples to store
            batch_size: Size of the batch for accuracy calculation
            baseline_accuracy: Expected accuracy of the classifier
            cusum_threshold: Detection threshold for the CUSUM algorithm
            cusum_delta: Expected magnitude of accuracy drop
            use_numpy: Whether to use NumPy for calculations
            metrics_callback: Optional callback for reporting metrics
            adaptation_callback: Optional callback for adaptation events
        """
        super().__init__(classifier, max_samples, metrics_callback, adaptation_callback)
        
        self.batch_size = batch_size
        self.use_numpy = use_numpy
        
        # Set up CUSUM detector for validated labels
        self.accuracy_cusum = AccuracyCUSUM(
            baseline_accuracy=baseline_accuracy,
            threshold=cusum_threshold,
            drift_magnitude=cusum_delta,
            use_numpy=use_numpy
        )
        
        # Training buffer - stores (features, label, timestamp) tuples
        self.training_samples = deque(maxlen=max_samples)
        
        # Buffer for tracking accuracy of validated samples
        self.validated_samples = []
    
    def provide_feedback(
        self,
        features: np.ndarray,
        label: Any,
        is_validated_label: bool = False,
        sample_id: Optional[str] = None,
        timestamp: Optional[float] = None
    ) -> Optional[AdaptationEvent]:
        """Process feedback according to the hybrid adaptation strategy.
        
        This method implements the core functionality of the HybridHandler:
        1. Makes an internal prediction using the classifier
        2. For all feedback (validated or not):
           - If prediction != label, adds to training samples and updates the classifier
        3. For validated labels only:
           - Updates the CUSUM detector with accuracy
           - If drift is detected, filters out pre-drift samples and retrains
        
        Args:
            features: Feature vector
            label: Suggested label (potential or validated)
            is_validated_label: Whether this is an externally validated label
            sample_id: Optional unique identifier for the sample
            timestamp: Optional timestamp (defaults to current time)
            
        Returns:
            AdaptationEvent if adaptation occurred, None otherwise
        """
        if not self.adaptation_enabled:
            logger.debug("Adaptation is disabled, skipping feedback")
            return None
        
        # Use current time if timestamp not provided
        if timestamp is None:
            timestamp = time.time()
            
        # Make internal prediction
        internal_prediction = self.classifier.predict(np.array([features]))[0]
        
        # Track whether any adaptation occurred
        passive_adaptation = False
        active_adaptation = False
        
        # Passive Update Logic (for all feedback)
        if internal_prediction != label:
            # Add to training buffer
            self.training_samples.append((features, label, timestamp))
            
            # Get performance before adaptation
            performance_before = self._get_current_performance()
            
            # Extract features, labels, and timestamps for training
            training_data = self._extract_training_data()
            
            # Train the classifier
            if training_data:
                X, y, _ = training_data
                self.classifier.fit(X, y)
            
            # Update stats
            self.n_adaptations += 1
            passive_adaptation = True
            
            # Get performance after adaptation
            performance_after = self._get_current_performance()
            
            logger.debug(
                f"Passive adaptation: added sample with {'' if is_validated_label else 'potential '}label '{label}', "
                f"training set size: {len(self.training_samples)}"
            )
        
        # Active Update Logic (for validated labels only)
        active_event = None
        if is_validated_label:
            # Track validation result (1 for correct, 0 for incorrect)
            is_correct = (internal_prediction == label)
            self.validated_samples.append((is_correct, timestamp))
            
            # If we have enough samples, check for drift
            if len(self.validated_samples) >= self.batch_size:
                # Calculate accuracy for this batch
                batch_accuracy = sum(correct for correct, _ in self.validated_samples[-self.batch_size:]) / self.batch_size
                
                # Update CUSUM detector
                drift_detected, drift_point = self.accuracy_cusum.update({"accuracy": batch_accuracy})
                
                # If drift detected, filter training samples and retrain
                if drift_detected:
                    # Get estimated drift time
                    if drift_point is not None:
                        drift_time = self.validated_samples[-(self.batch_size - drift_point)][1]
                    else:
                        drift_time = timestamp - 3600  # Default to 1 hour ago
                    
                    # Filter out samples before the drift time
                    filtered_samples = deque(
                        [(f, l, ts) for f, l, ts in self.training_samples if ts >= drift_time],
                        maxlen=self.max_samples
                    )
                    
                    # Calculate how many samples were removed
                    removed_count = len(self.training_samples) - len(filtered_samples)
                    
                    # Update training samples
                    self.training_samples = filtered_samples
                    
                    # Get performance before adaptation
                    performance_before = self._get_current_performance()
                    
                    # Extract features, labels, and timestamps for training
                    training_data = self._extract_training_data()
                    
                    # Retrain the classifier if we have enough samples
                    if training_data and len(training_data[0]) > 0:
                        X, y, _ = training_data
                        self.classifier.fit(X, y)
                        
                        # Update stats
                        self.n_adaptations += 1
                        active_adaptation = True
                        
                        # Get performance after adaptation
                        performance_after = self._get_current_performance()
                        
                        # Create an adaptation event
                        active_event = AdaptationEvent(
                            event_type="hybrid_active_drift",
                            timestamp=timestamp,
                            samples_added=0,
                            samples_removed=removed_count,
                            drift_detected=True,
                            drift_point_index=drift_point,
                            performance_before=performance_before,
                            performance_after=performance_after,
                            metadata={
                                "drift_time": drift_time,
                                "batch_accuracy": batch_accuracy,
                                "is_validated": True,
                                "cusum_value": self.accuracy_cusum.state.cumsum
                            }
                        )
                        
                        # Report adaptation event
                        self.report_adaptation(active_event)
                        
                        # Reset CUSUM detector
                        self.accuracy_cusum.reset()
                        
                        logger.info(
                            f"Active adaptation: filtered {removed_count} samples before drift time, "
                            f"retrained with {len(filtered_samples)} samples (batch accuracy: {batch_accuracy:.4f})"
                        )
        
        # Create and return adaptation event for passive adaptation (if not already handled by active)
        if passive_adaptation and not active_adaptation:
            passive_event = AdaptationEvent(
                event_type="hybrid_passive",
                timestamp=timestamp,
                samples_added=1,
                samples_removed=0,
                drift_detected=False,
                performance_before=performance_before,
                performance_after=performance_after,
                metadata={
                    "is_validated": is_validated_label,
                    "potential_label": None if is_validated_label else label
                }
            )
            
            # Report adaptation event
            self.report_adaptation(passive_event)
            
            return passive_event
        
        return active_event
    
    def _extract_training_data(self) -> Optional[Tuple[np.ndarray, List[Any], List[float]]]:
        """Extract features, labels, and timestamps from training samples.
        
        Returns:
            Tuple of (features_array, labels_list, timestamps_list) or None if empty
        """
        if not self.training_samples:
            return None
        
        # Extract features, labels, and timestamps
        features = []
        labels = []
        timestamps = []
        
        for f, l, ts in self.training_samples:
            features.append(f)
            labels.append(l)
            timestamps.append(ts)
        
        # Convert features to numpy array
        features_array = np.array(features)
        
        return features_array, labels, timestamps
    
    def _get_current_performance(self) -> Dict[str, Any]:
        """Get current performance metrics.
        
        Returns:
            Dictionary with performance metrics
        """
        # Validation accuracy (if available)
        if self.validated_samples:
            recent_samples = min(len(self.validated_samples), 100)
            recent_results = [correct for correct, _ in self.validated_samples[-recent_samples:]]
            validation_accuracy = sum(recent_results) / len(recent_results) if recent_results else 0.0
        else:
            validation_accuracy = None
        
        return {
            "n_training_samples": len(self.training_samples),
            "n_adaptations": self.n_adaptations,
            "validation_accuracy": validation_accuracy
        }
    
    def get_training_samples(self) -> List[Tuple[np.ndarray, Any, float]]:
        """Get the current training samples.
        
        Returns:
            List of (features, label, timestamp) tuples
        """
        return list(self.training_samples)
    
    def clear_training_samples(self) -> None:
        """Clear the training samples buffer."""
        self.training_samples.clear()
        logger.debug("Cleared training samples buffer")
    
    def get_state(self) -> Dict[str, Any]:
        """Get the current state of the handler.
        
        Returns:
            Dict containing the serializable state
        """
        base_state = super().get_state()
        
        # For serialization, store only the raw values
        serializable_training_samples = []
        for features, label, timestamp in self.training_samples:
            if isinstance(features, np.ndarray):
                features_list = features.tolist()
            else:
                features_list = features
            
            serializable_training_samples.append({
                "features": features_list,
                "label": label,
                "timestamp": timestamp
            })
        
        # Get drift detector state
        drift_detector_state = None
        if hasattr(self.accuracy_cusum, "get_state"):
            drift_detector_state = self.accuracy_cusum.get_state()
        
        # Create handler-specific state
        handler_state = {
            "batch_size": self.batch_size,
            "use_numpy": self.use_numpy,
            "training_samples": serializable_training_samples,
            "validated_samples": self.validated_samples,
            "accuracy_cusum": drift_detector_state
        }
        
        # Combine states
        return {**base_state, **handler_state}
    
    def set_state(self, state: Dict[str, Any]) -> None:
        """Restore the handler's state.
        
        Args:
            state: Previously saved state dictionary
        """
        # Restore base state
        super().set_state(state)
        
        # Restore handler-specific state
        self.batch_size = state.get("batch_size", self.batch_size)
        self.use_numpy = state.get("use_numpy", self.use_numpy)
        
        # Restore training samples
        self.training_samples = deque(maxlen=self.max_samples)
        
        for sample in state.get("training_samples", []):
            features = sample.get("features")
            if features is not None and self.use_numpy:
                features = np.array(features)
            
            label = sample.get("label")
            timestamp = sample.get("timestamp")
            
            self.training_samples.append((features, label, timestamp))
        
        # Restore validated samples
        self.validated_samples = state.get("validated_samples", [])
        
        # Restore drift detector
        if hasattr(self.accuracy_cusum, "set_state"):
            drift_detector_state = state.get("accuracy_cusum", {})
            if drift_detector_state:
                self.accuracy_cusum.set_state(drift_detector_state)
        
        # Train the classifier with the restored samples
        training_data = self._extract_training_data()
        if training_data and len(training_data[0]) > 0:
            X, y, _ = training_data
            self.classifier.fit(X, y)