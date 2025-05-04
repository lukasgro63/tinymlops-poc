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


class ActiveHandler(BaseAdaptiveHandler):
    """Active adaptation handler implementing Active Tiny kNN (Algorithm 4).
    
    This handler monitors the classification accuracy and detects concept drift
    using CUSUM. When drift is detected, it identifies recent samples associated
    with the drift and trains a new classifier from those samples.
    
    Use this when:
    - You want to adapt only upon detected drift
    - You have labeled samples available
    - The concept drift is abrupt rather than gradual
    """
    
    def __init__(
        self,
        classifier: BaseAdaptiveClassifier,
        max_samples: int = 100,
        batch_size: int = 30,
        baseline_accuracy: float = 0.9,
        cusum_threshold: float = 5.0,
        cusum_delta: float = 0.25,
        history_window_size: Optional[int] = None,
        metrics_callback: Optional[Callable[[Dict[str, Any]], None]] = None,
        adaptation_callback: Optional[Callable[[AdaptationEvent], None]] = None
    ):
        """Initialize the active handler.
        
        Args:
            classifier: The adaptive classifier to manage
            max_samples: Maximum number of training samples to store
            batch_size: Size of the batch for accuracy calculation
            baseline_accuracy: Expected accuracy of the classifier
            cusum_threshold: Detection threshold for the CUSUM algorithm
            cusum_delta: Expected magnitude of accuracy drop
            history_window_size: Size of the history window, defaults to max_samples * 2
            metrics_callback: Optional callback for reporting metrics
            adaptation_callback: Optional callback for adaptation events
        """
        super().__init__(classifier, max_samples, metrics_callback, adaptation_callback)
        
        self.batch_size = batch_size
        
        # Set up CUSUM detector
        self.drift_detector = AccuracyCUSUM(
            baseline_accuracy=baseline_accuracy,
            threshold=cusum_threshold,
            drift_magnitude=cusum_delta
        )
        
        # Set up accuracy buffer
        self.accuracy_buffer = deque(maxlen=batch_size)
        
        # Set up history window
        self.history_window_size = history_window_size or (max_samples * 2)
        self.history_window = deque(maxlen=self.history_window_size)
    
    def process_sample(self, sample: FeatureSample) -> Optional[AdaptationEvent]:
        """Process a new sample according to the active adaptation strategy.
        
        This implements Algorithm 4 (Active Tiny kNN):
        1. Store the sample in the history window
        2. If the sample has a label, add its result to the accuracy buffer
        3. When the buffer is full, calculate accuracy and update the CUSUM detector
        4. If drift is detected, identify the drift time and use samples after that
           point to train a new classifier
        
        Args:
            sample: A feature sample with extraction results and metadata
            
        Returns:
            AdaptationEvent if adaptation occurred, None otherwise
        """
        # Update counters
        self.n_samples_processed += 1
        
        # Add sample to history window
        self.history_window.append(sample)
        
        # Check if the sample has a label (supervised)
        if sample.label is None:
            # Cannot calculate accuracy without a label
            logger.debug("Sample has no label, skipping accuracy calculation")
            return None
        
        # Check if the sample has a prediction
        if sample.prediction is None:
            # Try to predict using the classifier
            predictions = self.classifier.predict(np.array([sample.features]))
            sample.prediction = predictions[0]
        
        # Update misclassification counter
        if sample.is_misclassified():
            self.n_samples_misclassified += 1
        
        # Add accuracy result to buffer (1 for correct, 0 for incorrect)
        is_correct = not sample.is_misclassified()
        self.accuracy_buffer.append(is_correct)
        
        # If accuracy buffer is full, update the drift detector
        if len(self.accuracy_buffer) == self.batch_size:
            # Calculate accuracy for the current batch
            accuracy = sum(self.accuracy_buffer) / self.batch_size
            
            # Update the CUSUM detector
            drift_detected, drift_point_index = self.drift_detector.update(accuracy)
            
            # Clear the accuracy buffer for the next batch
            self.accuracy_buffer.clear()
            
            # Report metrics
            metrics = {
                "batch_accuracy": accuracy,
                "cumulative_accuracy": 1.0 - (self.n_samples_misclassified / self.n_samples_processed),
                "drift_detected": drift_detected,
                "n_samples": self.n_samples_processed
            }
            self.report_metrics(metrics)
            
            logger.debug(
                f"Batch accuracy: {accuracy:.4f}, "
                f"cumulative accuracy: {metrics['cumulative_accuracy']:.4f}"
            )
            
            # If drift is detected and adaptation is enabled
            if drift_detected and self.adaptation_enabled:
                # Estimate the drift point in terms of samples
                drift_time = self.drift_detector.estimate_drift_timing(self.batch_size)
                
                # Get performance before adaptation
                performance_before = {
                    "accuracy": 1.0 - (self.n_samples_misclassified / self.n_samples_processed),
                    "n_samples": self.n_samples_processed,
                    "batch_accuracy": accuracy
                }
                
                # Calculate the cutoff index in the history window
                cutoff_index = max(0, len(self.history_window) - (self.n_samples_processed - drift_time))
                
                # Get samples after the drift point
                new_samples = list(self.history_window)[cutoff_index:]
                
                # Filter samples without labels
                new_training_samples = [s for s in new_samples if s.label is not None]
                
                # If we have enough labeled samples after the drift point
                if len(new_training_samples) > 0:
                    # Extract features and labels
                    features = np.array([s.features for s in new_training_samples])
                    labels = [s.label for s in new_training_samples]
                    
                    # Train a new classifier
                    self.classifier.fit(features, labels)
                    
                    # Increment adaptation counter
                    self.n_adaptations += 1
                    
                    # Reset the drift detector
                    self.drift_detector.reset()
                    
                    # Get performance after adaptation
                    performance_after = {
                        "accuracy": 1.0 - (self.n_samples_misclassified / self.n_samples_processed),
                        "n_samples": self.n_samples_processed,
                        "n_training_samples": len(new_training_samples)
                    }
                    
                    # Create adaptation event
                    event = AdaptationEvent(
                        event_type="active",
                        samples_added=len(new_training_samples),
                        samples_removed=0,
                        drift_detected=True,
                        drift_point_index=drift_point_index,
                        performance_before=performance_before,
                        performance_after=performance_after,
                        metadata={
                            "drift_time": drift_time,
                            "cutoff_index": cutoff_index,
                            "batch_accuracy": accuracy
                        }
                    )
                    
                    # Report adaptation event
                    self.report_adaptation(event)
                    
                    logger.info(
                        f"Active adaptation: trained with {len(new_training_samples)} samples "
                        f"after drift at point {drift_time} (batch accuracy: {accuracy:.4f})"
                    )
                    
                    return event
                else:
                    logger.warning(
                        f"Drift detected at point {drift_time}, but no labeled samples "
                        f"available after drift point"
                    )
        
        return None
    
    def get_state(self) -> Dict[str, Any]:
        """Get the current state of the handler.
        
        Returns:
            Dict containing the serializable state
        """
        base_state = super().get_state()
        
        # Serialize history window
        history_window = []
        for sample in self.history_window:
            history_window.append(sample.to_dict())
        
        # Serialize accuracy buffer
        accuracy_buffer = list(self.accuracy_buffer)
        
        # Get drift detector state
        drift_detector_state = self.drift_detector.get_state()
        
        handler_state = {
            "type": "ActiveHandler",
            "batch_size": self.batch_size,
            "history_window_size": self.history_window_size,
            "history_window": history_window,
            "accuracy_buffer": accuracy_buffer,
            "drift_detector": drift_detector_state
        }
        
        # Combine base and handler-specific state
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
        self.history_window_size = state.get("history_window_size", self.history_window_size)
        
        # Restore history window
        self.history_window = deque(maxlen=self.history_window_size)
        
        history_window = state.get("history_window", [])
        for sample_dict in history_window:
            sample = FeatureSample.from_dict(sample_dict)
            self.history_window.append(sample)
        
        # Restore accuracy buffer
        self.accuracy_buffer = deque(maxlen=self.batch_size)
        
        accuracy_buffer = state.get("accuracy_buffer", [])
        for result in accuracy_buffer:
            self.accuracy_buffer.append(result)
        
        # Restore drift detector
        drift_detector_state = state.get("drift_detector", {})
        self.drift_detector.set_state(drift_detector_state)