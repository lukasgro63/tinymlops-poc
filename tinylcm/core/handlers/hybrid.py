from typing import Any, Dict, List, Optional, Tuple, Callable
import numpy as np
from collections import deque
import time
import copy

from tinylcm.core.data_structures import FeatureSample, AdaptationEvent
from tinylcm.core.classifiers.base import BaseAdaptiveClassifier
from tinylcm.core.handlers.base import BaseAdaptiveHandler
from tinylcm.core.drift_detection.cusum import AccuracyCUSUM
from tinylcm.core.condensing import CondensingAlgorithm
from tinylcm.utils.logging import setup_logger

logger = setup_logger(__name__)


class HybridHandler(BaseAdaptiveHandler):
    """Hybrid adaptation handler implementing Hybrid Tiny kNN (Algorithm 5).
    
    This handler combines passive and active adaptation strategies:
    - Passively updates on misclassifications like PassiveHandler
    - Actively monitors for drift using CUSUM like ActiveHandler
    - When drift is detected, filters the training set to keep only recent samples
    - Optionally applies condensing to further reduce the training set size
    
    Use this when:
    - You want both incremental updates and drift detection
    - You have labeled samples available
    - You need to handle both gradual and abrupt concept drift
    """
    
    def __init__(
        self,
        classifier: BaseAdaptiveClassifier,
        max_samples: int = 100,
        batch_size: int = 30,
        baseline_accuracy: float = 0.9,
        cusum_threshold: float = 5.0,
        cusum_delta: float = 0.25,
        enable_condensing: bool = False,
        condensing_method: str = "class_balanced",
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
            enable_condensing: Whether to apply condensing after drift
            condensing_method: Method to use for condensing ("class_balanced", "distance_based", "random")
            use_numpy: Whether to use NumPy for calculations
            metrics_callback: Optional callback for reporting metrics
            adaptation_callback: Optional callback for adaptation events
        """
        super().__init__(classifier, max_samples, metrics_callback, adaptation_callback)
        
        self.batch_size = batch_size
        self.enable_condensing = enable_condensing
        self.condensing_method = condensing_method
        self.use_numpy = use_numpy
        
        # Set up CUSUM detector
        self.drift_detector = AccuracyCUSUM(
            baseline_accuracy=baseline_accuracy,
            threshold=cusum_threshold,
            drift_magnitude=cusum_delta,
            use_numpy=use_numpy
        )
        
        # Set up accuracy buffer
        self.accuracy_buffer = deque(maxlen=batch_size)
        
        # Training buffer - stores FeatureSample objects
        self.training_samples = deque(maxlen=max_samples)
    
    def process_sample(self, sample: FeatureSample) -> Optional[AdaptationEvent]:
        """Process a new sample according to the hybrid adaptation strategy.
        
        This implements Algorithm 5 (Hybrid Tiny kNN):
        1. If the sample is misclassified, add it to the training buffer (passive)
        2. If the sample has a label, add its result to the accuracy buffer
        3. When the buffer is full, calculate accuracy and update the CUSUM detector
        4. If drift is detected:
           a. Identify the drift time
           b. Filter the training buffer to keep only samples after the drift time
           c. Optionally apply condensing to further reduce the training set size
           d. Train the classifier on the filtered samples
        
        Args:
            sample: A feature sample with extraction results and metadata
            
        Returns:
            AdaptationEvent if adaptation occurred, None otherwise
        """
        # Update counters
        self.n_samples_processed += 1
        
        # Check if the sample has a label (supervised)
        if sample.label is None:
            # Cannot adapt without a label
            logger.debug("Sample has no label, skipping adaptation")
            return None
        
        # Check if the sample has a prediction
        if sample.prediction is None:
            # Try to predict using the classifier
            predictions = self.classifier.predict(np.array([sample.features]))
            sample.prediction = predictions[0]
        
        # Check if the prediction is correct
        is_correct = not sample.is_misclassified()
        if not is_correct:
            self.n_samples_misclassified += 1
        
        # Add accuracy result to buffer (1 for correct, 0 for incorrect)
        self.accuracy_buffer.append(is_correct)
        
        # Passive adaptation: If misclassified and adaptation is enabled, add to training buffer
        passive_adaptation = False
        if sample.is_misclassified() and self.adaptation_enabled:
            # Add the sample to the training buffer
            self.training_samples.append(sample)
            passive_adaptation = True
        
        # Active adaptation: Check for drift
        active_adaptation = False
        drift_event = None
        
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
                "n_samples": self.n_samples_processed,
                "n_training_samples": len(self.training_samples)
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
                    "batch_accuracy": accuracy,
                    "n_training_samples": len(self.training_samples)
                }
                
                # Filter training samples based on the drift time
                # Keep only samples that were added after the drift point
                filtered_samples = []
                removed_count = 0
                
                for s in self.training_samples:
                    if s.timestamp >= drift_time:
                        filtered_samples.append(s)
                    else:
                        removed_count += 1
                
                # If we have enough samples after filtering
                if len(filtered_samples) > 0:
                    # If condensing is enabled, apply it
                    if self.enable_condensing and len(filtered_samples) > 3:
                        # Extract features, labels, and timestamps
                        features = np.array([s.features for s in filtered_samples])
                        labels = [s.label for s in filtered_samples]
                        timestamps = [s.timestamp for s in filtered_samples]
                        
                        # Apply condensing algorithm
                        condensed_features, condensed_labels, condensed_timestamps = CondensingAlgorithm.condense_samples(
                            features, labels, self.max_samples, self.condensing_method, self.use_numpy, timestamps
                        )
                        
                        # Create new filtered samples list
                        condensed_filtered_samples = []
                        for i in range(len(condensed_labels)):
                            # Create a FeatureSample with the condensed data
                            condensed_sample = FeatureSample(
                                features=condensed_features[i],
                                label=condensed_labels[i],
                                timestamp=condensed_timestamps[i]
                            )
                            condensed_filtered_samples.append(condensed_sample)
                        
                        removed_count += (len(filtered_samples) - len(condensed_filtered_samples))
                        filtered_samples = condensed_filtered_samples
                        
                        logger.debug(
                            f"Condensing reduced training set: "
                            f"{len(features)} -> {len(condensed_features)} samples"
                        )
                    
                    # Update the training buffer with the filtered samples
                    self.training_samples = deque(filtered_samples, maxlen=self.max_samples)
                    
                    # Extract features and labels for training
                    features = np.array([s.features for s in self.training_samples])
                    labels = [s.label for s in self.training_samples]
                    
                    # Train the classifier on the filtered samples
                    self.classifier.fit(features, labels)
                    
                    # Increment adaptation counter
                    self.n_adaptations += 1
                    
                    # Reset the drift detector
                    self.drift_detector.reset()
                    
                    # Get performance after adaptation
                    performance_after = {
                        "accuracy": 1.0 - (self.n_samples_misclassified / self.n_samples_processed),
                        "n_samples": self.n_samples_processed,
                        "n_training_samples": len(self.training_samples)
                    }
                    
                    # Create adaptation event
                    drift_event = AdaptationEvent(
                        event_type="hybrid_active",
                        samples_added=0,
                        samples_removed=removed_count,
                        drift_detected=True,
                        drift_point_index=drift_point_index,
                        performance_before=performance_before,
                        performance_after=performance_after,
                        metadata={
                            "drift_time": drift_time,
                            "batch_accuracy": accuracy,
                            "condensing_applied": self.enable_condensing,
                            "condensing_method": self.condensing_method if self.enable_condensing else None
                        }
                    )
                    
                    # Report adaptation event
                    self.report_adaptation(drift_event)
                    
                    active_adaptation = True
                    
                    logger.info(
                        f"Hybrid (active) adaptation: trained with {len(self.training_samples)} samples "
                        f"after drift at point {drift_time} (batch accuracy: {accuracy:.4f})"
                    )
        
        # If we performed any adaptation, train the classifier
        if passive_adaptation and not active_adaptation:
            # For passive adaptation without drift detection
            # Get performance before adaptation (only if this is the first adaptation)
            performance_before = {
                "accuracy": 1.0 - (self.n_samples_misclassified / self.n_samples_processed),
                "n_samples": self.n_samples_processed,
                "n_training_samples": len(self.training_samples)
            }
            
            # Extract features and labels from training buffer
            features = np.array([s.features for s in self.training_samples])
            labels = [s.label for s in self.training_samples]
            
            # Train the classifier on the entire buffer
            self.classifier.fit(features, labels)
            
            # Increment adaptation counter
            self.n_adaptations += 1
            
            # Get performance after adaptation
            performance_after = {
                "accuracy": 1.0 - (self.n_samples_misclassified / self.n_samples_processed),
                "n_samples": self.n_samples_processed,
                "n_training_samples": len(self.training_samples)
            }
            
            # Create adaptation event
            event = AdaptationEvent(
                event_type="hybrid_passive",
                samples_added=1,
                samples_removed=0,
                drift_detected=False,
                performance_before=performance_before,
                performance_after=performance_after
            )
            
            # Report adaptation event
            self.report_adaptation(event)
            
            logger.debug(
                f"Hybrid (passive) adaptation: added 1 sample, "
                f"training set size: {len(self.training_samples)}, "
                f"current accuracy: {performance_after['accuracy']:.4f}"
            )
            
            return event
        
        # Return the drift event if it occurred
        return drift_event
    
    def get_state(self) -> Dict[str, Any]:
        """Get the current state of the handler.
        
        Returns:
            Dict containing the serializable state
        """
        base_state = super().get_state()
        
        # Serialize training samples
        training_samples = []
        for sample in self.training_samples:
            training_samples.append(sample.to_dict())
        
        # Serialize accuracy buffer
        accuracy_buffer = list(self.accuracy_buffer)
        
        # Get drift detector state
        drift_detector_state = self.drift_detector.get_state()
        
        handler_state = {
            "type": "HybridHandler",
            "batch_size": self.batch_size,
            "enable_condensing": self.enable_condensing,
            "condensing_method": self.condensing_method,
            "use_numpy": self.use_numpy,
            "training_samples": training_samples,
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
        self.enable_condensing = state.get("enable_condensing", self.enable_condensing)
        self.condensing_method = state.get("condensing_method", self.condensing_method)
        self.use_numpy = state.get("use_numpy", self.use_numpy)
        
        # Restore training samples
        self.training_samples = deque(maxlen=self.max_samples)
        
        training_samples = state.get("training_samples", [])
        for sample_dict in training_samples:
            sample = FeatureSample.from_dict(sample_dict)
            self.training_samples.append(sample)
        
        # Restore accuracy buffer
        self.accuracy_buffer = deque(maxlen=self.batch_size)
        
        accuracy_buffer = state.get("accuracy_buffer", [])
        for result in accuracy_buffer:
            self.accuracy_buffer.append(result)
        
        # Restore drift detector
        drift_detector_state = state.get("drift_detector", {})
        self.drift_detector.set_state(drift_detector_state)
        
        # If we have training samples, re-fit the classifier
        if len(self.training_samples) > 0:
            features = np.array([s.features for s in self.training_samples])
            labels = [s.label for s in self.training_samples]
            self.classifier.fit(features, labels)