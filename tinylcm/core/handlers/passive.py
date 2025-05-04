from typing import Any, Dict, List, Optional, Tuple, Callable
import numpy as np
from collections import deque
import time
import copy

from tinylcm.core.data_structures import FeatureSample, AdaptationEvent
from tinylcm.core.classifiers.base import BaseAdaptiveClassifier
from tinylcm.core.handlers.base import BaseAdaptiveHandler
from tinylcm.utils.logging import setup_logger

logger = setup_logger(__name__)


class PassiveHandler(BaseAdaptiveHandler):
    """Passive adaptation handler implementing Continuous Integration Training (CIT).
    
    This handler follows Algorithm 3 from the paper by incrementally updating
    the classifier when misclassifications occur. It adds misclassified samples
    to a training buffer and retrains the classifier on the entire buffer.
    
    Use this when:
    - You want continuous, incremental adaptation
    - You have labeled samples available
    - The concept drift is gradual rather than abrupt
    """
    
    def __init__(
        self,
        classifier: BaseAdaptiveClassifier,
        max_samples: int = 100,
        metrics_callback: Optional[Callable[[Dict[str, Any]], None]] = None,
        adaptation_callback: Optional[Callable[[AdaptationEvent], None]] = None
    ):
        """Initialize the passive handler.
        
        Args:
            classifier: The adaptive classifier to manage
            max_samples: Maximum number of training samples to store
            metrics_callback: Optional callback for reporting metrics
            adaptation_callback: Optional callback for adaptation events
        """
        super().__init__(classifier, max_samples, metrics_callback, adaptation_callback)
        
        # Training buffer - stores FeatureSample objects
        self.training_samples = deque(maxlen=max_samples)
    
    def process_sample(self, sample: FeatureSample) -> Optional[AdaptationEvent]:
        """Process a new sample according to the passive adaptation strategy.
        
        This implements Algorithm 3 (CIT):
        1. If the sample is misclassified, add it to the training buffer
        2. If the buffer exceeds the maximum size, remove the oldest sample
        3. Retrain the classifier on the entire buffer
        
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
        if sample.is_misclassified():
            # Increment misclassification counter
            self.n_samples_misclassified += 1
            
            # Only adapt if adaptation is enabled
            if not self.adaptation_enabled:
                logger.debug("Adaptation disabled, sample misclassified but not adapting")
                return None
            
            # Add the sample to the training buffer
            self.training_samples.append(sample)
            
            # Get performance before adaptation
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
                event_type="passive",
                samples_added=1,
                samples_removed=0,
                drift_detected=False,
                performance_before=performance_before,
                performance_after=performance_after
            )
            
            # Report adaptation event
            self.report_adaptation(event)
            
            logger.info(
                f"Passive adaptation: added 1 sample, "
                f"training set size: {len(self.training_samples)}, "
                f"current accuracy: {performance_after['accuracy']:.4f}"
            )
            
            return event
        
        return None
    
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
        
        handler_state = {
            "type": "PassiveHandler",
            "training_samples": training_samples
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
        
        # Restore training samples
        self.training_samples = deque(maxlen=self.max_samples)
        
        training_samples = state.get("training_samples", [])
        for sample_dict in training_samples:
            sample = FeatureSample.from_dict(sample_dict)
            self.training_samples.append(sample)
        
        # If we have training samples, re-fit the classifier
        if len(self.training_samples) > 0:
            features = np.array([s.features for s in self.training_samples])
            labels = [s.label for s in self.training_samples]
            self.classifier.fit(features, labels)