from abc import ABC, abstractmethod
from typing import Any, Dict, List, Optional, Union, Tuple, Set, Callable
import numpy as np
from collections import deque
import time
import copy

from tinylcm.core.data_structures import FeatureSample, AdaptationEvent
from tinylcm.core.classifiers.base import BaseAdaptiveClassifier
from tinylcm.core.base import AdaptiveClassifier
from tinylcm.core.drift_detection.cusum import AccuracyCUSUM
from tinylcm.utils.logging import setup_logger

logger = setup_logger(__name__)


class BaseAdaptiveHandler(ABC):
    """Base abstract class for all adaptive handlers.
    
    Adaptive handlers implement different strategies for model adaptation
    such as passive, active, or hybrid approaches. They control when and how
    the classifier is updated with new training samples.
    """
    
    def __init__(
        self,
        classifier: BaseAdaptiveClassifier,
        max_samples: int = 100,
        metrics_callback: Optional[Callable[[Dict[str, Any]], None]] = None,
        adaptation_callback: Optional[Callable[[AdaptationEvent], None]] = None
    ):
        """Initialize the adaptive handler.
        
        Args:
            classifier: The adaptive classifier to manage
            max_samples: Maximum number of training samples to store
            metrics_callback: Optional callback for reporting metrics
            adaptation_callback: Optional callback for adaptation events
        """
        self.classifier = classifier
        self.max_samples = max_samples
        self.metrics_callback = metrics_callback
        self.adaptation_callback = adaptation_callback
        
        # Track performance metrics
        self.n_samples_processed = 0
        self.n_samples_misclassified = 0
        self.n_adaptations = 0
        
        # Flag to enable or disable adaptation
        self.adaptation_enabled = True
    
    def get_classifier(self) -> BaseAdaptiveClassifier:
        """Get the internal classifier used by this handler.
        
        Returns:
            The adaptive classifier instance
        """
        return self.classifier
    
    @abstractmethod
    def process_sample(self, sample: FeatureSample) -> Optional[AdaptationEvent]:
        """Process a new sample according to the adaptation strategy.
        
        Args:
            sample: A feature sample with extraction results and metadata
            
        Returns:
            AdaptationEvent if adaptation occurred, None otherwise
        """
        pass
    
    def disable_adaptation(self) -> None:
        """Disable adaptation of the classifier."""
        self.adaptation_enabled = False
        logger.debug("Adaptation disabled")
    
    def enable_adaptation(self) -> None:
        """Enable adaptation of the classifier."""
        self.adaptation_enabled = True
        logger.debug("Adaptation enabled")
    
    def report_metrics(self, metrics: Dict[str, Any]) -> None:
        """Report metrics to the registered callback.
        
        Args:
            metrics: Dictionary of metrics to report
        """
        if self.metrics_callback:
            self.metrics_callback(metrics)
    
    def report_adaptation(self, event: AdaptationEvent) -> None:
        """Report an adaptation event to the registered callback.
        
        Args:
            event: The adaptation event that occurred
        """
        if self.adaptation_callback:
            self.adaptation_callback(event)
    
    def get_state(self) -> Dict[str, Any]:
        """Get the current state of the handler.
        
        Returns:
            Dict containing the serializable state
        """
        return {
            "max_samples": self.max_samples,
            "n_samples_processed": self.n_samples_processed,
            "n_samples_misclassified": self.n_samples_misclassified,
            "n_adaptations": self.n_adaptations,
            "adaptation_enabled": self.adaptation_enabled
        }
    
    def set_state(self, state: Dict[str, Any]) -> None:
        """Restore the handler's state.
        
        Args:
            state: Previously saved state dictionary
        """
        self.max_samples = state.get("max_samples", self.max_samples)
        self.n_samples_processed = state.get("n_samples_processed", 0)
        self.n_samples_misclassified = state.get("n_samples_misclassified", 0)
        self.n_adaptations = state.get("n_adaptations", 0)
        self.adaptation_enabled = state.get("adaptation_enabled", True)