from abc import ABC, abstractmethod
from typing import Any, Dict, List, Optional, Protocol, Tuple, Union, Callable
import numpy as np

from tinylcm.core.data_structures import FeatureSample, AdaptiveState, AdaptationEvent


class AdaptiveComponent(ABC):
    """Base abstract class for all adaptive learning components.
    
    This defines the common interface for components that need to be saved and loaded
    as part of an adaptive learning pipeline.
    """
    
    @abstractmethod
    def get_state(self) -> Dict[str, Any]:
        """Get the current state of this component for persistence.
        
        Returns:
            Dict containing the serializable state of this component
        """
        pass
    
    @abstractmethod
    def set_state(self, state: Dict[str, Any]) -> None:
        """Restore this component's state from a previously saved state.
        
        Args:
            state: Previously saved state dictionary to restore from
        """
        pass


class FeatureExtractor(Protocol):
    """Protocol defining the interface for feature extractors.
    
    Feature extractors transform input data into feature vectors suitable
    for use with adaptive classifiers.
    """
    
    def extract_features(self, input_data: Any) -> np.ndarray:
        """Extract feature vectors from the input data.
        
        Args:
            input_data: The input data to extract features from.
                This could be images, text, sensor data, etc.
                
        Returns:
            Feature vector as a numpy array
        """
        ...
    
    def get_state(self) -> Dict[str, Any]:
        """Get the current state of the feature extractor.
        
        Returns:
            Dict containing the serializable state
        """
        ...
    
    def set_state(self, state: Dict[str, Any]) -> None:
        """Restore the feature extractor's state.
        
        Args:
            state: Previously saved state dictionary
        """
        ...


class AdaptiveClassifier(Protocol):
    """Protocol defining the interface for adaptive classifiers.
    
    Adaptive classifiers learn incrementally from new samples and
    adapt to changing data distributions over time.
    """
    
    def fit(self, features: np.ndarray, labels: List[Any]) -> None:
        """Train the classifier on the provided data.
        
        This overwrites any existing training data.
        
        Args:
            features: Matrix of feature vectors, shape (n_samples, n_features)
            labels: List of corresponding labels
        """
        ...
    
    def incremental_fit(self, features: np.ndarray, labels: List[Any]) -> None:
        """Incrementally train the classifier on new data.
        
        This preserves existing training data and adds new samples.
        
        Args:
            features: Matrix of feature vectors, shape (n_samples, n_features)
            labels: List of corresponding labels
        """
        ...
    
    def predict(self, features: np.ndarray) -> List[Any]:
        """Predict labels for the provided feature vectors.
        
        Args:
            features: Matrix of feature vectors, shape (n_samples, n_features)
            
        Returns:
            List of predicted labels
        """
        ...
    
    def predict_proba(self, features: np.ndarray) -> np.ndarray:
        """Predict probability distributions over classes for the provided feature vectors.
        
        Args:
            features: Matrix of feature vectors, shape (n_samples, n_features)
            
        Returns:
            Matrix of class probabilities, shape (n_samples, n_classes)
        """
        ...
    
    def get_state(self) -> Dict[str, Any]:
        """Get the current state of the classifier.
        
        Returns:
            Dict containing the serializable state
        """
        ...
    
    def set_state(self, state: Dict[str, Any]) -> None:
        """Restore the classifier's state.
        
        Args:
            state: Previously saved state dictionary
        """
        ...


class DriftDetector(Protocol):
    """Protocol defining the interface for drift detectors.
    
    Drift detectors monitor data streams for changes in distribution
    that may indicate concept drift.
    """
    
    def update(self, metric: float) -> Tuple[bool, Optional[int]]:
        """Update the detector with a new observation.
        
        Args:
            metric: The metric to monitor for drift (e.g., accuracy)
            
        Returns:
            Tuple of (drift_detected, drift_point_index)
            - drift_detected: Boolean indicating if drift was detected
            - drift_point_index: Index of the drift point if detected, else None
        """
        ...
    
    def reset(self) -> None:
        """Reset the detector's state."""
        ...
    
    def get_state(self) -> Dict[str, Any]:
        """Get the current state of the detector.
        
        Returns:
            Dict containing the serializable state
        """
        ...
    
    def set_state(self, state: Dict[str, Any]) -> None:
        """Restore the detector's state.
        
        Args:
            state: Previously saved state dictionary
        """
        ...


class AdaptiveHandler(Protocol):
    """Protocol defining the interface for adaptive handlers.
    
    Adaptive handlers implement different strategies for model adaptation
    such as passive, active, or hybrid approaches.
    """
    
    def process_sample(self, sample: FeatureSample) -> AdaptationEvent:
        """Process a new sample according to the adaptation strategy.
        
        Args:
            sample: A feature sample with extraction results and metadata
            
        Returns:
            AdaptationEvent describing any adaptation that occurred
        """
        ...
    
    def get_classifier(self) -> AdaptiveClassifier:
        """Get the internal classifier used by this handler.
        
        Returns:
            The adaptive classifier instance
        """
        ...
    
    def get_state(self) -> Dict[str, Any]:
        """Get the current state of the handler.
        
        Returns:
            Dict containing the serializable state
        """
        ...
    
    def set_state(self, state: Dict[str, Any]) -> None:
        """Restore the handler's state.
        
        Args:
            state: Previously saved state dictionary
        """
        ...


class AdaptationCallback(Protocol):
    """Callback protocol for adaptation events.
    
    This is used to notify external components about adaptation events
    such as model updates and drift detection.
    """
    
    def on_adaptation(self, event: AdaptationEvent) -> None:
        """Called when an adaptation event occurs.
        
        Args:
            event: The adaptation event that occurred
        """
        ...