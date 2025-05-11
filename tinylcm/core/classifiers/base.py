from abc import ABC, abstractmethod
from typing import Any, Dict, List, Optional, Union, Tuple, Protocol
import numpy as np

from tinylcm.utils.logging import setup_logger

logger = setup_logger(__name__)


class AdaptiveClassifier(Protocol):
    """Protocol defining the interface for adaptive classifiers.
    
    Adaptive classifiers learn incrementally from new samples and
    adapt to changing data distributions over time.
    """
    
    def fit(self, features: np.ndarray, labels: List[Any], timestamps: Optional[List[float]] = None) -> None:
        """Train the classifier on the provided data.
        
        This overwrites any existing training data.
        
        Args:
            features: Matrix of feature vectors, shape (n_samples, n_features)
            labels: List of corresponding labels
            timestamps: Optional list of timestamps for each sample
        """
        ...
    
    def add_sample(self, feature: np.ndarray, label: Any, timestamp: Optional[float] = None) -> None:
        """Add a single sample to the classifier.
        
        This is more efficient than calling incremental_fit for a single sample.
        
        Args:
            feature: Feature vector, shape (n_features,)
            label: Label for the sample
            timestamp: Optional timestamp for the sample
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


class BaseAdaptiveClassifier(ABC):
    """Base abstract class for all adaptive classifiers.
    
    Adaptive classifiers can learn incrementally from new data, adapting
    to changing data distributions over time.
    """
    
    @abstractmethod
    def fit(self, features: np.ndarray, labels: List[Any], timestamps: Optional[List[float]] = None) -> None:
        """Train the classifier on the provided data.
        
        This overwrites any existing training data.
        
        Args:
            features: Matrix of feature vectors, shape (n_samples, n_features)
            labels: List of corresponding labels
            timestamps: Optional list of timestamps for each sample
        """
        pass
    
    @abstractmethod
    def add_sample(self, feature: np.ndarray, label: Any, timestamp: Optional[float] = None) -> None:
        """Add a single sample to the classifier.
        
        This is more efficient than calling incremental_fit for a single sample.
        
        Args:
            feature: Feature vector, shape (n_features,)
            label: Label for the sample
            timestamp: Optional timestamp for the sample
        """
        pass
    
    def incremental_fit(self, features: np.ndarray, labels: List[Any], timestamps: Optional[List[float]] = None) -> None:
        """Incrementally train the classifier on new data.
        
        This preserves existing training data and adds new samples.
        
        The default implementation calls add_sample for each sample,
        but subclasses should override this with a more efficient implementation if possible.
        
        Args:
            features: Matrix of feature vectors, shape (n_samples, n_features)
            labels: List of corresponding labels
            timestamps: Optional list of timestamps for each sample
        """
        if len(features) != len(labels):
            raise ValueError(f"Number of features ({len(features)}) and labels ({len(labels)}) must match")
        
        if timestamps is not None and len(timestamps) != len(features):
            raise ValueError(f"Number of timestamps ({len(timestamps)}) and features ({len(features)}) must match")
        
        for i in range(len(features)):
            if timestamps is not None:
                self.add_sample(features[i], labels[i], timestamps[i])
            else:
                self.add_sample(features[i], labels[i])
    
    @abstractmethod
    def predict(self, features: np.ndarray) -> List[Any]:
        """Predict labels for the provided feature vectors.
        
        Args:
            features: Matrix of feature vectors, shape (n_samples, n_features)
            
        Returns:
            List of predicted labels
        """
        pass
    
    @abstractmethod
    def predict_proba(self, features: np.ndarray) -> np.ndarray:
        """Predict probability distributions over classes for the provided feature vectors.
        
        Args:
            features: Matrix of feature vectors, shape (n_samples, n_features)
            
        Returns:
            Matrix of class probabilities, shape (n_samples, n_classes)
        """
        pass
    
    @abstractmethod
    def get_state(self) -> Dict[str, Any]:
        """Get the current state of the classifier.
        
        Returns:
            Dict containing the serializable state
        """
        pass
    
    @abstractmethod
    def set_state(self, state: Dict[str, Any]) -> None:
        """Restore the classifier's state.
        
        Args:
            state: Previously saved state dictionary
        """
        pass