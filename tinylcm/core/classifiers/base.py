from abc import ABC, abstractmethod
from typing import Any, Dict, List, Optional, Union, Tuple
import numpy as np

from tinylcm.utils.logging import setup_logger

logger = setup_logger(__name__)


class BaseAdaptiveClassifier(ABC):
    """Base abstract class for all adaptive classifiers.
    
    Adaptive classifiers can learn incrementally from new data, adapting
    to changing data distributions over time.
    """
    
    @abstractmethod
    def fit(self, features: np.ndarray, labels: List[Any]) -> None:
        """Train the classifier on the provided data.
        
        This overwrites any existing training data.
        
        Args:
            features: Matrix of feature vectors, shape (n_samples, n_features)
            labels: List of corresponding labels
        """
        pass
    
    @abstractmethod
    def incremental_fit(self, features: np.ndarray, labels: List[Any]) -> None:
        """Incrementally train the classifier on new data.
        
        This preserves existing training data and adds new samples.
        
        Args:
            features: Matrix of feature vectors, shape (n_samples, n_features)
            labels: List of corresponding labels
        """
        pass
    
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
        """Predict probability distributions over classes for the provided features.
        
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