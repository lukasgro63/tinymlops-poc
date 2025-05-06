from abc import ABC, abstractmethod
from typing import Any, Dict, List, Optional, Union, Tuple, Protocol
import numpy as np

from tinylcm.utils.logging import setup_logger

logger = setup_logger(__name__)


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


class BaseFeatureExtractor(ABC):
    """Base abstract class for all feature extractors.
    
    Feature extractors transform input data into feature vectors that can be used
    for classification. This is typically done by extracting intermediate outputs
    from neural networks or by applying transformations to the input data.
    """
    
    @abstractmethod
    def extract_features(self, input_data: Any) -> np.ndarray:
        """Extract feature vectors from input data.
        
        Args:
            input_data: The input data to extract features from
            
        Returns:
            Feature vector as a numpy array
        """
        pass
    
    @abstractmethod
    def get_state(self) -> Dict[str, Any]:
        """Get the current state of the feature extractor.
        
        Returns:
            Dict containing the serializable state
        """
        pass
    
    @abstractmethod
    def set_state(self, state: Dict[str, Any]) -> None:
        """Restore the feature extractor's state.
        
        Args:
            state: Previously saved state dictionary
        """
        pass


class NullFeatureExtractor(BaseFeatureExtractor):
    """A feature extractor that passes through input data unchanged.
    
    This is useful when the input data is already in a format suitable for
    classification, such as pre-extracted feature vectors.
    """
    
    def extract_features(self, input_data: Any) -> np.ndarray:
        """Pass through input data unchanged.
        
        Args:
            input_data: The input data, expected to be a numpy array
            
        Returns:
            The input data unchanged
        """
        # Ensure input is a numpy array
        if not isinstance(input_data, np.ndarray):
            if isinstance(input_data, (list, tuple)):
                input_data = np.array(input_data)
            else:
                raise TypeError(f"Expected numpy array or list/tuple, got {type(input_data)}")
                
        return input_data
    
    def get_state(self) -> Dict[str, Any]:
        """Get the current state of the feature extractor.
        
        Returns:
            Dict containing the serializable state
        """
        return {"type": "NullFeatureExtractor"}
    
    def set_state(self, state: Dict[str, Any]) -> None:
        """Restore the feature extractor's state.
        
        Args:
            state: Previously saved state dictionary
        """
        # Nothing to restore
        pass


class PreprocessingFeatureExtractor(BaseFeatureExtractor):
    """A feature extractor that applies preprocessing steps to input data.
    
    This feature extractor applies a sequence of preprocessing functions to
    transform input data before feature extraction.
    """
    
    def __init__(self, preprocessors: Optional[List[callable]] = None):
        """Initialize with preprocessing functions.
        
        Args:
            preprocessors: List of preprocessing functions to apply in sequence
        """
        self.preprocessors = preprocessors or []
    
    def extract_features(self, input_data: Any) -> np.ndarray:
        """Apply preprocessing steps and extract features.
        
        Args:
            input_data: The input data to preprocess and extract features from
            
        Returns:
            Processed feature vector
        """
        data = input_data
        
        # Apply each preprocessor in sequence
        for preprocessor in self.preprocessors:
            data = preprocessor(data)
        
        # Ensure output is a numpy array
        if not isinstance(data, np.ndarray):
            data = np.array(data)
        
        return data
    
    def get_state(self) -> Dict[str, Any]:
        """Get the current state of the feature extractor.
        
        Returns:
            Dict containing the serializable state
        """
        # Note: Cannot serialize preprocessor functions
        # This method should be overridden by subclasses that use specific preprocessors
        return {
            "type": "PreprocessingFeatureExtractor",
            "n_preprocessors": len(self.preprocessors)
        }
    
    def set_state(self, state: Dict[str, Any]) -> None:
        """Restore the feature extractor's state.
        
        Args:
            state: Previously saved state dictionary
        """
        # Preprocessors cannot be serialized directly
        # This method should be overridden by subclasses
        pass
    
    def add_preprocessor(self, preprocessor: callable) -> None:
        """Add a preprocessing function to the pipeline.
        
        Args:
            preprocessor: Function to add to the preprocessing pipeline
        """
        self.preprocessors.append(preprocessor)
        
    def clear_preprocessors(self) -> None:
        """Clear all preprocessing functions."""
        self.preprocessors = []