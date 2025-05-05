from typing import Any, Dict, List, Optional, Tuple, Union, Callable
import numpy as np
import os
import copy

from tinylcm.core.feature_extractors.base import BaseFeatureExtractor
from tinylcm.utils.logging import setup_logger

logger = setup_logger(__name__)

# Try to import tflite_runtime first (for embedded devices)
try:
    import tflite_runtime.interpreter as tflite
    USING_TFLITE_RUNTIME = True
    logger.info("Using tflite_runtime for TFLiteFeatureExtractor")
except ImportError:
    # Fall back to tensorflow.lite if tflite_runtime not available
    try:
        import tensorflow as tf
        USING_TFLITE_RUNTIME = False
        logger.info("Using tensorflow.lite for TFLiteFeatureExtractor")
    except ImportError:
        # Define a flag but don't raise an error yet - we'll do that when the class is instantiated
        USING_TFLITE_RUNTIME = None
        logger.warning("Neither tflite_runtime nor tensorflow could be imported. TFLiteFeatureExtractor will not work until one of them is installed.")


class TFLiteFeatureExtractor(BaseFeatureExtractor):
    """Feature extractor that uses a TFLite model to extract features.
    
    This extractor loads a TensorFlow Lite model and extracts intermediate
    layer outputs as feature vectors. This is useful for transfer learning
    and feature extraction from pre-trained models.
    """
    
    def __init__(
        self,
        model_path: str,
        feature_layer_index: int = -2,
        input_tensor_index: int = 0,
        output_tensor_index: Optional[int] = None,
        preprocessors: Optional[List[Callable]] = None,
        normalize_features: bool = True
    ):
        """Initialize the TFLite feature extractor.
        
        Args:
            model_path: Path to the TFLite model file
            feature_layer_index: Index of the layer to extract features from
                                 (default: -2, typically the penultimate layer)
            input_tensor_index: Index of the input tensor (default: 0)
            output_tensor_index: Index of the output tensor (default: determined by feature_layer_index)
            preprocessors: List of preprocessing functions to apply to the input
            normalize_features: Whether to normalize feature vectors to unit length
        """
        self.model_path = model_path
        self.feature_layer_index = feature_layer_index
        self.input_tensor_index = input_tensor_index
        self.output_tensor_index = output_tensor_index
        self.preprocessors = preprocessors or []
        self.normalize_features = normalize_features
        
        # Lazy loading of TFLite Interpreter
        self._interpreter = None
        
        logger.debug(
            f"Initialized TFLiteFeatureExtractor with model={model_path}, "
            f"feature_layer_index={feature_layer_index}"
        )
    
    def _ensure_interpreter_loaded(self) -> None:
        """Ensure the TFLite interpreter is loaded."""
        if self._interpreter is None:
            if USING_TFLITE_RUNTIME is None:
                raise ImportError(
                    "TensorFlow Lite is required for TFLiteFeatureExtractor. "
                    "Install it with 'pip install tflite-runtime' (recommended for embedded devices) "
                    "or 'pip install tensorflow' (for development environments)."
                )
                
            try:
                # Initialize the interpreter based on which library we have
                if USING_TFLITE_RUNTIME:
                    # Using tflite_runtime (lightweight for embedded devices)
                    self._interpreter = tflite.Interpreter(model_path=self.model_path)
                else:
                    # Using full tensorflow (heavier but more feature-rich)
                    self._interpreter = tf.lite.Interpreter(model_path=self.model_path)
                
                # Allocate tensors
                self._interpreter.allocate_tensors()
                
                # Get input and output details
                self._input_details = self._interpreter.get_input_details()
                self._output_details = self._interpreter.get_output_details()
                
                # Set up for feature extraction - here's the key part for using output as features
                if self.output_tensor_index is None:
                    # Default to using the specified feature layer index
                    self.output_tensor_index = self.feature_layer_index
                
                # Check tensor indices
                if isinstance(self.output_tensor_index, int):
                    if abs(self.output_tensor_index) >= len(self._output_details):
                        logger.warning(
                            f"Feature layer index {self.output_tensor_index} is out of range. "
                            f"Model has {len(self._output_details)} output tensors. "
                            f"Using the last output tensor."
                        )
                        self.output_tensor_index = -1
                
                logger.debug(f"TFLite interpreter loaded: {self.model_path}")
            except Exception as e:
                logger.error(f"Failed to load TFLite model: {str(e)}")
                raise
    
    def extract_features(self, input_data: Any) -> np.ndarray:
        """Extract features from the input data using the TFLite model.
        
        IMPORTANT: This is a synchronous, blocking operation. The execution time
        depends on the size of the input data, the complexity of the model, and
        the hardware capabilities of the device. For resource-constrained devices
        like the Raspberry Pi Zero, this operation may take significant time.
        
        If low-latency responses are required in your application, consider:
        1. Running this method in a separate thread
        2. Using a simpler model or pre-extracted features
        3. Batch processing inputs when possible
        
        Args:
            input_data: Input data in the format expected by the model
            
        Returns:
            Feature vector extracted from the specified layer
        """
        self._ensure_interpreter_loaded()
        
        # Apply preprocessors if any
        processed_data = input_data
        for preprocessor in self.preprocessors:
            processed_data = preprocessor(processed_data)
        
        # Prepare input data for the model
        input_details = self._input_details[self.input_tensor_index]
        if isinstance(processed_data, np.ndarray) and processed_data.shape != input_details['shape']:
            # If batch dimension is missing, add it
            if len(processed_data.shape) == len(input_details['shape']) - 1:
                processed_data = np.expand_dims(processed_data, axis=0)
            else:
                logger.warning(
                    f"Input shape mismatch: {processed_data.shape} vs {input_details['shape']}"
                )
        
        # Set input tensor
        self._interpreter.set_tensor(input_details['index'], processed_data)
        
        # Run inference
        self._interpreter.invoke()
        
        # Get output from the feature layer
        if isinstance(self.output_tensor_index, int):
            # If the index is an integer, get the tensor directly
            if self.output_tensor_index < 0:
                # Convert negative index to positive
                tensor_index = len(self._output_details) + self.output_tensor_index
            else:
                tensor_index = self.output_tensor_index
                
            output_details = self._output_details[tensor_index]
            feature_vector = self._interpreter.get_tensor(output_details['index'])
        else:
            # If the index is a string, find the tensor by name
            for details in self._output_details:
                if details['name'] == self.output_tensor_index:
                    feature_vector = self._interpreter.get_tensor(details['index'])
                    break
            else:
                raise ValueError(f"Tensor '{self.output_tensor_index}' not found in model outputs")
        
        # Remove batch dimension if present
        if feature_vector.shape[0] == 1:
            feature_vector = feature_vector[0]
        
        # Normalize features if requested
        if self.normalize_features:
            # Compute L2 norm
            norm = np.sqrt(np.sum(feature_vector * feature_vector, axis=-1, keepdims=True))
            # Avoid division by zero
            norm = np.maximum(norm, 1e-12)
            feature_vector = feature_vector / norm
        
        return feature_vector
    
    def get_state(self) -> Dict[str, Any]:
        """Get the current state of the feature extractor.
        
        Returns:
            Dict containing the serializable state
        """
        return {
            "type": "TFLiteFeatureExtractor",
            "model_path": self.model_path,
            "feature_layer_index": self.feature_layer_index,
            "input_tensor_index": self.input_tensor_index,
            "output_tensor_index": self.output_tensor_index,
            "normalize_features": self.normalize_features,
            # Cannot serialize preprocessor functions
            "n_preprocessors": len(self.preprocessors)
        }
    
    def set_state(self, state: Dict[str, Any]) -> None:
        """Restore the feature extractor's state.
        
        Args:
            state: Previously saved state dictionary
        """
        self.model_path = state.get("model_path", self.model_path)
        self.feature_layer_index = state.get("feature_layer_index", self.feature_layer_index)
        self.input_tensor_index = state.get("input_tensor_index", self.input_tensor_index)
        self.output_tensor_index = state.get("output_tensor_index", self.output_tensor_index)
        self.normalize_features = state.get("normalize_features", self.normalize_features)
        
        # Reset the interpreter to reload with new settings
        self._interpreter = None
    
    def add_preprocessor(self, preprocessor: Callable) -> None:
        """Add a preprocessing function to the pipeline.
        
        Args:
            preprocessor: Function to add to the preprocessing pipeline
        """
        self.preprocessors.append(preprocessor)
        
    def clear_preprocessors(self) -> None:
        """Clear all preprocessing functions."""
        self.preprocessors = []
        
    def __del__(self):
        """Clean up resources."""
        # Explicitly delete the interpreter to free resources
        if hasattr(self, '_interpreter') and self._interpreter is not None:
            del self._interpreter