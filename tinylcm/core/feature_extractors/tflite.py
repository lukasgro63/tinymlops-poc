from typing import Any, Dict, List, Optional, Tuple, Union, Callable
import numpy as np
import os
import copy
import time
import threading

from tinylcm.core.feature_extractors.base import BaseFeatureExtractor
from tinylcm.core.base import AdaptiveComponent
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


class TFLiteFeatureExtractor(BaseFeatureExtractor, AdaptiveComponent):
    """Feature extractor that uses a TFLite model to extract features.
    
    This extractor loads a TensorFlow Lite model and extracts intermediate
    layer outputs as feature vectors. This is useful for transfer learning
    and feature extraction from pre-trained models.
    
    The extractor is optimized for resource-constrained devices like the 
    Raspberry Pi Zero by using TFLite runtime and implementing non-blocking 
    model loading.
    """
    
    def __init__(
        self,
        model_path: str,
        feature_layer_index: int = -2,
        input_tensor_index: int = 0,
        output_tensor_index: Optional[int] = None,
        preprocessors: Optional[List[Callable]] = None,
        normalize_features: bool = True,
        lazy_loading: bool = True,
        input_shape: Optional[List[int]] = None,
        input_dtype: str = 'float32',
        thread_safe: bool = True
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
            lazy_loading: Whether to load the model only when needed (saves memory)
            input_shape: Expected input shape (used for validation), e.g. [1, 224, 224, 3]
            input_dtype: Expected input data type ('float32', 'uint8', etc.)
            thread_safe: Whether to use thread locking for multi-threaded safety
        """
        self.model_path = model_path
        self.feature_layer_index = feature_layer_index
        self.input_tensor_index = input_tensor_index
        self.output_tensor_index = output_tensor_index
        self.preprocessors = preprocessors or []
        self.normalize_features = normalize_features
        self.lazy_loading = lazy_loading
        self.input_shape = input_shape
        self.input_dtype = input_dtype
        self.thread_safe = thread_safe
        
        # Lazy loading of TFLite Interpreter
        self._interpreter = None
        self._input_details = None
        self._output_details = None
        self._lock = threading.RLock() if thread_safe else None
        self._is_loading = False
        self._load_error = None

        # Performance metrics
        self._total_inference_time = 0
        self._inference_count = 0
        
        logger.debug(
            f"Initialized TFLiteFeatureExtractor with model={model_path}, "
            f"feature_layer_index={feature_layer_index}, lazy_loading={lazy_loading}"
        )
        
        # Preload the model if not using lazy loading
        if not lazy_loading:
            self._load_interpreter()

    def _load_interpreter(self) -> None:
        """Load the TFLite interpreter.
        
        This method loads the TensorFlow Lite model and allocates tensors.
        If thread_safe is True, this method is protected by a lock.
        """
        if self._interpreter is not None:
            return  # Already loaded
            
        # Thread safety for loading
        if self.thread_safe:
            if self._is_loading:
                # Another thread is already loading the interpreter
                # Wait until it's done or an error is raised
                while self._is_loading and self._load_error is None:
                    time.sleep(0.01)
                    
                if self._load_error is not None:
                    raise self._load_error
                    
                return
                
            with self._lock:
                if self._interpreter is not None:
                    return  # Already loaded by another thread
                self._is_loading = True
        
        try:
            if USING_TFLITE_RUNTIME is None:
                raise ImportError(
                    "TensorFlow Lite is required for TFLiteFeatureExtractor. "
                    "Install it with 'pip install tflite-runtime' (recommended for embedded devices) "
                    "or 'pip install tensorflow' (for development environments)."
                )
                
            # Check if model file exists
            if not os.path.exists(self.model_path):
                raise FileNotFoundError(f"TFLite model file not found: {self.model_path}")
                
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
            
            # Validate input shape if provided
            if self.input_shape is not None:
                actual_shape = self._input_details[self.input_tensor_index]['shape']
                # Check that shapes are compatible (allowing for dynamic dimensions)
                if len(actual_shape) != len(self.input_shape):
                    logger.warning(
                        f"Input shape mismatch: expected {self.input_shape}, "
                        f"but model expects {actual_shape}"
                    )
                else:
                    for i, (expected, actual) in enumerate(zip(self.input_shape, actual_shape)):
                        if expected is not None and actual != -1 and expected != actual:
                            logger.warning(
                                f"Input shape mismatch at dimension {i}: "
                                f"expected {expected}, but model expects {actual}"
                            )
            
            logger.debug(f"TFLite interpreter loaded: {self.model_path}")
        except Exception as e:
            logger.error(f"Failed to load TFLite model: {str(e)}")
            self._load_error = e
            raise
        finally:
            if self.thread_safe:
                self._is_loading = False

    def _ensure_interpreter_loaded(self) -> None:
        """Ensure the TFLite interpreter is loaded.
        
        This method checks if the interpreter is loaded and loads it if needed.
        """
        if self._interpreter is None:
            self._load_interpreter()

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
        start_time = time.time()
        
        # Thread safety for inference
        if self.thread_safe:
            with self._lock:
                result = self._extract_features_internal(input_data)
        else:
            result = self._extract_features_internal(input_data)
        
        # Update performance metrics
        inference_time = time.time() - start_time
        self._total_inference_time += inference_time
        self._inference_count += 1
        
        return result

    def _extract_features_internal(self, input_data: Any) -> np.ndarray:
        """Internal method to extract features.
        
        This method is called by extract_features and handles the actual feature extraction.
        """
        self._ensure_interpreter_loaded()
        
        # Apply preprocessors if any
        processed_data = input_data
        for preprocessor in self.preprocessors:
            processed_data = preprocessor(processed_data)
        
        # Prepare input data for the model
        input_details = self._input_details[self.input_tensor_index]
        
        # Convert to numpy array if needed
        if not isinstance(processed_data, np.ndarray):
            processed_data = np.array(processed_data, dtype=np.float32)
        
        # Check shape compatibility
        if processed_data.shape != input_details['shape']:
            # If batch dimension is missing, add it
            if len(processed_data.shape) == len(input_details['shape']) - 1:
                processed_data = np.expand_dims(processed_data, axis=0)
            else:
                logger.warning(
                    f"Input shape mismatch: {processed_data.shape} vs {input_details['shape']}"
                )
        
        # Check data type
        if processed_data.dtype != input_details['dtype']:
            processed_data = processed_data.astype(input_details['dtype'])
        
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
            "lazy_loading": self.lazy_loading,
            "input_shape": self.input_shape,
            "input_dtype": self.input_dtype,
            "thread_safe": self.thread_safe,
            # Cannot serialize preprocessor functions
            "n_preprocessors": len(self.preprocessors),
            # Performance metrics
            "metrics": {
                "total_inference_time": self._total_inference_time,
                "inference_count": self._inference_count,
                "avg_inference_time": self._total_inference_time / max(1, self._inference_count)
            }
        }
    
    def set_state(self, state: Dict[str, Any]) -> None:
        """Restore the feature extractor's state.
        
        Args:
            state: Previously saved state dictionary
        """
        # Save old values that we might need to check if a reload is necessary
        old_model_path = self.model_path
        old_feature_layer_index = self.feature_layer_index
        old_input_tensor_index = self.input_tensor_index
        old_output_tensor_index = self.output_tensor_index
        
        # Update state
        self.model_path = state.get("model_path", self.model_path)
        self.feature_layer_index = state.get("feature_layer_index", self.feature_layer_index)
        self.input_tensor_index = state.get("input_tensor_index", self.input_tensor_index)
        self.output_tensor_index = state.get("output_tensor_index", self.output_tensor_index)
        self.normalize_features = state.get("normalize_features", self.normalize_features)
        self.lazy_loading = state.get("lazy_loading", self.lazy_loading)
        self.input_shape = state.get("input_shape", self.input_shape)
        self.input_dtype = state.get("input_dtype", self.input_dtype)
        self.thread_safe = state.get("thread_safe", self.thread_safe)
        
        # Check if we need to reload the interpreter
        need_reload = (
            old_model_path != self.model_path or
            old_feature_layer_index != self.feature_layer_index or
            old_input_tensor_index != self.input_tensor_index or
            old_output_tensor_index != self.output_tensor_index
        )
        
        # Reset the interpreter if needed
        if need_reload and self._interpreter is not None:
            # Thread safety for resource cleanup
            if self.thread_safe:
                with self._lock:
                    self._interpreter = None
                    self._input_details = None
                    self._output_details = None
            else:
                self._interpreter = None
                self._input_details = None
                self._output_details = None
            
            # Preload if not using lazy loading
            if not self.lazy_loading:
                self._load_interpreter()
    
    def add_preprocessor(self, preprocessor: Callable) -> None:
        """Add a preprocessing function to the pipeline.
        
        Args:
            preprocessor: Function to add to the preprocessing pipeline
        """
        self.preprocessors.append(preprocessor)
        
    def clear_preprocessors(self) -> None:
        """Clear all preprocessing functions."""
        self.preprocessors = []
    
    def get_performance_metrics(self) -> Dict[str, float]:
        """Get performance metrics for this feature extractor.
        
        Returns:
            Dictionary containing performance metrics:
            - total_inference_time: Total time spent in inference
            - inference_count: Number of inferences performed
            - avg_inference_time: Average time per inference
        """
        return {
            "total_inference_time": self._total_inference_time,
            "inference_count": self._inference_count,
            "avg_inference_time": 
                self._total_inference_time / max(1, self._inference_count)
        }
    
    def get_model_info(self) -> Dict[str, Any]:
        """Get information about the loaded model.
        
        Returns:
            Dictionary containing model information:
            - model_path: Path to the model file
            - input_details: Information about model inputs
            - output_details: Information about model outputs
            - is_loaded: Whether the model is currently loaded
        """
        if not self._interpreter:
            return {
                "model_path": self.model_path,
                "is_loaded": False
            }
        
        # Get information about the model
        return {
            "model_path": self.model_path,
            "input_details": self._input_details,
            "output_details": self._output_details,
            "is_loaded": True
        }
        
    def __del__(self):
        """Clean up resources."""
        # Explicitly delete the interpreter to free resources
        if hasattr(self, '_interpreter') and self._interpreter is not None:
            del self._interpreter