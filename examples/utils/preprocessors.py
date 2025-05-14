#!/usr/bin/env python3
"""
Image Preprocessors
-----------------
Utilities for image preprocessing optimized for quantized TensorFlow Lite models.
Provides functions for resizing and properly preparing input tensors for inference.
"""

import logging
from typing import Dict, List, Optional, Tuple, Union

import cv2
import numpy as np

logger = logging.getLogger(__name__)

def flatten_feature_tensor(features: np.ndarray) -> np.ndarray:
    """Flattens a multi-dimensional feature tensor to a 1D vector.
    
    This is particularly useful for converting convolutional feature maps
    (e.g., from the penultimate layer of a CNN) into a format suitable for KNN.
    
    Args:
        features: Multi-dimensional feature tensor (e.g., [7, 7, 1280])
        
    Returns:
        1D flattened feature vector
    """
    # If already 1D, return as is
    if len(features.shape) <= 1:
        return features
        
    # Flatten the tensor to 1D
    return features.flatten()


def resize_image(
    image: np.ndarray,
    target_size: Tuple[int, int] = (224, 224),
    keep_aspect_ratio: bool = False,
    padding_color: Tuple[int, int, int] = (0, 0, 0)
) -> np.ndarray:
    """Resize image to the target size.

    Args:
        image: Input image as numpy array (BGR or RGB)
        target_size: Target size (width, height)
        keep_aspect_ratio: Whether to maintain aspect ratio (will pad if True)
        padding_color: Color for padding (BGR or RGB, matching image)

    Returns:
        Resized image as numpy array
    """
    if image is None:
        raise ValueError("Input image is None")

    # Ensure we have a 3-channel image (RGB/BGR)
    # Sometimes picamera2 returns RGBA or other formats
    if len(image.shape) > 2:
        if image.shape[2] == 4:  # RGBA image
            image = image[:, :, :3]  # Take only the first 3 channels (RGB)
        elif image.shape[2] != 3:
            # Convert to 3 channels if we have something else (e.g., grayscale with alpha)
            image = cv2.cvtColor(image, cv2.COLOR_GRAY2BGR)
    else:
        # Convert single channel grayscale to 3 channel
        image = cv2.cvtColor(image, cv2.COLOR_GRAY2BGR)

    original_height, original_width = image.shape[:2]
    target_width, target_height = target_size

    if keep_aspect_ratio:
        # Calculate the scaling factor to keep aspect ratio
        scale = min(target_width / original_width, target_height / original_height)
        new_width = int(original_width * scale)
        new_height = int(original_height * scale)

        # Resize image
        resized = cv2.resize(image, (new_width, new_height), interpolation=cv2.INTER_AREA)

        # Create target-sized image with padding
        result = np.full((target_height, target_width, 3), padding_color, dtype=np.uint8)

        # Calculate position to place the resized image
        x_offset = (target_width - new_width) // 2
        y_offset = (target_height - new_height) // 2

        # Insert the resized image
        result[y_offset:y_offset+new_height, x_offset:x_offset+new_width] = resized

        return result
    else:
        # Simple resize without preserving aspect ratio
        return cv2.resize(image, target_size, interpolation=cv2.INTER_AREA)


def prepare_input_tensor_quantized(
    image: np.ndarray,
    input_details: Dict,
    convert_to_rgb: bool = True
) -> np.ndarray:
    """Prepare image as input tensor for quantized TFLite models.

    Args:
        image: Input image as numpy array (BGR by default from OpenCV)
        input_details: TFLite interpreter input details for normalization parameters
        convert_to_rgb: Whether to convert BGR to RGB (set True for most models)

    Returns:
        Preprocessed image ready for TFLite inference
    """
    if image is None:
        raise ValueError("Input image is None")

    # Ensure we have a 3-channel image
    if len(image.shape) > 2:
        if image.shape[2] == 4:  # RGBA image
            image = image[:, :, :3]  # Take only the first 3 channels (RGB/BGR)
        elif image.shape[2] != 3:
            # Convert to 3 channels if we have something else
            image = cv2.cvtColor(image, cv2.COLOR_GRAY2BGR)
    else:
        # Convert single channel grayscale to 3 channel
        image = cv2.cvtColor(image, cv2.COLOR_GRAY2BGR)

    # Convert from BGR to RGB if requested (most models expect RGB)
    if convert_to_rgb and image.shape[-1] == 3:
        processed = cv2.cvtColor(image, cv2.COLOR_BGR2RGB)
    else:
        processed = image.copy()

    # Get details about the input tensor's shape and data type
    input_shape = input_details['shape']
    input_dtype = input_details['dtype']

    # Get the quantization parameters if available
    if 'quantization' in input_details and len(input_details['quantization']) == 2:
        quantization_scale, quantization_zero_point = input_details['quantization']
        is_quantized = quantization_scale > 0
    else:
        is_quantized = False

    # Reshape image to match input tensor dimensions
    batch_size, height, width, channels = input_shape

    # Ensure image has correct size
    if processed.shape[:2] != (height, width):
        processed = resize_image(processed, (width, height))

    # Ensure the number of channels matches the expected input
    if processed.shape[2] != channels:
        logger.warning(f"Channel mismatch: Got {processed.shape[2]} but model expects {channels}. Adjusting...")
        if channels == 1 and processed.shape[2] == 3:
            processed = cv2.cvtColor(processed, cv2.COLOR_RGB2GRAY)
            processed = np.expand_dims(processed, axis=2)  # Add channel dimension back
        elif channels == 3 and processed.shape[2] == 1:
            processed = cv2.cvtColor(processed, cv2.COLOR_GRAY2RGB)
        else:
            # Take only the required number of channels
            processed = processed[:, :, :channels]

    # Get the appropriate shape for the tensor
    if len(input_shape) == 4:
        # Add batch dimension if needed
        processed = np.expand_dims(processed, axis=0)

    # Final check to ensure shape matches exactly what the model expects
    if processed.shape[-1] != channels:
        logger.error(f"Channel dimension mismatch: {processed.shape} vs expected {input_shape}")
        raise ValueError(f"Channel dimension mismatch: Got {processed.shape[-1]} but expected {channels}")

    # Convert to appropriate data type
    if is_quantized:
        logger.debug(f"Processing quantized model input with scale={quantization_scale}, zero_point={quantization_zero_point}")

        if input_dtype == np.uint8:
            # For uint8 quantized models (most common)
            # Input tensor is normalized as: real_input = (quantized_input - zero_point) * scale
            # So for quantization, we do: quantized_input = real_input / scale + zero_point
            # For most models, this is already done by the interpreter, just convert to uint8
            processed = processed.astype(np.uint8)
        elif input_dtype == np.int8:
            # For int8 quantized models
            # We may need to shift the range
            processed = (processed - 128).astype(np.int8)
        else:
            logger.warning(f"Unsupported quantized dtype: {input_dtype}, using as is")
            processed = processed.astype(input_dtype)
    else:
        # For floating-point models
        # Normalize to [0, 1] range by default
        processed = processed.astype(np.float32) / 255.0

    # Log the final shape for debugging
    logger.debug(f"Final input tensor shape: {processed.shape}, dtype: {processed.dtype}")

    return processed


def extract_features_from_interpreter(
    interpreter: 'tf.lite.Interpreter',
    output_layer_index: Optional[int] = None,
    input_tensor: Optional[np.ndarray] = None
) -> np.ndarray:
    """Extract features from a TFLite model at a specified layer.
    
    Args:
        interpreter: TFLite interpreter with the model loaded
        output_layer_index: Index of the layer to extract features from (None for final output)
        input_tensor: Input tensor to use (if None, uses the current input tensor)
        
    Returns:
        Features as numpy array
    """
    # If input tensor provided, set it
    if input_tensor is not None:
        input_details = interpreter.get_input_details()[0]
        interpreter.set_tensor(input_details['index'], input_tensor)
    
    # Run inference
    interpreter.invoke()
    
    # Get the output at the specified layer
    if output_layer_index is not None:
        # Get all output tensors
        output_details = interpreter.get_output_details()
        
        if output_layer_index < 0 or output_layer_index >= len(output_details):
            logger.warning(f"Invalid output layer index {output_layer_index}, defaulting to final output")
            output_layer_index = len(output_details) - 1
        
        # Extract the requested output tensor
        features = interpreter.get_tensor(output_details[output_layer_index]['index'])
    else:
        # Use the default output
        output_details = interpreter.get_output_details()[0]
        features = interpreter.get_tensor(output_details['index'])
    
    return features


def predict_with_threshold(
    features: np.ndarray,
    labels: List[str],
    threshold: float = 0.5,
    top_k: int = 1
) -> Tuple[List[str], List[float]]:
    """Get prediction results with confidence threshold.
    
    Args:
        features: Feature vector from model (usually softmax outputs)
        labels: List of class labels
        threshold: Confidence threshold to consider a valid prediction
        top_k: Number of top predictions to return
        
    Returns:
        Tuple of (predicted_labels, confidences)
    """
    # Ensure features has the right shape
    if len(features.shape) > 2:
        features = features.reshape(features.shape[0], -1)
    
    # If we have a batch dimension but only want one prediction
    if len(features.shape) == 2 and features.shape[0] == 1:
        features = features[0]
    
    # Get indices and values of top k predictions
    if len(features.shape) == 1:
        # Single prediction
        top_indices = np.argsort(-features)[:top_k]
        top_confidences = features[top_indices]
    else:
        # Batch of predictions (we'll just take the first one)
        top_indices = np.argsort(-features[0])[:top_k]
        top_confidences = features[0][top_indices]
    
    # Apply threshold
    valid_indices = []
    valid_confidences = []
    
    for i, conf in zip(top_indices, top_confidences):
        if conf >= threshold:
            valid_indices.append(i)
            valid_confidences.append(float(conf))
    
    # Convert indices to labels and extract the actual label without the index prefix
    # Example: "0 red" -> "red"
    predicted_labels = []
    for i in valid_indices:
        if i < len(labels):
            # Labels are formatted as "0 red", "1 green" - extract just the class name
            parts = labels[i].split(' ', 1)
            if len(parts) > 1:
                predicted_labels.append(parts[1])  # Use "red" instead of "0 red"
            else:
                predicted_labels.append(labels[i])  # Fallback to the original label
        else:
            predicted_labels.append(f"unknown_{i}")
    
    return predicted_labels, valid_confidences


# Example usage
def main():
    """Test the preprocessing utilities."""
    logging.basicConfig(level=logging.DEBUG)
    
    # Create a sample image
    image = np.ones((640, 480, 3), dtype=np.uint8) * 128
    
    # Sample input details for testing
    sample_input_details = {
        'shape': (1, 224, 224, 3),
        'dtype': np.uint8,
        'quantization': (0.007843137718737125, 0)  # Example quantization params
    }
    
    # Test image resizing
    resized = resize_image(image, (224, 224))
    logger.info(f"Resized image shape: {resized.shape}")
    
    # Test input tensor preparation
    input_tensor = prepare_input_tensor_quantized(resized, sample_input_details)
    logger.info(f"Input tensor shape: {input_tensor.shape}, dtype: {input_tensor.dtype}")


if __name__ == "__main__":
    main()