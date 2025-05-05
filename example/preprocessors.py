#!/usr/bin/env python3
"""
Image preprocessors for TinyLCM feature extractors
-------------------------------------------------
This module contains preprocessor functions for handling image data
before passing it to feature extractors.
"""

import numpy as np
import cv2
import logging

logger = logging.getLogger(__name__)

def resize_and_normalize(image, target_size=(160, 160)):
    """
    Resizes and normalizes the image to the target size and scale [0, 1].
    
    Args:
        image: Input image (numpy array)
        target_size: Target size as (width, height) tuple
        
    Returns:
        Resized and normalized image as float32 numpy array
    """
    try:
        # Resize image
        resized_image = cv2.resize(image, target_size, interpolation=cv2.INTER_AREA)
        
        # Convert RGBA to RGB if needed
        if len(resized_image.shape) > 2 and resized_image.shape[2] == 4:
            resized_image = cv2.cvtColor(resized_image, cv2.COLOR_RGBA2RGB)
        
        # Convert to float32 and normalize to [0, 1]
        normalized_image = resized_image.astype('float32') / 255.0
        
        # Expand dimensions for batch (if needed)
        if len(normalized_image.shape) == 3:
            normalized_image = np.expand_dims(normalized_image, axis=0)
            
        logger.debug(f"Preprocessed image from {image.shape} to {normalized_image.shape}")
        return normalized_image
        
    except Exception as e:
        logger.error(f"Error in resize_and_normalize: {e}")
        # Return a default tensor of the right shape and type
        if target_size and len(target_size) == 2:
            default_image = np.zeros((1, target_size[1], target_size[0], 3), dtype=np.float32)
            logger.warning(f"Returning default zero tensor of shape {default_image.shape}")
            return default_image
        else:
            raise

def convert_uint8_to_float32(image):
    """
    Converts UINT8 image to FLOAT32 without changing the pixel values.
    This is useful when a model expects FLOAT32 input but you don't want 
    to normalize the values.
    
    Args:
        image: Input image (numpy array)
        
    Returns:
        Image converted to float32 with original values preserved
    """
    return image.astype('float32')

def match_input_specs(image, input_shape):
    """
    Ensures the image matches the input specifications of a TFLite model.
    
    Args:
        image: Input image (numpy array)
        input_shape: Input shape from TFLite model
        
    Returns:
        Image processed to match input specifications
    """
    try:
        # Get target dimensions from input shape
        if len(input_shape) != 4:
            raise ValueError(f"Expected 4D input shape (NHWC), got {len(input_shape)}D")
            
        batch, height, width, channels = input_shape
        
        # Resize if needed
        if image.shape[0] != height or image.shape[1] != width:
            image = cv2.resize(image, (width, height), interpolation=cv2.INTER_AREA)
        
        # Ensure correct number of channels
        if len(image.shape) == 2:  # Grayscale
            if channels == 1:
                image = np.expand_dims(image, axis=-1)
            elif channels == 3:
                image = cv2.cvtColor(image, cv2.COLOR_GRAY2RGB)
        elif image.shape[2] == 3 and channels == 1:  # RGB to grayscale
            image = cv2.cvtColor(image, cv2.COLOR_RGB2GRAY)
            image = np.expand_dims(image, axis=-1)
        elif image.shape[2] == 4 and channels == 3:  # RGBA to RGB
            image = cv2.cvtColor(image, cv2.COLOR_RGBA2RGB)
        
        # Convert to float32 and normalize
        image = image.astype('float32') / 255.0
        
        # Add batch dimension if needed
        if batch == 1 and len(image.shape) == 3:
            image = np.expand_dims(image, axis=0)
            
        return image
        
    except Exception as e:
        logger.error(f"Error in match_input_specs: {e}")
        # Return a safe default
        return np.zeros(input_shape, dtype=np.float32)