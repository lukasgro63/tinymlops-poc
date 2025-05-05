#!/usr/bin/env python3
"""
Image preprocessors for TinyLCM feature extractors
-------------------------------------------------
This module contains preprocessor functions for handling image data
before passing it to feature extractors.
"""

import numpy as np
import cv2

def resize_and_normalize(image, target_size=(160, 160)):
    """
    Resizes and normalizes the image to the target size and scale [0, 1].
    
    Args:
        image: Input image (numpy array)
        target_size: Target size as (width, height) tuple
        
    Returns:
        Resized and normalized image as float32 numpy array
    """
    # Resize image
    resized_image = cv2.resize(image, target_size, interpolation=cv2.INTER_AREA)
    
    # Convert to float32 and normalize to [0, 1]
    normalized_image = resized_image.astype('float32') / 255.0
    
    # Expand dimensions for batch (if needed)
    if len(normalized_image.shape) == 3 and normalized_image.shape[0] != 1:
        normalized_image = np.expand_dims(normalized_image, axis=0)
        
    return normalized_image

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