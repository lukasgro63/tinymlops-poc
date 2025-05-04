#!/usr/bin/env python3
"""
Asynchronous Stone Detector using TensorFlow Lite
------------------------------------------------
Provides asynchronous functionality to detect stones in images using a TensorFlow Lite model.
Inference runs in a separate thread pool to avoid blocking.
"""

import asyncio
import logging
import os
from concurrent.futures import ThreadPoolExecutor
from typing import List, Tuple

import cv2
import numpy as np
import tflite_runtime.interpreter as tflite

logger = logging.getLogger("stone_detector.detector")

class StoneDetector:
    def __init__(self, model_path: str, labels_path: str, threshold: float = 0.5, executor: ThreadPoolExecutor = None):
        """
        Initialize the asynchronous stone detector.
        
        Args:
            model_path (str): Path to the TensorFlow Lite model
            labels_path (str): Path to the labels file
            threshold (float): Confidence threshold for detection
            executor (ThreadPoolExecutor): Thread pool executor for running blocking operations
        """
        self.model_path = model_path
        self.labels_path = labels_path
        self.threshold = threshold
        self.interpreter = None
        self.input_details = None
        self.output_details = None
        self.labels = []
        
        # Use provided executor or create a new one
        self.executor = executor or ThreadPoolExecutor(max_workers=2)
        
        # Load the TFLite model and labels (synchronously during init)
        self._load_model()
        self._load_labels()
        
        logger.info(f"Async stone detector initialized with model: {model_path}, labels: {labels_path}")
    
    def _load_model(self):
        """Load the TensorFlow Lite model"""
        try:
            # Check if model exists
            if not os.path.exists(self.model_path):
                raise FileNotFoundError(f"Model file not found: {self.model_path}")
            
            # Load the TFLite model
            self.interpreter = tflite.Interpreter(model_path=self.model_path)
            self.interpreter.allocate_tensors()
            
            # Get input and output details
            self.input_details = self.interpreter.get_input_details()
            self.output_details = self.interpreter.get_output_details()
            
            # Get model input shape
            self.input_shape = self.input_details[0]['shape']
            input_dtype = self.input_details[0]['dtype']
            logger.info(f"Model loaded. Input shape: {self.input_shape}, Input dtype: {input_dtype}")
            
            # Log more detailed information about the input
            logger.debug(f"Input details: {self.input_details}")
        
        except Exception as e:
            logger.error(f"Failed to load model: {e}", exc_info=True)
            raise
    
    def _load_labels(self):
        """Load labels from the labels file"""
        try:
            # Check if labels file exists
            if not os.path.exists(self.labels_path):
                raise FileNotFoundError(f"Labels file not found: {self.labels_path}")
            
            # Load labels
            with open(self.labels_path, 'r') as f:
                self.labels = [line.strip() for line in f.readlines()]
            
            logger.info(f"Loaded {len(self.labels)} labels: {self.labels}")
        
        except Exception as e:
            logger.error(f"Failed to load labels: {e}", exc_info=True)
            # Default label if none available
            self.labels = ["stone"]
    
    def preprocess_image(self, image: np.ndarray) -> np.ndarray:
        """
        Preprocess an image for the TensorFlow Lite model.
        
        Args:
            image (numpy.ndarray): The input image in BGR format
            
        Returns:
            numpy.ndarray: The preprocessed image
        """
        # Resize to input shape
        input_height, input_width = self.input_shape[1], self.input_shape[2]
        processed_img = cv2.resize(image, (input_width, input_height))
        
        # Convert BGR to RGB if needed
        if self.input_shape[3] == 3:  # Check if model expects 3 channels (RGB)
            processed_img = cv2.cvtColor(processed_img, cv2.COLOR_BGR2RGB)
        
        # Check the input data type required by the model
        input_dtype = self.input_details[0]['dtype']
        
        # Process according to the required data type
        if input_dtype == np.uint8:
            # Model expects uint8 (0-255 range)
            processed_img = processed_img.astype(np.uint8)
        else:
            # Model expects float32 (0-1 range)
            processed_img = processed_img.astype(np.float32) / 255.0
        
        # Add batch dimension
        processed_img = np.expand_dims(processed_img, axis=0)
        
        # Log the data type used for debugging
        logger.debug(f"Model input dtype: {input_dtype}, Image dtype: {processed_img.dtype}")
        
        return processed_img
    
    def _detect_sync(self, image: np.ndarray) -> List[Tuple[int, float, Tuple[int, int, int, int]]]:
        """
        Synchronous detection method (runs in thread pool)
        
        Args:
            image (numpy.ndarray): The input image in BGR format
            
        Returns:
            list: List of detections
        """
        # Check if model is loaded
        if self.interpreter is None:
            logger.error("Model not loaded")
            return []
        
        try:
            # Preprocess the image
            processed_img = self.preprocess_image(image)
            
            # Set the input tensor
            self.interpreter.set_tensor(self.input_details[0]['index'], processed_img)
            
            # Run inference
            self.interpreter.invoke()
            
        except Exception as e:
            logger.error(f"Error during inference preparation: {e}", exc_info=True)
            return []
        
        # Get detection results
        try:
            # For object detection models
            boxes = self.interpreter.get_tensor(
                self.output_details[0]['index'])[0]  # Boxes
            classes = self.interpreter.get_tensor(
                self.output_details[1]['index'])[0]  # Class IDs
            scores = self.interpreter.get_tensor(
                self.output_details[2]['index'])[0]  # Confidence scores
            
            # For SSD models, there might be a 4th output with number of detections
            if len(self.output_details) >= 4:
                num_detections = int(self.interpreter.get_tensor(
                    self.output_details[3]['index'])[0])
                boxes = boxes[:num_detections]
                classes = classes[:num_detections]
                scores = scores[:num_detections]
            
            # Filter by confidence threshold
            valid_detections = []
            img_height, img_width = image.shape[:2]
            
            for i, (box, class_id, score) in enumerate(zip(boxes, classes, scores)):
                if score >= self.threshold:
                    # Convert box coordinates [y1, x1, y2, x2] to [x, y, width, height]
                    y1, x1, y2, x2 = box
                    x = int(x1 * img_width)
                    y = int(y1 * img_height)
                    width = int((x2 - x1) * img_width)
                    height = int((y2 - y1) * img_height)
                    
                    valid_detections.append((int(class_id), float(score), (x, y, width, height)))
            
            logger.debug(f"Detected {len(valid_detections)} stones in image")
            return valid_detections
        
        except Exception as e:
            # Handle models with different output formats (like classification models)
            logger.warning(f"Standard object detection outputs not found, trying classification format: {e}")
            
            try:
                # For classification models
                scores = self.interpreter.get_tensor(self.output_details[0]['index'])[0]
                
                # Find the class with highest confidence
                class_id = np.argmax(scores)
                confidence = scores[class_id]
                
                if confidence >= self.threshold:
                    # For classification, use the whole image as bounding box
                    height, width = image.shape[:2]
                    bbox = (0, 0, width, height)
                    return [(int(class_id), float(confidence), bbox)]
                else:
                    return []
            
            except Exception as e2:
                logger.error(f"Failed to interpret model output: {e2}", exc_info=True)
                return []
    
    async def detect_async(self, image: np.ndarray) -> List[Tuple[int, float, Tuple[int, int, int, int]]]:
        """
        Detect stones in an image asynchronously.
        
        Args:
            image (numpy.ndarray): The input image in BGR format
            
        Returns:
            list: List of detections, each as (class_id, confidence, bounding_box)
                  where bounding_box is (x, y, width, height)
        """
        loop = asyncio.get_event_loop()
        return await loop.run_in_executor(self.executor, self._detect_sync, image)
    
    def draw_detections(self, image: np.ndarray, detections: List[Tuple[int, float, Tuple[int, int, int, int]]]) -> np.ndarray:
        """
        Draw detection boxes and labels on an image.
        
        Args:
            image (numpy.ndarray): The input image
            detections (list): List of detections from the detect method
            
        Returns:
            numpy.ndarray: The image with detections drawn on it
        """
        result_img = image.copy()
        
        for class_id, confidence, bbox in detections:
            # Get label text
            label = self.labels[class_id] if class_id < len(self.labels) else f"Class {class_id}"
            label_text = f"{label}: {confidence:.2f}"
            
            # Get bounding box coordinates
            x, y, width, height = bbox
            
            # Draw bounding box
            cv2.rectangle(result_img, (x, y), (x + width, y + height), (0, 255, 0), 2)
            
            # Draw label background
            text_size = cv2.getTextSize(label_text, cv2.FONT_HERSHEY_SIMPLEX, 0.5, 2)[0]
            cv2.rectangle(result_img, (x, y - text_size[1] - 5), (x + text_size[0], y), (0, 255, 0), -1)
            
            # Draw label text
            cv2.putText(result_img, label_text, (x, y - 5), cv2.FONT_HERSHEY_SIMPLEX, 0.5, (0, 0, 0), 2)
        
        return result_img