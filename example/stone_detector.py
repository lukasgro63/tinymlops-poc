#!/usr/bin/env python3
"""
Stone Detector
-------------
Detects stones in images using TensorFlow Lite models. Handles the loading of the model,
preprocessing of images, and post-processing of detection results.
"""

import os
import time
import asyncio
from concurrent.futures import ThreadPoolExecutor
from typing import List, Tuple, Optional

import cv2
import numpy as np
import tflite_runtime.interpreter as tflite

class StoneDetector:
    """Detects stones in images using TensorFlow Lite."""
    
    def __init__(self, 
                 model_path: str,
                 labels_path: str,
                 threshold: float = 0.5,
                 executor: Optional[ThreadPoolExecutor] = None):
        """Initialize the stone detector.
        
        Args:
            model_path: Path to the TFLite model file
            labels_path: Path to the labels file
            threshold: Confidence threshold for detections
            executor: ThreadPoolExecutor to use for inference (optional)
        """
        self.model_path = model_path
        self.labels_path = labels_path
        self.threshold = threshold
        
        # Initialize the TFLite interpreter
        self.interpreter = self._load_model(model_path)
        
        # Get input and output details
        self.input_details = self.interpreter.get_input_details()
        self.output_details = self.interpreter.get_output_details()
        
        # Load labels
        self.labels = self._load_labels(labels_path)
        
        # Use provided executor or create a new one
        self.executor = executor or ThreadPoolExecutor(max_workers=1)
    
    def _load_model(self, model_path: str) -> tflite.Interpreter:
        """Load the TFLite model from file."""
        try:
            interpreter = tflite.Interpreter(model_path=model_path)
            interpreter.allocate_tensors()
            return interpreter
        except Exception as e:
            raise RuntimeError(f"Failed to load model: {e}")
    
    def _load_labels(self, labels_path: str) -> List[str]:
        """Load labels from file."""
        try:
            with open(labels_path, 'r') as f:
                return [line.strip() for line in f.readlines()]
        except Exception as e:
            raise RuntimeError(f"Failed to load labels: {e}")
    
    def preprocess_image(self, image: np.ndarray) -> np.ndarray:
        """Preprocess the image for the model."""
        # Get input shape from model
        input_shape = self.input_details[0]['shape']
        height, width = input_shape[1], input_shape[2]
        
        # Resize image
        resized = cv2.resize(image, (width, height))
        
        # Convert to floating point and normalize
        normalized = resized.astype(np.float32) / 255.0
        
        # Expand dimensions if needed (batch dimension)
        if len(input_shape) == 4:
            normalized = np.expand_dims(normalized, axis=0)
        
        return normalized
    
    def detect(self, image: np.ndarray) -> List[Tuple[int, float, Tuple[int, int, int, int]]]:
        """Detect stones in an image.
        
        Args:
            image: The input image as a numpy array
            
        Returns:
            List of tuples (class_id, confidence, bounding_box)
        """
        # Preprocess the image
        processed_image = self.preprocess_image(image)
        
        # Set the input tensor
        self.interpreter.set_tensor(self.input_details[0]['index'], processed_image)
        
        # Run inference
        self.interpreter.invoke()
        
        # Get outputs
        detections = []
        
        # Get output tensors 
        # Note: Output format depends on your specific model
        # This assumes SSD-like output format with classes, boxes, scores
        # Adjust according to your model's output format
        
        try:
            # Get detection boxes
            boxes = self.interpreter.get_tensor(
                self.output_details[0]['index'])[0]
            
            # Get detection classes
            classes = self.interpreter.get_tensor(
                self.output_details[1]['index'])[0]
            
            # Get confidence scores
            scores = self.interpreter.get_tensor(
                self.output_details[2]['index'])[0]
            
            # Filter detections by threshold
            for i, score in enumerate(scores):
                if score > self.threshold:
                    # Convert normalized coordinates to pixel coordinates
                    y1, x1, y2, x2 = boxes[i]
                    h, w = image.shape[:2]
                    box = (
                        int(x1 * w),  # x
                        int(y1 * h),  # y
                        int((x2 - x1) * w),  # width
                        int((y2 - y1) * h)   # height
                    )
                    
                    # Add detection to list
                    detections.append((int(classes[i]), float(score), box))
        
        except Exception as e:
            print(f"Error processing detections: {e}")
        
        return detections
    
    async def detect_async(self, image: np.ndarray) -> List[Tuple[int, float, Tuple[int, int, int, int]]]:
        """Detect stones in an image asynchronously.
        
        Args:
            image: The input image as a numpy array
            
        Returns:
            List of tuples (class_id, confidence, bounding_box)
        """
        loop = asyncio.get_event_loop()
        return await loop.run_in_executor(self.executor, self.detect, image)
    
    def draw_detections(self, image: np.ndarray, 
                        detections: List[Tuple[int, float, Tuple[int, int, int, int]]]) -> np.ndarray:
        """Draw detection boxes and labels on the image.
        
        Args:
            image: The input image
            detections: List of detections from detect()
            
        Returns:
            Image with detections drawn on it
        """
        result = image.copy()
        
        for class_id, score, (x, y, w, h) in detections:
            # Draw bounding box
            cv2.rectangle(result, (x, y), (x + w, y + h), (0, 255, 0), 2)
            
            # Draw label
            label = f"{self.labels[class_id]}: {score:.2f}"
            cv2.putText(
                result, label, (x, y - 10),
                cv2.FONT_HERSHEY_SIMPLEX, 0.5, (0, 255, 0), 2
            )
        
        return result


# Test function
def main():
    """Test the StoneDetector functionality with a sample image."""
    import argparse
    
    parser = argparse.ArgumentParser(description='Test Stone Detector')
    parser.add_argument('--model', required=True, help='Path to TFLite model')
    parser.add_argument('--labels', required=True, help='Path to labels file')
    parser.add_argument('--image', required=True, help='Path to test image')
    parser.add_argument('--threshold', type=float, default=0.5, help='Detection threshold')
    
    args = parser.parse_args()
    
    # Initialize detector
    detector = StoneDetector(
        model_path=args.model,
        labels_path=args.labels,
        threshold=args.threshold
    )
    
    # Load test image
    image = cv2.imread(args.image)
    if image is None:
        print(f"Error: Could not load image {args.image}")
        return
    
    # Run detection
    start_time = time.time()
    detections = detector.detect(image)
    inference_time = (time.time() - start_time) * 1000
    
    # Draw detections
    result = detector.draw_detections(image, detections)
    
    # Print results
    print(f"Found {len(detections)} detections in {inference_time:.2f}ms")
    for class_id, score, box in detections:
        print(f"  {detector.labels[class_id]}: {score:.2f} at {box}")
    
    # Show result
    cv2.imshow("Detections", result)
    cv2.waitKey(0)
    cv2.destroyAllWindows()

if __name__ == "__main__":
    main()