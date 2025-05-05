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
            # Check if output_details has enough elements
            if len(self.output_details) < 3:
                print(f"Warning: Expected at least 3 output tensors, but got {len(self.output_details)}")
                # Try to infer the outputs based on shapes
                if len(self.output_details) == 1:
                    # Get the single output tensor
                    output = self.interpreter.get_tensor(self.output_details[0]['index'])
                    output_shape = output.shape
                    
                    print(f"Single output tensor shape: {output_shape}")
                    
                    # Handle MobileSSD-like single tensor output format (most common)
                    if len(output_shape) == 3 and output_shape[2] == 4:
                        # Likely a standard detection model with format [batch, detections, 4]
                        # where 4 = [y1, x1, y2, x2] - normalized coordinates
                        boxes = output[0]
                        
                        # For single output models, we may need to make a fixed assumption about class
                        # This assumes a single-class detector (e.g. a stone detector)
                        default_class_id = 0
                        default_score = 1.0  # We'll trust the model's detections
                        
                        # Process each detection
                        for i in range(len(boxes)):
                            # Parse the bounding box
                            if len(boxes[i]) >= 4:
                                y1, x1, y2, x2 = boxes[i][:4]
                                
                                # Convert to pixel coordinates
                                h, w = image.shape[:2]
                                box = (
                                    int(x1 * w),   # x
                                    int(y1 * h),   # y
                                    int((x2 - x1) * w),  # width
                                    int((y2 - y1) * h)   # height
                                )
                                
                                # Add to detections with default class and score
                                detections.append((default_class_id, default_score, box))
                                
                    # Handle common classification output format
                    elif len(output_shape) == 2 and output_shape[1] <= 1001:
                        # Likely a classification model with format [batch, classes]
                        scores = output[0]
                        
                        # Find the class with highest score
                        max_score_idx = np.argmax(scores)
                        max_score = scores[max_score_idx]
                        
                        if max_score > self.threshold:
                            # For classification, use the whole image as the detection box
                            h, w = image.shape[:2]
                            box = (0, 0, w, h)
                            
                            # Add to detections
                            detections.append((max_score_idx, float(max_score), box))
                    
                    # Try MobileNet SSD format - usually has 10 arrays of length 4 for coordinates
                    elif len(output_shape) == 3 and output_shape[1] == 10:
                        # MobileNet SSD detection model
                        output_data = output[0]
                        
                        # First 10 entries are boxes, then scores, then classes
                        num_detections = output_shape[1]
                        
                        # Check if we can guess the format - try to infer based on values
                        # MobileNet detections usually have score arrays after the 10 box arrays
                        max_values = np.max(output_data, axis=1)
                        
                        # Find arrays where max value is between 0-1 (likely scores)
                        score_candidates = [i for i, v in enumerate(max_values) if 0 < v <= 1.0]
                        
                        if score_candidates:
                            # Use the first probable score array
                            score_idx = score_candidates[0]
                            scores = output_data[score_idx]
                            
                            # Assume the next array contains classes
                            class_idx = min(score_idx + 1, len(output_data) - 1)
                            classes = output_data[class_idx]
                            
                            # Boxes are usually the first bunch of arrays
                            # Take the first 4 values of each of the first entries
                            for i in range(min(10, len(scores))):
                                if scores[i] > self.threshold:
                                    # Try to get box from first 4 arrays
                                    if i < len(output_data):
                                        box_data = output_data[i][:4]
                                        y1, x1, y2, x2 = box_data
                                        
                                        # Convert to pixel coordinates
                                        h, w = image.shape[:2]
                                        box = (
                                            int(x1 * w),   # x
                                            int(y1 * h),   # y
                                            int((x2 - x1) * w),  # width
                                            int((y2 - y1) * h)   # height
                                        )
                                        
                                        # Get class if available
                                        class_id = int(classes[i]) if i < len(classes) else 0
                                        
                                        # Add to detections
                                        detections.append((class_id, float(scores[i]), box))
                        
                    # Handle standard detection output in a single tensor
                    elif len(output_shape) >= 2:
                        # Some models output a single tensor with format [batch, num_detections, values]
                        # Try to determine if it matches common formats
                        
                        # Get the first batch
                        detections_data = output[0]
                        
                        # Check if each detection has at least 5 or 6 values 
                        # (4 for bbox + 1 for score + optional 1 for class)
                        if len(detections_data.shape) == 2 and detections_data.shape[1] >= 5:
                            # Process each detection
                            for i in range(len(detections_data)):
                                detection = detections_data[i]
                                
                                # Extract values - format depends on the model
                                # Most common formats are:
                                # [x1, y1, x2, y2, score, class_id]
                                # [y1, x1, y2, x2, score, class_id]
                                
                                # Try both formats
                                if len(detection) >= 6:
                                    # Format with class ID
                                    format1 = [detection[0], detection[1], detection[2], detection[3], detection[4], detection[5]]
                                    format2 = [detection[1], detection[0], detection[3], detection[2], detection[4], detection[5]]
                                    
                                    # Use format 1 by default
                                    x1, y1, x2, y2, score, class_id = format1
                                elif len(detection) >= 5:
                                    # Format without class ID
                                    format1 = [detection[0], detection[1], detection[2], detection[3], detection[4]]
                                    format2 = [detection[1], detection[0], detection[3], detection[2], detection[4]]
                                    
                                    # Use format 1 by default
                                    x1, y1, x2, y2, score = format1
                                    class_id = 0  # Default class
                                else:
                                    # Skip if not enough values
                                    continue
                                
                                # Check score threshold
                                if score > self.threshold:
                                    # Convert to pixel coordinates
                                    h, w = image.shape[:2]
                                    
                                    # Normalize coordinates if they're not already (values > 1.0)
                                    if x1 > 1.0 or y1 > 1.0 or x2 > 1.0 or y2 > 1.0:
                                        # Convert absolute pixel coordinates to normalized
                                        x1, y1, x2, y2 = x1/w, y1/h, x2/w, y2/h
                                    
                                    box = (
                                        int(x1 * w),   # x
                                        int(y1 * h),   # y
                                        int((x2 - x1) * w),  # width
                                        int((y2 - y1) * h)   # height
                                    )
                                    
                                    # Add detection
                                    detections.append((int(class_id), float(score), box))
                    
                    # For any other single-output format, try to just use it as is
                    # This works for many simple classification/detection models
                    elif len(output_shape) >= 1:
                        # Flatten output and treat it as scores
                        flat_output = output.flatten()
                        
                        # Get the class with highest score
                        if len(flat_output) > 0:
                            max_score_idx = np.argmax(flat_output)
                            max_score = flat_output[max_score_idx]
                            
                            if max_score > self.threshold:
                                # For classification, use the whole image as the detection box
                                h, w = image.shape[:2]
                                box = (0, 0, w, h)
                                
                                # Add to detections
                                detections.append((max_score_idx, float(max_score), box))
                    
                    # Print detection summary
                    print(f"Processed single output tensor and found {len(detections)} detections above threshold")
                    
                    return detections  # Return early with what we have
            
            # Standard SSD format with separate tensors
            # Get detection boxes - use safe indexing
            boxes_tensor = self.interpreter.get_tensor(self.output_details[0]['index'])
            if len(boxes_tensor) > 0:
                boxes = boxes_tensor[0]
            else:
                print("Warning: Boxes tensor is empty")
                return detections
            
            # Get detection classes - use safe indexing
            classes_idx = min(1, len(self.output_details)-1)
            classes_tensor = self.interpreter.get_tensor(self.output_details[classes_idx]['index'])
            if len(classes_tensor) > 0:
                classes = classes_tensor[0]
            else:
                print("Warning: Classes tensor is empty")
                return detections
            
            # Get confidence scores - use safe indexing
            scores_idx = min(2, len(self.output_details)-1)
            scores_tensor = self.interpreter.get_tensor(self.output_details[scores_idx]['index'])
            if len(scores_tensor) > 0:
                scores = scores_tensor[0]
            else:
                print("Warning: Scores tensor is empty")
                return detections
            
            # Validate lengths match to avoid index errors
            if len(scores) != len(classes) or len(scores) != len(boxes):
                print(f"Warning: Inconsistent detection arrays - scores:{len(scores)}, classes:{len(classes)}, boxes:{len(boxes)}")
                # Use the minimum length to avoid index errors
                num_detections = min(len(scores), len(classes), len(boxes))
            else:
                num_detections = len(scores)
                
            # Filter detections by threshold
            for i in range(num_detections):
                if i < len(scores) and scores[i] > self.threshold:
                    # Safely get bbox coordinates
                    if i < len(boxes) and len(boxes[i]) >= 4:
                        # Get coordinates with safe indexing
                        y1, x1, y2, x2 = boxes[i][:4]
                        
                        # Ensure image has shape
                        if hasattr(image, 'shape') and len(image.shape) >= 2:
                            h, w = image.shape[:2]
                            box = (
                                int(x1 * w),  # x
                                int(y1 * h),  # y
                                int((x2 - x1) * w),  # width
                                int((y2 - y1) * h)   # height
                            )
                            
                            # Safely get class
                            class_id = int(classes[i]) if i < len(classes) else 0
                            
                            # Add detection to list
                            detections.append((class_id, float(scores[i]), box))
        
        except Exception as e:
            print(f"Error processing detections: {e}")
            # Log more detailed error information
            import traceback
            traceback.print_exc()
        
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