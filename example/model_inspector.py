#!/usr/bin/env python3
"""
TFLite Model Inspector
---------------------
Utility for inspecting TFLite model input and output formats.
Used for debugging model-related issues.
"""

import argparse
import numpy as np
import cv2
import os
import time

try:
    import tflite_runtime.interpreter as tflite
    using_tflite_runtime = True
    print("Using TFLite Runtime")
except ImportError:
    try:
        import tensorflow as tf
        using_tflite_runtime = False
        print("Using TensorFlow")
    except ImportError:
        raise ImportError(
            "Neither TensorFlow nor TFLite Runtime found. "
            "Please install at least one with: "
            "pip install tflite-runtime or pip install tensorflow"
        )

def load_model(model_path):
    """Load TFLite model and get details about inputs and outputs."""
    try:
        print(f"Loading model from {model_path}")
        if using_tflite_runtime:
            interpreter = tflite.Interpreter(model_path=model_path)
        else:
            interpreter = tf.lite.Interpreter(model_path=model_path)
        
        interpreter.allocate_tensors()
        
        # Get input and output details
        input_details = interpreter.get_input_details()
        output_details = interpreter.get_output_details()
        
        print("\n=== Model Summary ===")
        print(f"Model loaded successfully: {os.path.basename(model_path)}")
        print(f"File size: {os.path.getsize(model_path) / (1024*1024):.2f} MB")
        
        print("\n=== Input Details ===")
        for i, detail in enumerate(input_details):
            print(f"Input #{i}:")
            print(f"  name: {detail['name']}")
            print(f"  shape: {detail['shape']}")
            print(f"  type: {detail['dtype']}")
            if 'quantization' in detail and detail['quantization'][0] != 0:
                print(f"  quantized: scale={detail['quantization'][0]}, zero_point={detail['quantization'][1]}")
        
        print("\n=== Output Details ===")
        for i, detail in enumerate(output_details):
            print(f"Output #{i}:")
            print(f"  name: {detail['name']}")
            print(f"  shape: {detail['shape']}")
            print(f"  type: {detail['dtype']}")
            if 'quantization' in detail and detail['quantization'][0] != 0:
                print(f"  quantized: scale={detail['quantization'][0]}, zero_point={detail['quantization'][1]}")
        
        return interpreter, input_details, output_details
    
    except Exception as e:
        print(f"Error loading model: {e}")
        import traceback
        traceback.print_exc()
        return None, None, None

def preprocess_image(image_path, input_details):
    """Preprocess an image for model input."""
    try:
        # Read image
        print(f"Reading image: {image_path}")
        image = cv2.imread(image_path)
        if image is None:
            raise ValueError(f"Could not read image: {image_path}")
        
        print(f"Original image shape: {image.shape}")
        
        # Get target shape from model input
        if len(input_details) > 0:
            input_shape = input_details[0]['shape']
            height, width = input_shape[1], input_shape[2]
            
            # Resize image
            resized = cv2.resize(image, (width, height))
            print(f"Resized to: {resized.shape}")
            
            # Handle channels
            if input_shape[3] == 1 and len(resized.shape) == 3 and resized.shape[2] == 3:
                # Convert to grayscale
                gray = cv2.cvtColor(resized, cv2.COLOR_BGR2GRAY)
                processed = np.expand_dims(gray, axis=-1)
                print("Converted to grayscale")
            else:
                # Ensure RGB order (OpenCV uses BGR)
                processed = cv2.cvtColor(resized, cv2.COLOR_BGR2RGB)
            
            # Convert to float32 and normalize to [0, 1]
            if input_details[0]['dtype'] == np.float32:
                processed = processed.astype(np.float32) / 255.0
                print("Converted to float32 and normalized to [0,1]")
            
            # Add batch dimension
            processed = np.expand_dims(processed, axis=0)
            print(f"Final processed shape: {processed.shape}")
            
            return processed
        else:
            print("No input details available - returning original image")
            return np.expand_dims(image, axis=0)
            
    except Exception as e:
        print(f"Error preprocessing image: {e}")
        import traceback
        traceback.print_exc()
        return None

def run_inference(interpreter, preprocessed_image):
    """Run inference and time it."""
    try:
        input_details = interpreter.get_input_details()
        output_details = interpreter.get_output_details()
        
        # Set input tensor
        interpreter.set_tensor(input_details[0]['index'], preprocessed_image)
        
        # Time the inference
        start_time = time.time()
        
        # Run inference
        interpreter.invoke()
        
        inference_time = (time.time() - start_time) * 1000  # ms
        print(f"\nInference time: {inference_time:.2f} ms")
        
        # Get output tensors
        print("\n=== Output Tensors ===")
        outputs = []
        for i, output in enumerate(output_details):
            output_data = interpreter.get_tensor(output['index'])
            outputs.append(output_data)
            print(f"Output #{i} {output['name']}:")
            print(f"  Shape: {output_data.shape}")
            
            # Print summary statistics
            flat_data = output_data.flatten()
            print(f"  Min value: {np.min(flat_data)}")
            print(f"  Max value: {np.max(flat_data)}")
            print(f"  Mean value: {np.mean(flat_data)}")
            
            # If it's a small tensor, print all values
            if flat_data.size <= 10:
                print(f"  Values: {flat_data}")
            else:
                # Print first few values
                print(f"  First 5 values: {flat_data[:5]}")
        
        return outputs
    
    except Exception as e:
        print(f"Error running inference: {e}")
        import traceback
        traceback.print_exc()
        return None

def analyze_detector_outputs(outputs, score_threshold=0.5):
    """Analyze the output of an object detection model."""
    print("\n=== Detection Analysis ===")
    
    # Handle different model output formats
    if len(outputs) == 4:  # Standard TF detection format
        # Format: boxes, classes, scores, num_detections
        boxes = outputs[0][0]
        classes = outputs[1][0].astype(np.int32)
        scores = outputs[2][0]
        num_detections = int(outputs[3][0])
        
        print(f"Number of detections: {num_detections}")
        print(f"Detection boxes shape: {boxes.shape}")
        print(f"Detection classes shape: {classes.shape}")
        print(f"Detection scores shape: {scores.shape}")
        
        # Print top 5 detections
        for i in range(min(5, num_detections)):
            if scores[i] >= score_threshold:
                print(f"Detection #{i+1}: class={classes[i]}, score={scores[i]:.4f}, box={boxes[i]}")
    
    elif len(outputs) == 3:  # Some models use 3 outputs: boxes, scores, classes
        boxes = outputs[0][0]
        scores = outputs[1][0]
        classes = outputs[2][0].astype(np.int32)
        
        # Find detections above threshold
        valid_detections = np.where(scores >= score_threshold)[0]
        print(f"Detections above threshold: {len(valid_detections)}")
        
        # Print top 5 detections
        for i in range(min(5, len(valid_detections))):
            idx = valid_detections[i]
            print(f"Detection #{i+1}: class={classes[idx]}, score={scores[idx]:.4f}, box={boxes[idx]}")
    
    elif len(outputs) == 1:  # Single output tensor format
        # Try to determine format based on shape
        output = outputs[0]
        shape = output.shape
        
        print(f"Single output tensor shape: {shape}")
        
        # Print tensor statistics to help analyze
        flat_output = output.flatten()
        print(f"Min: {np.min(flat_output)}, Max: {np.max(flat_output)}, Mean: {np.mean(flat_output)}")
        
        # Analyze output tensor contents in detail
        output_data = output[0]  # Take first batch
        
        # Print the first few rows to inspect the data
        if len(output_data) > 0:
            print("\nFirst few rows of output data:")
            for i in range(min(5, len(output_data))):
                print(f"Row {i}: {output_data[i]}")
            
            # Try to detect data patterns
            # 1. Check for coordinates
            has_coords = False
            for i in range(min(5, len(output_data))):
                if len(output_data[i]) >= 4:
                    coords = output_data[i][:4]
                    # Coordinates are usually between 0-1 (normalized) 
                    if all(0 <= v <= 1 for v in coords):
                        has_coords = True
                        print(f"\nDetected possible normalized coordinates at row {i}: {coords}")
                        break
            
            # 2. Check for score-like values
            has_scores = False
            for i in range(min(10, len(output_data))):
                if len(output_data[i]) >= 5:
                    score = output_data[i][4]
                    if 0 <= score <= 1:
                        has_scores = True
                        print(f"Detected possible score at row {i}: {score}")
                        break
            
            # Specific formats
            if len(shape) == 3 and shape[2] == 4:
                # Likely bounding boxes
                print("\nLikely a detection model with output format [batch, detections, 4]")
                print("Format: [y1, x1, y2, x2] (normalized coordinates)")
                print("Using default class_id=0 and score=1.0")
                
                # Visualize the first few detections
                for i in range(min(5, len(output_data))):
                    print(f"Detection #{i+1}: box={output_data[i]}")
            
            elif len(shape) == 3 and shape[2] > 5:  # [batch, detections, values]
                # Try to interpret as [y1, x1, y2, x2, score, class_id, ...]
                print("\nAttempting to interpret as standard detection format:")
                
                # Count detections above threshold
                valid_count = 0
                for i in range(len(output_data)):
                    if len(output_data[i]) >= 5 and output_data[i][4] >= score_threshold:
                        valid_count += 1
                
                print(f"Detections above threshold: {valid_count}")
                
                # Print top 5 detections
                count = 0
                for i in range(len(output_data)):
                    if len(output_data[i]) >= 5 and output_data[i][4] >= score_threshold:
                        count += 1
                        
                        # Extract box and score
                        box = output_data[i][:4]
                        score = output_data[i][4]
                        
                        # Extract class if available
                        class_id = int(output_data[i][5]) if len(output_data[i]) >= 6 else 0
                        
                        print(f"Detection #{count}: class={class_id}, score={score:.4f}, box={box}")
                        
                        if count >= 5:
                            break
            
            elif len(shape) == 2 and shape[1] == 1:
                # Special case for binary classification
                score = float(output_data[0])
                print("\nBinary classifier with single output")
                print(f"Score: {score:.6f}")
                
                # Interpret the score
                if 0 <= score <= 1:
                    prediction = "Positive (class 1)" if score > 0.5 else "Negative (class 0)"
                    confidence = max(score, 1-score)
                    print(f"Prediction: {prediction} with confidence {confidence:.2f}")
                    print(f"This is {'above' if score > score_threshold else 'below'} the threshold {score_threshold}")
                else:
                    # Not a standard probability score
                    print("Score is not in the range [0,1], may need different interpretation")
                    print("Possible logit output (pre-sigmoid) or custom scoring")
            
            elif len(shape) == 2 and shape[1] <= 1001:
                # Likely a classification model
                print("\nLikely a classification model with output format [batch, classes]")
                
                # Find top classes
                top_classes = np.argsort(output_data)[-5:][::-1]
                
                print("Top 5 classes:")
                for i, class_id in enumerate(top_classes):
                    print(f"#{i+1}: class={class_id}, score={output_data[class_id]:.4f}")
            
            else:
                print("\nOutput format doesn't match common patterns")
                print("You may need to implement custom parsing for this model format")
                
                # Last resort: just check if there are any values > threshold
                high_values = np.where(output_data.flatten() > score_threshold)[0]
                if len(high_values) > 0:
                    print(f"Found {len(high_values)} values above threshold {score_threshold}")
                    print(f"First few high values: {output_data.flatten()[high_values[:5]]}")
        else:
            print("Empty output data")
    else:
        print(f"Unknown output format with {len(outputs)} tensors - cannot parse detections")

def main():
    parser = argparse.ArgumentParser(description="TFLite Model Inspector")
    parser.add_argument("--model", required=True, help="Path to TFLite model file")
    parser.add_argument("--image", help="Path to test image for inference")
    parser.add_argument("--threshold", type=float, default=0.5, help="Detection score threshold")
    
    args = parser.parse_args()
    
    # Load and inspect model
    interpreter, input_details, output_details = load_model(args.model)
    
    if interpreter is None:
        return
    
    # Run inference if image is provided
    if args.image:
        image = preprocess_image(args.image, input_details)
        if image is not None:
            outputs = run_inference(interpreter, image)
            if outputs:
                analyze_detector_outputs(outputs, args.threshold)

if __name__ == "__main__":
    main()