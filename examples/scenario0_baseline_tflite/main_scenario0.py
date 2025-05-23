#!/usr/bin/env python3
"""
TinyLCM Performance Baseline - Scenario 0
----------------------------------------
Pure TFLite inference without TinyLCM library for performance comparison.
This script performs inference directly with TFLite and manually logs performance metrics.

Designed for Raspberry Pi Zero 2W performance evaluation.
"""

import argparse
import json
import logging
import os
import psutil
import signal
import sys
import time
import uuid
from datetime import datetime
from pathlib import Path
from typing import Any, Dict, List, Optional, Tuple

import cv2
import numpy as np

# Try to import tflite_runtime (for Raspberry Pi) or fall back to TensorFlow's interpreter
try:
    from tflite_runtime.interpreter import Interpreter as tflite_Interpreter
    TFLITE_AVAILABLE = True
except ImportError:
    try:
        import tensorflow as tf
        tflite_Interpreter = tf.lite.Interpreter
        TFLITE_AVAILABLE = True
    except ImportError:
        TFLITE_AVAILABLE = False
        print("TFLite not available. Please install tflite_runtime or tensorflow.")
        sys.exit(1)

# Import utils from parent directory
sys.path.append(os.path.dirname(os.path.dirname(os.path.abspath(__file__))))
from utils.camera_handler import CameraHandler
from utils.preprocessors import resize_image

# Global variables
running = True
logger = None
process = psutil.Process(os.getpid())


def load_config(config_path: str) -> Dict:
    """Load the configuration from a JSON file."""
    try:
        with open(config_path, 'r') as f:
            config = json.load(f)
        return config
    except Exception as e:
        logging.error(f"Failed to load config: {e}")
        sys.exit(1)


def setup_logging(log_level: str = "INFO") -> None:
    """Set up logging configuration."""
    global logger
    
    numeric_level = getattr(logging, log_level.upper(), None)
    if not isinstance(numeric_level, int):
        numeric_level = logging.INFO
    
    # Create logs directory if it doesn't exist
    logs_dir = Path("./logs")
    logs_dir.mkdir(exist_ok=True)
    
    # Set up logging
    log_file = logs_dir / f"scenario0_baseline_{datetime.now().strftime('%Y%m%d_%H%M%S')}.log"
    
    logging.basicConfig(
        level=numeric_level,
        format='%(asctime)s - %(name)s - %(levelname)s - %(message)s',
        handlers=[
            logging.FileHandler(log_file),
            logging.StreamHandler(sys.stdout)
        ]
    )
    
    logger = logging.getLogger(__name__)


def load_labels(labels_path: str) -> List[str]:
    """Load labels from a text file."""
    try:
        with open(labels_path, 'r') as f:
            labels = [line.strip() for line in f.readlines()]
        return labels
    except Exception as e:
        logger.error(f"Failed to load labels: {e}")
        return []


class TFLiteModel:
    """Simple TFLite model wrapper for inference."""
    
    def __init__(self, model_path: str):
        """Initialize the TFLite interpreter."""
        self.interpreter = tflite_Interpreter(model_path=model_path)
        self.interpreter.allocate_tensors()
        
        # Get input and output details
        self.input_details = self.interpreter.get_input_details()
        self.output_details = self.interpreter.get_output_details()
        
        # Get input shape
        self.input_shape = self.input_details[0]['shape']
        self.input_height = self.input_shape[1]
        self.input_width = self.input_shape[2]
    
    def predict(self, image: np.ndarray) -> np.ndarray:
        """Run inference on an image."""
        # Preprocess image
        if image.shape[:2] != (self.input_height, self.input_width):
            image = cv2.resize(image, (self.input_width, self.input_height))
        
        # Prepare input tensor
        input_tensor = np.expand_dims(image, axis=0)
        
        # Check if model expects float32 or uint8
        if self.input_details[0]['dtype'] == np.uint8:
            input_tensor = input_tensor.astype(np.uint8)
        else:
            input_tensor = input_tensor.astype(np.float32) / 255.0
        
        # Set input tensor
        self.interpreter.set_tensor(self.input_details[0]['index'], input_tensor)
        
        # Run inference
        self.interpreter.invoke()
        
        # Get output
        output_data = self.interpreter.get_tensor(self.output_details[0]['index'])
        return output_data[0]  # Remove batch dimension


class PerformanceLogger:
    """Logs performance metrics to file."""
    
    def __init__(self, log_dir: str = "./logs"):
        """Initialize the performance logger."""
        self.log_dir = Path(log_dir)
        self.log_dir.mkdir(exist_ok=True)
        
        # Create performance log file
        timestamp = datetime.now().strftime('%Y%m%d_%H%M%S')
        self.log_file = self.log_dir / f"performance_scenario0_{timestamp}.json"
        
        # Performance metrics storage
        self.metrics = []
        self.start_time = time.time()
        
        # Log system info at start
        self.log_system_info()
    
    def log_system_info(self):
        """Log system information at startup."""
        system_info = {
            "timestamp": datetime.now().isoformat(),
            "type": "system_info",
            "cpu_count": psutil.cpu_count(),
            "total_memory_mb": psutil.virtual_memory().total / (1024 * 1024),
            "python_version": sys.version,
            "opencv_version": cv2.__version__
        }
        self._append_to_log(system_info)
    
    def log_inference(self, inference_time: float, prediction: Dict[str, Any]):
        """Log a single inference with performance metrics."""
        # Get current resource usage
        cpu_percent = process.cpu_percent(interval=None)
        memory_info = process.memory_info()
        memory_mb = memory_info.rss / (1024 * 1024)
        
        metric = {
            "timestamp": datetime.now().isoformat(),
            "type": "inference",
            "inference_time_ms": inference_time * 1000,
            "cpu_percent": cpu_percent,
            "memory_mb": memory_mb,
            "prediction": prediction,
            "uptime_seconds": time.time() - self.start_time
        }
        
        self.metrics.append(metric)
        self._append_to_log(metric)
    
    def log_summary(self):
        """Log summary statistics."""
        if not self.metrics:
            return
        
        inference_times = [m["inference_time_ms"] for m in self.metrics if m["type"] == "inference"]
        cpu_percents = [m["cpu_percent"] for m in self.metrics if m["type"] == "inference"]
        memory_mbs = [m["memory_mb"] for m in self.metrics if m["type"] == "inference"]
        
        summary = {
            "timestamp": datetime.now().isoformat(),
            "type": "summary",
            "total_inferences": len(inference_times),
            "avg_inference_time_ms": np.mean(inference_times) if inference_times else 0,
            "std_inference_time_ms": np.std(inference_times) if inference_times else 0,
            "min_inference_time_ms": np.min(inference_times) if inference_times else 0,
            "max_inference_time_ms": np.max(inference_times) if inference_times else 0,
            "avg_cpu_percent": np.mean(cpu_percents) if cpu_percents else 0,
            "avg_memory_mb": np.mean(memory_mbs) if memory_mbs else 0,
            "max_memory_mb": np.max(memory_mbs) if memory_mbs else 0,
            "total_runtime_seconds": time.time() - self.start_time
        }
        
        self._append_to_log(summary)
        logger.info(f"Performance Summary: {json.dumps(summary, indent=2)}")
    
    def _append_to_log(self, data: Dict):
        """Append data to the log file."""
        try:
            with open(self.log_file, 'a') as f:
                f.write(json.dumps(data) + '\n')
        except Exception as e:
            logger.error(f"Failed to write to performance log: {e}")


def signal_handler(sig, frame):
    """Handle shutdown signals gracefully."""
    global running
    logger.info("Shutdown signal received. Stopping...")
    running = False


def main(config_path: str):
    """Main execution function."""
    global running, logger
    
    # Load configuration
    config = load_config(config_path)
    
    # Set up logging
    setup_logging(config.get("device", {}).get("log_level", "INFO"))
    logger.info("Starting Scenario 0 - Baseline TFLite Performance Test")
    
    # Set up signal handler
    signal.signal(signal.SIGINT, signal_handler)
    signal.signal(signal.SIGTERM, signal_handler)
    
    # Initialize camera
    camera_config = config.get("camera", {})
    camera = CameraHandler(
        resolution=tuple(camera_config.get("resolution", [640, 480])),
        framerate=camera_config.get("framerate", 1),
        rotation=camera_config.get("rotation", 0)
    )
    
    if not camera.start():
        logger.error("Failed to start camera")
        sys.exit(1)
    
    # Initialize TFLite model
    model_config = config.get("model", {})
    model_path = model_config.get("model_path", "./model/model_object.tflite")
    labels_path = model_config.get("labels_path", "./model/labels_object.txt")
    threshold = model_config.get("threshold", 0.75)
    
    model = TFLiteModel(model_path)
    labels = load_labels(labels_path)
    
    # Initialize performance logger
    performance_logger = PerformanceLogger()
    
    # Get inference interval from config
    app_config = config.get("application", {})
    inference_interval_ms = app_config.get("inference_interval_ms", 2000)
    inference_interval = inference_interval_ms / 1000.0
    
    # Main loop
    inference_count = 0
    last_inference_time = 0
    
    logger.info("Starting inference loop...")
    
    try:
        while running:
            current_time = time.time()
            
            # Check if it's time for the next inference
            if current_time - last_inference_time >= inference_interval:
                # Capture frame
                frame = camera.capture_frame()
                if frame is None:
                    logger.warning("Failed to capture frame")
                    continue
                
                # Perform inference and measure time
                inference_start = time.time()
                predictions = model.predict(frame)
                inference_time = time.time() - inference_start
                
                # Process predictions
                if len(predictions) > 0:
                    # Get top prediction
                    top_idx = np.argmax(predictions)
                    confidence = float(predictions[top_idx])
                    predicted_class = labels[top_idx] if top_idx < len(labels) else f"class_{top_idx}"
                    
                    prediction_result = {
                        "class": predicted_class,
                        "confidence": confidence,
                        "all_scores": predictions.tolist()
                    }
                    
                    # Log if above threshold
                    if confidence >= threshold:
                        logger.info(f"Inference #{inference_count}: {predicted_class} ({confidence:.2%})")
                else:
                    prediction_result = {
                        "class": "unknown",
                        "confidence": 0.0,
                        "all_scores": []
                    }
                
                # Log performance metrics
                performance_logger.log_inference(inference_time, prediction_result)
                
                inference_count += 1
                last_inference_time = current_time
                
                # Log progress every 10 inferences
                if inference_count % 10 == 0:
                    logger.info(f"Completed {inference_count} inferences")
            
            # Small sleep to prevent busy waiting
            time.sleep(0.01)
    
    except Exception as e:
        logger.error(f"Error in main loop: {e}")
    finally:
        # Clean up
        logger.info("Shutting down...")
        camera.stop()
        
        # Log final summary
        performance_logger.log_summary()
        
        logger.info(f"Completed {inference_count} inferences")
        logger.info("Scenario 0 completed")


if __name__ == "__main__":
    parser = argparse.ArgumentParser(description="Scenario 0 - Baseline TFLite Performance Test")
    parser.add_argument(
        "--config",
        type=str,
        default="config_scenario0.json",
        help="Path to configuration file"
    )
    
    args = parser.parse_args()
    main(args.config)