#!/usr/bin/env python3
"""
TinyLCM Performance Test - Scenario 1
------------------------------------
TinyLCM inference pipeline WITHOUT drift detection for performance comparison.
This script uses TinyLCM components (feature extraction, KNN classifier) but disables
all drift detection to measure the overhead of the core TinyLCM pipeline.

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
        TFLITE_AVAILABLE = True
    except ImportError:
        TFLITE_AVAILABLE = False

# Import utils from parent directory
sys.path.append(os.path.dirname(os.path.dirname(os.path.abspath(__file__))))
from utils.camera_handler import CameraHandler
from utils.device_id_manager import DeviceIDManager
from utils.preprocessors import resize_image

from tinylcm.core.classifiers.knn import LightweightKNN
from tinylcm.core.data_logger.logger import DataLogger
from tinylcm.core.data_structures import FeatureSample
from tinylcm.core.feature_extractors.tflite import TFLiteFeatureExtractor
from tinylcm.core.feature_transformers.standard_scaler_pca import StandardScalerPCATransformer
from tinylcm.core.operational_monitor.monitor import OperationalMonitor
from tinylcm.core.pipeline import InferencePipeline
from tinylcm.utils.logging import setup_logger

# Global variables
logger = None
running = True
current_frame = None
config = None
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
    log_file = logs_dir / f"tinylcm_scenario1_{datetime.now().strftime('%Y%m%d_%H%M%S')}.log"
    
    logging.basicConfig(
        level=numeric_level,
        format='%(asctime)s - %(name)s - %(levelname)s - %(message)s',
        handlers=[
            logging.FileHandler(log_file),
            logging.StreamHandler(sys.stdout)
        ]
    )
    
    logger = logging.getLogger(__name__)


class PerformanceLogger:
    """Enhanced performance logger for TinyLCM scenario."""
    
    def __init__(self, log_dir: str = "./logs"):
        """Initialize the performance logger."""
        self.log_dir = Path(log_dir)
        self.log_dir.mkdir(exist_ok=True)
        
        # Create performance log file
        timestamp = datetime.now().strftime('%Y%m%d_%H%M%S')
        self.log_file = self.log_dir / f"performance_scenario1_{timestamp}.json"
        
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
            "scenario": "scenario1_tinylcm_no_drift",
            "cpu_count": psutil.cpu_count(),
            "total_memory_mb": psutil.virtual_memory().total / (1024 * 1024),
            "python_version": sys.version,
            "opencv_version": cv2.__version__
        }
        self._append_to_log(system_info)
    
    def log_inference(self, 
                     total_time: float,
                     feature_extraction_time: float,
                     knn_inference_time: float,
                     prediction: Dict[str, Any]):
        """Log a single inference with detailed timing breakdown."""
        # Get current resource usage
        cpu_percent = process.cpu_percent(interval=None)
        memory_info = process.memory_info()
        memory_mb = memory_info.rss / (1024 * 1024)
        
        metric = {
            "timestamp": datetime.now().isoformat(),
            "type": "inference",
            "total_time_ms": total_time * 1000,
            "feature_extraction_time_ms": feature_extraction_time * 1000,
            "knn_inference_time_ms": knn_inference_time * 1000,
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
        
        inference_metrics = [m for m in self.metrics if m["type"] == "inference"]
        if not inference_metrics:
            return
        
        total_times = [m["total_time_ms"] for m in inference_metrics]
        feature_times = [m["feature_extraction_time_ms"] for m in inference_metrics]
        knn_times = [m["knn_inference_time_ms"] for m in inference_metrics]
        cpu_percents = [m["cpu_percent"] for m in inference_metrics]
        memory_mbs = [m["memory_mb"] for m in inference_metrics]
        
        summary = {
            "timestamp": datetime.now().isoformat(),
            "type": "summary",
            "scenario": "scenario1_tinylcm_no_drift",
            "total_inferences": len(inference_metrics),
            "avg_total_time_ms": np.mean(total_times),
            "std_total_time_ms": np.std(total_times),
            "min_total_time_ms": np.min(total_times),
            "max_total_time_ms": np.max(total_times),
            "avg_feature_extraction_time_ms": np.mean(feature_times),
            "avg_knn_inference_time_ms": np.mean(knn_times),
            "avg_cpu_percent": np.mean(cpu_percents),
            "avg_memory_mb": np.mean(memory_mbs),
            "max_memory_mb": np.max(memory_mbs),
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


def setup_tinylcm_components(config: Dict) -> Tuple[TFLiteFeatureExtractor, StandardScalerPCATransformer, LightweightKNN]:
    """Initialize TinyLCM components without drift detection."""
    tinylcm_config = config.get("tinylcm", {})
    
    # Feature extractor
    extractor_config = tinylcm_config.get("feature_extractor", {})
    feature_extractor = TFLiteFeatureExtractor(
        model_path=extractor_config.get("model_path"),
        feature_layer_index=extractor_config.get("feature_layer_index", 0),
        normalize_features=extractor_config.get("normalize_features", False)
    )
    
    # Feature transformer
    transformer_config = tinylcm_config.get("feature_transformation", {})
    feature_transformer = None
    if transformer_config.get("enabled", False):
        feature_transformer = StandardScalerPCATransformer.load_pretrained(
            transformer_config.get("model_path")
        )
    
    # KNN classifier
    classifier_config = tinylcm_config.get("adaptive_classifier", {})
    knn_classifier = LightweightKNN(
        k=classifier_config.get("k", 5),
        max_samples=classifier_config.get("max_samples", 200),
        distance_metric=classifier_config.get("distance_metric", "euclidean"),
        use_numpy=classifier_config.get("use_numpy", True)
    )
    
    # Load initial state if provided
    initial_state_path = classifier_config.get("initial_state_path")
    if initial_state_path and Path(initial_state_path).exists():
        knn_classifier.load_state(initial_state_path)
        logger.info(f"Loaded KNN initial state from {initial_state_path}")
    
    return feature_extractor, feature_transformer, knn_classifier


def main(config_path: str):
    """Main execution function."""
    global running, logger, config
    
    # Load configuration
    config = load_config(config_path)
    
    # Set up logging
    setup_logging(config.get("device", {}).get("log_level", "INFO"))
    logger.info("Starting Scenario 1 - TinyLCM without Drift Detection")
    
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
    
    # Initialize TinyLCM components BEFORE starting camera
    feature_extractor, feature_transformer, knn_classifier = setup_tinylcm_components(config)
    
    # Initialize performance logger
    performance_logger = PerformanceLogger()
    
    # Get inference interval from config
    app_config = config.get("application", {})
    inference_interval_ms = app_config.get("inference_interval_ms", 2000)
    inference_interval = inference_interval_ms / 1000.0
    
    # NOW start camera after all initialization
    camera.start()
    logger.info("Camera started")
    
    # Warm-up: Wait for camera to initialize and provide frames
    logger.info("Warming up camera...")
    frame = None
    for _ in range(20):
        frame = camera.get_frame()
        if frame is not None:
            break
        time.sleep(0.2)
    
    if frame is None:
        logger.error("Failed to get frame from camera after warm-up")
        sys.exit(1)
    
    # Main loop
    inference_count = 0
    last_inference_time = 0
    
    logger.info("Starting inference loop with TinyLCM (no drift detection)...")
    
    try:
        while running:
            current_time = time.time()
            
            # Check if it's time for the next inference
            if current_time - last_inference_time >= inference_interval:
                # Capture frame
                frame = camera.get_frame()
                if frame is None:
                    logger.warning("Failed to capture frame")
                    continue
                
                # Start total timing
                total_start = time.time()
                
                # Feature extraction
                feature_start = time.time()
                features = feature_extractor.extract(frame)
                
                # Apply transformation if available
                if feature_transformer:
                    features = feature_transformer.transform(features)
                
                feature_time = time.time() - feature_start
                
                # KNN inference
                knn_start = time.time()
                prediction, confidence = knn_classifier.predict(features)
                knn_time = time.time() - knn_start
                
                total_time = time.time() - total_start
                
                # Prepare prediction result
                prediction_result = {
                    "class": prediction,
                    "confidence": float(confidence),
                    "knn_samples": len(knn_classifier.data)
                }
                
                # Log performance metrics
                performance_logger.log_inference(
                    total_time=total_time,
                    feature_extraction_time=feature_time,
                    knn_inference_time=knn_time,
                    prediction=prediction_result
                )
                
                # Log if high confidence
                if confidence >= config.get("model", {}).get("threshold", 0.75):
                    logger.info(f"Inference #{inference_count}: {prediction} ({confidence:.2%})")
                
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
        logger.info("Scenario 1 completed")


if __name__ == "__main__":
    parser = argparse.ArgumentParser(description="Scenario 1 - TinyLCM without Drift Detection")
    parser.add_argument(
        "--config",
        type=str,
        default="config_scenario1.json",
        help="Path to configuration file"
    )
    
    args = parser.parse_args()
    main(args.config)