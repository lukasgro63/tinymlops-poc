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
import queue
import signal
import sys
import threading
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
feature_transformer = None  # Added global variable for feature transformer
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
                     drift_check_time: float,
                     prediction: Dict[str, Any],
                     drift_detected: bool = False):
        """Log a single inference with detailed timing breakdown."""
        # Get current resource usage
        cpu_percent = psutil.cpu_percent(interval=0.1)  # System-wide CPU usage
        memory_info = process.memory_info()
        memory_mb = memory_info.rss / (1024 * 1024)
        
        metric = {
            "timestamp": datetime.now().isoformat(),
            "type": "inference",
            "total_time_ms": total_time * 1000,
            "feature_extraction_time_ms": feature_extraction_time * 1000,
            "knn_inference_time_ms": knn_inference_time * 1000,
            "drift_check_time_ms": drift_check_time * 1000,
            "cpu_percent": cpu_percent,
            "memory_mb": memory_mb,
            "prediction": prediction,
            "drift_detected": drift_detected,
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
        drift_times = [m.get("drift_check_time_ms", 0) for m in inference_metrics]
        cpu_percents = [m["cpu_percent"] for m in inference_metrics]
        memory_mbs = [m["memory_mb"] for m in inference_metrics]
        drift_detections = [m.get("drift_detected", False) for m in inference_metrics]
        
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
            "avg_drift_check_time_ms": np.mean(drift_times),
            "avg_cpu_percent": np.mean(cpu_percents),
            "avg_memory_mb": np.mean(memory_mbs),
            "max_memory_mb": np.max(memory_mbs),
            "total_runtime_seconds": time.time() - self.start_time,
            "total_drift_events": 0,  # Always 0 for scenario1
            "drift_detection_rate": 0.0  # Always 0 for scenario1
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


def setup_tinylcm_components(config: Dict) -> InferencePipeline:
    """Initialize TinyLCM components without drift detection."""
    global feature_transformer  # Make feature_transformer accessible globally
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
        feature_transformer = StandardScalerPCATransformer(
            processor_path=transformer_config.get("model_path")
        )
        logger.info(f"Initialized StandardScalerPCATransformer from {transformer_config.get('model_path')}")
    else:
        logger.warning("Feature transformation is disabled in config!")
    
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
    loaded_initial_state = False
    
    if initial_state_path and Path(initial_state_path).exists():
        try:
            with open(initial_state_path, 'r') as f:
                loaded_state_data = json.load(f)
            
            if "classifier" in loaded_state_data and isinstance(loaded_state_data["classifier"], dict):
                knn_classifier.set_state(loaded_state_data["classifier"])
                logger.info(f"Loaded KNN initial state from {initial_state_path}")
                loaded_initial_state = True
            else:
                logger.warning(f"Invalid state file format in {initial_state_path}")
        except Exception as e:
            logger.error(f"Failed to load KNN state: {e}")
    
    # If no initial state was loaded, create random fallback data
    if not loaded_initial_state:
        logger.warning("No initial KNN state loaded. Creating random fallback data.")
        
        # Get feature dimensions by processing a test image
        test_image = np.zeros((224, 224, 3), dtype=np.uint8)
        test_features = feature_extractor.extract_features(test_image)
        logger.info(f"Test image raw features shape: {test_features.shape}")
        
        # Apply transformation if available
        if feature_transformer:
            test_features = feature_transformer.transform(test_features)
            logger.info(f"Test image transformed features shape: {test_features.shape}")
        else:
            logger.warning("No feature transformer during initialization - using raw features!")
        
        feature_dim = test_features.shape[0]
        logger.info(f"Final feature dimension for KNN: {feature_dim}")
        
        # Define classes to initialize (from config or default)
        classes = ["lego", "stone", "leaf", "negative"]  # Default classes
        samples_per_class = 10
        
        # Generate random features for each class
        X_init = []
        y_init = []
        
        for class_label in classes:
            # Generate random features for this class
            class_features = np.random.randn(samples_per_class, feature_dim).astype(np.float32)
            X_init.extend(class_features)
            y_init.extend([class_label] * samples_per_class)
        
        # Fit the classifier with random data
        knn_classifier.fit(np.array(X_init), y_init)
        logger.info(f"Initialized KNN with {len(X_init)} random samples for {len(classes)} classes")
    
    # Initialize operational monitor (same as scenario2_1)
    monitor_config = tinylcm_config.get("operational_monitor", {})
    operational_monitor = OperationalMonitor(
        storage_dir=tinylcm_config.get("data_logger", {}).get("log_dir", "./logs"),
        collect_system_metrics=monitor_config.get("track_system_metrics", True),
        system_metrics_interval=monitor_config.get("report_interval_seconds", 30)
    )
    logger.info(f"Operational monitor initialized with:")
    logger.info(f"- collect_system_metrics: {monitor_config.get('track_system_metrics', True)}")
    logger.info(f"- system_metrics_interval: {monitor_config.get('report_interval_seconds', 30)}")
    
    # Initialize data logger if enabled (same as scenario2_1)
    logger_config = tinylcm_config.get("data_logger", {})
    if logger_config.get("enabled", True):
        data_logger = DataLogger(
            storage_dir=logger_config.get("log_dir", "./logs")
        )
    else:
        data_logger = None
    
    # Initialize InferencePipeline WITHOUT drift detectors
    pipeline = InferencePipeline(
        feature_extractor=feature_extractor,
        classifier=knn_classifier,
        autonomous_monitors=[],  # Empty list - no drift detectors
        operational_monitor=operational_monitor,
        data_logger=data_logger
    )
    
    return pipeline


def main(config_path: str):
    """Main execution function."""
    global running, logger, config
    
    # Load configuration
    config = load_config(config_path)
    
    # Set up logging
    setup_logging(config.get("device", {}).get("log_level", "INFO"))
    logger.info("Starting Scenario 1 - TinyLCM without Drift Detection")
    
    # Create necessary directories (same as scenario2_1)
    logs_dir = Path("./logs")
    logs_dir.mkdir(exist_ok=True)
    state_dir = Path("./state")
    state_dir.mkdir(exist_ok=True)
    
    # Set up signal handler
    signal.signal(signal.SIGINT, signal_handler)
    signal.signal(signal.SIGTERM, signal_handler)
    
    # Initialize DeviceIDManager (same as scenario2_1)
    device_id_manager = DeviceIDManager(device_id_file="device_id.txt")
    device_id = device_id_manager.get_device_id()
    logger.info(f"Device ID: {device_id}")
    
    # Initialize camera
    camera_config = config.get("camera", {})
    camera = CameraHandler(
        resolution=tuple(camera_config.get("resolution", [640, 480])),
        framerate=camera_config.get("framerate", 1),
        rotation=camera_config.get("rotation", 0),
        auto_start=False  # Don't auto-start, we'll start manually after model init
    )
    
    # Initialize TinyLCM pipeline BEFORE starting camera
    pipeline = setup_tinylcm_components(config)
    
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
    warmup_attempts = 0
    max_warmup_attempts = 50  # Increase attempts
    
    while warmup_attempts < max_warmup_attempts and frame is None:
        frame = camera.get_frame()
        if frame is None:
            time.sleep(0.2)  # Wait 200ms between attempts
            warmup_attempts += 1
            if warmup_attempts % 10 == 0:
                logger.info(f"Still warming up camera... attempt {warmup_attempts}/{max_warmup_attempts}")
    
    if frame is None:
        logger.error(f"Failed to get frame from camera after {max_warmup_attempts} warm-up attempts")
        sys.exit(1)
    else:
        logger.info(f"Camera ready after {warmup_attempts} warm-up attempts")
    
    # Main loop
    inference_count = 0
    last_inference_time = 0
    
    logger.info("Starting inference loop with TinyLCM (no drift detection)...")
    
    try:
        while running:
            loop_start_time = time.time()
            
            # Capture frame
            frame = camera.get_frame()
            if frame is None:
                logger.warning("Failed to capture frame")
                time.sleep(0.1)
                continue
            
            # Convert RGBA to RGB if necessary
            if frame.shape[2] == 4:
                frame = cv2.cvtColor(frame, cv2.COLOR_RGBA2RGB)
            
            # Resize image to inference resolution
            target_size = tuple(camera_config.get("inference_resolution", [224, 224]))
            resized_frame = resize_image(frame, target_size)
            
            # Create sample ID (same as scenario2_1)
            sample_id = f"{device_id}_{int(time.time())}_{uuid.uuid4().hex[:8]}"
            
            # Start timing for the complete inference
            total_start_time = time.time()
            
            # Time feature extraction separately
            feature_start_time = time.time()
            
            # Extract features manually first (same as scenario2_1)
            features = pipeline.feature_extractor.extract_features(resized_frame)
            
            # Apply feature transformation if available
            if feature_transformer is not None:
                features = feature_transformer.transform(features)
            
            feature_extraction_time = time.time() - feature_start_time
            
            # Time KNN inference (which happens inside the pipeline)
            knn_start_time = time.time()
            
            # Use the pipeline.process() method (same as scenario2_1)
            try:
                result = pipeline.process(
                    input_data=features,  # Pass the transformed features
                    label=None,  # No ground truth label available
                    sample_id=sample_id,
                    timestamp=time.time(),
                    extract_features=False  # Tell pipeline not to extract features again
                )
                
                knn_inference_time = time.time() - knn_start_time
                
            except Exception as e:
                logger.error(f"Error processing frame: {e}")
                result = None
                knn_inference_time = 0
            
            # Calculate total time
            total_time = time.time() - total_start_time
            
            # Extract prediction and confidence from result
            if result:
                prediction = result.get("prediction", "unknown")
                confidence = result.get("confidence", 0.0)
                prediction_result = {
                    "class": prediction,
                    "confidence": float(confidence),
                    "knn_samples": len(pipeline.classifier.X_train) if hasattr(pipeline.classifier, 'X_train') else 0
                }
            else:
                prediction_result = {
                    "class": "error",
                    "confidence": 0.0,
                    "knn_samples": 0
                }
            
            # Since no drift detection, drift_check_time is 0
            drift_check_time = 0
            
            # Log performance metrics (same format as scenario2_1)
            performance_logger.log_inference(
                total_time=total_time,
                feature_extraction_time=feature_extraction_time,
                knn_inference_time=knn_inference_time,
                drift_check_time=drift_check_time,
                prediction=prediction_result,
                drift_detected=False  # Always False since no drift detection
            )
            
            # Log if high confidence
            if confidence >= config.get("model", {}).get("threshold", 0.75):
                logger.info(f"Inference #{inference_count}: {prediction} ({confidence:.2%})")
            
            inference_count += 1
            
            # Log progress every 10 inferences
            if inference_count % 10 == 0:
                logger.info(f"Completed {inference_count} inferences")
            
            # Sleep to maintain the desired framerate
            elapsed = time.time() - loop_start_time
            if elapsed < inference_interval:
                time.sleep(inference_interval - elapsed)
    
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