#!/usr/bin/env python3
"""
TinyLCM Autonomous Monitoring Example (Scenario 1)
--------------------------------------------------
This example demonstrates autonomous drift monitoring using the InferencePipeline
from the tinylcm library. It captures frames from a camera, processes them through 
the TinyLCM pipeline, and monitors for drift using PageHinkleyFeatureMonitor.

This is designed for a headless Raspberry Pi Zero 2W.
"""

import argparse
import cv2
import json
import logging
import os
import signal
import sys
import time
import uuid
from datetime import datetime
from pathlib import Path
from typing import Any, Dict, List, Optional, Tuple

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

# Import tinylcm components
from tinylcm.core.feature_extractors.tflite import TFLiteFeatureExtractor
from tinylcm.core.classifiers.knn import LightweightKNN
from tinylcm.core.drift_detection.features import PageHinkleyFeatureMonitor
from tinylcm.core.operational_monitor.monitor import OperationalMonitor
from tinylcm.core.data_logger.logger import DataLogger
from tinylcm.core.pipeline import InferencePipeline
from tinylcm.core.data_structures import FeatureSample
from tinylcm.client.sync_client import SyncClient
from tinylcm.utils.logging import setup_logger

# Global variables
logger = None
running = True
current_frame = None
config = None
sync_client = None


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
            logging.StreamHandler()
        ]
    )
    
    logger = logging.getLogger("tinylcm_scenario1")
    logger.info(f"Logging initialized at level {log_level}")


def create_directory_structure(config: Dict) -> None:
    """Create necessary directories from configuration."""
    # Create state directory
    state_dir = Path(config["tinylcm"]["state_manager"]["state_dir"])
    state_dir.mkdir(exist_ok=True)
    
    # Create log directory
    log_dir = Path(config["tinylcm"]["data_logger"]["log_dir"])
    log_dir.mkdir(exist_ok=True)
    
    # Create debug directory if needed
    if config["application"]["save_debug_frames"]:
        debug_dir = Path(config["application"]["debug_output_dir"])
        debug_dir.mkdir(exist_ok=True)
    
    # Create drift images directory
    if config["tinylcm"]["features"]["save_drift_images"]:
        drift_dir = Path("./drift_images")
        drift_dir.mkdir(exist_ok=True)


def on_drift_detected(drift_info: Dict[str, Any]) -> None:
    """Callback function for drift detection events.
    
    Args:
        drift_info: Dictionary containing information about the detected drift
    """
    timestamp = datetime.now().strftime("%Y-%m-%d %H:%M:%S")
    
    # Extract information from drift info
    detector_name = drift_info.get("detector", "Unknown")
    
    # Determine reason based on the detector type
    if "metric" in drift_info:
        metric = drift_info["metric"]
        current_value = drift_info.get("current_value", "unknown")
        reason = f"{metric} drift detected (current: {current_value})"
    else:
        reason = "Drift detected"
    
    # Log the drift detection
    logger.warning(f"DRIFT DETECTED by {detector_name}: {reason}")
    
    # Extract metrics from drift info
    metrics = {k: v for k, v in drift_info.items() if k not in ["detector", "timestamp"]}
    
    for key, value in metrics.items():
        logger.info(f"  {key}: {value}")
    
    # Global reference to the sync client and current frame
    global sync_client, current_frame
    
    # Save the current frame if drift image saving is enabled
    image_path = None
    if config["tinylcm"]["features"]["save_drift_images"] and current_frame is not None:
        drift_dir = Path("./drift_images")
        drift_dir.mkdir(exist_ok=True)
        
        image_path = drift_dir / f"drift_{timestamp.replace(' ', '_').replace(':', '-')}_{detector_name}.jpg"
        cv2.imwrite(str(image_path), current_frame)
        logger.info(f"Saved drift image to {image_path}")
    
    # If sync client is available, create and send a drift event package
    if sync_client:
        try:
            # Send metrics package with drift event details
            sync_client.synchronize_all()
            logger.info(f"Synchronized data with TinySphere after drift event")
        except Exception as e:
            logger.error(f"Failed to synchronize after drift event: {e}")


def handle_sigterm(signum, frame):
    """Handle SIGTERM signal for clean shutdown."""
    global running
    logger.info("Received termination signal, shutting down...")
    running = False


def process_labels(labels_path: str) -> List[str]:
    """Process label file to extract class names.
    
    Args:
        labels_path: Path to the labels file
        
    Returns:
        List of class names
    """
    with open(labels_path, 'r') as f:
        labels_raw = [line.strip() for line in f.readlines()]
    
    # Extract actual class names (format: "0 red" -> "red")
    labels = []
    for label in labels_raw:
        parts = label.split(' ', 1)
        if len(parts) > 1:
            labels.append(parts[1])
        else:
            labels.append(label)
    
    return labels


def main():
    """Main function for the example application."""
    global running, current_frame, config, sync_client
    
    parser = argparse.ArgumentParser(description="TinyLCM Autonomous Monitoring Example (Scenario 1)")
    parser.add_argument("--config", type=str, default="config_scenario1.json",
                        help="Path to the configuration file")
    args = parser.parse_args()
    
    # Load configuration
    config = load_config(args.config)
    
    # Set up logging
    setup_logging(config["device"]["log_level"])
    
    # Create directory structure
    create_directory_structure(config)
    
    # Register signal handlers for clean shutdown
    signal.signal(signal.SIGTERM, handle_sigterm)
    signal.signal(signal.SIGINT, handle_sigterm)
    
    logger.info("Starting TinyLCM Autonomous Monitoring Example with InferencePipeline (Scenario 1)")
    
    try:
        # Initialize DeviceIDManager
        device_id_manager = DeviceIDManager(device_id_file="device_id.txt")
        device_id = device_id_manager.get_device_id()
        logger.info(f"Device ID: {device_id}")
        
        # Initialize CameraHandler
        camera_config = config["camera"]
        camera = CameraHandler(
            resolution=tuple(camera_config["resolution"]),
            framerate=camera_config["framerate"],
            rotation=camera_config["rotation"]
        )
        
        # Load TFLite model labels
        model_config = config["model"]
        labels_path = model_config["labels_path"]
        labels = process_labels(labels_path)
        logger.info(f"Loaded {len(labels)} labels: {labels}")
        
        # Initialize TinyLCM components
        tinylcm_config = config["tinylcm"]
        
        # Initialize feature extractor
        feature_extractor_config = tinylcm_config["feature_extractor"]

        # Create preprocessor function for image conversion
        def preprocess_image(image):
            """Preprocess image for the TFLite model."""
            # Convert to RGB if needed
            if image.ndim == 3 and image.shape[2] == 3:
                # Check if image is already RGB (not BGR)
                if isinstance(image, np.ndarray) and image.flags['C_CONTIGUOUS']:
                    # Image is likely BGR from OpenCV
                    image = cv2.cvtColor(image, cv2.COLOR_BGR2RGB)
            elif image.ndim == 3 and image.shape[2] == 4:
                # BGRA to RGB
                image = cv2.cvtColor(image, cv2.COLOR_BGRA2RGB)

            return image

        feature_extractor = TFLiteFeatureExtractor(
            model_path=feature_extractor_config["model_path"],
            feature_layer_index=feature_extractor_config["feature_layer_index"],
            normalize_features=feature_extractor_config.get("normalize_features", True),
            lazy_loading=feature_extractor_config.get("lazy_loading", False),
            preprocessors=[preprocess_image]
        )
        
        # Initialize lightweight KNN
        knn_config = tinylcm_config["adaptive_classifier"]
        classifier = LightweightKNN(
            k=knn_config["k"],
            max_samples=knn_config["max_samples"], 
            distance_metric=knn_config["distance_metric"],
            use_numpy=knn_config["use_numpy"]
        )
        
        # Initialize some initial samples for the KNN classifier
        # This is necessary even in monitoring-only mode

        # First extract features from a test image to determine dimension
        # Create a small test image for feature extraction
        test_image = np.zeros((224, 224, 3), dtype=np.uint8)
        features = feature_extractor.extract_features(test_image)
        feature_dimension = features.shape[0] if features.ndim == 1 else features.shape[1]
        logger.info(f"Feature extractor output dimension: {feature_dimension}")

        # Generate some initial samples for classifier
        initial_features = np.random.randn(len(labels), feature_dimension)
        initial_timestamps = [time.time() - (i*10) for i in range(len(labels))]
        classifier.fit(initial_features, labels, initial_timestamps)
        
        # Initialize drift detector
        drift_detector_config = tinylcm_config["drift_detectors"][0]  # Use first detector
        
        # Create a function that extracts a specific feature dimension
        feature_index = drift_detector_config["feature_index"]
        feature_extractor_fn = lambda features: features[feature_index]
        
        drift_detector = PageHinkleyFeatureMonitor(
            feature_statistic_fn=feature_extractor_fn,
            delta=drift_detector_config["delta"],
            lambda_threshold=drift_detector_config["lambda_param"],
            warm_up_samples=drift_detector_config["min_samples"],
            reference_update_interval=drift_detector_config["warmup_samples"]
        )
        
        # Register drift callback
        drift_detector.register_callback(on_drift_detected)
        
        # Initialize operational monitor
        monitor_config = tinylcm_config["operational_monitor"]
        operational_monitor = OperationalMonitor(
            storage_dir=tinylcm_config["data_logger"]["log_dir"],
            collect_system_metrics=monitor_config.get("track_system_metrics", True),
            system_metrics_interval=monitor_config.get("report_interval_seconds", 30)
        )
        
        # Initialize data logger if enabled
        logger_config = tinylcm_config["data_logger"]
        if logger_config["enabled"]:
            data_logger = DataLogger(
                storage_dir=logger_config["log_dir"]
            )
        else:
            data_logger = None
        
        # Initialize SyncClient for communication with TinySphere
        sync_config = tinylcm_config["sync_client"]
        sync_client = SyncClient(
            server_url=sync_config["server_url"],
            api_key=sync_config["api_key"],
            device_id=device_id,
            max_retries=sync_config["max_retries"],
            auto_register=sync_config["auto_register"]
        )
        
        # Initialize InferencePipeline with the configured components
        pipeline = InferencePipeline(
            feature_extractor=feature_extractor,
            classifier=classifier,
            autonomous_monitors=[drift_detector],
            operational_monitor=operational_monitor,
            data_logger=data_logger
        )
        
        # Perform initial connection test to TinySphere
        try:
            status = sync_client.check_server_status()
            logger.info(f"Successfully connected to TinySphere server: {status}")
        except Exception as e:
            logger.warning(f"Could not connect to TinySphere server: {e}")
        
        # Start camera
        camera.start()
        logger.info("Camera started")
        
        # Warm-up: Wait for camera to initialize and provide frames
        logger.info("Warming up camera...")
        for _ in range(10):
            frame = camera.get_frame()
            if frame is not None:
                break
            time.sleep(0.1)
        
        if frame is None:
            logger.error("Failed to get frames from camera. Exiting.")
            running = False
        
        # Main inference loop
        logger.info("Starting main inference loop")
        frame_count = 0
        start_time = time.time()
        inference_interval = config["application"]["inference_interval_ms"] / 1000.0
        
        last_drift_check_time = time.time()
        drift_check_interval = 1.0  # Check for drift every 1 second
        
        last_sync_time = time.time()
        sync_interval = sync_config["sync_interval_seconds"]
        
        while running:
            loop_start_time = time.time()
            
            # Get frame from camera
            frame = camera.get_frame()
            if frame is None:
                logger.warning("Failed to get frame, skipping iteration")
                time.sleep(0.1)
                continue
            
            # Store current frame for drift visualization
            current_frame = frame.copy()
            
            # Resize image to target size
            target_size = tuple(camera_config["inference_resolution"])
            resized_frame = resize_image(frame, target_size)

            # Create sample ID
            sample_id = f"{device_id}_{int(time.time())}_{uuid.uuid4().hex[:8]}"

            # Process the frame directly with TinyLCM pipeline
            # The pipeline will use the preprocessor function to convert BGR to RGB
            result = pipeline.process(
                input_data=resized_frame,
                label=None,  # No ground truth label available
                sample_id=sample_id,
                timestamp=time.time(),
                extract_features=True  # Tell the pipeline to extract features from the input
            )
            
            # Extract prediction and confidence from result
            if result:
                prediction = result.get("prediction", "unknown")
                confidence = result.get("confidence", 0.0)
                if prediction:
                    logger.debug(f"Frame {frame_count}: Prediction={prediction}, Confidence={confidence:.4f}")
            
            # Update frame counter
            frame_count += 1
            if frame_count % 100 == 0:
                elapsed = time.time() - start_time
                fps = frame_count / elapsed
                logger.info(f"Processed {frame_count} frames. Current FPS: {fps:.2f}")
            
            # Periodically check for drift
            current_time = time.time()
            if current_time - last_drift_check_time >= drift_check_interval:
                pipeline.check_autonomous_drifts()
                last_drift_check_time = current_time
            
            # Periodically sync with TinySphere
            if current_time - last_sync_time >= sync_interval:
                try:
                    sync_client.synchronize_all()
                    last_sync_time = current_time
                except Exception as e:
                    logger.warning(f"Failed to synchronize with TinySphere: {e}")
            
            # Sleep to maintain the desired framerate
            elapsed = time.time() - loop_start_time
            if elapsed < inference_interval:
                time.sleep(inference_interval - elapsed)
        
    except KeyboardInterrupt:
        logger.info("Interrupted by user")
    except Exception as e:
        logger.error(f"Error in main loop: {e}", exc_info=True)
    finally:
        logger.info("Cleaning up resources")
        
        # Stop camera
        if 'camera' in locals():
            camera.stop()
            logger.info("Camera stopped")
        
        # Perform final sync with TinySphere
        logger.info("Performing final sync with TinySphere")
        if 'sync_client' in locals() and sync_client:
            try:
                sync_client.synchronize_all()
            except Exception:
                pass
            sync_client.close()
            logger.info("Sync client closed")
        
        logger.info("TinyLCM Autonomous Monitoring Example completed")


if __name__ == "__main__":
    main()