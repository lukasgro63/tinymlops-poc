#!/usr/bin/env python3
"""
TinyLCM Autonomous Monitoring Example (Scenario 1)
-------------------------------------------------
Demonstrates the use of TinyLCM for autonomous drift monitoring without adaptation.
- Uses a quantized TFLite model for object classification
- Implements autonomous drift detection using proxy metrics
- Sends detected drift events to TinySphere server
- Logs operational metrics for monitoring
"""

import argparse
import json
import logging
import os
import signal
import sys
import threading
import time
import uuid
from datetime import datetime
from pathlib import Path
from typing import Dict, List, Optional, Tuple, Union

import cv2
import numpy as np

# Add the parent directory to sys.path to be able to import utils
sys.path.append(os.path.dirname(os.path.dirname(os.path.abspath(__file__))))

# Import utils and TinyLCM
from utils.camera_handler import CameraHandler
from utils.device_id_manager import DeviceIDManager
from utils.preprocessors import (
    prepare_input_tensor_quantized,
    resize_image,
    extract_features_from_interpreter,
    predict_with_threshold
)

# Import TinyLCM modules
import tinylcm
from tinylcm.core.drift_detection import (
    EWMAConfidenceMonitor,
    PageHinkleyFeatureMonitor,
    PredictionDistributionMonitor
)
from tinylcm.core.operational_monitor.monitor import OperationalMonitor
from tinylcm.core.data_logger import DataLogger
from tinylcm.core.data_structures import FeatureSample
from utils.sync_client import ExtendedSyncClient
from tinylcm.utils.logging import setup_logger

# Try to import TFLite
try:
    import tflite_runtime.interpreter as tflite
    TFLITE_AVAILABLE = True
except ImportError:
    try:
        import tensorflow.lite as tflite
        TFLITE_AVAILABLE = True
    except ImportError:
        TFLITE_AVAILABLE = False
        logger = logging.getLogger(__name__)
        logger.error("Neither tflite_runtime nor tensorflow could be imported. TFLite inference will not be available.")

# Global variables
running = True
logger = None
sync_client = None
current_sample = None
current_frame = None
config = None


class StoneDetector:
    """Performs object detection using a quantized TFLite model."""
    
    def __init__(
        self,
        model_path: str,
        labels_path: str,
        threshold: float = 0.6,
        input_mean: float = 127.5,
        input_std: float = 127.5,
        feature_layer_index: int = -2
    ):
        """Initialize the stone detector.
        
        Args:
            model_path: Path to the TFLite model file
            labels_path: Path to the text file containing class labels
            threshold: Confidence threshold for predictions
            input_mean: Mean value for input normalization
            input_std: Standard deviation value for input normalization
            feature_layer_index: Index of the layer to extract features from
        """
        self.model_path = os.path.abspath(model_path)
        self.labels_path = os.path.abspath(labels_path)
        self.threshold = threshold
        self.input_mean = input_mean
        self.input_std = input_std
        self.feature_layer_index = feature_layer_index
        
        # Load the labels
        self.labels = self._load_labels()
        
        # Load the TFLite model
        if TFLITE_AVAILABLE:
            self.interpreter = self._load_model()
            self.input_details = self.interpreter.get_input_details()[0]
            self.output_details = self.interpreter.get_output_details()[0]
            logger.info(f"Model loaded with input shape {self.input_details['shape']} "
                       f"and dtype {self.input_details['dtype']}")
        else:
            logger.error("TFLite is not available. Cannot load model.")
            self.interpreter = None
            self.input_details = None
            self.output_details = None
    
    def _load_model(self) -> 'tflite.Interpreter':
        """Load the TFLite model."""
        logger.info(f"Loading TFLite model from {self.model_path}")
        try:
            # Load the TFLite model
            interpreter = tflite.Interpreter(model_path=self.model_path)
            interpreter.allocate_tensors()
            return interpreter
        except Exception as e:
            logger.error(f"Failed to load TFLite model: {e}")
            raise
    
    def _load_labels(self) -> List[str]:
        """Load class labels from a text file."""
        logger.info(f"Loading labels from {self.labels_path}")
        try:
            with open(self.labels_path, 'r') as f:
                labels = [line.strip() for line in f.readlines()]
            logger.info(f"Loaded {len(labels)} labels: {labels}")
            return labels
        except Exception as e:
            logger.error(f"Failed to load labels: {e}")
            return [f"class_{i}" for i in range(10)]  # Default placeholder labels
    
    def detect(self, image: np.ndarray) -> Tuple[str, float, np.ndarray]:
        """Perform object detection on an image.
        
        Args:
            image: Input image (BGR format from OpenCV)
            
        Returns:
            Tuple of (prediction_label, confidence, features)
        """
        if self.interpreter is None:
            logger.error("Interpreter is not available")
            return "unknown", 0.0, np.zeros((10,))
        
        # Preprocess the image for the quantized model
        input_tensor = prepare_input_tensor_quantized(
            image, 
            self.input_details,
            convert_to_rgb=True
        )
        
        # Set the input tensor
        self.interpreter.set_tensor(self.input_details['index'], input_tensor)
        
        # Run inference
        self.interpreter.invoke()
        
        # Get output tensor
        output_tensor = self.interpreter.get_tensor(self.output_details['index'])
        
        # Extract features for drift detection (from an earlier layer if specified)
        if self.feature_layer_index != -1:
            features = extract_features_from_interpreter(
                self.interpreter, 
                output_layer_index=self.feature_layer_index
            )
            # Flatten features if they're multi-dimensional
            features = features.flatten()
        else:
            # Use the output probabilities as features
            features = output_tensor.flatten()
        
        # Get prediction
        predicted_labels, confidences = predict_with_threshold(
            output_tensor, 
            self.labels, 
            threshold=self.threshold
        )
        
        if predicted_labels and confidences:
            return predicted_labels[0], confidences[0], features
        else:
            return "unknown", 0.0, features


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


def on_drift_detected(detector_name: str, reason: str, metrics: Dict) -> None:
    """Callback function for drift detection events."""
    timestamp = datetime.now().strftime("%Y-%m-%d %H:%M:%S")
    logger.warning(f"DRIFT DETECTED by {detector_name}: {reason}")
    
    for key, value in metrics.items():
        logger.info(f"  {key}: {value}")
    
    # Global reference to the sync client and current sample
    global sync_client, current_sample, current_frame
    
    # Send the drift event to TinySphere if sync client is available
    if sync_client:
        # Save the current frame if drift image saving is enabled
        image_path = None
        if config["tinylcm"]["features"]["save_drift_images"] and current_frame is not None:
            drift_dir = Path("./drift_images")
            drift_dir.mkdir(exist_ok=True)
            
            image_path = drift_dir / f"drift_{timestamp.replace(' ', '_').replace(':', '-')}_{detector_name}.jpg"
            cv2.imwrite(str(image_path), current_frame)
            logger.info(f"Saved drift image to {image_path}")
        
        # Send the drift event to TinySphere
        success = sync_client.create_and_send_drift_event_package(
            detector_name=detector_name,
            reason=reason,
            metrics=metrics,
            sample=current_sample,
            image_path=str(image_path) if image_path else None
        )
        
        logger.info(f"Drift event sent to TinySphere: {'Success' if success else 'Failed'}")


def handle_sigterm(signum, frame):
    """Handle SIGTERM signal for clean shutdown."""
    global running
    logger.info("Received termination signal, shutting down...")
    running = False


class TinyLCMMonitoringPipeline:
    """Simple monitoring pipeline for autonomous drift detection.
    
    This is a simplified version of the TinyLCM InferencePipeline
    that only focuses on autonomous drift detection and monitoring
    without using the full LightweightKNN and feature extraction components.
    """
    
    def __init__(
        self,
        drift_detectors: List,
        operational_monitor: Optional[OperationalMonitor] = None,
        data_logger: Optional[DataLogger] = None
    ):
        """Initialize the monitoring pipeline.
        
        Args:
            drift_detectors: List of autonomous drift detectors
            operational_monitor: Optional operational monitor
            data_logger: Optional data logger
        """
        self.drift_detectors = drift_detectors
        self.operational_monitor = operational_monitor
        self.data_logger = data_logger
        logger.info(f"Initialized TinyLCMMonitoringPipeline with {len(drift_detectors)} drift detectors")
    
    def process(self, sample: FeatureSample) -> None:
        """Process a sample with the monitoring pipeline."""
        # Start timing for performance monitoring
        start_time = time.time()
        
        # Update all drift detectors with the sample
        for detector in self.drift_detectors:
            # Create record with proper format for drift detectors
            record = {
                "features": sample.features,
                "prediction": sample.prediction,
                "confidence": sample.metadata.get("confidence", 0.0),
                "sample_id": sample.sample_id,
                "timestamp": sample.timestamp
            }
            detector.update(record)
        
        # Track inference time with the operational monitor
        if self.operational_monitor:
            inference_time = time.time() - start_time
            self.operational_monitor.track_inference(
                input_id=sample.sample_id,
                prediction=sample.prediction,
                confidence=sample.metadata.get("confidence", 0.0),
                latency_ms=inference_time * 1000,  # Convert seconds to milliseconds
                features=sample.features
            )
        
        # Log the sample if a data logger is available
        if self.data_logger:
            self.data_logger.log_sample(
                input_data=sample.features,
                prediction=sample.prediction,
                confidence=sample.metadata.get("confidence", 0.0),
                metadata={
                    "sample_id": sample.sample_id,
                    "timestamp": sample.timestamp
                }
            )
    
    def check_autonomous_drifts(self) -> bool:
        """Check all autonomous drift detectors for drift."""
        drift_detected = False
        
        for detector in self.drift_detectors:
            result, drift_info = detector.check_for_drift()
            if result:
                drift_detected = True
        
        return drift_detected


def main():
    """Main function for the TinyLCM Scenario 1 example."""
    global running, config, sync_client
    
    # Parse command-line arguments
    parser = argparse.ArgumentParser(description="TinyLCM Autonomous Monitoring Example")
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
    
    logger.info("Starting TinyLCM Autonomous Monitoring Example (Scenario 1)")
    
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
        
        # Initialize StoneDetector (TFLite classifier)
        model_config = config["model"]
        stone_detector = StoneDetector(
            model_path=model_config["model_path"],
            labels_path=model_config["labels_path"],
            threshold=model_config["threshold"],
            input_mean=model_config["input_mean"],
            input_std=model_config["input_std"],
            feature_layer_index=model_config["feature_layer_index"]
        )
        
        # Initialize TinyLCM components
        tinylcm_config = config["tinylcm"]
        
        # Initialize drift detectors
        drift_detectors = []
        for detector_config in tinylcm_config["drift_detectors"]:
            if detector_config["type"] == "EWMAConfidenceMonitor":
                drift_detector = EWMAConfidenceMonitor(
                    lambda_param=detector_config["alpha"],
                    threshold_factor=detector_config["threshold_factor"],
                    drift_window=detector_config["window_size"],
                    warm_up_samples=detector_config["min_samples"],
                    reference_update_interval=detector_config["warmup_samples"]
                )
                drift_detectors.append(drift_detector)
                logger.info(f"Initialized EWMAConfidenceMonitor with alpha={detector_config['alpha']}")
            
            elif detector_config["type"] == "PageHinkleyFeatureMonitor":
                # Create a function that extracts a specific feature
                feature_index = detector_config["feature_index"]
                feature_extractor = lambda features: features[feature_index]
                
                drift_detector = PageHinkleyFeatureMonitor(
                    feature_statistic_fn=feature_extractor,
                    delta=detector_config["delta"],
                    lambda_threshold=detector_config["lambda_param"],
                    warm_up_samples=detector_config["min_samples"],
                    reference_update_interval=detector_config["warmup_samples"]
                )
                drift_detectors.append(drift_detector)
                logger.info(f"Initialized PageHinkleyFeatureMonitor for feature index {detector_config['feature_index']}")
            
            elif detector_config["type"] == "PredictionDistributionMonitor":
                drift_detector = PredictionDistributionMonitor(
                    window_size=detector_config.get("window_size", 100),
                    threshold=detector_config.get("threshold", 0.15),
                    method=detector_config.get("method", "block"),
                    min_samples=detector_config.get("min_samples", 100)
                )
                drift_detectors.append(drift_detector)
                logger.info(f"Initialized PredictionDistributionMonitor with window_size={detector_config.get('window_size', 100)}, threshold={detector_config.get('threshold', 0.15)}")
        
        # Initialize operational monitor
        monitor_config = tinylcm_config["operational_monitor"]
        operational_monitor = OperationalMonitor(
            storage_dir=monitor_config.get("storage_dir", "./metrics"),
            collect_system_metrics=monitor_config.get("track_system_metrics", True),
            system_metrics_interval=monitor_config.get("report_interval_seconds", 30)
        )
        
        # Initialize data logger if enabled
        if tinylcm_config["data_logger"]["enabled"]:
            logger_config = tinylcm_config["data_logger"]
            data_logger = DataLogger(
                storage_dir=logger_config["log_dir"]
            )
        else:
            data_logger = None
        
        # Initialize SyncClient for communication with TinySphere
        sync_config = tinylcm_config["sync_client"]
        sync_client = ExtendedSyncClient(
            server_url=sync_config["server_url"],
            device_id=device_id,
            api_key=sync_config["api_key"],
            sync_dir=sync_config.get("sync_dir", "./sync_data"),
            sync_interval_seconds=sync_config["sync_interval_seconds"],
            max_retries=sync_config["max_retries"],
            auto_register=sync_config["auto_register"]
        )
        
        # Perform initial connection test
        if sync_client.check_connection():
            logger.info("Successfully connected to TinySphere server")
        else:
            logger.warning("Could not connect to TinySphere server - will continue and try again later")
        
        # Initialize simplified monitoring pipeline (without LightweightKNN and FeatureExtractor)
        pipeline = TinyLCMMonitoringPipeline(
            drift_detectors=drift_detectors,
            operational_monitor=operational_monitor,
            data_logger=data_logger
        )
        
        # Register drift detection callbacks
        for detector in drift_detectors:
            detector.register_callback(on_drift_detected)
        
        # Start camera
        camera.start()
        logger.info("Camera started")
        
        # Warm-up: Wait for a few frames
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
        inference_interval = config["application"]["inference_interval_ms"] / 1000.0
        
        last_drift_check_time = time.time()
        drift_check_interval = 1.0  # Check for drift every 1 second
        
        while running:
            loop_start_time = time.time()
            
            # Get frame from camera
            frame = camera.get_frame()
            if frame is None:
                logger.warning("Failed to get frame, skipping iteration")
                time.sleep(0.1)
                continue
            
            # Resize image to target size
            target_size = tuple(camera_config["inference_resolution"])
            resized_frame = resize_image(frame, target_size)
            
            # Perform object detection
            prediction, confidence, features = stone_detector.detect(resized_frame)
            
            # Create sample ID
            sample_id = f"{device_id}_{int(time.time())}_{uuid.uuid4().hex[:8]}"
            
            # Create FeatureSample for TinyLCM
            sample = FeatureSample(
                sample_id=sample_id,
                features=features,
                prediction=prediction,
                timestamp=time.time(),
                metadata={"confidence": confidence}
            )
            
            # Update global current sample and frame for drift detection callback
            global current_sample, current_frame
            current_sample = sample
            current_frame = frame.copy() if frame is not None else None
            
            # Process sample with TinyLCM monitoring pipeline
            pipeline.process(sample)
            
            # Check for drift periodically
            current_time = time.time()
            if current_time - last_drift_check_time >= drift_check_interval:
                if pipeline.check_autonomous_drifts():
                    logger.info("Drift detected during periodic check")
                last_drift_check_time = current_time
                
                # Check if it's time to sync with TinySphere
                if sync_client and sync_client.should_sync():
                    try:
                        # Get operational metrics from monitor
                        if operational_monitor:
                            try:
                                metrics = operational_monitor.get_current_metrics()
                                # Send metrics to TinySphere
                                logger.debug("Sending metrics to TinySphere")
                                sync_client.create_and_send_metrics_package(metrics)
                            except Exception as e:
                                logger.error(f"Error getting metrics: {e}")
                        
                        # Sync all pending packages
                        logger.debug("Syncing pending packages with TinySphere")
                        sync_results = sync_client.sync_all_pending_packages()
                        if sync_results:
                            logger.info(f"Synced {len(sync_results)} packages with TinySphere")
                        else:
                            logger.debug("No packages to sync with TinySphere")
                    except Exception as e:
                        logger.error(f"Error during TinySphere sync: {e}")
            
            # Display frames if enabled
            if config["application"]["display_frames"]:
                # Add prediction text to frame
                text = f"{prediction}: {confidence:.2f}"
                cv2.putText(frame, text, (10, 30), cv2.FONT_HERSHEY_SIMPLEX, 
                           1, (0, 255, 0), 2, cv2.LINE_AA)
                
                cv2.imshow('TinyLCM Monitoring', frame)
                key = cv2.waitKey(1) & 0xFF
                if key == ord('q'):
                    logger.info("User requested exit")
                    running = False
                    break
            
            # Save debug frames if enabled
            if config["application"]["save_debug_frames"] and frame_count % 100 == 0:
                debug_dir = Path(config["application"]["debug_output_dir"])
                frame_path = debug_dir / f"frame_{frame_count:06d}_{prediction}.jpg"
                cv2.imwrite(str(frame_path), frame)
            
            # Increment frame counter
            frame_count += 1
            
            # Maintain inference rate
            elapsed = time.time() - loop_start_time
            sleep_time = max(0, inference_interval - elapsed)
            if sleep_time > 0:
                time.sleep(sleep_time)
            
            # Log stats occasionally
            if frame_count % 100 == 0:
                fps = 1.0 / (time.time() - loop_start_time + sleep_time)
                logger.info(f"Processed {frame_count} frames. Current FPS: {fps:.2f}")
        
        logger.info("Inference loop stopped")
    
    except Exception as e:
        logger.exception(f"Error in main loop: {e}")
    
    finally:
        logger.info("Cleaning up resources")
        
        # Stop the camera
        try:
            camera.stop()
            logger.info("Camera stopped")
        except:
            pass
        
        # Close the sync client
        try:
            if sync_client:
                # Perform one final sync before closing
                logger.info("Performing final sync with TinySphere")
                sync_client.sync_all_pending_packages()
                sync_client.close()
                logger.info("Sync client closed")
        except Exception as e:
            logger.error(f"Error closing sync client: {e}")
        
        # Clean up OpenCV windows
        if config and config.get("application", {}).get("display_frames", False):
            cv2.destroyAllWindows()
        
        logger.info("TinyLCM Autonomous Monitoring Example completed")


if __name__ == "__main__":
    main()