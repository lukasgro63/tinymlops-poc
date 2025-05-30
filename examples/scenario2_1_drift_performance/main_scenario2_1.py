#!/usr/bin/env python3
"""
TinyLCM Autonomous Monitoring Example (Scenario 2)
--------------------------------------------------
This example demonstrates autonomous drift monitoring using the InferencePipeline
from the tinylcm library. It captures frames from a camera, processes them through 
the TinyLCM pipeline, and monitors for drift for a 5-class model using KNNDistanceMonitor.

This is designed for a headless Raspberry Pi Zero 2W.
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
from tinylcm.core.drift_detection.confidence import (
    EWMAConfidenceMonitor, PageHinkleyConfidenceMonitor)
from tinylcm.core.drift_detection.features import (FeatureMonitor,
                                                   KNNDistanceMonitor,
                                                   PageHinkleyFeatureMonitor)
# Import tinylcm components
from tinylcm.core.feature_extractors.tflite import TFLiteFeatureExtractor
from tinylcm.core.feature_transformers.pca import PCATransformer
from tinylcm.core.feature_transformers.standard_scaler_pca import \
    StandardScalerPCATransformer
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
    log_file = logs_dir / f"tinylcm_scenario2_{datetime.now().strftime('%Y%m%d_%H%M%S')}.log"
    
    logging.basicConfig(
        level=numeric_level,
        format='%(asctime)s - %(name)s - %(levelname)s - %(message)s',
        handlers=[
            logging.FileHandler(log_file),
            logging.StreamHandler()
        ]
    )
    
    logger = logging.getLogger("tinylcm_scenario2")
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


class PerformanceLogger:
    """Enhanced performance logger for TinyLCM scenario with drift detection."""
    
    def __init__(self, log_dir: str = "./logs"):
        """Initialize the performance logger."""
        self.log_dir = Path(log_dir)
        self.log_dir.mkdir(exist_ok=True)
        
        # Create performance log file
        timestamp = datetime.now().strftime('%Y%m%d_%H%M%S')
        self.log_file = self.log_dir / f"performance_scenario2_1_{timestamp}.json"
        
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
            "scenario": "scenario2_1_drift_performance",
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
    
    def log_drift_event(self, detector_name: str, drift_type: str, metrics: Dict[str, Any]):
        """Log a drift detection event."""
        drift_event = {
            "timestamp": datetime.now().isoformat(),
            "type": "drift_event",
            "detector_name": detector_name,
            "drift_type": drift_type,
            "metrics": metrics,
            "uptime_seconds": time.time() - self.start_time
        }
        self._append_to_log(drift_event)
    
    def log_summary(self):
        """Log summary statistics."""
        if not self.metrics:
            return
        
        inference_metrics = [m for m in self.metrics if m["type"] == "inference"]
        drift_events = [m for m in self.metrics if m["type"] == "drift_event"]
        
        if not inference_metrics:
            return
        
        total_times = [m["total_time_ms"] for m in inference_metrics]
        feature_times = [m["feature_extraction_time_ms"] for m in inference_metrics]
        knn_times = [m["knn_inference_time_ms"] for m in inference_metrics]
        drift_times = [m["drift_check_time_ms"] for m in inference_metrics]
        cpu_percents = [m["cpu_percent"] for m in inference_metrics]
        memory_mbs = [m["memory_mb"] for m in inference_metrics]
        drift_detections = [m["drift_detected"] for m in inference_metrics]
        
        summary = {
            "timestamp": datetime.now().isoformat(),
            "type": "summary",
            "scenario": "scenario2_1_drift_performance",
            "total_inferences": len(inference_metrics),
            "total_drift_events": len(drift_events),
            "drift_detection_rate": sum(drift_detections) / len(drift_detections) if drift_detections else 0,
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


def on_drift_detected(drift_info: Dict[str, Any], *args) -> None:
    """Callback function for drift detection events.

    Args:
        drift_info: Dictionary containing information about the detected drift
        *args: Additional arguments (ignored, but needed to handle the callback)
    """
    global performance_logger
    timestamp = datetime.now().strftime("%Y-%m-%d %H:%M:%S")

    # Check if we're in cooldown - this shouldn't happen normally since this callback
    # should only be called when NOT in cooldown, but let's be safe
    if drift_info.get("in_cooldown_period", False):
        # This is informational only as it shouldn't normally happen
        logger.debug(f"Drift callback executed while in cooldown period - this is unexpected")
        return

    # Extract information from drift info
    detector_name = drift_info.get("detector_type", "Unknown")
    if detector_name == "Unknown":
        # Fallback if detector_type is not set
        detector_name = drift_info.get("detector", "Unknown")
    
    # Ensure detector_name is directly usable in logs and drift events
    if detector_name == "KNNDistanceMonitor":
        drift_type = "knn_distance"
    elif detector_name == "EWMAConfidenceMonitor":
        drift_type = "confidence"
    elif detector_name == "FeatureMonitor":
        drift_type = "feature"
    else:
        drift_type = detector_name.lower()

    # Determine reason based on the detector type
    if "metric" in drift_info:
        metric = drift_info["metric"]
        current_value = drift_info.get("current_value", "unknown")
        threshold = drift_info.get("threshold", "unknown")
        reason = f"{metric} drift detected (current: {current_value}, threshold: {threshold})"
    else:
        reason = "Drift detected"

    # Log the drift detection with more visibility (use WARNING level)
    logger.warning(f"DRIFT DETECTED by {detector_name} (drift_type={drift_type}): {reason}")
    
    # Add specific indicators based on detector name
    if detector_name == "KNNDistanceMonitor" and "neighbor_distance" in reason:
        logger.warning(f"!!! KNN DISTANCE-BASED DRIFT DETECTED !!! Capturing image for analysis...")

    # Extract metrics from drift info - but exclude some known keys to reduce noise
    excluded_keys = ["detector", "detector_type", "detector_id", "timestamp", "sample_id",
                     "drift_detected", "in_cooldown_period"]
    metrics = {k: v for k, v in drift_info.items() if k not in excluded_keys}

    # Log detailed drift metrics
    logger.info(f"Drift detailed metrics from {detector_name}:")
    for key, value in metrics.items():
        logger.info(f"  {key}: {value}")
    
    # Log to performance logger if available
    if 'performance_logger' in globals() and performance_logger:
        performance_logger.log_drift_event(detector_name, drift_type, metrics)
    
    # Track drift event in operational logs
    if 'operational_monitor' in globals() and 'operational_monitor' in locals():
        # Create unique operation ID
        drift_op_id = f"drift_{int(time.time())}_{uuid.uuid4().hex[:6]}"
        
        # Prepare metadata with all drift information
        drift_metadata = {
            "detector": detector_name,
            "drift_type": drift_type,
            "reason": reason,
            "metrics": metrics,
            "current_value": drift_info.get("current_value"),
            "threshold": drift_info.get("threshold"),
            "metric": drift_info.get("metric")
        }
        
        # Track operation in operational monitor
        operational_monitor.track_operation(
            operation_id=drift_op_id,
            operation_type="drift_detection",
            result=drift_type,
            success=True,
            metadata=drift_metadata,
            timestamp=time.time()
        )
        logger.info(f"Drift event recorded in operational logs with ID: {drift_op_id}")
    
    # Global reference to the current frame
    global current_frame
    
    # Save the current frame if drift image saving is enabled (but not for sync)
    image_path = None
    if config["tinylcm"]["features"]["save_drift_images"] and current_frame is not None:
        # Get device ID
        device_id = None
        if 'device_id_manager' in globals() and 'device_id_manager' in locals():
            device_id = device_id_manager.get_device_id()
        else:
            import socket
            device_id = f"device-{socket.gethostname()}"

        # Get current date in YYYYMMDD format for bucket structure
        current_date = datetime.now().strftime("%Y%m%d")

        # Create drift directory structure for compatibility with tinysphere bucket format:
        # device_id/drift_type/date/filename.jpg
        # drift_type is already set above, no need to recompute it here

        # Create base drift directory
        drift_dir = Path("./drift_images")
        drift_dir.mkdir(exist_ok=True)

        # Create device directory
        device_dir = drift_dir / device_id
        device_dir.mkdir(exist_ok=True)

        # Create drift type directory
        drift_type_dir = device_dir / drift_type
        drift_type_dir.mkdir(exist_ok=True)

        # Create date directory
        date_dir = drift_type_dir / current_date
        date_dir.mkdir(exist_ok=True)

        # Create filename with timestamp and event ID
        event_id = f"event_{timestamp.replace(' ', '_').replace(':', '-')}_{uuid.uuid4().hex[:8]}"
        image_filename = f"{event_id}.jpg"

        # Convert BGR to RGB for correct color visualization
        rgb_frame = cv2.cvtColor(current_frame, cv2.COLOR_BGR2RGB)

        # Full path to the image
        image_path = date_dir / image_filename
        cv2.imwrite(str(image_path), rgb_frame)
        logger.warning(f"DRIFT EVENT IMAGE SAVED ({drift_type}): {image_path}")


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


def apply_feature_transformation(features: np.ndarray) -> np.ndarray:
    """Apply feature transformation (like PCA) to extracted features.
    
    Args:
        features: Raw features extracted from the model
        
    Returns:
        Transformed features with potentially reduced dimensionality
    """
    global feature_transformer
    
    if feature_transformer is None:
        logger.warning("Feature transformer not initialized, returning original features")
        return features
    
    try:
        # Log the transformer type for debugging
        transformer_type = type(feature_transformer).__name__
            
        # Apply transformation
        transformed_features = feature_transformer.transform(features)
        
        # Log occasional debug information
        if hasattr(apply_feature_transformation, 'call_count'):
            apply_feature_transformation.call_count += 1
        else:
            apply_feature_transformation.call_count = 1
            
        # Log every 100th transformation for regular monitoring
        if apply_feature_transformation.call_count % 100 == 0:
            logger.info(f"Feature transformation ({transformer_type}): input shape: {features.shape}, output shape: {transformed_features.shape}")
        
        # Log the first few transformations in more detail to verify correct operation
        if apply_feature_transformation.call_count <= 5:
            logger.info(f"Initial transformation ({transformer_type}) - Details:")
            logger.info(f"  Input shape: {features.shape}")
            logger.info(f"  Output shape: {transformed_features.shape}")
            logger.info(f"  Input range: [{np.min(features):.4f}, {np.max(features):.4f}]")
            logger.info(f"  Output range: [{np.min(transformed_features):.4f}, {np.max(transformed_features):.4f}]")
            
        return transformed_features
    
    except Exception as e:
        logger.error(f"Error applying feature transformation: {e}")
        # Return original features on error
        return features


def main():
    """Main function for the example application."""
    global running, current_frame, config, feature_transformer, performance_logger
    
    parser = argparse.ArgumentParser(description="TinyLCM Autonomous Monitoring Example (Scenario 2.1 - Performance)")
    parser.add_argument("--config", type=str, default="config_scenario2_1.json",
                        help="Path to the configuration file")
    args = parser.parse_args()
    
    # Load configuration
    config = load_config(args.config)
    
    # Set up logging
    setup_logging(config["device"]["log_level"])
    
    # Create directory structure
    create_directory_structure(config)
    
    # Initialize performance logger
    performance_logger = PerformanceLogger()
    
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
        
        # Initialize feature transformer if enabled in config
        if "feature_transformation" in tinylcm_config and tinylcm_config["feature_transformation"].get("enabled", False):
            transform_config = tinylcm_config["feature_transformation"]
            
            if transform_config["type"] == "PCA":
                logger.info(f"Initializing PCA transformer with model: {transform_config['model_path']}")
                feature_transformer = PCATransformer(
                    model_path=transform_config["model_path"],
                    n_components=transform_config.get("n_components", 50)
                )
                logger.info(f"PCA transformer initialized, will reduce features to {transform_config.get('n_components', 50)} dimensions")
            elif transform_config["type"] == "StandardScalerPCA":
                logger.info(f"Initializing StandardScalerPCA transformer with processor: {transform_config['model_path']}")
                feature_transformer = StandardScalerPCATransformer(
                    processor_path=transform_config["model_path"]
                )
                logger.info(f"StandardScalerPCA transformer initialized, applying standardization and PCA transformation")
            else:
                logger.warning(f"Unknown feature transformation type: {transform_config['type']}")
        else:
            logger.info("Feature transformation not enabled in config")
        
        # Initialize lightweight KNN
        knn_config = tinylcm_config["adaptive_classifier"]
        classifier = LightweightKNN(
            k=knn_config["k"],
            max_samples=knn_config["max_samples"],
            distance_metric=knn_config["distance_metric"],
            use_numpy=knn_config["use_numpy"]
        )

        # Log KNN classifier information
        logger.info(f"KNN INIT INFO: KNN class type = {type(classifier).__name__}")
        logger.info(f"KNN INIT INFO: KNN class module = {type(classifier).__module__}")
        logger.info(f"KNN INIT INFO: KNN predict_proba method ID = {id(classifier.predict_proba)}")
        logger.info(f"KNN INIT INFO: Distance metric = {knn_config['distance_metric']}")
        
        # Lade den initialen KNN-Zustand, falls vorhanden
        initial_state_path_str = knn_config.get("initial_state_path")
        loaded_initial_state = False
        
        if initial_state_path_str:
            initial_state_path = Path(initial_state_path_str)
            if initial_state_path.exists():
                logger.info(f"Versuche, initialen k-NN Zustand zu laden von: {initial_state_path}")
                try:
                    with open(initial_state_path, 'r') as f:
                        loaded_state_data = json.load(f)
                    
                    if "classifier" in loaded_state_data and isinstance(loaded_state_data["classifier"], dict):
                        classifier.set_state(loaded_state_data["classifier"])
                        logger.info(f"LightweightKNN initialisiert mit {len(classifier.X_train)} Samples aus Zustand: {initial_state_path}")

                        # Log KNN info after loading state to check if methods changed
                        logger.info(f"KNN AFTER LOAD: KNN class type = {type(classifier).__name__}")
                        logger.info(f"KNN AFTER LOAD: KNN predict_proba method ID = {id(classifier.predict_proba)}")
                        logger.info(f"KNN AFTER LOAD: Classes = {sorted(list(classifier._classes))}")
                        logger.info(f"KNN AFTER LOAD: Sample count = {len(classifier.X_train)}")

                        loaded_initial_state = True
                    elif isinstance(loaded_state_data, dict) and "X_train" in loaded_state_data:
                        # Fallback, falls nur KNN state direkt gespeichert wurde
                        classifier.set_state(loaded_state_data)
                        logger.info(f"LightweightKNN initialisiert mit {len(classifier.X_train)} Samples aus direktem Zustand: {initial_state_path}")

                        # Log KNN info after loading state to check if methods changed (fallback case)
                        logger.info(f"KNN AFTER LOAD (FALLBACK): KNN class type = {type(classifier).__name__}")
                        logger.info(f"KNN AFTER LOAD (FALLBACK): KNN predict_proba method ID = {id(classifier.predict_proba)}")
                        logger.info(f"KNN AFTER LOAD (FALLBACK): Classes = {sorted(list(classifier._classes))}")
                        logger.info(f"KNN AFTER LOAD (FALLBACK): Sample count = {len(classifier.X_train)}")

                        loaded_initial_state = True
                    else:
                        logger.error(f"Schlüssel 'classifier' nicht in Zustandsdatei gefunden oder ungültiges Format: {initial_state_path}")
                except Exception as e:
                    logger.error(f"Fehler beim Laden des initialen k-NN Zustands von {initial_state_path}: {e}")
            else:
                logger.warning(f"Initiale k-NN Zustandsdatei nicht gefunden: {initial_state_path}")
        
        # Nur wenn kein initialer Zustand geladen wurde, verwenden wir zufällige Daten als Fallback
        if not loaded_initial_state:
            logger.warning("Kein initialer k-NN Zustand geladen. Verwende zufällige Daten als Fallback.")
            
            # First extract features from a test image to determine dimension
            # Create a small test image for feature extraction
            test_image = np.zeros((224, 224, 3), dtype=np.uint8)
            features = feature_extractor.extract_features(test_image)
            
            # Apply feature transformation if available
            if feature_transformer is not None:
                features = feature_transformer.transform(features)
                
            feature_dimension = features.shape[0] if features.ndim == 1 else features.shape[1]
            logger.info(f"Feature dimension after transformation: {feature_dimension}")

            # Generate some initial samples for classifier
            initial_features = np.random.randn(len(labels), feature_dimension)
            initial_timestamps = [time.time() - (i*10) for i in range(len(labels))]
            classifier.fit(initial_features, labels, initial_timestamps)
        
        # Initialize drift detectors
        drift_detectors = []

        # Process each drift detector configuration
        for detector_idx, detector_config in enumerate(tinylcm_config["drift_detectors"]):
            detector_config_type = detector_config.get("type", "PageHinkleyFeatureMonitor")

            if detector_config_type == "PageHinkleyFeatureMonitor":
                # Get feature index from config
                feature_index = detector_config["feature_index"]

                # Make a more robust feature extraction function that handles edge cases
                def safe_feature_extractor(features):
                    """Extract a feature safely with boundary checking."""
                    if features is None:
                        return 0.0

                    # Convert to numpy array if needed
                    if not isinstance(features, np.ndarray):
                        try:
                            features = np.array(features)
                        except:
                            return 0.0

                    # Check if feature index is in range
                    if feature_index < 0 or feature_index >= features.size:
                        # Use last feature if index out of range
                        if features.size > 0:
                            return float(features.flatten()[-1])
                        return 0.0

                    # Get the feature at the specified index
                    try:
                        feature_value = float(features.flatten()[feature_index])

                        # Log some feature statistics occasionally to understand the scale issue
                        if hasattr(safe_feature_extractor, 'feature_count'):
                            safe_feature_extractor.feature_count += 1
                        else:
                            safe_feature_extractor.feature_count = 1

                        if safe_feature_extractor.feature_count % 50 == 0:
                            # Calculate some statistics about the features
                            flat_features = features.flatten()
                            try:
                                stats = {
                                    'min': float(np.min(flat_features)),
                                    'max': float(np.max(flat_features)),
                                    'mean': float(np.mean(flat_features)),
                                    'std': float(np.std(flat_features)),
                                    'selected_value': feature_value,
                                    'dtype': str(features.dtype)
                                }
                                logger.info(f"Feature statistics: {stats}")
                            except Exception as e:
                                logger.error(f"Error calculating feature statistics: {e}")

                        return feature_value
                    except:
                        return 0.0

                # Alternative feature statistic that uses normalized feature values
                def normalized_feature_extractor(features):
                    """Extract feature and normalize to reduce extreme values."""
                    raw_value = safe_feature_extractor(features)
                    # Normalize to a more reasonable range for drift detection
                    # The value 1e14 is based on the observed magnitude in logs
                    normalized_value = raw_value / 1e14
                    return normalized_value

                # Create feature detector with normalized values
                feature_detector = PageHinkleyFeatureMonitor(
                    feature_statistic_fn=normalized_feature_extractor,
                    delta=detector_config["delta"],
                    lambda_threshold=detector_config["lambda_param"],
                    warm_up_samples=detector_config["min_samples"],
                    reference_update_interval=detector_config["warmup_samples"]
                )

                # Set the cooldown period to a higher value to reduce frequent drift events
                # The cooldown mechanism is now properly implemented in the library
                feature_detector.drift_cooldown_period = 500  # Samples to wait before detecting drift again

                # Register drift callback
                feature_detector.register_callback(on_drift_detected)

                # Add to detector list
                drift_detectors.append(feature_detector)
                logger.info(f"Initialized PageHinkleyFeatureMonitor with feature index {feature_index}")

            elif detector_config_type == "EWMAConfidenceMonitor":
                # Create confidence monitor to detect when confidence drops
                confidence_monitor = EWMAConfidenceMonitor(
                    lambda_param=detector_config.get("lambda_param", 0.2),
                    threshold_factor=detector_config.get("threshold_factor", 2.0),
                    drift_window=detector_config.get("drift_window", 3),
                    min_confidence=detector_config.get("min_confidence", 0.5),
                    warm_up_samples=detector_config.get("warm_up_samples", 30),
                    reference_update_interval=detector_config.get("reference_update_interval", 50)
                )

                # Register drift callback
                confidence_monitor.register_callback(on_drift_detected)

                # Add to detector list
                drift_detectors.append(confidence_monitor)
                logger.info(f"Initialized EWMAConfidenceMonitor to detect confidence changes")
                
            elif detector_config_type == "KNNDistanceMonitor":
                # Create KNN distance monitor to detect when neighbor distances increase
                knn_distance_monitor = KNNDistanceMonitor(
                    delta=detector_config.get("delta", 0.1),
                    lambda_threshold=detector_config.get("lambda_threshold", 5.0),
                    exit_threshold_factor=detector_config.get("exit_threshold_factor", 0.7),
                    high_confidence_threshold=detector_config.get("high_confidence_threshold", 0.9),
                    stable_known_classes=detector_config.get("stable_known_classes", ["lego", "stone", "leaf"]),
                    warm_up_samples=detector_config.get("warm_up_samples", 100),
                    reference_update_interval=detector_config.get("reference_update_interval", 50),
                    reference_update_factor=detector_config.get("reference_update_factor", 0.05),
                    pause_reference_update_during_drift=detector_config.get("pause_reference_update_during_drift", True),
                    drift_cooldown_period=detector_config.get("drift_cooldown_period", 30),
                    reference_stats_path=detector_config.get("reference_stats_path", None)
                )
                
                # Register drift callback
                knn_distance_monitor.register_callback(on_drift_detected)
                
                # Add to detector list
                drift_detectors.append(knn_distance_monitor)
                
                # Log information about the initialized detector
                reference_info = ""
                if detector_config.get("reference_stats_path"):
                    reference_info = f" using reference statistics from {detector_config.get('reference_stats_path')}"
                    
                logger.info(f"Initialized KNNDistanceMonitor to detect neighbor distance changes{reference_info}")

            else:
                logger.warning(f"Unknown drift detector type: {detector_config_type}")

        # Make sure we have at least one detector
        if not drift_detectors:
            logger.warning("No drift detectors configured - creating default PageHinkleyFeatureMonitor")
            default_detector = PageHinkleyFeatureMonitor(
                feature_statistic_fn=lambda features: float(np.mean(features)) if isinstance(features, np.ndarray) else 0.0,
                lambda_threshold=15.0,
                warm_up_samples=50
            )
            default_detector.register_callback(on_drift_detected)
            drift_detectors.append(default_detector)

        # Keep backward compatibility by using the first detector as primary
        drift_detector = drift_detectors[0]
        
        # Initialize operational monitor
        monitor_config = tinylcm_config["operational_monitor"]
        operational_monitor = OperationalMonitor(
            storage_dir=tinylcm_config["data_logger"]["log_dir"],
            collect_system_metrics=monitor_config.get("track_system_metrics", True),
            system_metrics_interval=monitor_config.get("report_interval_seconds", 30)
        )

        # Verify operational monitor configuration
        logger.info(f"Operational monitor initialized with:")
        logger.info(f"- collect_system_metrics: {monitor_config.get('track_system_metrics', True)}")
        logger.info(f"- system_metrics_interval: {monitor_config.get('report_interval_seconds', 30)}")

        # Make sure the system metrics collection is running
        # Try to restart collection if the monitor has the methods
        if hasattr(operational_monitor, '_start_system_metrics_collection'):
            try:
                operational_monitor._start_system_metrics_collection()
                logger.info("Explicitly started system metrics collection thread")
            except Exception as e:
                logger.warning(f"Failed to explicitly start system metrics thread: {e}")

        # Track basic system metrics manually if needed
        if monitor_config.get("track_system_metrics", True):
            try:
                # If psutil is available, collect basic metrics
                import psutil
                basic_metrics = {
                    "timestamp": time.time(),
                    "cpu_percent": psutil.cpu_percent(interval=0.1),
                    "memory_percent": psutil.virtual_memory().percent
                }
                if hasattr(operational_monitor, 'track_system_metrics'):
                    operational_monitor.track_system_metrics(basic_metrics)
                    logger.info("Added initial system metrics to operational monitor")
            except ImportError:
                logger.warning("psutil not available, cannot add basic metrics")
        logger.info(f"Operational monitor initialized with:")
        logger.info(f"- collect_system_metrics: {monitor_config.get('track_system_metrics', True)}")
        logger.info(f"- system_metrics_interval: {monitor_config.get('report_interval_seconds', 30)}")
        
        # Initialize data logger if enabled
        logger_config = tinylcm_config["data_logger"]
        if logger_config["enabled"]:
            data_logger = DataLogger(
                storage_dir=logger_config["log_dir"]
            )
        else:
            data_logger = None
            
        # Initialize InferencePipeline with the configured components
        pipeline = InferencePipeline(
            feature_extractor=feature_extractor,
            classifier=classifier,
            autonomous_monitors=drift_detectors,  # Use all configured detectors
            operational_monitor=operational_monitor,
            data_logger=data_logger
        )

        # Also register the drift callback at the pipeline level to ensure it's called
        # when drift is detected during both normal processing and explicit drift checks
        pipeline.register_drift_callback(on_drift_detected)

        # Manually collect system metrics once to verify they're working
        try:
            system_metrics = operational_monitor.get_system_metrics()
            logger.info(f"Initial system metrics: {system_metrics}")
        except Exception as e:
            logger.error(f"Failed to collect initial system metrics: {e}", exc_info=True)
        
        # Start camera
        camera.start()
        logger.info("Camera started")
        
        # Warm-up: Wait for camera to initialize and provide frames
        logger.info("Warming up camera...")
        for _ in range(20):  # Increase warm-up frames
            frame = camera.get_frame()
            if frame is not None:
                break
            time.sleep(0.2)  # Longer delay between warm-up attempts
        
        if frame is None:
            logger.error("Failed to get frames from camera. Exiting.")
            running = False
        
        # Main inference loop
        logger.info("Starting main inference loop for Scenario 2")
        frame_count = 0
        start_time = time.time()
        inference_interval = config["application"]["inference_interval_ms"] / 1000.0
        
        last_drift_check_time = time.time()
        drift_check_interval = 0.1  # Check for drift very frequently (now used for logging only)
        
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

            # Add direct debugging of confidence value computation
            timestamp = time.time()

            # Extract features manually to verify they're correct
            if frame_count % 5 == 0:  # Only do this occasionally to avoid log spam
                try:
                    # Extract raw features
                    test_features = feature_extractor.extract_features(resized_frame)
                    logger.info(f"MANUAL DEBUG: Raw features shape = {test_features.shape}")
                    logger.info(f"MANUAL DEBUG: Raw features sample = {test_features[:3]}")
                    
                    # Apply PCA transformation if available
                    if feature_transformer is not None:
                        transformed_features = feature_transformer.transform(test_features)
                        logger.info(f"MANUAL DEBUG: Transformed features shape = {transformed_features.shape}")
                        logger.info(f"MANUAL DEBUG: Transformed features sample = {transformed_features[:3]}")
                        # Use the transformed features for prediction
                        test_features = transformed_features

                    # Test direct prediction and confidence calculation
                    logger.info("MANUAL DEBUG: About to call classifier.predict_proba directly")
                    test_probas = classifier.predict_proba(np.array([test_features]))
                    logger.info(f"MANUAL DEBUG: Raw probas from direct call = {test_probas[0]}")
                except Exception as e:
                    logger.error(f"MANUAL DEBUG: Error in direct feature extraction or prediction: {str(e)}")

            # Modified process function to intercept and transform features
            def feature_interceptor(input_data, extract_features=True):
                """Intercept feature extraction to apply PCA transformation."""
                if extract_features:
                    # Extract features using the feature extractor
                    features = feature_extractor.extract_features(input_data)
                    
                    # Apply feature transformation if available
                    if feature_transformer is not None:
                        features = feature_transformer.transform(features)
                    
                    return features
                else:
                    # Input is already features
                    return input_data

            # Start timing for the complete inference
            total_start_time = time.time()
            
            # Proceed with normal pipeline processing but use our custom feature extraction
            # Rather than modifying the pipeline directly, we'll extract features manually
            # and pass them to the pipeline with extract_features=False
            
            # Extract and transform features
            try:
                # Time feature extraction
                feature_start_time = time.time()
                
                # Extract features
                features = feature_extractor.extract_features(resized_frame)
                
                # Apply feature transformation if available
                if feature_transformer is not None:
                    features = feature_transformer.transform(features)
                
                feature_extraction_time = time.time() - feature_start_time
                
                # Time KNN inference (which happens inside the pipeline)
                knn_start_time = time.time()
                
                # Use the processed features directly in the pipeline
                result = pipeline.process(
                    input_data=features,  # Pass the transformed features instead of raw image
                    label=None,  # No ground truth label available
                    sample_id=sample_id,
                    timestamp=timestamp,
                    extract_features=False  # Tell pipeline not to extract features again
                )
                
                knn_inference_time = time.time() - knn_start_time
                
            except Exception as e:
                logger.error(f"Error processing frame: {e}")
                result = None
                feature_extraction_time = 0
                knn_inference_time = 0

            # Filter out "negative" class predictions for display if desired
            if result and result.get("prediction") == "negative":
                logger.debug(f"Filtered out 'negative' class prediction with confidence: {result.get('confidence', 0.0):.4f}")
            
            # Extract prediction and confidence from result
            if result:
                prediction = result.get("prediction", "unknown")
                confidence = result.get("confidence", 0.0)
                features = result.get("features")
                confidence_threshold = config["model"]["threshold"]

                # Debug output for confidence values - always log these
                logger.info(f"MAIN DEBUG: Result confidence: {confidence}, Prediction: {prediction}")

                # Add raw probabilities if available in the result
                probabilities = result.get("probabilities")
                if probabilities is not None:
                    logger.info(f"MAIN DEBUG: Raw probabilities: {probabilities}")

                # Process positive class predictions with high confidence, ignore negative class
                if prediction not in ["negative"] and confidence >= confidence_threshold:
                    # Log more details for better analysis
                    if frame_count % 10 == 0:  # Log every 10th frame to avoid flooding
                        logger.info(f"Frame {frame_count}: Prediction={prediction}, Confidence={confidence:.4f}")
                        if features is not None:
                            # Log first 5 feature values to understand scale
                            feature_sample = features[:5]
                            logger.info(f"Feature sample: {feature_sample}")

                    # Save Object predictions for validation
                    image_path = None
                    try:
                        # Create directory for prediction images if it doesn't exist
                        pred_dir = Path("./prediction_images")
                        pred_dir.mkdir(exist_ok=True)

                        # Convert BGR to RGB for correct color visualization
                        rgb_frame = cv2.cvtColor(current_frame, cv2.COLOR_BGR2RGB)

                        # Save the image with frame number and prediction information
                        timestamp_str = datetime.now().strftime("%H%M%S")
                        image_path = pred_dir / f"frame{frame_count:06d}_{timestamp_str}_{prediction}_{confidence:.2f}.jpg"
                        cv2.imwrite(str(image_path), rgb_frame)

                        # Only log occasionally to avoid flooding logs
                        if frame_count % 10 == 0:
                            logger.info(f"Saved positive class {prediction} image to {image_path}")

                    except Exception as e:
                        logger.error(f"Error saving prediction image: {e}")
                elif prediction == "negative":
                    # Log negative predictions at debug level
                    logger.debug(f"Frame {frame_count}: Filtered negative prediction with confidence: {confidence:.4f}")
                elif confidence < confidence_threshold:
                    # Log low confidence predictions at debug level
                    logger.debug(f"Frame {frame_count}: Filtered low confidence prediction {prediction} with confidence: {confidence:.4f}")
            
            # Update frame counter
            frame_count += 1
            if frame_count % 100 == 0:
                elapsed = time.time() - start_time
                fps = frame_count / elapsed
                logger.info(f"Processed {frame_count} frames. Current FPS: {fps:.2f}")
            
            # Check for drift only periodically to reduce frequency of drift events
            current_time = time.time()

            # Check for drift every frame for controlled experiment
            drift_detected_in_frame = False
            drift_check_time = 0
            
            if True:
                # Time drift checking
                drift_start_time = time.time()
                
                # The updated pipeline.check_autonomous_drifts() method now handles cooldown internally,
                # so we don't need to manually call callbacks anymore. If drift is detected,
                # and the detector is not in cooldown, it will automatically call our registered
                # callbacks through its _notify_callbacks mechanism.
                drift_results = pipeline.check_autonomous_drifts()
                
                drift_check_time = time.time() - drift_start_time
                
                # Always log drift check result to operational logs, regardless of detection outcome
                if operational_monitor:
                    drift_check_id = f"drift_check_{int(time.time())}_{uuid.uuid4().hex[:6]}"
                    drift_detected = any(r.get("drift_detected", False) for r in drift_results) if drift_results else False
                    drift_detected_in_frame = drift_detected
                    
                    # Create metadata with drift check results
                    check_metadata = {
                        "drift_detected": drift_detected,
                        "check_timestamp": time.time(),
                        "frame_count": frame_count,
                        "results_count": len(drift_results) if drift_results else 0,
                        "drift_details": drift_results if drift_results else [],
                        "prediction_confidence": confidence,
                        "prediction_class": prediction
                    }
                    
                    # Record drift check operation in operational logs
                    operational_monitor.track_operation(
                        operation_id=drift_check_id,
                        operation_type="drift_check",
                        result=str(drift_detected),
                        success=True,
                        metadata=check_metadata,
                        timestamp=time.time()
                    )
                    logger.info(f"Drift check recorded in operational logs with ID: {drift_check_id}, drift_detected={drift_detected}")
                
                if drift_results:
                    # Just log information about the drift results
                    drift_count = 0
                    cooldown_count = 0

                    for result in drift_results:
                        # Get detector name, use detector_type as primary key, detector as fallback
                        detector_name = result.get('detector_type', result.get('detector', 'unknown'))
                        
                        # Get consistent drift type
                        if detector_name == "KNNDistanceMonitor":
                            drift_type = "knn_distance"
                        elif detector_name == "EWMAConfidenceMonitor":
                            drift_type = "confidence"
                        elif detector_name == "FeatureMonitor":
                            drift_type = "feature"
                        else:
                            drift_type = detector_name.lower()

                        if result.get("drift_detected", False):
                            drift_count += 1

                            # Log more detailed information about the detected drift
                            metric = result.get("metric", "unknown")
                            current = result.get("current_value", "unknown")
                            threshold = result.get("threshold", "unknown")

                            logger.debug(f"Drift detected by {detector_name} (drift_type={drift_type}) - "
                                        f"Metric: {metric}, Current: {current}, Threshold: {threshold}")
                            
                            # Track drift event in operational logs (from drift check)
                            if operational_monitor:
                                # Create unique operation ID
                                drift_op_id = f"drift_check_{int(time.time())}_{uuid.uuid4().hex[:6]}"
                                
                                # Prepare metadata with drift information
                                drift_metadata = {
                                    "detector": detector_name,
                                    "drift_type": drift_type,
                                    "metric": metric,
                                    "current_value": current,
                                    "threshold": threshold,
                                    "reason": f"{metric} drift detected (current: {current}, threshold: {threshold})",
                                    "metrics": {k: v for k, v in result.items() if k not in ["detector", "detector_type", "detector_id", "timestamp", "sample_id", "drift_detected", "in_cooldown_period"]}
                                }
                                
                                # Track operation in operational monitor
                                operational_monitor.track_operation(
                                    operation_id=drift_op_id,
                                    operation_type="drift_detection",
                                    result=drift_type,
                                    success=True,
                                    metadata=drift_metadata,
                                    timestamp=time.time()
                                )
                                logger.debug(f"Drift event from drift check recorded in operational logs with ID: {drift_op_id}")
                                
                        elif result.get("in_cooldown_period", False):
                            cooldown_count += 1

                            # Log detailed cooldown information
                            samples_since = result.get("samples_since_last_drift", "unknown")
                            cooldown_period = result.get("drift_cooldown_period", "unknown")

                            logger.debug(f"Detector {detector_name} (drift_type={drift_type}) in cooldown period "
                                        f"({samples_since}/{cooldown_period} samples since last drift)")

                    if drift_count > 0:
                        logger.debug(f"Found {drift_count} drift detections (callbacks handled internally by library)")
                    if cooldown_count > 0:
                        logger.debug(f"Found {cooldown_count} detectors in cooldown period")

                    # No need to manually call on_drift_detected - the library now handles
                    # this with proper cooldown mechanisms

            last_drift_check_time = current_time
            
            # Calculate total time
            total_time = time.time() - total_start_time
            
            # Log performance metrics for this frame
            if result:
                prediction_result = {
                    "class": prediction,
                    "confidence": float(confidence),
                    "knn_samples": len(classifier.X_train) if hasattr(classifier, 'X_train') else 0
                }
            else:
                prediction_result = {
                    "class": "error",
                    "confidence": 0.0,
                    "knn_samples": 0
                }
            
            performance_logger.log_inference(
                total_time=total_time,
                feature_extraction_time=feature_extraction_time,
                knn_inference_time=knn_inference_time,
                drift_check_time=drift_check_time,
                prediction=prediction_result,
                drift_detected=drift_detected_in_frame
            )
            
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
        
        # Log final performance summary
        if 'performance_logger' in locals():
            performance_logger.log_summary()
        
        logger.info("TinyLCM Autonomous Monitoring Example completed")


if __name__ == "__main__":
    main()