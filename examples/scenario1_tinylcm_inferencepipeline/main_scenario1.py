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
from utils.sync_client import \
    ExtendedSyncClient  # Use the extended version with additional features

from tinylcm.core.classifiers.knn import LightweightKNN
from tinylcm.core.data_logger.logger import DataLogger
from tinylcm.core.data_structures import FeatureSample
from tinylcm.core.drift_detection.features import PageHinkleyFeatureMonitor
from tinylcm.core.drift_detection.confidence import EWMAConfidenceMonitor
# Import tinylcm components
from tinylcm.core.feature_extractors.tflite import TFLiteFeatureExtractor
from tinylcm.core.operational_monitor.monitor import OperationalMonitor
from tinylcm.core.pipeline import InferencePipeline
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
        
        # Convert BGR to RGB for correct color visualization
        rgb_frame = cv2.cvtColor(current_frame, cv2.COLOR_BGR2RGB)

        image_path = drift_dir / f"drift_{timestamp.replace(' ', '_').replace(':', '-')}_{detector_name}.jpg"
        cv2.imwrite(str(image_path), rgb_frame)
        logger.info(f"Saved drift image to {image_path}")
    
    # If sync client is available, create and send a drift event package
    if sync_client:
        try:
            # Send drift event information to server (create a custom package)
            # First check if method exists and use appropriate one

            # Prepare drift info for sending
            drift_data = {
                "detector_name": detector_name,
                "reason": reason,
                "metrics": metrics,
                "timestamp": datetime.now().isoformat()
            }

            # Get sync directory - check if it's available in various ways
            sync_dir = "./sync_data"  # Default fallback

            if hasattr(sync_client, 'sync_dir'):
                sync_dir = sync_client.sync_dir
            elif hasattr(sync_client, 'sync_interface') and hasattr(sync_client.sync_interface, 'storage_dir'):
                sync_dir = sync_client.sync_interface.storage_dir
            elif "sync_client" in config["tinylcm"]:
                sync_dir = config["tinylcm"]["sync_client"].get("sync_dir", "./sync_data")

            # Include image path if available
            if image_path:
                drift_data["image_path"] = str(image_path)

            # Try to use the ExtendedSyncClient method if available
            if hasattr(sync_client, 'create_and_send_drift_event_package'):
                # This is the best case - use the extended client's method
                # Create a FeatureSample-like object for the current state
                from tinylcm.core.data_structures import FeatureSample
                current_sample_obj = FeatureSample(
                    sample_id=f"drift_{int(time.time())}",
                    features=None,
                    prediction=None,
                    timestamp=time.time(),
                    metadata={"confidence": 0.0}
                )

                # Use the specialized method
                success = sync_client.create_and_send_drift_event_package(
                    detector_name=detector_name,
                    reason=reason,
                    metrics=metrics,
                    sample=current_sample_obj,
                    image_path=str(image_path) if image_path else None
                )

                logger.info(f"Drift event sent using extended client: {'Success' if success else 'Failed'}")
                return  # Exit early if we used this method

            # Fall back to manual package creation
            # Get the correct directory structure - usually it's in the sync_interface
            if hasattr(sync_client, 'sync_interface') and hasattr(sync_client.sync_interface, 'create_package'):
                # Use the sync_interface to create package with proper structure
                try:
                    # Get device_id for the package
                    dev_id = None
                    if 'device_id_manager' in globals():
                        dev_id = device_id_manager.get_device_id()
                    elif hasattr(sync_client, 'device_id'):
                        dev_id = sync_client.device_id
                    else:
                        import socket
                        dev_id = f"device-{socket.gethostname()}"

                    package_id = sync_client.sync_interface.create_package(
                        device_id=dev_id,
                        package_type="drift_event",
                        description=f"Drift detected by {detector_name}: {reason}"
                    )

                    # Now create a temp file for the drift data
                    import tempfile
                    temp_dir = tempfile.mkdtemp()
                    drift_file = os.path.join(temp_dir, "drift_event.json")

                    with open(drift_file, 'w') as f:
                        json.dump(drift_data, f)

                    # Add to package
                    sync_client.sync_interface.add_file_to_package(
                        package_id=package_id,
                        file_path=drift_file,
                        file_type="drift_event"
                    )

                    # Add image if available
                    if image_path:
                        sync_client.sync_interface.add_file_to_package(
                            package_id=package_id,
                            file_path=str(image_path),
                            file_type="image"
                        )

                    # Finalize the package
                    sync_client.sync_interface.finalize_package(package_id)

                    # We'll use this package_id for future references
                    return  # Exit early - we're done!
                except Exception as e:
                    logger.warning(f"Failed to create package using sync_interface: {e}")

            # If we're still here, fall back to the most manual approach
            packages_dir = os.path.join(sync_dir, "packages")
            os.makedirs(packages_dir, exist_ok=True)

            # Create a unique package ID
            package_id = f"drift_{int(time.time())}_{uuid.uuid4().hex[:8]}"

            # Create the package directory - try different structures that SyncClient might recognize
            # Structure 1: packages/package_id/
            package_dir = os.path.join(packages_dir, package_id)
            os.makedirs(package_dir, exist_ok=True)

            # Create metadata.json - this is what SyncClient looks for
            # Get device_id safely
            dev_id = None
            if 'device_id_manager' in globals():
                dev_id = device_id_manager.get_device_id()
            elif hasattr(sync_client, 'device_id'):
                dev_id = sync_client.device_id
            else:
                # Last resort - use hostname
                import socket
                dev_id = f"device-{socket.gethostname()}"

            metadata = {
                "id": package_id,
                "type": "drift_event",
                "timestamp": time.time(),
                "device_id": dev_id,  # Use the safely obtained device ID
                "status": "pending"
            }

            metadata_file = os.path.join(package_dir, "metadata.json")
            with open(metadata_file, 'w') as f:
                json.dump(metadata, f)

            # Create data.json with the drift information
            data_file = os.path.join(package_dir, "data.json")
            with open(data_file, 'w') as f:
                json.dump(drift_data, f)

            # If we have an image, copy it to the package directory
            if image_path:
                img_dest = os.path.join(package_dir, "image.jpg")
                try:
                    import shutil
                    shutil.copy(str(image_path), img_dest)
                except Exception as e:
                    logger.warning(f"Failed to copy drift image: {e}")

            logger.info(f"Created drift event package: {package_id}")

            # Now try to trigger synchronization
            try:
                # Force a sync cycle
                if hasattr(sync_client, 'sync_all_pending_packages'):
                    sync_client.sync_all_pending_packages()
                    logger.info(f"Triggered sync after creating drift event package")
            except Exception as e:
                logger.warning(f"Failed to trigger sync: {e}")
        except Exception as e:
            logger.error(f"Failed to save/send drift event: {e}")


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
            feature_dimension = features.shape[0] if features.ndim == 1 else features.shape[1]
            logger.info(f"Feature extractor output dimension: {feature_dimension}")

            # Generate some initial samples for classifier
            initial_features = np.random.randn(len(labels), feature_dimension)
            initial_timestamps = [time.time() - (i*10) for i in range(len(labels))]
            classifier.fit(initial_features, labels, initial_timestamps)
        
        # Initialize drift detectors
        drift_detectors = []

        # Process each drift detector configuration
        for detector_idx, detector_config in enumerate(tinylcm_config["drift_detectors"]):
            detector_type = detector_config.get("type", "PageHinkleyFeatureMonitor")

            if detector_type == "PageHinkleyFeatureMonitor":
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

                # Add a filter to prevent multiple sequential drift detections
                feature_detector.drift_cooldown_period = 100  # Samples to wait before detecting drift again
                feature_detector.samples_since_last_drift = 0

                # Override update method to handle detection cooldown
                original_update = feature_detector.update
                def update_with_cooldown(record, *args, **kwargs):
                    # Increment counter since last drift
                    feature_detector.samples_since_last_drift += 1

                    # Check if we're in cooldown period after drift detected
                    if hasattr(feature_detector, 'drift_detected') and feature_detector.drift_detected:
                        # During cooldown period, don't try to update stats
                        if feature_detector.samples_since_last_drift < feature_detector.drift_cooldown_period:
                            return False, {}
                        else:
                            # Cooldown period over, reset drift flag
                            feature_detector.drift_detected = False

                    try:
                        # Try to extract feature safely
                        if not record or 'features' not in record:
                            return False, {}

                        # Call original update
                        result = original_update(record, *args, **kwargs)

                        # If drift detected, set flag
                        is_drift, _ = result
                        if is_drift:
                            feature_detector.drift_detected = True
                            feature_detector.samples_since_last_drift = 0

                        return result
                    except Exception as e:
                        logger.warning(f"Error in feature detector update: {e}")
                        return False, {}

                # Replace the update method
                feature_detector.update = update_with_cooldown

                # Register drift callback
                feature_detector.register_callback(on_drift_detected)

                # Add to detector list
                drift_detectors.append(feature_detector)
                logger.info(f"Initialized PageHinkleyFeatureMonitor with feature index {feature_index}")

            elif detector_type == "EWMAConfidenceMonitor":
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

            else:
                logger.warning(f"Unknown drift detector type: {detector_type}")

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
        
        # Initialize ExtendedSyncClient for communication with TinySphere
        sync_config = tinylcm_config["sync_client"]
        sync_client = ExtendedSyncClient(
            server_url=sync_config["server_url"],
            device_id=device_id,
            api_key=sync_config["api_key"],
            sync_dir=sync_config.get("sync_dir", "./sync_data"),
            sync_interval_seconds=sync_config["sync_interval_seconds"],
            max_retries=sync_config["max_retries"],
            auto_register=sync_config["auto_register"],
            enable_image_transfer=sync_config.get("enable_prediction_images", False)
        )

        # Log prediction image transfer status
        if sync_client.enable_prediction_images:
            logger.info("Prediction image transfer to TinySphere enabled")
        else:
            logger.info("Prediction image transfer to TinySphere disabled")
        
        # Initialize InferencePipeline with the configured components
        pipeline = InferencePipeline(
            feature_extractor=feature_extractor,
            classifier=classifier,
            autonomous_monitors=drift_detectors,  # Use all configured detectors
            operational_monitor=operational_monitor,
            data_logger=data_logger
        )
        
        # Perform initial connection test to TinySphere
        try:
            status = sync_client.check_connection()
            if status:
                logger.info("Successfully connected to TinySphere server")
            else:
                logger.warning("Could not connect to TinySphere server - will continue and try again later")
        except Exception as e:
            logger.warning(f"Error checking connection to TinySphere server: {e}")

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

            # Add direct debugging of confidence value computation
            timestamp = time.time()

            # Extract features manually to verify they're correct
            if frame_count % 20 == 0:  # Only do this occasionally to avoid log spam
                try:
                    test_features = feature_extractor.extract_features(resized_frame)
                    logger.info(f"MANUAL DEBUG: Raw features shape = {test_features.shape}")
                    logger.info(f"MANUAL DEBUG: Raw features sample = {test_features[:3]}")

                    # Test direct prediction and confidence calculation
                    logger.info("MANUAL DEBUG: About to call classifier.predict_proba directly")
                    test_probas = classifier.predict_proba(np.array([test_features]))
                    logger.info(f"MANUAL DEBUG: Raw probas from direct call = {test_probas[0]}")
                except Exception as e:
                    logger.error(f"MANUAL DEBUG: Error in direct feature extraction or prediction: {str(e)}")

            # Proceed with normal pipeline processing
            result = pipeline.process(
                input_data=resized_frame,
                label=None,  # No ground truth label available
                sample_id=sample_id,
                timestamp=timestamp,
                extract_features=True  # Tell the pipeline to extract features from the input
            )

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

                # Process lego and stone predictions with high confidence, ignore negative class
                if prediction in ["lego", "stone"] and confidence >= confidence_threshold:
                    # Log more details for better analysis
                    if frame_count % 10 == 0:  # Log every 10th frame to avoid flooding
                        logger.info(f"Frame {frame_count}: Prediction={prediction}, Confidence={confidence:.4f}")
                        if features is not None:
                            # Log first 5 feature values to understand scale
                            feature_sample = features[:5]
                            logger.info(f"Feature sample: {feature_sample}")

                    # Save LEGO brick predictions for validation
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
                            logger.info(f"Saved {prediction} image to {image_path}")

                        # Add image to sync client if enabled
                        if config["tinylcm"]["features"].get("save_prediction_images", False) and sync_client and sync_client.enable_prediction_images:
                            added = sync_client.add_prediction_image(
                                image_path=str(image_path),
                                prediction=prediction,
                                confidence=confidence
                            )
                            if added and frame_count % 10 == 0:
                                logger.debug(f"Added image {image_path} to sync queue")
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
            
            # Periodically check for drift
            current_time = time.time()
            if current_time - last_drift_check_time >= drift_check_interval:
                pipeline.check_autonomous_drifts()
                last_drift_check_time = current_time
            
            # Periodically sync with TinySphere
            if current_time - last_sync_time >= sync_interval:
                try:
                    # Get operational metrics from monitor
                    if operational_monitor:
                        try:
                            metrics = operational_monitor.get_current_metrics()
                            # Add detailed log to debug the metrics collection
                            logger.info(f"Collected metrics keys: {list(metrics.keys())}")
                            logger.info(f"System metrics included: {'system' in metrics}")

                            # Send metrics to TinySphere
                            logger.info("Sending metrics to TinySphere")
                            success = sync_client.create_and_send_metrics_package(metrics)
                            logger.info(f"Metrics package sent: {'Success' if success else 'Failed'}")
                        except Exception as e:
                            logger.error(f"Error getting or sending metrics: {e}", exc_info=True)

                    # Sync all pending packages
                    logger.debug("Syncing pending packages with TinySphere")
                    sync_results = sync_client.sync_all_pending_packages()

                    # Process sync results
                    if sync_results:
                        logger.info(f"Synced {len(sync_results)} packages with TinySphere")

                        # Clean up transferred images if enabled
                        if sync_client.enable_prediction_images and sync_config.get("cleanup_transferred_images", False):
                            # Check if any prediction image packages were successfully synced
                            prediction_image_packages = [r for r in sync_results
                                                     if r.get("success") and r.get("package_type", "") == "prediction_images"]

                            # Get list of transferred images for cleanup
                            images_to_clean = []
                            for pkg in prediction_image_packages:
                                # Extract images from package details if available
                                if "processed_images" in pkg:
                                    images_to_clean.extend(pkg["processed_images"])

                            # Delete transferred images
                            if images_to_clean:
                                success_count, fail_count = sync_client.delete_transferred_images(images_to_clean)
                                logger.info(f"Cleaned up {success_count} transferred images, {fail_count} failed")
                    else:
                        logger.debug("No packages to sync with TinySphere")

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
                # Sync all pending packages
                sync_results = sync_client.sync_all_pending_packages()
                logger.info("Final sync completed")

                # Final cleanup of transferred images if enabled
                if sync_client.enable_prediction_images and sync_config.get("cleanup_transferred_images", False):
                    # Check if any prediction image packages were successfully synced
                    prediction_image_packages = [r for r in sync_results
                                             if r.get("success") and r.get("package_type", "") == "prediction_images"]

                    # Get list of transferred images for cleanup
                    images_to_clean = []
                    for pkg in prediction_image_packages:
                        if "processed_images" in pkg:
                            images_to_clean.extend(pkg["processed_images"])

                    # Delete transferred images
                    if images_to_clean:
                        success_count, fail_count = sync_client.delete_transferred_images(images_to_clean)
                        logger.info(f"Final cleanup: Removed {success_count} transferred images, {fail_count} failed")
            except Exception as e:
                logger.warning(f"Error during final sync: {e}")

            # Close the client
            sync_client.close()
            logger.info("Sync client closed")
        
        logger.info("TinyLCM Autonomous Monitoring Example completed")


if __name__ == "__main__":
    main()