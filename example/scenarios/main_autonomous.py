#!/usr/bin/env python3
"""
Autonomous Drift Detector with Proxy Metrics for Stone Detection
---------------------------------------------------------------
This application runs on a Raspberry Pi with a Pi Camera to detect stones.
It uses TinyLCM's autonomous drift detection system with proxy metrics
(confidence, distribution, features) to detect when the model is drifting,
without requiring ground truth labels.

Key features:
- Label-free drift detection using multiple proxy metrics
- Quarantine buffer to collect potentially drifting samples
- Heuristic adaptation to update the model without ground truth
- Full synchronization with central server

The application can run in full autonomous mode, adapting to changes in data
distribution at the edge without requiring external validation.
"""

import argparse
import asyncio
import json
import logging
import os
import sys
import tempfile
import time
from concurrent.futures import ThreadPoolExecutor
from pathlib import Path
from queue import Empty, Queue
from threading import Event, Thread

import cv2
import numpy as np

# Add parent directory to path so we can import common modules
parent_dir = os.path.dirname(os.path.dirname(os.path.abspath(__file__)))
if parent_dir not in sys.path:
    sys.path.append(parent_dir)
    
# Import preprocessors
from preprocessors import convert_uint8_to_float32, resize_and_normalize

# Add base directory to path (for tinylcm)
base_dir = os.path.dirname(os.path.dirname(os.path.dirname(os.path.abspath(__file__))))
if base_dir not in sys.path:
    sys.path.append(base_dir)

# Try to find tinylcm directory
tinylcm_dir = os.path.join(base_dir, "tinylcm")
if os.path.exists(tinylcm_dir) and tinylcm_dir not in sys.path:
    sys.path.append(tinylcm_dir)

# Now try to import tinylcm
try:
    import tinylcm
except ImportError:
    print("Error: Could not import tinylcm module.")
    print(f"Current Python path: {sys.path}")
    print("Please ensure tinylcm is installed or in the Python path.")
    sys.exit(1)

from camera_handler import CameraHandler
from device_id_manager import DeviceIDManager
from stone_detector import StoneDetector
from sync_client import SyncClient
from system_metrics import SystemMetricsCollector

# Configure logging
logging.basicConfig(
    level=logging.INFO,
    format="%(asctime)s - %(name)s - %(levelname)s - %(message)s",
    handlers=[
        logging.StreamHandler(),
        logging.FileHandler("stone_detector_autonomous.log")
    ]
)
logger = logging.getLogger("stone_detector_autonomous")

class AutonomousStoneDetectorApp:
    def __init__(self, config_path="scenarios/config_autonomous.json"):
        # Load configuration
        self.config = self.load_config(config_path)
        logger.info(f"Loaded configuration from {config_path}")
        
        # Create necessary directories
        self.ensure_directories()
        
        # Get or create device ID
        self.device_id_manager = DeviceIDManager()
        self.device_id = self.device_id_manager.get_device_id()
        self.config["tinysphere"]["device_id"] = self.device_id
        logger.info(f"Using device ID: {self.device_id}")
        
        # Thread pool executor for blocking operations
        self.executor = ThreadPoolExecutor(max_workers=4)
        
        # Queues for async processing
        self.inference_queue = Queue(maxsize=10)
        self.result_queue = Queue(maxsize=10)
        self.logging_queue = Queue(maxsize=100)
        self.sync_queue = Queue(maxsize=10)
        
        # Initialize TinyLCM components
        self.initialize_tinylcm()
        
        # Initialize camera
        self.camera = CameraHandler(
            resolution=self.config["camera"]["resolution"],
            framerate=self.config["camera"]["framerate"]
        )
        
        # Initialize detector
        self.detector = StoneDetector(
            model_path=self.config["model"]["path"],
            labels_path=self.config["model"]["labels"],
            threshold=self.config["model"]["threshold"],
            executor=self.executor
        )
        
        # Event for clean shutdown
        self.stop_event = Event()
        
        # Worker threads
        self.workers = []
        
        # System metrics collector
        self.metrics_collector = SystemMetricsCollector(interval=60)
        
        logger.info("Autonomous Stone Detector Application initialized")

    def load_config(self, config_path):
        """Load configuration from JSON file"""
        try:
            with open(config_path, 'r') as f:
                return json.load(f)
        except Exception as e:
            logger.error(f"Failed to load config: {e}")
            # Return default configuration
            return self._get_default_config()

    def _get_default_config(self):
        """Return default configuration"""
        return {
            "camera": {
                "resolution": (640, 480),
                "framerate": 10
            },
            "model": {
                "path": "models/model.tflite",
                "labels": "models/labels.txt",
                "threshold": 0.5
            },
            "tinylcm": {
                "model_dir": "tinylcm_data/models",
                "data_dir": "tinylcm_data/data_logs",
                "inference_dir": "tinylcm_data/inference_logs",
                "quarantine_dir": "tinylcm_data/quarantine",
                "heuristic_dir": "tinylcm_data/heuristic_logs",
                "sync_interval_seconds": 60,
                "enable_autonomous_detection": True,
                "enable_quarantine": True,
                "enable_heuristic_adaptation": True,
                "external_validation": False
            },
            "tinysphere": {
                "server_url": "http://192.168.0.66:8000",
                "api_key": "your-api-key",
                "device_id": "pi-zero-stone-detector"
            },
            "application": {
                "detection_interval": 1,
                "save_detected_stones": True,
                "data_dir": "data",
            }
        }

    def ensure_directories(self):
        """Create necessary directories if they don't exist"""
        directories = [
            "data",
            "models",
            "tinylcm_data",
            "tinylcm_data/models",
            "tinylcm_data/data_logs",
            "tinylcm_data/inference_logs",
            "tinylcm_data/quarantine",
            "tinylcm_data/heuristic_logs",
            "tinylcm_data/sync",
            "tinylcm_data/sync/packages",
            "tinylcm_data/sync/history"
        ]
        for directory in directories:
            os.makedirs(directory, exist_ok=True)
        logger.debug("Ensured all required directories exist")

    def initialize_tinylcm(self):
        """Initialize TinyLCM components with autonomous drift detection"""
        try:
            # Import core components
            from tinylcm.core.adaptation_tracker import AdaptationTracker
            from tinylcm.core.classifiers.knn import LightweightKNN
            from tinylcm.core.data_logger.logger import DataLogger
            from tinylcm.core.feature_extractors.tflite import \
                TFLiteFeatureExtractor
            from tinylcm.core.handlers.hybrid import HybridHandler
            from tinylcm.core.inference_monitor.monitor import InferenceMonitor
            from tinylcm.core.state_manager import StateManager

            # Initialize feature extractor without preprocessors
            # We'll handle preprocessing directly in the process_frame_async method
            self.feature_extractor = TFLiteFeatureExtractor(
                model_path=self.config["model"]["path"],
                feature_layer_index=-1  # Use the last layer for features
            )
            
            # Initialize classifier
            self.classifier = LightweightKNN(
                k=3,
                distance_metric="euclidean",
                max_samples=100,
                use_numpy=True,
                weight_by_distance=False
            )
            
            # Initialize State Manager
            self.state_manager = StateManager(
                storage_dir=self.config["tinylcm"]["model_dir"]
            )
            
            # Initialize Adaptation Tracker
            self.adaptation_tracker = AdaptationTracker(
                log_dir=self.config["tinylcm"]["model_dir"],
                auto_create_dir=True,
                export_format="json"
            )
            
            # Initialize DataLogger for data collection
            self.data_logger = DataLogger(
                storage_dir=self.config["tinylcm"]["data_dir"]
            )
            
            # Initialize InferenceMonitor for basic metrics
            self.inference_monitor = InferenceMonitor(
                storage_dir=self.config["tinylcm"]["inference_dir"]
            )
            
        except ImportError as e:
            logger.error(f"Failed to import core TinyLCM components: {str(e)}")
            print(f"Error: Could not initialize TinyLCM core components: {str(e)}")
            print("Please ensure tinylcm is properly installed with all required components.")
            sys.exit(1)
        
        # Set up autonomous drift detectors
        drift_config = self.config["tinylcm"].get("drift_detection", {})
        self.autonomous_monitors = self._setup_autonomous_detectors(drift_config)
        
        # Set up quarantine buffer if enabled
        self.quarantine_buffer = None
        if self.config["tinylcm"].get("enable_quarantine", True):
            try:
                from tinylcm.core.quarantine.buffer import (QuarantineBuffer,
                                                            QuarantineStrategy)

                # Use default quarantine directory if not specified
                quarantine_dir = self.config["tinylcm"].get("quarantine_dir", "tinylcm_data/quarantine")
                
                # Ensure quarantine directory exists
                os.makedirs(quarantine_dir, exist_ok=True)
                
                self.quarantine_buffer = QuarantineBuffer(
                    storage_dir=quarantine_dir,
                    max_size=1000,
                    auto_persist=True,
                    quarantine_strategy=QuarantineStrategy.ALL_DRIFT_SAMPLES
                )
                logger.debug("Quarantine buffer initialized")
            except ImportError as e:
                logger.error(f"Failed to import quarantine buffer: {str(e)}")
                logger.warning("Quarantine buffer will be disabled")
                print(f"Error importing quarantine buffer: {str(e)}")
                print("Please ensure tinylcm and its quarantine modules are properly installed.")
        
        # Set up heuristic adapter if enabled
        self.heuristic_adapter = None
        if (self.config["tinylcm"].get("enable_heuristic_adaptation", False) and 
            self.quarantine_buffer is not None):
            try:
                from tinylcm.core.heuristics.adapter import (HeuristicAdapter,
                                                             HeuristicStrategy)

                # Use default heuristic directory if not specified
                heuristic_dir = self.config["tinylcm"].get("heuristic_dir", "tinylcm_data/heuristic_logs")
                
                # Ensure heuristic directory exists
                os.makedirs(heuristic_dir, exist_ok=True)
                
                self.heuristic_adapter = HeuristicAdapter(
                    quarantine_buffer=self.quarantine_buffer,
                    strategy=HeuristicStrategy.HYBRID,
                    min_cluster_size=5,
                    min_samples_for_adaptation=10,
                    confidence_threshold=self.config["tinylcm"].get("heuristic_confidence_threshold", 0.7),
                    log_dir=heuristic_dir,
                    max_new_classes=3
                )
                logger.debug("Heuristic adapter initialized")
            except ImportError as e:
                logger.error(f"Failed to import heuristic adapter: {str(e)}")
                logger.warning("Heuristic adaptation will be disabled")
                print(f"Error importing heuristic adapter: {str(e)}")
                print("Please ensure tinylcm and its heuristic modules are properly installed.")
        
        # Initialize Hybrid Handler with classifier
        try:
            # Already imported in the try block above
            self.hybrid_handler = HybridHandler(
                classifier=self.classifier,
                max_samples=100,
                batch_size=30,
                baseline_accuracy=0.9,
                cusum_threshold=5.0,
                cusum_delta=0.25,
                enable_condensing=False,
                use_numpy=True
            )
            
            # Initialize Adaptive Pipeline with all components
            from tinylcm.core.pipeline import AdaptivePipeline
            
            self.adaptive_pipeline = AdaptivePipeline(
                feature_extractor=self.feature_extractor,
                classifier=self.classifier,
                handler=self.hybrid_handler,
                state_manager=self.state_manager,
                adaptation_tracker=self.adaptation_tracker,
                data_logger=self.data_logger,
                config={
                    "enable_autonomous_detection": self.config["tinylcm"].get("enable_autonomous_detection", True),
                    "enable_quarantine": self.config["tinylcm"].get("enable_quarantine", True),
                    "enable_heuristic_adaptation": self.config["tinylcm"].get("enable_heuristic_adaptation", False),
                    "external_validation": self.config["tinylcm"].get("external_validation", False),
                    "quarantine_check_interval": self.config["tinylcm"].get("quarantine_check_interval", 50),
                    "heuristic_confidence_threshold": self.config["tinylcm"].get("heuristic_confidence_threshold", 0.7)
                },
                autonomous_monitors=self.autonomous_monitors,
                quarantine_buffer=self.quarantine_buffer,
                heuristic_adapter=self.heuristic_adapter
            )
            
            # Register drift detection callback
            self.adaptive_pipeline.register_drift_callback(self._on_drift_detected)
            
            logger.info("Adaptive Pipeline initialized with autonomous components")
            
            # Initialize SyncInterface for data synchronization
            from tinylcm.client.sync_interface import SyncInterface
            
            self.sync_interface = SyncInterface(
                sync_dir="tinylcm_data/sync"
            )
            
        except ImportError as e:
            logger.error(f"Failed to import required components: {str(e)}")
            print(f"Error: Could not initialize components: {str(e)}")
            print("This likely means some components of the enhanced system are not available.")
            sys.exit(1)
        
        # Initialize SyncClient
        self.sync_client = None
        try:
            if hasattr(self, 'sync_interface') and self.sync_interface is not None:
                self.sync_client = SyncClient(
                    server_url=self.config["tinysphere"]["server_url"],
                    api_key=self.config["tinysphere"]["api_key"],
                    device_id=self.config["tinysphere"]["device_id"],
                    sync_interface=self.sync_interface
                )
                logger.info("SyncClient initialized successfully")
            else:
                logger.warning("SyncInterface not available, skipping SyncClient initialization")
        except Exception as e:
            logger.error(f"Failed to initialize SyncClient: {e}")
            self.sync_client = None
        
        logger.info("TinyLCM components initialized with autonomous drift detection")

    def _setup_autonomous_detectors(self, drift_config):
        """Set up autonomous drift detectors based on configuration"""
        detectors = []
        
        try:
            # Try to import the autonomous drift detector modules
            from tinylcm.core.drift_detection.confidence import (
                EWMAConfidenceMonitor, PageHinkleyConfidenceMonitor)
            from tinylcm.core.drift_detection.distribution import \
                PredictionDistributionMonitor
            from tinylcm.core.drift_detection.features import FeatureMonitor

            # Default configuration if none provided
            default_drift_config = {
                "ewma_confidence": {
                    "enabled": True,
                    "lambda_param": 0.1,
                    "threshold_factor": 3.0,
                    "drift_window": 5,
                    "training_size": 30
                },
                "distribution": {
                    "enabled": True,
                    "window_size": 50,
                    "threshold": 0.25,
                    "method": "block"
                },
                "feature": {
                    "enabled": True,
                    "window_size": 100,
                    "threshold": 3.0,
                    "reference_size": 50,
                    "max_features": 50,
                    "distance_metric": "euclidean"
                }
            }
            
            # Add EWMA Confidence Monitor if enabled
            ewma_enabled = drift_config.get("ewma_confidence", default_drift_config["ewma_confidence"]).get("enabled", True)
            if ewma_enabled:
                ewma_config = drift_config.get("ewma_confidence", default_drift_config["ewma_confidence"])
                ewma_detector = EWMAConfidenceMonitor(
                    lambda_param=ewma_config.get("lambda_param", 0.1),
                    threshold_factor=ewma_config.get("threshold_factor", 3.0),
                    drift_window=ewma_config.get("drift_window", 5),
                    training_size=ewma_config.get("training_size", 30)
                )
                ewma_detector.register_callback(self._on_confidence_drift)
                detectors.append(ewma_detector)
                logger.debug("EWMA Confidence Monitor initialized")
            
            # Add Distribution Monitor if enabled
            dist_enabled = drift_config.get("distribution", default_drift_config["distribution"]).get("enabled", True)
            if dist_enabled:
                dist_config = drift_config.get("distribution", default_drift_config["distribution"])
                dist_detector = PredictionDistributionMonitor(
                    window_size=dist_config.get("window_size", 50),
                    threshold=dist_config.get("threshold", 0.25),
                    method=dist_config.get("method", "block")
                )
                dist_detector.register_callback(self._on_distribution_drift)
                detectors.append(dist_detector)
                logger.debug("Prediction Distribution Monitor initialized")
            
            # Add Feature Monitor if enabled
            feature_enabled = drift_config.get("feature", default_drift_config["feature"]).get("enabled", True)
            if feature_enabled:
                feature_config = drift_config.get("feature", default_drift_config["feature"])
                feature_detector = FeatureMonitor(
                    window_size=feature_config.get("window_size", 100),
                    threshold=feature_config.get("threshold", 3.0),
                    reference_size=feature_config.get("reference_size", 50),
                    max_features=feature_config.get("max_features", 50),
                    distance_metric=feature_config.get("distance_metric", "euclidean")
                )
                feature_detector.register_callback(self._on_feature_drift)
                detectors.append(feature_detector)
                logger.debug("Feature Monitor initialized")
        
        except ImportError as e:
            logger.error(f"Failed to import autonomous drift detectors: {str(e)}")
            logger.warning("Autonomous drift detection will be disabled")
            print(f"Error importing drift detectors: {str(e)}")
            print("Please ensure tinylcm and its autonomous drift detection modules are properly installed.")
        
        logger.info(f"Set up {len(detectors)} autonomous drift detectors")
        return detectors

    def _on_drift_detected(self, drift_info, result):
        """Callback for when drift is detected by any detector"""
        detector_name = drift_info.get("detector", "unknown")
        metric = drift_info.get("metric", "unknown")
        current_value = drift_info.get("current_value", 0.0)
        threshold = drift_info.get("threshold", 0.0)
        
        logger.warning(
            f"Drift detected by {detector_name} - "
            f"metric: {metric}, value: {current_value:.4f}, threshold: {threshold:.4f}"
        )
        
        # Log drift event for further analysis
        self.logging_queue.put(("log_metadata", {
            "data": {
                "drift_detected": True,
                "detector": detector_name,
                "metric": metric,
                "current_value": current_value,
                "threshold": threshold,
                "timestamp": time.time(),
                "sample_id": result.get("sample_id", "unknown")
            },
            "key": f"drift_{int(time.time())}",
            "category": "drift_events"
        }))

    def _on_confidence_drift(self, drift_info):
        """Callback for confidence-based drift"""
        logger.info(f"Confidence drift detected: average confidence dropped to {drift_info.get('current_value', 0.0):.4f}")

    def _on_distribution_drift(self, drift_info):
        """Callback for distribution-based drift"""
        logger.info(f"Distribution drift detected: JS divergence is {drift_info.get('current_value', 0.0):.4f}")

    def _on_feature_drift(self, drift_info):
        """Callback for feature-based drift"""
        logger.info(f"Feature drift detected: feature distance is {drift_info.get('current_value', 0.0):.4f}")

    async def save_image_async(self, img_path, image_data):
        """Save image asynchronously"""
        loop = asyncio.get_event_loop()
        await loop.run_in_executor(self.executor, cv2.imwrite, img_path, image_data)

    def logging_worker(self):
        """Worker thread for handling TinyLCM logging operations"""
        logger.info("Logging worker thread started")
        
        while not self.stop_event.is_set():
            try:
                # Get logging operation from queue
                log_item = self.logging_queue.get(timeout=1.0)
                
                if log_item is None:  # Shutdown signal
                    logger.info("Received shutdown signal in logging worker")
                    break
                
                operation, data = log_item
                logger.debug(f"Processing logging operation: {operation}")
                
                try:
                    if operation == "log_image":
                        self.data_logger.log_image(**data)
                    elif operation == "track_inference":
                        self.inference_monitor.track_inference(**data)
                    elif operation == "log_metadata":
                        if not hasattr(self.data_logger, "log_metadata"):
                            # Use log_data as fallback
                            logger.warning("DataLogger has no log_metadata method, using log_data instead")
                            self.data_logger.log_data(
                                input_data=json.dumps(data.get("data", {})), 
                                input_type="json",
                                metadata=data
                            )
                        else:
                            self.data_logger.log_metadata(**data)
                    else:
                        logger.warning(f"Unknown logging operation: {operation}")
                    
                    logger.debug(f"Successfully completed operation: {operation}")
                except Exception as inner_e:
                    logger.error(f"Error processing {operation}: {inner_e}", exc_info=True)
                
                self.logging_queue.task_done()
                
            except Empty:
                # This is normal, just continue
                continue
            except Exception as e:
                logger.error(f"Error in logging worker: {e}", exc_info=True)

    def sync_worker(self):
        """Worker thread for handling sync operations"""
        logger.info("Sync worker thread started")
        
        # Don't sync immediately - wait for the first interval
        # This prevents blocking during startup when network might not be available
        time.sleep(10)  # Short initial delay
        
        sync_count = 0
        
        while not self.stop_event.is_set():
            try:
                # Check if it's time to sync
                sync_interval = self.config["tinylcm"]["sync_interval_seconds"]
                logger.debug(f"Waiting {sync_interval}s for next sync operation (count: {sync_count})")
                
                # Use a loop with shorter sleeps to check stop_event more frequently
                for _ in range(int(sync_interval / 2)):
                    if self.stop_event.is_set():
                        break
                    time.sleep(2)
                
                if self.stop_event.is_set():
                    break
                
                if self.sync_client:
                    try:
                        logger.info(f"Starting sync operation #{sync_count+1}")
                        sync_start_time = time.time()
                        
                        # Create package with component data
                        logger.debug("Creating sync package...")
                        package_id = self.sync_interface.create_package_from_components(
                            device_id=self.config["tinysphere"]["device_id"],
                            state_manager=self.state_manager,
                            adaptation_tracker=self.adaptation_tracker,
                            inference_monitor=self.inference_monitor,
                            data_logger=self.data_logger
                        )
                        logger.debug(f"Created package with ID: {package_id}")
                        
                        # Sync quarantine data if available
                        if self.quarantine_buffer:
                            try:
                                # Synchronize quarantined samples
                                logger.debug("Getting quarantine samples for sync...")
                                quarantine_samples = self.quarantine_buffer.get_samples_for_sync()
                                if quarantine_samples:
                                    logger.debug(f"Found {len(quarantine_samples)} quarantine samples to sync")
                                    # Create a temporary JSON file with the quarantine samples
                                    with tempfile.NamedTemporaryFile(mode='w', suffix='.json', delete=False) as tmp:
                                        tmp_path = tmp.name
                                        json.dump(quarantine_samples, tmp)
                                    
                                    # Add the temporary file to the package
                                    logger.debug(f"Adding quarantine samples to package {package_id}")
                                    self.sync_interface.add_file_to_package(
                                        package_id=package_id,
                                        file_path=tmp_path,
                                        file_type="quarantine_samples",
                                        metadata={"count": len(quarantine_samples)}
                                    )
                                    
                                    # Remove the temporary file
                                    os.unlink(tmp_path)
                                    
                                    # Mark samples as synced
                                    sample_ids = []
                                    for sample in quarantine_samples:
                                        if isinstance(sample, dict) and "sample_id" in sample:
                                            sample_ids.append(sample["sample_id"])
                                        elif hasattr(sample, "sample_id"):
                                            sample_ids.append(sample.sample_id)
                                    
                                    if sample_ids:
                                        logger.debug(f"Marking {len(sample_ids)} samples as synced")
                                        self.quarantine_buffer.mark_as_synced(sample_ids)
                                    
                                    logger.info(f"Added {len(quarantine_samples)} quarantined samples to sync package")
                            except Exception as e:
                                logger.error(f"Failed to sync quarantine samples: {e}", exc_info=True)
                        
                        # Finalize package
                        logger.debug(f"Finalizing package {package_id}")
                        self.sync_interface.finalize_package(package_id)
                        
                        # Send package with timeout to prevent blocking indefinitely
                        logger.debug(f"Sending package {package_id} to server")
                        response = self.sync_client.send_package(package_id)
                        
                        # Check for validation results if using external validation
                        if self.config["tinylcm"].get("external_validation", False):
                            validation_results = response.get("validation_results", [])
                            if validation_results:
                                logger.debug(f"Processing {len(validation_results)} validation results")
                                # Process validation results
                                count = self.adaptive_pipeline.process_validation_results(validation_results)
                                logger.info(f"Processed {count} validation results from server")
                        
                        sync_duration = time.time() - sync_start_time
                        logger.info(f"Sync operation #{sync_count+1} completed successfully in {sync_duration:.2f}s")
                        sync_count += 1
                        
                    except Exception as e:
                        logger.error(f"Failed to sync data: {e}", exc_info=True)
            
            except Exception as e:
                logger.error(f"Error in sync worker: {e}", exc_info=True)

    async def process_frame_async(self, frame, frame_id):
        """Process a frame asynchronously with autonomous drift detection"""
        start_time = time.time()
        
        # Detect stones in the frame (runs in executor)
        detections = await self.detector.detect_async(frame)
        
        # Extract features based on configuration
        if self.config["tinylcm"].get("use_detection_scores_as_features", True):  # Default to TRUE for safer execution
            # Use detection scores as features (more efficient for embedded devices)
            # This avoids using TensorFlow directly for feature extraction
            if len(detections) > 0:  # Explicit length check instead of bool conversion
                # Create a feature vector from confidence scores
                confidence_scores = [conf for _, conf, _ in detections]
                # Pad with zeros if less than expected features
                while len(confidence_scores) < 5:  # Pad to expected feature size
                    confidence_scores.append(0.0)
                # Convert to numpy array
                features = np.array(confidence_scores, dtype=np.float32)
            else:
                # If no detections, create a zero vector
                features = np.zeros(5, dtype=np.float32)  # Default feature size
        else:
            # Only enter this branch if explicitly configured to use the feature extractor
            try:
                # Create a proper preprocessor like in the example code
                preprocessed_frame = None
                
                # Get input shape from model
                if hasattr(self.feature_extractor, '_interpreter') and self.feature_extractor._interpreter is not None:
                    input_details = self.feature_extractor._interpreter.get_input_details()
                    if input_details and len(input_details) > 0:
                        input_shape = input_details[0]['shape']
                        if len(input_shape) >= 3:
                            target_height, target_width = input_shape[1], input_shape[2]
                            
                            # Use the exact preprocessing approach from the example
                            resized_image = cv2.resize(frame, (target_width, target_height), interpolation=cv2.INTER_AREA)
                            
                            # Convert to 3 channels if needed
                            if len(resized_image.shape) > 2 and resized_image.shape[2] == 4:
                                resized_image = cv2.cvtColor(resized_image, cv2.COLOR_RGBA2RGB)
                                
                            # Convert to float32 and normalize
                            normalized_image = resized_image.astype('float32') / 255.0
                            
                            # Add batch dimension
                            preprocessed_frame = np.expand_dims(normalized_image, axis=0)
                            
                            # Extract features
                            features = self.feature_extractor.extract_features(preprocessed_frame)
                            
                            # Always flatten to 1D
                            features = features.flatten()
                        else:
                            raise ValueError(f"Invalid input shape: {input_shape}")
                    else:
                        raise ValueError("No input details available")
                else:
                    raise ValueError("Feature extractor not properly initialized")
                
            except Exception as e:
                logger.error(f"Feature extraction error: {e}")
                # ALWAYS fallback to detection scores on error
                if len(detections) > 0:
                    features = np.array([conf for _, conf, _ in detections], dtype=np.float32)
                else:
                    features = np.zeros(5, dtype=np.float32)
        
        # Calculate inference time
        inference_time_ms = (time.time() - start_time) * 1000
        
        # Process each detection
        for i, detection in enumerate(detections):
            class_id, confidence, bbox = detection
            prediction = self.detector.labels[class_id]
            
            # Process through adaptive pipeline with autonomous monitoring
            try:
                # NOTE: We previously hit an error with AdaptivePipeline.process()
                # getting unexpected keyword arguments 'prediction' and 'confidence'.
                # Looking at the method signature in pipeline.py, these aren't direct 
                # parameters but should go in metadata.
                
                # Create metadata dictionary with all detection information
                metadata = {
                    "inference_time_ms": inference_time_ms,
                    "bbox": bbox,
                    "frame_id": frame_id,
                    "detection_id": i,
                    "prediction": prediction,
                    "confidence": float(confidence),  # Ensure it's a scalar
                    "label": prediction,  # Default label is the current prediction
                }
                
                # Check the signature of AdaptivePipeline.process and only pass valid arguments
                # The AdaptivePipeline.process method accepts: input_data, label, timestamp, sample_id, metadata, extract_features
                
                # For autonomous mode, we don't have ground truth labels
                # We can pass prediction as the label if ground truth is available
                # In a real autonomous system, we would likely not pass a label here
                # unless using self-supervision or pseudo-labeling techniques
                result = self.adaptive_pipeline.process(
                    input_data=frame,  # Pass the frame for logging
                    label=prediction,  # Pass the predicted label - this is used for feature samples
                    timestamp=time.time(),
                    sample_id=f"frame_{frame_id}_{i}",
                    extract_features=False,  # We already have features
                    metadata=metadata
                )
            except Exception as e:
                logger.error(f"Error in adaptive pipeline processing: {e}")
                # Create a minimal result to continue operation
                result = {
                    "sample_id": f"frame_{frame_id}_{i}",
                    "quarantined": False,
                    "autonomous_drift_detected": False
                }
            
            # Queue inference tracking
            self.logging_queue.put(("track_inference", {
                "input_id": f"frame_{frame_id}_{i}",
                "prediction": prediction,
                "confidence": confidence,
                "latency_ms": inference_time_ms
            }))
            
            # Log if sample was quarantined
            if result.get("quarantined", False):
                logger.info(f"Sample quarantined: frame_{frame_id}_{i} (confidence: {confidence:.4f})")
            
            # Save stone image if enabled
            if self.config["application"]["save_detected_stones"]:
                # Extract the stone region
                x, y, w, h = bbox
                stone_img = frame[y:y+h, x:x+w]
                
                # Save stone image asynchronously
                timestamp = int(time.time())
                img_filename = f"stone_{timestamp}_{i}.jpg"
                img_path = os.path.join(self.config["application"]["data_dir"], img_filename)
                
                # Save asynchronously
                await self.save_image_async(img_path, stone_img)
                
                # Convert to bytes for DataLogger
                _, img_bytes = cv2.imencode('.jpg', stone_img)
                
                # Queue image logging
                self.logging_queue.put(("log_image", {
                    "image_data": img_bytes.tobytes(),
                    "prediction": prediction,
                    "confidence": confidence,
                    "metadata": {
                        "bbox": bbox,
                        "inference_time_ms": inference_time_ms,
                        "original_filename": img_filename,
                        "quarantined": result.get("quarantined", False),
                        "autonomous_drift_detected": result.get("autonomous_drift_detected", False)
                    }
                }))
                
                logger.info(f"Detected stone with confidence {confidence:.2f}, saved to {img_path}")
        
        # Draw detections on the frame
        frame_with_detections = self.detector.draw_detections(frame, detections)
        
        return frame_with_detections

    async def main_loop(self):
        """Main asynchronous processing loop"""
        frame_id = 0
        consecutive_errors = 0
        last_stats_time = time.time()
        
        logger.info("Starting main processing loop")
        
        while not self.stop_event.is_set():
            try:
                # Capture frame asynchronously
                logger.debug(f"Capturing frame {frame_id}")
                capture_start = time.time()
                frame = await self.camera.capture_frame_async()
                capture_time = time.time() - capture_start
                
                if frame is None:
                    logger.warning(f"Failed to capture frame (attempt {consecutive_errors+1}), retrying...")
                    consecutive_errors += 1
                    
                    # If we've failed too many times, try to restart the camera
                    if consecutive_errors > 10:
                        logger.error("Too many consecutive frame capture failures, trying to reset camera...")
                        try:
                            # Stop and restart camera
                            await self.camera.stop()
                            await asyncio.sleep(1)
                            await self.camera.start()
                            await asyncio.sleep(2)  # Give it time to initialize
                            consecutive_errors = 0
                            logger.info("Camera restarted")
                        except Exception as reset_e:
                            logger.error(f"Failed to reset camera: {reset_e}")
                    
                    await asyncio.sleep(0.2)  # Slightly longer delay between retries
                    continue
                
                # Successfully captured a frame
                consecutive_errors = 0
                logger.debug(f"Frame {frame_id} captured in {capture_time*1000:.1f}ms, shape: {frame.shape}")
                
                # Process frame asynchronously
                process_start = time.time()
                frame_with_detections = await self.process_frame_async(frame, frame_id)
                process_time = time.time() - process_start
                
                # Log successful processing
                logger.debug(f"Frame {frame_id} processed in {process_time*1000:.1f}ms")
                frame_id += 1
                
                # Display results if not headless
                if not self.config.get("headless", False):
                    cv2.imshow("Autonomous Stone Detector", frame_with_detections)
                    key = cv2.waitKey(1) & 0xFF
                    if key == ord('q'):
                        logger.info("User pressed 'q', stopping...")
                        self.stop_event.set()
                        break
                
                # Periodically output quarantine and adaptation statistics
                # Check time-based instead of frame-based since frames might be slow
                current_time = time.time()
                if current_time - last_stats_time > 30:  # Every 30 seconds
                    try:
                        last_stats_time = current_time
                        stats = self.adaptive_pipeline.get_statistics()
                        logger.info(
                            f"Stats after {frame_id} frames: {stats['n_samples_processed']} processed, "
                            f"{stats['n_samples_quarantined']} quarantined, "
                            f"{stats['n_autonomous_drift_detected']} drift events, "
                            f"{stats['n_samples_adapted']} adapted"
                        )
                        
                        # Also log queue sizes to help diagnose backups
                        logger.info(
                            f"Queue sizes - inference: {self.inference_queue.qsize()}, "
                            f"logging: {self.logging_queue.qsize()}, "
                            f"result: {self.result_queue.qsize()}, "
                            f"sync: {self.sync_queue.qsize()}"
                        )
                    except Exception as stats_e:
                        logger.error(f"Error getting statistics: {stats_e}")
                
                # Wait for the detection interval
                detection_interval = self.config["application"]["detection_interval"]
                logger.debug(f"Waiting {detection_interval}s before next frame")
                await asyncio.sleep(detection_interval)
            
            except asyncio.CancelledError:
                logger.info("Main loop cancelled")
                break
            except Exception as e:
                logger.error(f"Error in main loop: {e}", exc_info=True)
                consecutive_errors += 1
                await asyncio.sleep(0.5)  # Longer delay after errors
                
                # If we're having persistent errors, try to recover
                if consecutive_errors > 5:
                    logger.warning(f"Persistent errors in main loop (count: {consecutive_errors}), trying to recover...")
                    await asyncio.sleep(2)  # Extra delay for recovery

    def start_workers(self):
        """Start all worker threads"""
        # Start logging worker
        logging_thread = Thread(target=self.logging_worker, daemon=True)
        logging_thread.start()
        self.workers.append(logging_thread)
        
        # Start sync worker
        if self.sync_client:
            sync_thread = Thread(target=self.sync_worker, daemon=True)
            sync_thread.start()
            self.workers.append(sync_thread)
        
        logger.info("Worker threads started")

    async def run(self):
        """Run the asynchronous stone detector application"""
        logger.info("Starting Autonomous Stone Detector Application")
        
        try:
            # Start camera
            await self.camera.start()
            logger.info("Camera started")
            
            # Start system metrics collector
            metrics_task = asyncio.create_task(
                self.metrics_collector.start(self.data_logger)
            )
            
            # Start worker threads
            self.start_workers()
            
            # Run main loop
            await self.main_loop()
            
            # Cancel metrics task
            metrics_task.cancel()
            try:
                await metrics_task
            except asyncio.CancelledError:
                pass
            
        except KeyboardInterrupt:
            logger.info("Keyboard interrupt detected, shutting down...")
        except Exception as e:
            logger.error(f"Error in application: {e}", exc_info=True)
        finally:
            # Signal workers to stop
            self.stop_event.set()
            
            # Stop camera
            await self.camera.stop()
            
            # Close TinyLCM components
            self.inference_monitor.close()
            self.data_logger.close()
            
            # Close adaptive pipeline
            self.adaptive_pipeline.cleanup()
            
            # Close display window if open
            cv2.destroyAllWindows()
            
            # Wait for workers to finish
            for worker in self.workers:
                worker.join(timeout=5)
            
            # Shutdown executor
            self.executor.shutdown(wait=True)
            
            logger.info("Application shut down cleanly")

async def main():
    parser = argparse.ArgumentParser(description="Autonomous Stone Detector with TinyLCM")
    parser.add_argument("--config", default="scenarios/config_autonomous.json", help="Path to configuration file")
    parser.add_argument("--headless", action="store_true", help="Run in headless mode without GUI")
    parser.add_argument("--debug", action="store_true", help="Enable debug logging")
    parser.add_argument("--timeout", type=int, default=0, help="Auto-stop after specified seconds (0 for no timeout)")
    args = parser.parse_args()
    
    # Set up logging level
    if args.debug:
        logging.getLogger("stone_detector_autonomous").setLevel(logging.DEBUG)
        logging.getLogger("stone_detector.camera").setLevel(logging.DEBUG)
        logging.getLogger("tinylcm").setLevel(logging.DEBUG)
    
    logger.info(f"Starting Autonomous Stone Detector with config: {args.config}")
    
    try:
        # Create and run the application
        app = AutonomousStoneDetectorApp(config_path=args.config)
        
        # Set headless mode if specified
        if args.headless:
            app.config["headless"] = True
            logger.info("Running in headless mode")
        
        # Set up auto-stop if requested
        if args.timeout > 0:
            logger.info(f"Will automatically stop after {args.timeout} seconds")
            
            def auto_stop():
                logger.info(f"Auto-stop timer of {args.timeout}s elapsed, stopping application")
                app.stop_event.set()
            
            # Schedule the auto-stop
            loop = asyncio.get_event_loop()
            loop.call_later(args.timeout, auto_stop)
        
        # Run the application with timeout watch
        try:
            logger.info("Starting application run")
            await asyncio.wait_for(app.run(), timeout=None)  # No timeout here, we control it separately
        except asyncio.TimeoutError:
            logger.error("Application timed out")
        except KeyboardInterrupt:
            logger.info("Application interrupted by user")
        except Exception as e:
            logger.error(f"Application run failed: {e}", exc_info=True)
        
        logger.info("Application run completed")
    
    except Exception as e:
        logger.error(f"Failed to start application: {e}", exc_info=True)
    
    finally:
        # Final cleanup
        logger.info("Exiting main function")

if __name__ == "__main__":
    asyncio.run(main())