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
                "sync_interval_seconds": 300,
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
            "tinylcm_data/heuristic_logs"
        ]
        for directory in directories:
            os.makedirs(directory, exist_ok=True)
        logger.debug("Ensured all required directories exist")

    def initialize_tinylcm(self):
        """Initialize TinyLCM components with autonomous drift detection"""
        # Initialize feature extractor
        self.feature_extractor = tinylcm.TFLiteFeatureExtractor(
            model_path=self.config["model"]["path"]
        )
        
        # Initialize classifier
        self.classifier = tinylcm.LightweightKNN(
            k=3,
            distance_metric="euclidean",
            max_samples=100,
            use_numpy=True,
            weight_by_distance=False
        )
        
        # Initialize State Manager
        self.state_manager = tinylcm.StateManager(
            storage_dir=self.config["tinylcm"]["model_dir"]
        )
        
        # Initialize Adaptation Tracker
        self.adaptation_tracker = tinylcm.AdaptationTracker(
            log_dir=self.config["tinylcm"]["model_dir"],
            auto_create_dir=True,
            export_format="json"
        )
        
        # Initialize DataLogger for data collection
        self.data_logger = tinylcm.DataLogger(
            storage_dir=self.config["tinylcm"]["data_dir"]
        )
        
        # Initialize InferenceMonitor for basic metrics
        self.inference_monitor = tinylcm.InferenceMonitor(
            storage_dir=self.config["tinylcm"]["inference_dir"]
        )
        
        # Set up autonomous drift detectors
        drift_config = self.config["tinylcm"].get("drift_detection", {})
        self.autonomous_monitors = self._setup_autonomous_detectors(drift_config)
        
        # Set up quarantine buffer if enabled
        if self.config["tinylcm"].get("enable_quarantine", True):
            self.quarantine_buffer = tinylcm.QuarantineBuffer(
                storage_dir=self.config["tinylcm"]["quarantine_dir"],
                max_size=1000,
                auto_persist=True,
                quarantine_strategy=tinylcm.QuarantineStrategy.ALL_DRIFT_SAMPLES
            )
        else:
            self.quarantine_buffer = None
        
        # Set up heuristic adapter if enabled
        if (self.config["tinylcm"].get("enable_heuristic_adaptation", False) and 
            self.quarantine_buffer is not None):
            self.heuristic_adapter = tinylcm.HeuristicAdapter(
                quarantine_buffer=self.quarantine_buffer,
                strategy=tinylcm.HeuristicStrategy.HYBRID,
                min_cluster_size=5,
                min_samples_for_adaptation=10,
                confidence_threshold=self.config["tinylcm"].get("heuristic_confidence_threshold", 0.7),
                log_dir=self.config["tinylcm"]["heuristic_dir"],
                max_new_classes=3
            )
        else:
            self.heuristic_adapter = None
        
        # Initialize Hybrid Handler with classifier
        self.hybrid_handler = tinylcm.HybridHandler(
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
        self.adaptive_pipeline = tinylcm.AdaptivePipeline(
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
        
        # Initialize SyncInterface for data synchronization
        self.sync_interface = tinylcm.SyncInterface(
            sync_dir="tinylcm_data/sync"
        )
        
        # Initialize SyncClient
        try:
            self.sync_client = SyncClient(
                server_url=self.config["tinysphere"]["server_url"],
                api_key=self.config["tinysphere"]["api_key"],
                device_id=self.config["tinysphere"]["device_id"],
                sync_interface=self.sync_interface
            )
            logger.info("SyncClient initialized successfully")
        except Exception as e:
            logger.error(f"Failed to initialize SyncClient: {e}")
            self.sync_client = None
        
        logger.info("TinyLCM components initialized with autonomous drift detection")

    def _setup_autonomous_detectors(self, drift_config):
        """Set up autonomous drift detectors based on configuration"""
        detectors = []
        
        # Add EWMA Confidence Monitor if enabled
        if drift_config.get("ewma_confidence", {}).get("enabled", True):
            ewma_config = drift_config.get("ewma_confidence", {})
            ewma_detector = tinylcm.EWMAConfidenceMonitor(
                lambda_param=ewma_config.get("lambda_param", 0.1),
                threshold_factor=ewma_config.get("threshold_factor", 3.0),
                drift_window=ewma_config.get("drift_window", 5),
                training_size=ewma_config.get("training_size", 30)
            )
            ewma_detector.register_callback(self._on_confidence_drift)
            detectors.append(ewma_detector)
            logger.debug("EWMA Confidence Monitor initialized")
        
        # Add Distribution Monitor if enabled
        if drift_config.get("distribution", {}).get("enabled", True):
            dist_config = drift_config.get("distribution", {})
            dist_detector = tinylcm.PredictionDistributionMonitor(
                window_size=dist_config.get("window_size", 50),
                threshold=dist_config.get("threshold", 0.25),
                method=dist_config.get("method", "block")
            )
            dist_detector.register_callback(self._on_distribution_drift)
            detectors.append(dist_detector)
            logger.debug("Prediction Distribution Monitor initialized")
        
        # Add Feature Monitor if enabled
        if drift_config.get("feature", {}).get("enabled", True):
            feature_config = drift_config.get("feature", {})
            feature_detector = tinylcm.FeatureMonitor(
                window_size=feature_config.get("window_size", 100),
                threshold=feature_config.get("threshold", 3.0),
                reference_size=feature_config.get("reference_size", 50),
                max_features=feature_config.get("max_features", 50),
                distance_metric=feature_config.get("distance_metric", "euclidean")
            )
            feature_detector.register_callback(self._on_feature_drift)
            detectors.append(feature_detector)
            logger.debug("Feature Monitor initialized")
        
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
        while not self.stop_event.is_set():
            try:
                # Get logging operation from queue
                log_item = self.logging_queue.get(timeout=1.0)
                
                if log_item is None:  # Shutdown signal
                    break
                
                operation, data = log_item
                
                if operation == "log_image":
                    self.data_logger.log_image(**data)
                elif operation == "track_inference":
                    self.inference_monitor.track_inference(**data)
                elif operation == "log_metadata":
                    self.data_logger.log_metadata(**data)
                
                self.logging_queue.task_done()
                
            except Empty:
                continue
            except Exception as e:
                logger.error(f"Error in logging worker: {e}")

    def sync_worker(self):
        """Worker thread for handling sync operations"""
        while not self.stop_event.is_set():
            try:
                # Check if it's time to sync
                sync_interval = self.config["tinylcm"]["sync_interval_seconds"]
                time.sleep(sync_interval)
                
                if self.sync_client:
                    try:
                        logger.info("Starting sync operation")
                        
                        # Create package with component data
                        package_id = self.sync_interface.create_package_from_components(
                            device_id=self.config["tinysphere"]["device_id"],
                            state_manager=self.state_manager,
                            adaptation_tracker=self.adaptation_tracker,
                            inference_monitor=self.inference_monitor,
                            data_logger=self.data_logger
                        )
                        
                        # Sync quarantine data if available
                        if self.quarantine_buffer:
                            # Synchronize quarantined samples
                            quarantine_samples = self.quarantine_buffer.get_samples_for_sync()
                            if quarantine_samples:
                                self.sync_interface.add_quarantine_samples(
                                    package_id=package_id,
                                    samples=quarantine_samples
                                )
                                logger.info(f"Added {len(quarantine_samples)} quarantined samples to sync package")
                        
                        # Finalize package
                        self.sync_interface.finalize_package(package_id)
                        
                        # Send package
                        response = self.sync_client.send_package(package_id)
                        
                        # Check for validation results if using external validation
                        if self.config["tinylcm"].get("external_validation", False):
                            validation_results = response.get("validation_results", [])
                            if validation_results:
                                # Process validation results
                                count = self.adaptive_pipeline.process_validation_results(validation_results)
                                logger.info(f"Processed {count} validation results from server")
                        
                        logger.info("Sync operation completed successfully")
                        
                    except Exception as e:
                        logger.error(f"Failed to sync data: {e}")
            
            except Exception as e:
                logger.error(f"Error in sync worker: {e}")

    async def process_frame_async(self, frame, frame_id):
        """Process a frame asynchronously with autonomous drift detection"""
        start_time = time.time()
        
        # Extract features
        features = self.feature_extractor.extract_features(frame)
        
        # Detect stones in the frame (runs in executor)
        detections = await self.detector.detect_async(frame)
        
        # Calculate inference time
        inference_time_ms = (time.time() - start_time) * 1000
        
        # Process each detection
        for i, detection in enumerate(detections):
            class_id, confidence, bbox = detection
            prediction = self.detector.labels[class_id]
            
            # Process through adaptive pipeline with autonomous monitoring
            result = self.adaptive_pipeline.process(
                input_data=frame,  # Pass the frame for logging
                timestamp=time.time(),
                sample_id=f"frame_{frame_id}_{i}",
                extract_features=False,  # We already extracted features
                metadata={
                    "inference_time_ms": inference_time_ms,
                    "bbox": bbox,
                    "frame_id": frame_id,
                    "detection_id": i
                }
            )
            
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
        
        while not self.stop_event.is_set():
            try:
                # Capture frame asynchronously
                frame = await self.camera.capture_frame_async()
                
                if frame is None:
                    logger.warning("Failed to capture frame, retrying...")
                    await asyncio.sleep(0.1)
                    continue
                
                # Process frame asynchronously
                frame_with_detections = await self.process_frame_async(frame, frame_id)
                frame_id += 1
                
                # Display results if not headless
                if not self.config.get("headless", False):
                    cv2.imshow("Autonomous Stone Detector", frame_with_detections)
                    key = cv2.waitKey(1) & 0xFF
                    if key == ord('q'):
                        self.stop_event.set()
                        break
                
                # Periodically output quarantine and adaptation statistics
                if frame_id % 100 == 0:
                    stats = self.adaptive_pipeline.get_statistics()
                    logger.info(
                        f"Stats: {stats['n_samples_processed']} processed, "
                        f"{stats['n_samples_quarantined']} quarantined, "
                        f"{stats['n_autonomous_drift_detected']} drift events, "
                        f"{stats['n_samples_adapted']} adapted"
                    )
                
                # Wait for the detection interval
                await asyncio.sleep(self.config["application"]["detection_interval"])
            
            except asyncio.CancelledError:
                break
            except Exception as e:
                logger.error(f"Error in main loop: {e}", exc_info=True)
                await asyncio.sleep(0.1)

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
    args = parser.parse_args()
    
    # Create and run the application
    app = AutonomousStoneDetectorApp(config_path=args.config)
    
    # Set headless mode if specified
    if args.headless:
        app.config["headless"] = True
    
    # Run the application
    await app.run()

if __name__ == "__main__":
    asyncio.run(main())