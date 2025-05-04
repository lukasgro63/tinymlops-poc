#!/usr/bin/env python3
"""
Asynchronous Stone Detector Application with TinyLCM Integration
---------------------------------------------------------------
This application runs on a Raspberry Pi Zero 2W with a Pi Camera to detect stones.
It uses TensorFlow Lite for inference and TinyLCM for monitoring and data management.
All critical operations are asynchronous to prevent blocking.
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
import tinylcm

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
        logging.FileHandler("stone_detector.log")
    ]
)
logger = logging.getLogger("stone_detector")

class AsyncStoneDetectorApp:
    def __init__(self, config_path="config.json"):
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
        
        logger.info("Asynchronous Stone Detector Application initialized")

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
                "sync_interval_seconds": 30
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
            "tinylcm_data/inference_logs"
        ]
        for directory in directories:
            os.makedirs(directory, exist_ok=True)
        logger.debug("Ensured all required directories exist")

    def initialize_tinylcm(self):
        """Initialize TinyLCM components"""
        # Initialize the new adaptive pipeline components
        feature_extractor = tinylcm.TFLiteFeatureExtractor(
            model_path=self.config["model"]["path"]
        )
        
        # LightweightKNN with correct parameters
        classifier = tinylcm.LightweightKNN(
            k=3,
            distance_metric="euclidean",
            max_samples=100,
            use_numpy=True,
            weight_by_distance=False
        )
        
        # Initialize State Manager (replaces ModelManager)
        self.state_manager = tinylcm.StateManager(
            storage_dir=self.config["tinylcm"]["model_dir"]
        )
        
        # Initialize Adaptation Tracker
        self.adaptation_tracker = tinylcm.AdaptationTracker(
            storage_dir=self.config["tinylcm"]["model_dir"]
        )
        
        # Initialize Hybrid Handler with classifier
        hybrid_handler = tinylcm.HybridHandler(
            classifier=classifier,
            max_samples=100,
            batch_size=30,
            baseline_accuracy=0.9,
            cusum_threshold=5.0,
            cusum_delta=0.25,
            enable_condensing=False,
            use_numpy=True
        )
        
        # Initialize Adaptive Pipeline with Hybrid Handler
        self.adaptive_pipeline = tinylcm.AdaptivePipeline(
            feature_extractor=feature_extractor,
            handler=hybrid_handler,
            state_manager=self.state_manager,
            adaptation_tracker=self.adaptation_tracker,
            data_logger=None,  # We'll set this up separately
            config={"auto_save": True}  # Use config dict instead of auto_save parameter
        )
        
        # Initialize DataLogger for data collection
        self.data_logger = tinylcm.DataLogger(
            storage_dir=self.config["tinylcm"]["data_dir"]
        )
        
        # Initialize InferenceMonitor for basic metrics
        self.inference_monitor = tinylcm.InferenceMonitor(
            storage_dir=self.config["tinylcm"]["inference_dir"]
        )
        
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
        
        logger.info("TinyLCM components initialized with adaptive pipeline")

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
                        
                        # Finalize package
                        self.sync_interface.finalize_package(package_id)
                        
                        # Send package
                        self.sync_client.send_package(package_id)
                        logger.info("Sync operation completed successfully")
                        
                    except Exception as e:
                        logger.error(f"Failed to sync data: {e}")
            
            except Exception as e:
                logger.error(f"Error in sync worker: {e}")

    async def process_frame_async(self, frame, frame_id):
        """Process a frame asynchronously"""
        start_time = time.time()
        
        # Detect stones in the frame (runs in executor)
        detections = await self.detector.detect_async(frame)
        
        # Calculate inference time
        inference_time_ms = (time.time() - start_time) * 1000
        
        # Process each detection
        for i, detection in enumerate(detections):
            class_id, confidence, bbox = detection
            
            # Queue inference tracking
            self.logging_queue.put(("track_inference", {
                "input_id": f"frame_{frame_id}_{i}",
                "prediction": self.detector.labels[class_id],
                "confidence": confidence,
                "latency_ms": inference_time_ms
            }))
            
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
                    "prediction": self.detector.labels[class_id],
                    "confidence": confidence,
                    "metadata": {
                        "bbox": bbox,
                        "inference_time_ms": inference_time_ms,
                        "original_filename": img_filename
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
                    cv2.imshow("Stone Detector", frame_with_detections)
                    key = cv2.waitKey(1) & 0xFF
                    if key == ord('q'):
                        self.stop_event.set()
                        break
                
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
        logger.info("Starting Asynchronous Stone Detector Application")
        
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
            
            # Close display window if open
            cv2.destroyAllWindows()
            
            # Wait for workers to finish
            for worker in self.workers:
                worker.join(timeout=5)
            
            # Shutdown executor
            self.executor.shutdown(wait=True)
            
            logger.info("Application shut down cleanly")

async def main():
    parser = argparse.ArgumentParser(description="Asynchronous Stone Detector with TinyLCM")
    parser.add_argument("--config", default="config.json", help="Path to configuration file")
    parser.add_argument("--headless", action="store_true", help="Run in headless mode without GUI")
    args = parser.parse_args()
    
    # Create and run the application
    app = AsyncStoneDetectorApp(config_path=args.config)
    
    # Set headless mode if specified
    if args.headless:
        app.config["headless"] = True
    
    # Run the application
    await app.run()

if __name__ == "__main__":
    asyncio.run(main())