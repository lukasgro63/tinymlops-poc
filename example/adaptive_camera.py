#!/usr/bin/env python3
"""
Adaptive Camera Object Detection for Raspberry Pi Zero 2W
--------------------------------------------------------
This application demonstrates the full TinyLCM adaptive workflow:
1. Capturing images with picamera2
2. Processing with TFLite feature extraction
3. Classification with adaptive k-NN
4. Performance monitoring
5. Feedback mechanism
6. Drift detection and handling
7. State management
8. Clean shutdown

Requirements:
- Raspberry Pi Zero 2W
- Raspberry Pi Camera Module
- OpenCV (cv2)
- Picamera2 
- NumPy
- TinyLCM with adaptive core

Author: TinyLCM Team
"""

import argparse
import asyncio
import concurrent.futures
import json
import os
import signal
import sys
import threading
import time
import uuid
from dataclasses import dataclass
from datetime import datetime
from pathlib import Path
from queue import Empty, Queue, Full
from typing import Any, Dict, List, Optional, Tuple, Union

import cv2
import numpy as np

# Import picamera2 with fallback for development environments
try:
    from picamera2 import Picamera2
    PICAMERA_AVAILABLE = True
except ImportError:
    print("Warning: picamera2 not available - using fallback camera implementation")
    PICAMERA_AVAILABLE = False

# Import TinyLCM
import tinylcm

# Configure paths
MODEL_DIR = os.path.join(os.path.dirname(__file__), "models")
DATA_DIR = os.path.join(os.path.dirname(__file__), "data")
STATE_DIR = os.path.join(DATA_DIR, "states")
ADAPTATION_DIR = os.path.join(DATA_DIR, "adaptation")
INFERENCE_DIR = os.path.join(DATA_DIR, "inference")

# Ensure directories exist
for directory in [MODEL_DIR, DATA_DIR, STATE_DIR, ADAPTATION_DIR, INFERENCE_DIR]:
    os.makedirs(directory, exist_ok=True)

# Configure logging
import logging
logging.basicConfig(
    level=logging.INFO,
    format='%(asctime)s - %(name)s - %(levelname)s - %(message)s',
    handlers=[
        logging.StreamHandler(),
        logging.FileHandler(os.path.join(DATA_DIR, "adaptive_camera.log"))
    ]
)
logger = logging.getLogger("adaptive_camera")

# Default model and labels paths
DEFAULT_MODEL_PATH = os.path.join(MODEL_DIR, "mobilenet_v2_1.0_224_quant.tflite")
DEFAULT_LABELS_PATH = os.path.join(MODEL_DIR, "labels.txt")

@dataclass
class ProcessingResult:
    """Holds the result of a single frame processing."""
    frame_id: str
    frame: np.ndarray
    result: Any
    latency_ms: float
    timestamp: float


class FallbackCamera:
    """Fallback camera implementation using OpenCV (for development without Pi Camera)."""
    
    def __init__(self, width: int = 640, height: int = 480, fps: int = 30):
        self.width = width
        self.height = height
        self.fps = fps
        self.camera = None
        logger.info(f"Initializing fallback camera: {width}x{height} @ {fps}fps")
        
    def start(self):
        """Start the camera."""
        self.camera = cv2.VideoCapture(0)
        self.camera.set(cv2.CAP_PROP_FRAME_WIDTH, self.width)
        self.camera.set(cv2.CAP_PROP_FRAME_HEIGHT, self.height)
        self.camera.set(cv2.CAP_PROP_FPS, self.fps)
        if not self.camera.isOpened():
            raise RuntimeError("Failed to open camera")
        logger.info("Fallback camera started")
        
    def capture_array(self):
        """Capture a frame as a numpy array."""
        if self.camera is None:
            raise RuntimeError("Camera not started")
        ret, frame = self.camera.read()
        if not ret:
            raise RuntimeError("Failed to read frame from camera")
        return frame
    
    def close(self):
        """Stop the camera."""
        if self.camera is not None:
            self.camera.release()
            self.camera = None
        logger.info("Fallback camera stopped")


class AdaptiveCameraApp:
    """
    Adaptive Camera Application using TinyLCM for continuous learning.
    
    This application captures frames from the camera, processes them through
    an adaptive ML pipeline, and displays the results. It supports:
    - Interactive feedback to improve the model
    - Drift detection and automatic adaptation
    - Performance monitoring
    - State saving and loading
    - GUI visualization of adaptive model state
    """
    
    def __init__(self, config_path: Optional[str] = None):
        """Initialize the adaptive camera application."""
        # Load configuration
        self.config = self._load_config(config_path)
        logger.info(f"Loaded configuration: {self.config}")
        
        # Initialize camera
        self._init_camera()
        
        # Load model labels
        self.labels = self._load_labels(self.config["model"]["labels_path"])
        logger.info(f"Loaded {len(self.labels)} labels")
        
        # Initialize TinyLCM components
        self._init_tinylcm_components()
        
        # Signal handling for clean shutdown
        signal.signal(signal.SIGINT, self._signal_handler)
        signal.signal(signal.SIGTERM, self._signal_handler)
        
        # Initialize processing queues
        self.frame_queue = Queue(maxsize=self.config["processing"]["queue_size"])
        self.result_queue = Queue(maxsize=self.config["processing"]["queue_size"])
        self.feedback_queue = Queue(maxsize=self.config["processing"]["queue_size"])
        
        # Initialize threading primitives
        self.stop_event = threading.Event()
        self.workers = []
        
        # Statistics
        self.stats = {
            "frames_processed": 0,
            "frames_dropped": 0,
            "feedback_provided": 0,
            "drift_events": 0,
            "start_time": time.time()
        }
        
        # Last processed result for display
        self.last_result = None
        self.last_frame = None
        self.last_frame_lock = threading.Lock()
        
        # UI state
        self.selected_label = None
        self.user_feedback_enabled = self.config["user_interface"]["enable_feedback"]
        
        # Initialize framerate counter
        self.frame_times = []
        
        logger.info("AdaptiveCameraApp initialized")
        
    def _load_config(self, config_path: Optional[str]) -> Dict[str, Any]:
        """Load configuration from JSON file with defaults."""
        # Default configuration
        default_config = {
            "camera": {
                "width": 640,
                "height": 480,
                "fps": 30,
                "rotation": 0
            },
            "model": {
                "path": DEFAULT_MODEL_PATH,
                "labels_path": DEFAULT_LABELS_PATH,
                "feature_layer": -2,
                "input_size": [224, 224]
            },
            "processing": {
                "queue_size": 10,
                "max_workers": 2
            },
            "adaptive": {
                "strategy": "hybrid",
                "k_neighbors": 5,
                "distance_metric": "euclidean",
                "cusum_threshold": 5.0,
                "cusum_delta": 0.25,
                "save_on_drift": True
            },
            "user_interface": {
                "enable_display": True,
                "enable_feedback": True,
                "window_name": "TinyLCM Adaptive Camera",
                "display_confidence": True,
                "display_latency": True
            }
        }
        
        # If config path provided, load and merge with defaults
        if config_path:
            try:
                with open(config_path, 'r') as f:
                    user_config = json.load(f)
                
                # Recursive dictionary merge
                def merge_dicts(d1, d2):
                    for k, v in d2.items():
                        if k in d1 and isinstance(d1[k], dict) and isinstance(v, dict):
                            merge_dicts(d1[k], v)
                        else:
                            d1[k] = v
                
                merge_dicts(default_config, user_config)
                logger.info(f"Loaded configuration from {config_path}")
            except Exception as e:
                logger.error(f"Failed to load config from {config_path}: {e}")
                logger.warning("Using default configuration")
        
        return default_config

    def _load_labels(self, labels_path: str) -> List[str]:
        """Load class labels from file."""
        try:
            with open(labels_path, 'r') as f:
                return [line.strip() for line in f.readlines()]
        except Exception as e:
            logger.error(f"Failed to load labels from {labels_path}: {e}")
            # Return some default labels as fallback
            return ["background", "person", "cat", "dog", "object", "vehicle"]

    def _init_camera(self):
        """Initialize the camera (picamera2 or fallback)."""
        camera_config = self.config["camera"]
        
        if PICAMERA_AVAILABLE:
            logger.info("Initializing Picamera2")
            self.camera = Picamera2()
            
            # Configure camera
            config = self.camera.create_still_configuration(
                main={"size": (camera_config["width"], camera_config["height"])}
            )
            self.camera.configure(config)
            
            # Set rotation if needed
            if camera_config.get("rotation"):
                self.camera.set_controls({"RotateISP": camera_config["rotation"]})
        else:
            logger.info("Using fallback camera (OpenCV)")
            self.camera = FallbackCamera(
                width=camera_config["width"],
                height=camera_config["height"],
                fps=camera_config["fps"]
            )

    def _init_tinylcm_components(self):
        """Initialize TinyLCM components for the adaptive ML pipeline."""
        logger.info("Initializing TinyLCM components")
        
        # Initialize feature extractor with the TFLite model
        self.feature_extractor = tinylcm.TFLiteFeatureExtractor(
            model_path=self.config["model"]["path"],
            feature_layer_index=self.config["model"]["feature_layer"],
            input_shape=self.config["model"]["input_size"]
        )
        
        # Initialize the adaptive k-NN classifier
        self.classifier = tinylcm.LightweightKNN(
            k=self.config["adaptive"]["k_neighbors"],
            distance_metric=self.config["adaptive"]["distance_metric"],
            distance_threshold=0.8,  # Limit influence of distant neighbors
            enable_condensing=True,  # Keep dataset compact
            condensing_method="class_balanced"
        )
        
        # Select the appropriate handler based on configuration
        adaptive_strategy = self.config["adaptive"]["strategy"].lower()
        if adaptive_strategy == "passive":
            self.handler = tinylcm.PassiveHandler()
        elif adaptive_strategy == "active":
            self.handler = tinylcm.ActiveHandler(
                cusum_threshold=self.config["adaptive"]["cusum_threshold"],
                cusum_delta=self.config["adaptive"]["cusum_delta"],
                batch_size=30
            )
        else:  # default to hybrid
            self.handler = tinylcm.HybridHandler(
                cusum_threshold=self.config["adaptive"]["cusum_threshold"],
                cusum_delta=self.config["adaptive"]["cusum_delta"],
                batch_size=30
            )
        
        # Initialize state manager for saving and loading model states
        self.state_manager = tinylcm.StateManager(
            storage_dir=STATE_DIR,
            max_states=10
        )
        
        # Initialize adaptation tracker for logging adaptation events
        self.adaptation_tracker = tinylcm.AdaptationTracker(
            log_dir=ADAPTATION_DIR,
            max_events_in_memory=100
        )
        
        # Initialize inference monitor for tracking performance metrics
        self.inference_monitor = tinylcm.InferenceMonitor(
            storage_dir=INFERENCE_DIR
        )
        
        # Finally, create the adaptive pipeline to coordinate all components
        self.pipeline = tinylcm.AdaptivePipeline(
            feature_extractor=self.feature_extractor,
            classifier=self.classifier,
            handler=self.handler,
            state_manager=self.state_manager,
            adaptation_tracker=self.adaptation_tracker,
            auto_save=self.config["adaptive"]["save_on_drift"]
        )
        
        # Register drift callback
        self.pipeline.register_drift_callback(self._on_drift_detected)
        
        logger.info(f"Initialized adaptive pipeline with {adaptive_strategy} strategy")
    
    def _on_drift_detected(self, event):
        """Callback for drift detection events."""
        self.stats["drift_events"] += 1
        logger.info(f"Drift detected! Reason: {event.reason}, Type: {event.adaptation_type}")
        
        # You could take additional actions here like alerting, logging, etc.
    
    def _preprocess_frame(self, frame):
        """Preprocess a frame for model input."""
        # Resize to the expected input size
        input_size = self.config["model"]["input_size"]
        return cv2.resize(frame, (input_size[1], input_size[0]))
    
    def _signal_handler(self, sig, frame):
        """Handle termination signals for clean shutdown."""
        logger.info(f"Received signal {sig}, initiating shutdown")
        self.stop()
    
    def frame_capture_worker(self):
        """Worker thread for capturing frames from the camera."""
        logger.info("Frame capture worker started")
        
        try:
            frame_id = 0
            
            while not self.stop_event.is_set():
                try:
                    # Capture frame
                    frame = self.camera.capture_array()
                    
                    # Try to put in queue, drop if full
                    try:
                        self.frame_queue.put((frame_id, frame), block=False)
                        frame_id += 1
                    except Full:
                        self.stats["frames_dropped"] += 1
                        logger.debug("Frame queue full, dropping frame")
                    
                    # Brief pause to control frame rate
                    time.sleep(1.0 / self.config["camera"]["fps"])
                    
                except Exception as e:
                    logger.error(f"Error in frame capture: {e}")
                    time.sleep(0.1)  # Avoid tight loop in case of errors
            
            logger.info("Frame capture worker stopped")
        except Exception as e:
            logger.error(f"Fatal error in frame capture worker: {e}")
    
    def processing_worker(self):
        """Worker thread for processing frames through the adaptive pipeline."""
        logger.info("Processing worker started")
        
        try:
            while not self.stop_event.is_set():
                try:
                    # Get frame from queue
                    frame_id, frame = self.frame_queue.get(timeout=0.1)
                    
                    # Preprocess frame
                    input_tensor = self._preprocess_frame(frame)
                    
                    # Process with adaptive pipeline
                    start_time = time.time()
                    result = self.pipeline.process(input_tensor)
                    latency_ms = (time.time() - start_time) * 1000
                    
                    # Track inference metrics
                    self.inference_monitor.track_inference(
                        input_id=f"frame_{frame_id}",
                        prediction=self.labels[int(result.prediction)],
                        confidence=result.confidence,
                        latency_ms=latency_ms,
                        metadata={"shape": frame.shape}
                    )
                    
                    # Create processing result
                    proc_result = ProcessingResult(
                        frame_id=f"frame_{frame_id}",
                        frame=frame,
                        result=result,
                        latency_ms=latency_ms,
                        timestamp=time.time()
                    )
                    
                    # Put result in queue
                    try:
                        self.result_queue.put(proc_result, block=False)
                    except Full:
                        logger.warning("Result queue full, dropping result")
                    
                    # Update statistics
                    self.stats["frames_processed"] += 1
                    
                    # Update last result for display
                    with self.last_frame_lock:
                        self.last_result = proc_result
                        self.last_frame = frame.copy()
                    
                    # Mark task as done
                    self.frame_queue.task_done()
                    
                except Empty:
                    # No frame available
                    continue
                except Exception as e:
                    logger.error(f"Error processing frame: {e}")
                    if 'frame_id' in locals():
                        self.frame_queue.task_done()
            
            logger.info("Processing worker stopped")
        except Exception as e:
            logger.error(f"Fatal error in processing worker: {e}")
    
    def feedback_worker(self):
        """Worker thread for handling feedback."""
        logger.info("Feedback worker started")
        
        try:
            while not self.stop_event.is_set():
                try:
                    # Get feedback from queue
                    feedback = self.feedback_queue.get(timeout=0.1)
                    
                    # Process feedback
                    result, label = feedback
                    
                    # Provide feedback to the pipeline
                    self.pipeline.provide_feedback(
                        features=result.result.features,
                        true_label=label,
                        metadata={
                            "source": "user_input",
                            "timestamp": time.time()
                        }
                    )
                    
                    # Update statistics
                    self.stats["feedback_provided"] += 1
                    
                    # Mark task as done
                    self.feedback_queue.task_done()
                    
                except Empty:
                    # No feedback available
                    continue
                except Exception as e:
                    logger.error(f"Error processing feedback: {e}")
                    if 'feedback' in locals():
                        self.feedback_queue.task_done()
            
            logger.info("Feedback worker stopped")
        except Exception as e:
            logger.error(f"Fatal error in feedback worker: {e}")
    
    def display_worker(self):
        """Worker thread for displaying processed frames."""
        if not self.config["user_interface"]["enable_display"]:
            return
        
        logger.info("Display worker started")
        window_name = self.config["user_interface"]["window_name"]
        cv2.namedWindow(window_name, cv2.WINDOW_NORMAL)
        
        # Setup UI elements
        if self.user_feedback_enabled:
            # Create buttons for each label
            button_height = 30
            button_width = 100
            button_margin = 10
            self.button_regions = []
            
            for i, label in enumerate(self.labels):
                x = button_margin + (button_width + button_margin) * (i % 3)
                y = button_margin + (button_height + button_margin) * (i // 3)
                self.button_regions.append((label, (x, y, button_width, button_height)))
        
        # Initialize FPS calculation
        fps_start_time = time.time()
        fps_counter = 0
        fps = 0
        
        try:
            while not self.stop_event.is_set():
                # Get the latest frame and result
                with self.last_frame_lock:
                    if self.last_frame is None or self.last_result is None:
                        time.sleep(0.01)
                        continue
                    
                    frame = self.last_frame.copy()
                    result = self.last_result
                
                # Draw prediction
                prediction_idx = int(result.result.prediction)
                prediction_label = self.labels[prediction_idx]
                confidence = result.result.confidence if hasattr(result.result, 'confidence') else None
                
                # Prepare display text
                text = prediction_label
                if confidence is not None and self.config["user_interface"]["display_confidence"]:
                    text += f": {confidence:.2f}"
                if self.config["user_interface"]["display_latency"]:
                    text += f" | {result.latency_ms:.1f}ms"
                
                # Draw text on the image
                cv2.putText(
                    frame, text, (10, 30),
                    cv2.FONT_HERSHEY_SIMPLEX, 1, (0, 255, 0), 2
                )
                
                # Calculate and display FPS
                fps_counter += 1
                if (time.time() - fps_start_time) > 1.0:
                    fps = fps_counter / (time.time() - fps_start_time)
                    fps_counter = 0
                    fps_start_time = time.time()
                
                cv2.putText(
                    frame, f"FPS: {fps:.1f}", (10, 70),
                    cv2.FONT_HERSHEY_SIMPLEX, 0.7, (0, 255, 0), 2
                )
                
                # Draw adaptation status
                if self.stats["drift_events"] > 0:
                    cv2.putText(
                        frame, f"Adaptations: {self.stats['drift_events']}", (10, 110),
                        cv2.FONT_HERSHEY_SIMPLEX, 0.7, (0, 150, 255), 2
                    )
                
                # Draw UI elements for feedback
                if self.user_feedback_enabled:
                    # First, draw a dark background for buttons
                    overlay = frame.copy()
                    cv2.rectangle(
                        overlay, (0, frame.shape[0] - 200), (frame.shape[1], frame.shape[0]),
                        (60, 60, 60), -1
                    )
                    cv2.addWeighted(overlay, 0.7, frame, 0.3, 0, frame)
                    
                    # Draw buttons
                    for label, (x, y, w, h) in self.button_regions:
                        # Adjust y position to bottom of frame
                        adjusted_y = frame.shape[0] - 180 + y
                        
                        # Highlight selected label
                        if label == self.selected_label:
                            cv2.rectangle(frame, (x, adjusted_y), (x + w, adjusted_y + h), (0, 255, 0), -1)
                            text_color = (0, 0, 0)
                        else:
                            cv2.rectangle(frame, (x, adjusted_y), (x + w, adjusted_y + h), (200, 200, 200), 2)
                            text_color = (255, 255, 255)
                        
                        # Add label text
                        cv2.putText(
                            frame, label, (x + 5, adjusted_y + 20),
                            cv2.FONT_HERSHEY_SIMPLEX, 0.5, text_color, 1
                        )
                    
                    # Draw instructions
                    cv2.putText(
                        frame, "Click label to provide feedback", (10, frame.shape[0] - 30),
                        cv2.FONT_HERSHEY_SIMPLEX, 0.6, (255, 255, 255), 1
                    )
                
                # Show the frame
                cv2.imshow(window_name, frame)
                
                # Process UI interactions
                key = cv2.waitKey(1) & 0xFF
                
                # Handle key presses
                if key == ord('q'):
                    logger.info("User requested exit via keyboard")
                    self.stop_event.set()
                elif key == ord('s'):
                    # Save state manually
                    state_id = self.pipeline.save_state()
                    logger.info(f"Manually saved state: {state_id}")
                elif key == ord('r'):
                    # Reset metrics
                    self.inference_monitor.reset_metrics()
                    logger.info("Reset performance metrics")
                elif key == ord('f'):
                    # Toggle feedback mode
                    self.user_feedback_enabled = not self.user_feedback_enabled
                    logger.info(f"Feedback mode toggled to {self.user_feedback_enabled}")
                
                # Handle mouse clicks for feedback
                def mouse_callback(event, x, y, flags, param):
                    if event == cv2.EVENT_LBUTTONDOWN and self.user_feedback_enabled:
                        # Check if a button was clicked
                        for label, (bx, by, bw, bh) in self.button_regions:
                            # Adjust y position to bottom of frame
                            adjusted_by = frame.shape[0] - 180 + by
                            
                            if (bx <= x <= bx + bw) and (adjusted_by <= y <= adjusted_by + bh):
                                # Button clicked
                                self.selected_label = label
                                
                                # Add feedback to queue
                                if self.last_result is not None:
                                    logger.info(f"User provided feedback: {label}")
                                    self.feedback_queue.put((self.last_result, label))
                                break
                
                cv2.setMouseCallback(window_name, mouse_callback)
                
                # Brief pause to limit CPU usage
                time.sleep(0.01)
            
            logger.info("Display worker stopped")
        except Exception as e:
            logger.error(f"Fatal error in display worker: {e}")
        finally:
            cv2.destroyAllWindows()
    
    def start(self):
        """Start the adaptive camera application."""
        logger.info("Starting AdaptiveCameraApp")
        
        # Start the camera
        self.camera.start()
        
        # Start the adaptation session
        self.adaptation_tracker.start_session(
            metadata={
                "app": "AdaptiveCameraApp",
                "model": os.path.basename(self.config["model"]["path"]),
                "strategy": self.config["adaptive"]["strategy"]
            }
        )
        
        # Start worker threads
        self.workers = []
        
        # Frame capture worker
        capture_thread = threading.Thread(target=self.frame_capture_worker, daemon=True)
        capture_thread.start()
        self.workers.append(capture_thread)
        
        # Processing workers
        for _ in range(self.config["processing"]["max_workers"]):
            process_thread = threading.Thread(target=self.processing_worker, daemon=True)
            process_thread.start()
            self.workers.append(process_thread)
        
        # Feedback worker
        feedback_thread = threading.Thread(target=self.feedback_worker, daemon=True)
        feedback_thread.start()
        self.workers.append(feedback_thread)
        
        # Display worker (if display enabled)
        if self.config["user_interface"]["enable_display"]:
            display_thread = threading.Thread(target=self.display_worker, daemon=True)
            display_thread.start()
            self.workers.append(display_thread)
        
        logger.info("All worker threads started")
    
    def run(self):
        """Run the application until stopped."""
        self.start()
        
        try:
            # Wait for stop event
            while not self.stop_event.is_set():
                time.sleep(0.1)
        except KeyboardInterrupt:
            logger.info("Keyboard interrupt detected")
        finally:
            self.stop()
    
    def stop(self):
        """Stop the application and clean up resources."""
        if self.stop_event.is_set():
            return
        
        logger.info("Stopping AdaptiveCameraApp")
        self.stop_event.set()
        
        # Print final stats
        runtime = time.time() - self.stats["start_time"]
        fps = self.stats["frames_processed"] / runtime if runtime > 0 else 0
        logger.info(f"Runtime: {runtime:.1f}s, Frames: {self.stats['frames_processed']}, "
                    f"FPS: {fps:.1f}, Adaptations: {self.stats['drift_events']}")
        
        # Close the camera
        try:
            self.camera.close()
            logger.info("Camera closed")
        except Exception as e:
            logger.error(f"Error closing camera: {e}")
        
        # Wait for worker threads to finish processing current items
        for worker in self.workers:
            worker.join(timeout=2.0)
        
        # Save final state
        try:
            state_id = self.pipeline.save_state()
            logger.info(f"Final state saved: {state_id}")
        except Exception as e:
            logger.error(f"Error saving final state: {e}")
        
        # Close TinyLCM components
        try:
            self.inference_monitor.close()
            logger.info("Inference monitor closed")
        except Exception as e:
            logger.error(f"Error closing inference monitor: {e}")
        
        logger.info("AdaptiveCameraApp stopped cleanly")


def download_model_if_needed():
    """Download MobileNet model if it doesn't exist."""
    if os.path.exists(DEFAULT_MODEL_PATH) and os.path.exists(DEFAULT_LABELS_PATH):
        return
    
    import urllib.request
    import shutil
    
    logger.info("Downloading default model files...")
    
    # Create model directory
    os.makedirs(MODEL_DIR, exist_ok=True)
    
    # List of files to download
    files = [
        ("https://storage.googleapis.com/download.tensorflow.org/models/tflite/mobilenet_v1_1.0_224_quant_and_labels.zip", 
         os.path.join(MODEL_DIR, "mobilenet_v1.zip")),
    ]
    
    # Download and extract
    for url, dest in files:
        try:
            # Download
            logger.info(f"Downloading {url}...")
            urllib.request.urlretrieve(url, dest)
            
            # Extract
            logger.info(f"Extracting {dest}...")
            shutil.unpack_archive(dest, MODEL_DIR)
            
            # Remove zip file
            os.remove(dest)
        except Exception as e:
            logger.error(f"Error downloading model file: {e}")
            return
    
    # Rename/move files if needed
    if not os.path.exists(DEFAULT_MODEL_PATH):
        unpacked_models = [f for f in os.listdir(MODEL_DIR) if f.endswith('.tflite')]
        if unpacked_models:
            shutil.move(os.path.join(MODEL_DIR, unpacked_models[0]), DEFAULT_MODEL_PATH)
    
    if not os.path.exists(DEFAULT_LABELS_PATH):
        unpacked_labels = [f for f in os.listdir(MODEL_DIR) if f.endswith('labels.txt')]
        if unpacked_labels:
            shutil.move(os.path.join(MODEL_DIR, unpacked_labels[0]), DEFAULT_LABELS_PATH)
    
    logger.info("Model download completed")


def main():
    """Main entry point."""
    parser = argparse.ArgumentParser(description="TinyLCM Adaptive Camera Demo")
    parser.add_argument("--config", type=str, help="Path to configuration file")
    parser.add_argument("--download-model", action="store_true", help="Download model if not present")
    args = parser.parse_args()
    
    # Download model if requested
    if args.download_model:
        download_model_if_needed()
    
    # Create and run the application
    app = AdaptiveCameraApp(config_path=args.config)
    app.run()


if __name__ == "__main__":
    main()