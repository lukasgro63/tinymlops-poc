#!/usr/bin/env python3
# -*- coding: utf-8 -*-
"""
Adaptive Monitoring Example with PiCamera2

This example demonstrates the dual drift detection and adaptive learning system
of TinyLCM on a Raspberry Pi with camera input. It shows how to:

1. Use the label-based adaptive learning system with AdaptivePipeline
2. Use the autonomous drift detection system without requiring labels
3. Combine both systems for comprehensive model monitoring
4. Handle non-blocking I/O operations for resource efficiency

Requirements:
- Raspberry Pi (preferably Pi 4 or Pi Zero 2 W)
- Camera module (supports both v1 and v2 camera modules)
- TinyLCM library with tflite and picamera dependencies:
  pip install tinylcm[tflite,picamera]

You can run this example headless on a Raspberry Pi. It will save drift events
and adaptation logs to disk. If a display is connected, it will also show the
camera feed with predictions overlaid.
"""

import argparse
import os
import sys
import time
import threading
import queue
import random
import uuid
import json
import datetime
import traceback
import signal
import numpy as np
from pathlib import Path

# Import TinyLCM components
from tinylcm.core.pipeline import AdaptivePipeline
from tinylcm.core.feature_extractors.tflite import TFLiteFeatureExtractor
from tinylcm.core.classifiers.knn import LightweightKNN
from tinylcm.core.handlers.hybrid import HybridHandler
from tinylcm.core.inference_monitor.monitor import InferenceMonitor
from tinylcm.core.state_manager import StateManager
from tinylcm.core.adaptation_tracker import AdaptationTracker

# Import autonomous drift detectors
from tinylcm.core.drift_detection.confidence import EWMAConfidenceMonitor
from tinylcm.core.drift_detection.distribution import PredictionDistributionMonitor
from tinylcm.core.drift_detection.features import FeatureMonitor

# Import utils
from tinylcm.utils.logging import setup_logger

# Set up logging
logger = setup_logger(__name__)

# Global flag to control program execution
running = True

# Default paths
DEFAULT_MODEL_PATH = "models/model.tflite"
DEFAULT_LABELS_PATH = "models/labels.txt"
DEFAULT_OUTPUT_DIR = "output"

# Simulation parameters
SIMULATION_INTERVAL = 30  # Seconds between simulated changes

class AdaptiveMonitoringSystem:
    """Main system for adaptive monitoring with camera input.
    
    This class integrates the adaptive pipeline with autonomous drift detection,
    camera input, and visualization.
    """
    
    def __init__(self, config):
        """Initialize the system with the provided configuration.
        
        Args:
            config: Dictionary of configuration parameters
        """
        self.config = config
        
        # Create output directory
        self.output_dir = Path(config.get("output_dir", DEFAULT_OUTPUT_DIR))
        self.output_dir.mkdir(parents=True, exist_ok=True)
        
        # Load labels
        self.labels = self._load_labels(config.get("labels_path", DEFAULT_LABELS_PATH))
        logger.info(f"Loaded {len(self.labels)} labels")
        
        # Initialize components
        self._init_components()
        
        # Set up queues for threading
        self.inference_queue = queue.Queue(maxsize=10)
        self.feedback_queue = queue.Queue(maxsize=10)
        self.visualization_queue = queue.Queue(maxsize=5)
        
        # Set up results cache
        self.results_cache = {}
        self.results_lock = threading.Lock()
        
        # Drift and adaptation event counters
        self.drift_events = {
            "confidence": 0,
            "distribution": 0,
            "feature": 0,
            "accuracy": 0
        }
        
        # Thread flags and locks
        self.threads = []
        self.stop_event = threading.Event()
        
        logger.info("AdaptiveMonitoringSystem initialized")
    
    def _load_labels(self, labels_path):
        """Load class labels from a text file.
        
        Args:
            labels_path: Path to the labels file
            
        Returns:
            List of label strings
        """
        try:
            with open(labels_path, 'r') as f:
                return [line.strip() for line in f.readlines()]
        except Exception as e:
            logger.error(f"Error loading labels: {str(e)}")
            # Provide some default labels
            return ["class_0", "class_1", "class_2", "class_3", "class_4"]
    
    def _init_components(self):
        """Initialize all system components."""
        # Initialize feature extractor
        self.feature_extractor = TFLiteFeatureExtractor(
            model_path=self.config.get("model_path", DEFAULT_MODEL_PATH),
            feature_layer_index=self.config.get("feature_layer_index", -2),
            input_mean=127.5,
            input_std=127.5,
            input_size=(224, 224)
        )
        
        # Initialize classifier
        self.classifier = LightweightKNN(
            k=self.config.get("k_neighbors", 5),
            distance_metric=self.config.get("distance_metric", "euclidean"),
            max_samples=self.config.get("max_samples", 100),
            use_numpy=True
        )
        
        # Initialize handler
        self.handler = HybridHandler(
            classifier=self.classifier,
            max_samples=self.config.get("max_samples", 100),
            batch_size=self.config.get("batch_size", 10),
            baseline_accuracy=self.config.get("baseline_accuracy", 0.8),
            cusum_threshold=self.config.get("cusum_threshold", 5.0),
            cusum_delta=self.config.get("cusum_delta", 0.2),
            enable_condensing=self.config.get("enable_condensing", True),
            use_numpy=True
        )
        
        # Initialize state manager
        self.state_manager = StateManager(
            storage_dir=str(self.output_dir / "adaptive_states"),
            max_states=self.config.get("max_states", 10)
        )
        
        # Initialize adaptation tracker
        self.adaptation_tracker = AdaptationTracker(
            log_dir=str(self.output_dir / "adaptation_logs"),
            max_events_in_memory=100
        )
        
        # Initialize inference monitor
        self.inference_monitor = InferenceMonitor(
            storage_dir=str(self.output_dir / "inference_logs"),
            memory_window_size=1000
        )
        
        # Initialize adaptive pipeline
        self.pipeline = AdaptivePipeline(
            feature_extractor=self.feature_extractor,
            classifier=self.classifier,
            handler=self.handler,
            state_manager=self.state_manager,
            adaptation_tracker=self.adaptation_tracker
        )
        
        # Initialize autonomous drift detectors
        self.confidence_monitor = EWMAConfidenceMonitor(
            lambda_param=0.1,
            threshold_factor=3.0,
            drift_window=5,
            training_size=30
        )
        
        self.distribution_monitor = PredictionDistributionMonitor(
            window_size=50,
            threshold=0.25,
            min_samples=50
        )
        
        self.feature_monitor = FeatureMonitor(
            window_size=100,
            threshold=3.0,
            reference_size=50,
            max_features=50
        )
        
        # Register callbacks for autonomous detectors
        self.confidence_monitor.register_callback(
            lambda info: self._on_drift_detected("confidence", info)
        )
        
        self.distribution_monitor.register_callback(
            lambda info: self._on_drift_detected("distribution", info)
        )
        
        self.feature_monitor.register_callback(
            lambda info: self._on_drift_detected("feature", info)
        )
        
        # Start a new session for adaptation tracking
        self.adaptation_tracker.start_session({
            "model_path": self.config.get("model_path", DEFAULT_MODEL_PATH),
            "labels": self.labels,
            "config": self.config
        })
        
        # Register adaptation callback
        self.pipeline.register_adaptation_callback(self._on_adaptation_event)
        
        # Initialize camera if headless flag is not set
        if not self.config.get("headless", False):
            self._init_camera()
        
        logger.info("All components initialized")
    
    def _init_camera(self):
        """Initialize the camera module."""
        try:
            # Try to import picamera2
            from picamera2 import Picamera2
            from libcamera import Transform
            
            self.picam2 = Picamera2()
            
            # Configure camera
            preview_config = self.picam2.create_preview_configuration(
                main={"size": (640, 480)},
                transform=Transform(hflip=True, vflip=True)
            )
            self.picam2.configure(preview_config)
            
            # Start camera
            self.picam2.start()
            logger.info("PiCamera2 initialized")
            
            # Set camera initialized flag
            self.camera_initialized = True
            
        except ImportError:
            logger.warning("PiCamera2 not available. Using OpenCV if available.")
            try:
                import cv2
                self.cap = cv2.VideoCapture(0)
                if not self.cap.isOpened():
                    logger.error("Could not open camera")
                    self.camera_initialized = False
                else:
                    self.camera_initialized = True
                    logger.info("OpenCV camera initialized")
            except ImportError:
                logger.error("Neither PiCamera2 nor OpenCV available")
                self.camera_initialized = False
        
        except Exception as e:
            logger.error(f"Error initializing camera: {str(e)}")
            self.camera_initialized = False
    
    def start(self):
        """Start all system threads and begin processing."""
        logger.info("Starting AdaptiveMonitoringSystem")
        
        # Start inference worker thread
        inference_thread = threading.Thread(
            target=self._inference_worker,
            name="inference_thread",
            daemon=True
        )
        self.threads.append(inference_thread)
        inference_thread.start()
        
        # Start feedback worker thread
        feedback_thread = threading.Thread(
            target=self._feedback_worker,
            name="feedback_thread",
            daemon=True
        )
        self.threads.append(feedback_thread)
        feedback_thread.start()
        
        # Start visualization thread if not in headless mode
        if not self.config.get("headless", False) and self.camera_initialized:
            viz_thread = threading.Thread(
                target=self._visualization_worker,
                name="visualization_thread",
                daemon=True
            )
            self.threads.append(viz_thread)
            viz_thread.start()
        
        # Start camera capture thread
        camera_thread = threading.Thread(
            target=self._camera_worker,
            name="camera_thread",
            daemon=True
        )
        self.threads.append(camera_thread)
        camera_thread.start()
        
        # Start simulation thread if enabled
        if self.config.get("simulate_drift", False):
            sim_thread = threading.Thread(
                target=self._simulation_worker,
                name="simulation_thread",
                daemon=True
            )
            self.threads.append(sim_thread)
            sim_thread.start()
        
        # Register signal handlers
        signal.signal(signal.SIGINT, self._handle_signal)
        signal.signal(signal.SIGTERM, self._handle_signal)
        
        logger.info("All system threads started")
    
    def _handle_signal(self, sig, frame):
        """Handle termination signals."""
        logger.info(f"Received signal {sig}, shutting down...")
        self.stop()
    
    def stop(self):
        """Stop all system threads and clean up resources."""
        global running
        running = False
        
        # Set stop event for threads
        self.stop_event.set()
        
        # Wait for threads to finish
        for thread in self.threads:
            if thread.is_alive():
                thread.join(timeout=2.0)
        
        # Close components
        self.inference_monitor.close()
        self.state_manager.join()
        self.adaptation_tracker.join()
        
        # Save final state
        try:
            self.pipeline.save_state(
                metadata={"reason": "system_shutdown", "drift_events": self.drift_events}
            )
        except Exception as e:
            logger.error(f"Error saving final state: {str(e)}")
        
        # Close camera
        if hasattr(self, 'picam2'):
            self.picam2.close()
        elif hasattr(self, 'cap'):
            self.cap.release()
        
        # Close display if using OpenCV
        try:
            import cv2
            cv2.destroyAllWindows()
        except:
            pass
        
        logger.info("System shutdown complete")
    
    def _camera_worker(self):
        """Thread to capture camera frames and queue them for processing."""
        logger.info("Camera worker started")
        
        frame_counter = 0
        
        while not self.stop_event.is_set() and running:
            try:
                # Capture frame
                if hasattr(self, 'picam2') and self.camera_initialized:
                    # PiCamera2 capture
                    frame = self.picam2.capture_array()
                elif hasattr(self, 'cap') and self.camera_initialized:
                    # OpenCV capture
                    ret, frame = self.cap.read()
                    if not ret:
                        logger.warning("Failed to capture frame")
                        time.sleep(0.1)
                        continue
                else:
                    # Simulated frame for headless mode
                    logger.debug("Using simulated input data")
                    # Create a random colored frame
                    frame = np.random.randint(0, 255, (224, 224, 3), dtype=np.uint8)
                
                # Generate a unique ID for this frame
                frame_id = f"frame_{frame_counter}_{int(time.time())}"
                frame_counter += 1
                
                # Preprocess frame for the model
                try:
                    # Put frame in the inference queue
                    if not self.inference_queue.full():
                        self.inference_queue.put((frame, frame_id), block=False)
                    else:
                        # If queue is full, skip this frame
                        logger.debug("Inference queue full, skipping frame")
                except queue.Full:
                    pass
                
                # Visualize the frame (if not headless)
                if not self.config.get("headless", False) and self.camera_initialized:
                    try:
                        if not self.visualization_queue.full():
                            self.visualization_queue.put(frame, block=False)
                    except queue.Full:
                        pass
                
                # Simulate feedback with some probability
                if random.random() < self.config.get("feedback_probability", 0.1):
                    # Wait for the inference to complete
                    time.sleep(0.1)
                    
                    with self.results_lock:
                        if frame_id in self.results_cache:
                            result = self.results_cache[frame_id]
                            prediction = result["prediction"]
                            
                            # Randomly modify the ground truth label for simulation
                            if self.config.get("simulate_drift", False) and random.random() < 0.3:
                                # Simulate wrong prediction during drift
                                prediction_idx = self.labels.index(prediction)
                                possible_labels = [i for i in range(len(self.labels)) if i != prediction_idx]
                                if possible_labels:
                                    true_label = self.labels[random.choice(possible_labels)]
                                else:
                                    true_label = prediction
                            else:
                                # Use the prediction as the true label (simulating correct predictions)
                                true_label = prediction
                            
                            # Add to feedback queue
                            try:
                                if not self.feedback_queue.full():
                                    self.feedback_queue.put((frame_id, true_label), block=False)
                            except queue.Full:
                                pass
                
                # Sleep to control frame rate
                time.sleep(1.0 / self.config.get("fps", 10))
                
            except Exception as e:
                logger.error(f"Error in camera thread: {str(e)}")
                traceback.print_exc()
                time.sleep(0.5)
        
        logger.info("Camera worker stopped")
    
    def _inference_worker(self):
        """Thread to process frames from the inference queue."""
        logger.info("Inference worker started")
        
        while not self.stop_event.is_set() and running:
            try:
                # Get frame from queue with timeout
                try:
                    frame, frame_id = self.inference_queue.get(timeout=0.5)
                except queue.Empty:
                    continue
                
                # Start timing for latency measurement
                start_time = time.time()
                
                # Process frame with adaptive pipeline
                result = self.pipeline.process(frame)
                
                # Calculate latency
                latency_ms = (time.time() - start_time) * 1000
                
                # Get prediction, confidence, features
                prediction = result["prediction"]
                confidence = result["confidence"]
                features = result.get("features")
                
                # Convert prediction index to label if needed
                if isinstance(prediction, (int, np.integer)):
                    prediction_label = self.labels[prediction] if prediction < len(self.labels) else f"unknown_{prediction}"
                else:
                    prediction_label = str(prediction)
                
                # Store result in cache for feedback
                with self.results_lock:
                    self.results_cache[frame_id] = {
                        "prediction": prediction_label,
                        "confidence": confidence,
                        "features": features,
                        "latency_ms": latency_ms,
                        "timestamp": time.time()
                    }
                
                # Clean up old cache entries (keep only last 100)
                cache_size = 100
                with self.results_lock:
                    if len(self.results_cache) > cache_size:
                        oldest_keys = sorted(self.results_cache.keys(), 
                                           key=lambda k: self.results_cache[k]["timestamp"])[:len(self.results_cache) - cache_size]
                        for key in oldest_keys:
                            del self.results_cache[key]
                
                # Track inference with monitor
                self.inference_monitor.track_inference(
                    input_id=frame_id,
                    prediction=prediction_label,
                    confidence=confidence,
                    latency_ms=latency_ms,
                    features=features
                )
                
                # Update autonomous drift detectors
                record = {
                    "prediction": prediction_label,
                    "confidence": confidence,
                    "features": features
                }
                
                # Check confidence monitor
                _, drift_info = self.confidence_monitor.update(record)
                
                # Check prediction distribution monitor
                _, drift_info = self.distribution_monitor.update(record)
                
                # Check feature monitor if features are available
                if features is not None:
                    _, drift_info = self.feature_monitor.update(record)
                
                # Log inference completion
                logger.debug(f"Processed frame {frame_id}: {prediction_label} ({confidence:.2f}) in {latency_ms:.1f}ms")
                
                # Mark task as done
                self.inference_queue.task_done()
                
            except Exception as e:
                logger.error(f"Error in inference thread: {str(e)}")
                traceback.print_exc()
                time.sleep(0.5)
        
        logger.info("Inference worker stopped")
    
    def _feedback_worker(self):
        """Thread to process feedback from the feedback queue."""
        logger.info("Feedback worker started")
        
        while not self.stop_event.is_set() and running:
            try:
                # Get feedback from queue with timeout
                try:
                    frame_id, true_label = self.feedback_queue.get(timeout=0.5)
                except queue.Empty:
                    continue
                
                # Get the corresponding result
                result = None
                with self.results_lock:
                    if frame_id in self.results_cache:
                        result = self.results_cache[frame_id]
                
                if result is None:
                    logger.warning(f"No cached result for frame {frame_id}")
                    self.feedback_queue.task_done()
                    continue
                
                # Get features from cached result
                features = result.get("features")
                if features is None:
                    logger.warning(f"No features available for frame {frame_id}")
                    self.feedback_queue.task_done()
                    continue
                
                # Process feedback
                start_time = time.time()
                
                # Process with the pipeline to provide feedback
                feedback_result = self.pipeline.process(
                    input_data=features,
                    label=true_label,
                    extract_features=False
                )
                
                # Log adaptation if it occurred
                if feedback_result.get("adaptation_occurred", False):
                    logger.info(
                        f"Adaptation occurred: {feedback_result.get('adaptation_event_type')} "
                        f"(drift: {feedback_result.get('drift_detected', False)})"
                    )
                    
                    # Count accuracy-based drift detection
                    if feedback_result.get("drift_detected", False):
                        self.drift_events["accuracy"] += 1
                
                # Log feedback processing
                latency_ms = (time.time() - start_time) * 1000
                logger.debug(f"Processed feedback for {frame_id}: {true_label} in {latency_ms:.1f}ms")
                
                # Mark task as done
                self.feedback_queue.task_done()
                
            except Exception as e:
                logger.error(f"Error in feedback thread: {str(e)}")
                traceback.print_exc()
                time.sleep(0.5)
        
        logger.info("Feedback worker stopped")
    
    def _visualization_worker(self):
        """Thread to visualize frames with predictions."""
        logger.info("Visualization worker started")
        
        try:
            import cv2
            cv2.namedWindow("Adaptive Monitoring", cv2.WINDOW_NORMAL)
            
            while not self.stop_event.is_set() and running:
                try:
                    # Get frame from queue with timeout
                    try:
                        frame = self.visualization_queue.get(timeout=0.5)
                    except queue.Empty:
                        continue
                    
                    # Get latest metrics
                    metrics = self.inference_monitor.get_current_metrics()
                    
                    # Get drift status
                    confidence_drift, _ = self.confidence_monitor.check_for_drift()
                    distribution_drift, _ = self.distribution_monitor.check_for_drift()
                    feature_drift, _ = self.feature_monitor.check_for_drift()
                    
                    # Draw frame information
                    predictions = metrics.get("prediction_distribution", {})
                    
                    # Sort predictions by frequency
                    sorted_preds = sorted(predictions.items(), key=lambda x: x[1], reverse=True)
                    
                    # Draw prediction distribution
                    y_pos = 30
                    cv2.putText(frame, "Predictions:", (10, y_pos), 
                                cv2.FONT_HERSHEY_SIMPLEX, 0.5, (0, 255, 0), 1)
                    y_pos += 20
                    
                    for i, (pred, count) in enumerate(sorted_preds[:5]):
                        cv2.putText(frame, f"{pred}: {count}", (10, y_pos + i*20), 
                                    cv2.FONT_HERSHEY_SIMPLEX, 0.5, (0, 255, 0), 1)
                    
                    # Draw drift status
                    y_pos += 120
                    cv2.putText(frame, "Drift Status:", (10, y_pos), 
                                cv2.FONT_HERSHEY_SIMPLEX, 0.5, (0, 255, 0), 1)
                    y_pos += 20
                    
                    # Color map for drift status
                    green = (0, 255, 0)
                    red = (0, 0, 255)
                    
                    cv2.putText(frame, f"Confidence: {'DRIFT' if confidence_drift else 'OK'}", 
                                (10, y_pos), cv2.FONT_HERSHEY_SIMPLEX, 0.5, 
                                red if confidence_drift else green, 1)
                    y_pos += 20
                    
                    cv2.putText(frame, f"Distribution: {'DRIFT' if distribution_drift else 'OK'}", 
                                (10, y_pos), cv2.FONT_HERSHEY_SIMPLEX, 0.5, 
                                red if distribution_drift else green, 1)
                    y_pos += 20
                    
                    cv2.putText(frame, f"Features: {'DRIFT' if feature_drift else 'OK'}", 
                                (10, y_pos), cv2.FONT_HERSHEY_SIMPLEX, 0.5, 
                                red if feature_drift else green, 1)
                    
                    # Draw inference stats
                    y_pos += 40
                    cv2.putText(frame, "Inference Stats:", (10, y_pos), 
                                cv2.FONT_HERSHEY_SIMPLEX, 0.5, (0, 255, 0), 1)
                    y_pos += 20
                    
                    if "latency" in metrics and "mean" in metrics["latency"]:
                        cv2.putText(frame, f"Latency: {metrics['latency']['mean']:.1f}ms", 
                                    (10, y_pos), cv2.FONT_HERSHEY_SIMPLEX, 0.5, (0, 255, 0), 1)
                        y_pos += 20
                    
                    if "confidence" in metrics and "mean" in metrics["confidence"]:
                        cv2.putText(frame, f"Confidence: {metrics['confidence']['mean']:.2f}", 
                                    (10, y_pos), cv2.FONT_HERSHEY_SIMPLEX, 0.5, (0, 255, 0), 1)
                    
                    # Display the frame
                    cv2.imshow("Adaptive Monitoring", frame)
                    
                    # Check for key press (q to quit)
                    key = cv2.waitKey(1) & 0xFF
                    if key == ord('q'):
                        self.stop_event.set()
                    
                    # Mark task as done
                    self.visualization_queue.task_done()
                    
                except Exception as e:
                    logger.error(f"Error in visualization thread: {str(e)}")
                    traceback.print_exc()
                    time.sleep(0.5)
            
            # Clean up
            cv2.destroyAllWindows()
            
        except ImportError:
            logger.error("OpenCV not available for visualization")
        
        logger.info("Visualization worker stopped")
    
    def _simulation_worker(self):
        """Thread to simulate drift by injecting distribution changes."""
        logger.info("Simulation worker started")
        
        current_phase = 0
        phase_start_time = time.time()
        
        while not self.stop_event.is_set() and running:
            try:
                current_time = time.time()
                time_in_phase = current_time - phase_start_time
                
                # Change phase every SIMULATION_INTERVAL seconds
                if time_in_phase > SIMULATION_INTERVAL:
                    current_phase = (current_phase + 1) % 3
                    phase_start_time = current_time
                    
                    logger.info(f"Simulation phase changed to {current_phase}")
                    
                    # Simulate a change in the distribution
                    if current_phase == 1:
                        # Phase 1: Simulate confidence drop
                        logger.info("Simulating confidence drop")
                        
                    elif current_phase == 2:
                        # Phase 2: Simulate distribution shift
                        logger.info("Simulating distribution shift")
                
                # Sleep to avoid busy wait
                time.sleep(1.0)
                
            except Exception as e:
                logger.error(f"Error in simulation thread: {str(e)}")
                traceback.print_exc()
                time.sleep(0.5)
        
        logger.info("Simulation worker stopped")
    
    def _on_drift_detected(self, detector_type, info):
        """Handle drift detected by autonomous detectors.
        
        Args:
            detector_type: Type of detector that found drift
            info: Information about the detected drift
        """
        self.drift_events[detector_type] += 1
        
        logger.warning(f"Drift detected by {detector_type} detector")
        
        # Log drift event
        timestamp = datetime.datetime.now().strftime("%Y-%m-%d_%H-%M-%S")
        drift_file = self.output_dir / f"drift_{detector_type}_{timestamp}.json"
        
        try:
            with open(drift_file, 'w') as f:
                json.dump(info, f, indent=2, default=str)
            logger.info(f"Drift event logged to {drift_file}")
        except Exception as e:
            logger.error(f"Error logging drift event: {str(e)}")
        
        # Save current state after drift
        try:
            state_id = self.pipeline.save_state(
                metadata={
                    "reason": f"drift_detected_{detector_type}",
                    "detector": detector_type,
                    "drift_events": self.drift_events
                }
            )
            logger.info(f"State saved after drift detection: {state_id}")
        except Exception as e:
            logger.error(f"Error saving state after drift: {str(e)}")
    
    def _on_adaptation_event(self, event, result):
        """Handle adaptation events from the pipeline.
        
        Args:
            event: AdaptationEvent object
            result: Result dictionary from the pipeline
        """
        logger.info(f"Adaptation event: {event.event_type} (drift: {event.drift_detected})")
        
        # Save state after adaptation
        try:
            state_id = self.pipeline.save_state(
                metadata={
                    "reason": f"adaptation_event_{event.event_type}",
                    "drift_detected": event.drift_detected,
                    "drift_events": self.drift_events
                }
            )
            logger.info(f"State saved after adaptation: {state_id}")
        except Exception as e:
            logger.error(f"Error saving state after adaptation: {str(e)}")

def parse_args():
    """Parse command line arguments."""
    parser = argparse.ArgumentParser(description="Adaptive Monitoring with PiCamera2")
    
    # Input settings
    parser.add_argument("--model-path", type=str, default=DEFAULT_MODEL_PATH,
                        help=f"Path to the TFLite model (default: {DEFAULT_MODEL_PATH})")
    parser.add_argument("--labels-path", type=str, default=DEFAULT_LABELS_PATH,
                        help=f"Path to the labels file (default: {DEFAULT_LABELS_PATH})")
    
    # Output settings
    parser.add_argument("--output-dir", type=str, default=DEFAULT_OUTPUT_DIR,
                        help=f"Output directory for logs and states (default: {DEFAULT_OUTPUT_DIR})")
    
    # System settings
    parser.add_argument("--headless", action="store_true",
                        help="Run in headless mode without visualization")
    parser.add_argument("--fps", type=float, default=10.0,
                        help="Target frames per second (default: 10.0)")
    parser.add_argument("--simulate-drift", action="store_true",
                        help="Simulate drift by injecting distribution changes")
    parser.add_argument("--feedback-probability", type=float, default=0.1,
                        help="Probability of generating feedback for a frame (default: 0.1)")
    
    # Algorithm settings
    parser.add_argument("--k-neighbors", type=int, default=5,
                        help="Number of neighbors for KNN classifier (default: 5)")
    parser.add_argument("--cusum-threshold", type=float, default=5.0,
                        help="Threshold for CUSUM drift detection (default: 5.0)")
    
    return parser.parse_args()

def main():
    """Main function."""
    # Parse command line arguments
    args = parse_args()
    
    # Convert args to config dictionary
    config = vars(args)
    
    try:
        # Create and start the adaptive monitoring system
        system = AdaptiveMonitoringSystem(config)
        system.start()
        
        # Keep the main thread running
        while running:
            time.sleep(0.1)
    
    except KeyboardInterrupt:
        logger.info("Keyboard interrupt received")
    except Exception as e:
        logger.error(f"Error in main thread: {str(e)}")
        traceback.print_exc()
    finally:
        if 'system' in locals():
            system.stop()

if __name__ == "__main__":
    main()