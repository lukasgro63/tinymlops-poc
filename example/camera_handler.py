#!/usr/bin/env python3
"""
Camera Handler for Raspberry Pi Camera using picamera2
-----------------------------------------------------------
Compatible with Raspberry Pi OS Bookworm and the new libcamera stack.
Provides async-compatible capture methods for use with the stone detector app.
"""

import logging
import time
from threading import Lock, Thread

import numpy as np

# Import picamera2 with fallback mechanism for development
try:
    from picamera2 import Picamera2
    PICAMERA_AVAILABLE = True
except ImportError:
    import cv2
    PICAMERA_AVAILABLE = False

logger = logging.getLogger("stone_detector.camera")

class CameraHandler:
    def __init__(self, resolution=(640, 480), framerate=10):
        """
        Initialize the camera handler.
        
        Args:
            resolution (tuple): The resolution to capture images (width, height)
            framerate (int): The target framerate
        """
        self.resolution = resolution
        self.framerate = framerate
        self.camera = None
        self.frame = None
        self.frame_lock = Lock()
        self.running = False
        self.thread = None
        
        logger.info(f"Camera handler initialized with resolution {resolution} and framerate {framerate}")
    
    def _camera_thread_picamera2(self):
        """Background thread that continuously captures frames using picamera2"""
        logger.info("Picamera2 capture thread started")
        
        try:
            # Initialize the camera
            self.camera = Picamera2()
            
            # Configure the camera
            config = self.camera.create_still_configuration(
                main={"size": self.resolution, "format": "RGB888"}
            )
            self.camera.configure(config)
            
            # Calculate frame delay to approximate the requested framerate
            frame_delay = 1.0 / self.framerate
            
            # Start the camera
            self.camera.start()
            
            # Allow camera to warm up
            time.sleep(1)
            
            # Main capture loop
            while self.running:
                start_time = time.time()
                
                # Capture frame
                frame = self.camera.capture_array()
                
                # Update the frame with lock protection
                with self.frame_lock:
                    self.frame = frame.copy()
                
                # Calculate delay to maintain framerate
                elapsed = time.time() - start_time
                sleep_time = max(0, frame_delay - elapsed)
                if sleep_time > 0:
                    time.sleep(sleep_time)
        
        except Exception as e:
            logger.error(f"Error in picamera2 thread: {e}", exc_info=True)
        finally:
            # Clean up camera
            if self.camera:
                self.camera.stop()
                self.camera.close()
                self.camera = None
            logger.info("Picamera2 thread stopped")
    
    def _camera_thread_fallback(self):
        """Fallback thread using OpenCV for development environments"""
        logger.info("OpenCV fallback camera thread started")
        
        try:
            # Initialize OpenCV camera
            self.camera = cv2.VideoCapture(0)
            self.camera.set(cv2.CAP_PROP_FRAME_WIDTH, self.resolution[0])
            self.camera.set(cv2.CAP_PROP_FRAME_HEIGHT, self.resolution[1])
            self.camera.set(cv2.CAP_PROP_FPS, self.framerate)
            
            if not self.camera.isOpened():
                raise RuntimeError("Failed to open OpenCV camera")
            
            # Calculate frame delay
            frame_delay = 1.0 / self.framerate
            
            # Main capture loop
            while self.running:
                start_time = time.time()
                
                # Capture frame
                ret, frame = self.camera.read()
                
                if ret:
                    # Convert BGR to RGB to match picamera2 output
                    frame_rgb = cv2.cvtColor(frame, cv2.COLOR_BGR2RGB)
                    
                    # Update the frame with lock protection
                    with self.frame_lock:
                        self.frame = frame_rgb.copy()
                
                # Calculate delay to maintain framerate
                elapsed = time.time() - start_time
                sleep_time = max(0, frame_delay - elapsed)
                if sleep_time > 0:
                    time.sleep(sleep_time)
        
        except Exception as e:
            logger.error(f"Error in OpenCV fallback thread: {e}", exc_info=True)
        finally:
            # Clean up camera
            if self.camera:
                self.camera.release()
                self.camera = None
            logger.info("OpenCV fallback thread stopped")
    
    async def start(self):
        """Start the camera capture thread"""
        if not self.running:
            self.running = True
            
            # Use appropriate camera implementation
            if PICAMERA_AVAILABLE:
                self.thread = Thread(target=self._camera_thread_picamera2)
            else:
                logger.warning("Using OpenCV fallback camera instead of picamera2")
                self.thread = Thread(target=self._camera_thread_fallback)
                
            self.thread.daemon = True
            self.thread.start()
            logger.info("Camera started")
    
    async def stop(self):
        """Stop the camera capture thread"""
        self.running = False
        if self.thread:
            self.thread.join(timeout=5)
            self.thread = None
        logger.info("Camera stopped")
    
    def capture_frame(self):
        """
        Capture the latest frame from the camera.
        
        Returns:
            numpy.ndarray: The captured frame or None if no frame is available
        """
        with self.frame_lock:
            if self.frame is not None:
                return self.frame.copy()
        return None
        
    async def capture_frame_async(self):
        """
        Capture the latest frame from the camera asynchronously.
        This is a thin wrapper around capture_frame to provide async compatibility.
        
        Returns:
            numpy.ndarray: The captured frame or None if no frame is available
        """
        return self.capture_frame()
    
    def __del__(self):
        """Ensure camera resources are released"""
        self.running = False
        if hasattr(self, 'thread') and self.thread:
            self.thread.join(timeout=1)
        if hasattr(self, 'camera') and self.camera:
            if PICAMERA_AVAILABLE:
                self.camera.close()
            else:
                self.camera.release()