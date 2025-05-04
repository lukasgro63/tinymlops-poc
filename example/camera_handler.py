#!/usr/bin/env python3
"""
Camera Handler for Raspberry Pi Camera using legacy picamera
-----------------------------------------------------------
Compatible with Raspberry Pi OS Buster.
"""

import logging
import time
from threading import Lock, Thread

import numpy as np
import picamera
import picamera.array

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
    
    def _camera_thread(self):
        """Background thread that continuously captures frames from the camera"""
        logger.info("Camera capture thread started")
        
        try:
            # Initialize the camera
            self.camera = picamera.PiCamera()
            self.camera.resolution = self.resolution
            self.camera.framerate = self.framerate
            
            # Allow camera to warm up
            time.sleep(2)
            
            # Calculate the delay between frames to maintain the framerate
            frame_delay = 1.0 / self.framerate
            
            # Main capture loop
            with picamera.array.PiRGBArray(self.camera, size=self.resolution) as stream:
                while self.running:
                    start_time = time.time()
                    
                    # Capture frame
                    self.camera.capture(stream, format='rgb', use_video_port=True)
                    frame = stream.array
                    
                    # Update the frame with lock protection
                    with self.frame_lock:
                        self.frame = frame.copy()
                    
                    # Clear the stream for the next frame
                    stream.truncate(0)
                    
                    # Calculate delay to maintain framerate
                    elapsed = time.time() - start_time
                    sleep_time = max(0, frame_delay - elapsed)
                    if sleep_time > 0:
                        time.sleep(sleep_time)
        
        except Exception as e:
            logger.error(f"Error in camera thread: {e}", exc_info=True)
        finally:
            # Clean up camera
            if self.camera:
                self.camera.close()
                self.camera = None
            logger.info("Camera thread stopped")
    
    async def start(self):
        """Start the camera capture thread"""
        if not self.running:
            self.running = True
            self.thread = Thread(target=self._camera_thread)
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
            self.camera.close()