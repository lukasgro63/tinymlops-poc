#!/usr/bin/env python3
"""
Camera Handler
-------------
Provides functionality to capture frames using picamera2 with OpenCV fallback.
Captures frames continuously in a background thread.
"""

import logging
import threading
import time
from typing import Optional, Tuple, Union

import cv2
import numpy as np

logger = logging.getLogger(__name__)

# Try to import picamera2
try:
    from picamera2 import Picamera2
    PICAMERA2_AVAILABLE = True
except ImportError:
    logger.warning("picamera2 not available, falling back to OpenCV for camera access")
    PICAMERA2_AVAILABLE = False


class CameraHandler:
    """Handler for camera operations with thread-safe frame capture."""
    
    def __init__(
        self,
        resolution: Tuple[int, int] = (640, 480),
        framerate: int = 30,
        camera_index: int = 0,
        rotation: int = 0,
        force_cv2: bool = False,
        auto_start: bool = True
    ):
        """Initialize the camera handler.
        
        Args:
            resolution: Camera resolution (width, height)
            framerate: Camera framerate
            camera_index: Camera index for OpenCV (ignored for picamera2)
            rotation: Rotation angle (0, 90, 180, 270)
            force_cv2: Force using OpenCV instead of picamera2 even if available
            auto_start: Automatically start the camera thread upon initialization
        """
        self.resolution = resolution
        self.framerate = framerate
        self.camera_index = camera_index
        self.rotation = rotation
        self.force_cv2 = force_cv2
        
        # Thread control variables
        self._running = False
        self._thread = None
        self._lock = threading.Lock()
        self._current_frame = None
        
        # Camera initialization
        self._camera = None
        self._use_picamera = PICAMERA2_AVAILABLE and not force_cv2
        
        logger.info(f"Initializing camera with resolution {resolution}, "
                   f"framerate {framerate}, using picamera2: {self._use_picamera}")
        
        if auto_start:
            self.start()
    
    def _init_picamera(self) -> bool:
        """Initialize the picamera2 camera."""
        try:
            # Check if camera is available first
            self._camera = Picamera2()
            
            # Try to detect available cameras
            cameras = Picamera2.global_camera_info()
            if not cameras:
                logger.warning("No cameras detected by picamera2")
                self._use_picamera = False
                return False
            
            width, height = self.resolution
            
            # Configure the camera
            config = self._camera.create_preview_configuration(
                main={"size": (width, height)},
                controls={"FrameRate": self.framerate}
            )
            self._camera.configure(config)
            
            # Apply rotation if needed (picamera2 does this differently)
            # We'll handle rotation in the frame processing
            
            return True
        except Exception as e:
            logger.error(f"Failed to initialize picamera2: {e}")
            self._use_picamera = False
            return False
    
    def _init_cv2_camera(self) -> bool:
        """Initialize the OpenCV camera."""
        try:
            self._camera = cv2.VideoCapture(self.camera_index)
            width, height = self.resolution
            
            # Configure camera properties
            self._camera.set(cv2.CAP_PROP_FRAME_WIDTH, width)
            self._camera.set(cv2.CAP_PROP_FRAME_HEIGHT, height)
            self._camera.set(cv2.CAP_PROP_FPS, self.framerate)
            
            if not self._camera.isOpened():
                logger.error("Failed to open camera with OpenCV")
                return False
            
            return True
        except Exception as e:
            logger.error(f"Failed to initialize OpenCV camera: {e}")
            return False
    
    def _capture_frames(self):
        """Continuously capture frames in a loop (runs in a separate thread)."""
        logger.info("Starting frame capture thread")
        
        success = False
        
        # Try picamera2 first if available and not forced to use CV2
        if self._use_picamera:
            success = self._init_picamera()
            if success:
                self._camera.start()
                logger.info("Successfully initialized picamera2")
            else:
                logger.warning("picamera2 initialization failed, trying OpenCV fallback")
                self._use_picamera = False
        
        # Fallback to OpenCV if picamera2 failed or not available
        if not success:
            success = self._init_cv2_camera()
            if success:
                logger.info("Successfully initialized OpenCV camera")
        
        if not success:
            logger.error("Camera initialization failed for both picamera2 and OpenCV, stopping capture thread")
            return
        
        logger.info("Camera initialized successfully, beginning capture loop")
        
        # Main capture loop
        while self._running:
            try:
                if self._use_picamera:
                    # Capture frame from picamera2
                    frame = self._camera.capture_array()
                else:
                    # Capture frame from OpenCV
                    ret, frame = self._camera.read()
                    if not ret:
                        logger.warning("Failed to capture frame from OpenCV camera")
                        time.sleep(0.1)
                        continue
                
                # Apply rotation if needed
                if self.rotation == 90:
                    frame = cv2.rotate(frame, cv2.ROTATE_90_CLOCKWISE)
                elif self.rotation == 180:
                    frame = cv2.rotate(frame, cv2.ROTATE_180)
                elif self.rotation == 270:
                    frame = cv2.rotate(frame, cv2.ROTATE_90_COUNTERCLOCKWISE)
                
                # Update the current frame (with thread lock)
                with self._lock:
                    self._current_frame = frame
                
            except Exception as e:
                logger.error(f"Error capturing frame: {e}")
                time.sleep(0.1)
        
        # Clean up
        logger.info("Stopping camera and cleaning up")
        if self._use_picamera:
            try:
                self._camera.stop()
                self._camera.close()
            except Exception as e:
                logger.error(f"Error closing picamera2: {e}")
        else:
            try:
                self._camera.release()
            except Exception as e:
                logger.error(f"Error releasing OpenCV camera: {e}")
    
    def start(self):
        """Start the camera capture thread."""
        if self._running:
            logger.warning("Camera is already running")
            return
        
        self._running = True
        self._thread = threading.Thread(target=self._capture_frames, daemon=True)
        self._thread.start()
        
        # Wait briefly to ensure camera is starting
        time.sleep(0.5)
    
    def stop(self):
        """Stop the camera capture thread."""
        logger.info("Stopping camera capture thread")
        if not self._running:
            logger.warning("Camera is not running")
            return
        
        self._running = False
        if self._thread and self._thread.is_alive():
            self._thread.join(timeout=3.0)
            if self._thread.is_alive():
                logger.warning("Camera thread did not terminate properly")
    
    def get_frame(self) -> Optional[np.ndarray]:
        """Get the most recent captured frame.
        
        Returns:
            The most recent frame as a numpy array, or None if no frame available
        """
        with self._lock:
            if self._current_frame is None:
                return None
            # Return a copy to avoid thread safety issues
            return self._current_frame.copy()
    
    def __enter__(self):
        """Context manager entry."""
        self.start()
        return self
    
    def __exit__(self, exc_type, exc_val, exc_tb):
        """Context manager exit."""
        self.stop()


# Test function
def main():
    """Test the CameraHandler functionality."""
    logging.basicConfig(level=logging.INFO)
    
    # Create and start the camera
    camera = CameraHandler(resolution=(640, 480), framerate=30)
    
    try:
        # Display frames for 10 seconds
        start_time = time.time()
        while time.time() - start_time < 10:
            frame = camera.get_frame()
            if frame is not None:
                # Display the frame
                cv2.imshow('Camera Test', frame)
                if cv2.waitKey(1) & 0xFF == ord('q'):
                    break
            time.sleep(0.03)  # ~30 fps
    finally:
        # Clean up
        camera.stop()
        cv2.destroyAllWindows()


if __name__ == "__main__":
    main()