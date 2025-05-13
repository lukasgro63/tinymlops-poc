#!/usr/bin/env python3
"""
SyncClient
---------
TinySphere synchronization client for the TinyLCM application.
Extends the base SyncClient with application-specific functionality.
"""

import json
import logging
import os
import tempfile  # Add this import
import time
from pathlib import Path
from typing import Any, Dict, List, Optional, Tuple

import shutil
from datetime import datetime

from tinylcm.client.sync_client import SyncClient as TinyLCMSyncClient
from tinylcm.client.sync_interface import SyncInterface, SyncPackage
from tinylcm.core.data_structures import FeatureSample
from tinylcm.utils.errors import TinyLCMConnectionError, SyncError

logger = logging.getLogger(__name__)

# Global configuration variable that will be set from main_scenario1.py
config = None


class ExtendedSyncClient:
    """Extended SyncClient for the TinyLCM application."""

    def __init__(
        self,
        server_url: str,
        device_id: str,
        api_key: str = "tinylcm-demo-key",
        sync_dir: str = "./sync_data",
        sync_interval_seconds: int = 30,
        max_retries: int = 3,
        auto_register: bool = True
    ):
        """Initialize the extended SyncClient.

        Args:
            server_url: URL of the TinySphere server
            device_id: Unique device identifier
            api_key: API key for authentication
            sync_dir: Directory for storing sync data
            sync_interval_seconds: Interval between sync attempts
            max_retries: Maximum number of connection retry attempts
            auto_register: Whether to automatically register the device
        """
        # Create sync directory if it doesn't exist
        os.makedirs(sync_dir, exist_ok=True)
        
        # Create sync interface
        self.sync_interface = SyncInterface(sync_dir=sync_dir)
        
        # Custom headers for API key authentication
        self.headers = {
            "X-API-Key": api_key,
            "Content-Type": "application/json"
        }
        
        # Initialize the core TinyLCM SyncClient
        self.client = TinyLCMSyncClient(
            server_url=server_url,
            api_key=api_key,
            device_id=device_id,
            sync_interface=self.sync_interface,
            max_retries=max_retries,
            auto_register=auto_register
        )
        
        self.device_id = device_id
        self.sync_interval_seconds = sync_interval_seconds
        self.sync_dir = sync_dir
        self.last_sync_time = 0
        self._running = False

        # Image transfer attributes (disabled by default)
        self.enable_prediction_images = False
        self.pending_prediction_images = []

        logger.info(f"Extended SyncClient initialized for device {device_id}")
    
    def check_connection(self) -> bool:
        """Check connection to the TinySphere server.

        Returns:
            True if connected, False otherwise
        """
        try:
            # Don't use check_server_status which may not be available
            # Use the connection_manager directly which is guaranteed to be there
            connected = self.client.connection_manager.connect()
            logger.info(f"Server connection status: {'Connected' if connected else 'Disconnected'}")
            return connected
        except TinyLCMConnectionError as e:
            logger.warning(f"Server connection check failed: {e}")
            return False
    
    def register_device(self) -> bool:
        """Register the device with TinySphere.

        Returns:
            True if registration was successful, False otherwise
        """
        try:
            success = self.client.register_device()
            return success
        except TinyLCMConnectionError as e:
            logger.error(f"Device registration failed: {e}")
            return False

    def send_model(self, model_path: Optional[str] = None, labels_path: Optional[str] = None) -> bool:
        """Send the device's model to the TinySphere server.

        Args:
            model_path: Path to the model file (optional, will use default if not specified)
            labels_path: Path to the labels file (optional, will use default if not specified)

        Returns:
            True if the model was sent successfully, False otherwise
        """
        try:
            # Use the provided paths or get them from the global config
            global config
            if config and "model" in config:
                model_config = config.get("model", {})
                model_path = model_path or model_config.get("model_path")
                labels_path = labels_path or model_config.get("labels_path")

            if not model_path:
                logger.error("No model path specified and no config available")
                return False

            # Get current directory
            current_dir = os.getcwd()
            logger.info(f"Current working directory: {current_dir}")

            # Ensure we have absolute paths
            if not os.path.isabs(model_path):
                model_path = os.path.join(current_dir, model_path)

            if labels_path and not os.path.isabs(labels_path):
                labels_path = os.path.join(current_dir, labels_path)

            logger.info(f"Using model path: {model_path}")
            if labels_path:
                logger.info(f"Using labels path: {labels_path}")

            # Check if files exist
            if not os.path.exists(model_path):
                logger.error(f"Model file not found: {model_path}")
                return False

            # Create package for model
            description = f"Initial model from device {self.device_id}"
            package_id = self.sync_interface.create_package(
                device_id=self.device_id,
                package_type="model",
                description=description
            )

            # Add model file to package
            if not self.sync_interface.add_file_to_package(
                package_id=package_id,
                file_path=model_path,
                file_type="model"
            ):
                logger.error(f"Failed to add model file {model_path} to package")
                return False

            # Add labels file if available
            if labels_path and os.path.exists(labels_path):
                if not self.sync_interface.add_file_to_package(
                    package_id=package_id,
                    file_path=labels_path,
                    file_type="labels"
                ):
                    logger.warning(f"Failed to add labels file {labels_path} to package")

            # Create metadata
            metadata = {
                "device_id": self.device_id,
                "timestamp": time.time(),
                "model_info": {
                    "source": "device_registration",
                    "model_type": "tflite",
                    "description": "Initial device model"
                }
            }

            # Create a properly named metadata file (model_info.json for the model transformer)
            # Create a temp directory to store our properly named files
            temp_dir = tempfile.mkdtemp()
            metadata_file_path = os.path.join(temp_dir, "model_info.json")

            # Write the data to the proper file
            with open(metadata_file_path, 'w') as metadata_file:
                json.dump(metadata, metadata_file)

            # Add the metadata file to the package
            success = self.sync_interface.add_file_to_package(
                package_id=package_id,
                file_path=metadata_file_path,
                file_type="metadata"
            )

            # Clean up the temporary files
            try:
                os.remove(metadata_file_path)
                os.rmdir(temp_dir)
            except Exception as e:
                logger.warning(f"Failed to remove temporary metadata files: {e}")

            # Finalize the package
            self.sync_interface.finalize_package(package_id)

            # Send the package
            result = self.send_package(package_id)
            if result:
                logger.info(f"Successfully sent initial model to server")
            else:
                logger.error(f"Failed to send initial model to server")

            return result

        except Exception as e:
            logger.error(f"Failed to send model to server: {e}")
            return False
    
    def send_package(self, package_id: str) -> bool:
        """Send a specific package to TinySphere.
        
        Args:
            package_id: ID of the package to send
            
        Returns:
            True if sending was successful, False otherwise
        """
        try:
            return self.client.send_package(package_id)
        except (TinyLCMConnectionError, SyncError) as e:
            logger.error(f"Failed to send package {package_id}: {e}")
            return False
    
    def sync_all_pending_packages(self) -> List[Dict[str, Any]]:
        """Synchronize all pending packages and prediction images.

        Returns:
            List of results for each package
        """
        try:
            results = []

            # First sync prediction images if enabled
            if self.enable_prediction_images and self.pending_prediction_images:
                logger.info(f"Syncing {len(self.pending_prediction_images)} pending prediction images")
                img_results = self.sync_prediction_images()
                if img_results:
                    results.extend(img_results)

            # Then sync all other pending packages
            pkg_results = self.client.sync_all_pending_packages()
            if pkg_results:
                results.extend(pkg_results)

            self.last_sync_time = time.time()
            return results
        except Exception as e:
            logger.error(f"Failed to sync pending packages: {e}")
            return []
    
    def should_sync(self) -> bool:
        """Check if it's time for a sync operation.
        
        Returns:
            True if it's time to sync, False otherwise
        """
        return time.time() - self.last_sync_time >= self.sync_interval_seconds
    
    def create_and_send_drift_event_package(
        self,
        detector_name: str,
        reason: str,
        metrics: Dict[str, Any],
        sample: Optional[FeatureSample] = None,
        image_path: Optional[str] = None
    ) -> bool:
        """Create and send a package containing drift event information.

        Args:
            detector_name: Name of the drift detector that triggered the event
            reason: Reason for the drift event
            metrics: Metrics associated with the drift event
            sample: FeatureSample that triggered the drift event
            image_path: Path to the image that triggered the drift event (optional)

        Returns:
            True if the operation was successful, False otherwise
        """
        try:
            # Generate a package ID with description instead of metadata
            description = f"Drift event from {detector_name}: {reason}"
            package_id = self.sync_interface.create_package(
                device_id=self.device_id,
                package_type="drift_event",
                description=description
            )

            # Create the drift event data
            drift_data = {
                "timestamp": time.time(),
                "detector_name": detector_name,
                "reason": reason,
                "metrics": metrics,
                "sample": {
                    "sample_id": sample.sample_id if sample else None,
                    "prediction": sample.prediction if sample else None,
                    "confidence": float(sample.metadata.get("confidence", 0.0)) if sample else None,
                    "timestamp": sample.timestamp if sample else None
                }
            }

            # Create a properly named drift event file
            # Create a temp directory to store our properly named files
            temp_dir = tempfile.mkdtemp()
            drift_file_path = os.path.join(temp_dir, "drift_event.json")

            # Write the data to the proper file
            with open(drift_file_path, 'w') as drift_file:
                json.dump(drift_data, drift_file)

            # Add the drift event file to the package
            success = self.sync_interface.add_file_to_package(
                package_id=package_id,
                file_path=drift_file_path,
                file_type="drift_event"
            )

            # Clean up the temporary files
            try:
                os.remove(drift_file_path)
                os.rmdir(temp_dir)
            except Exception as e:
                logger.warning(f"Failed to remove temporary drift files: {e}")

            # Add image to the package if provided
            if image_path and os.path.exists(image_path):
                # Path to image is structured as device_id/drift_type/date/filename.jpg
                # Extract the complete path structure to maintain it in the package
                rel_path = os.path.relpath(image_path, "./drift_images") if image_path.startswith("./drift_images") else os.path.basename(image_path)

                # Log path information for debugging
                logger.info(f"Adding drift image to package: {image_path}")
                logger.info(f"Relative path for drift image: {rel_path}")

                # Create a metadata file for the image with the proper destination path
                metadata = {
                    "relative_path": rel_path,  # Store this for the server to reconstruct the path
                    "drift_type": detector_name,
                    "timestamp": time.time(),
                    "target_bucket": "drift"  # Set target bucket explicitly to 'drift'
                }

                # Create a temp metadata file
                temp_dir = tempfile.mkdtemp()
                try:
                    metadata_file_path = os.path.join(temp_dir, "image_metadata.json")
                    with open(metadata_file_path, 'w') as meta_file:
                        json.dump(metadata, meta_file)

                    # Add metadata file first
                    self.sync_interface.add_file_to_package(
                        package_id=package_id,
                        file_path=metadata_file_path,
                        file_type="image_metadata"
                    )

                    # Add the image itself
                    self.sync_interface.add_file_to_package(
                        package_id=package_id,
                        file_path=image_path,
                        file_type="image"
                    )

                    # Clean up temp dir
                    os.remove(metadata_file_path)
                    os.rmdir(temp_dir)
                except Exception as e:
                    logger.warning(f"Error adding drift image metadata: {e}")
                    # Add the image without metadata as fallback
                    self.sync_interface.add_file_to_package(
                        package_id=package_id,
                        file_path=image_path,
                        file_type="image"
                    )

            # Finalize the package
            self.sync_interface.finalize_package(package_id)

            # Send the package
            return self.send_package(package_id)

        except Exception as e:
            logger.error(f"Failed to create and send drift event package: {e}")
            return False
    
    def create_and_send_metrics_package(self, metrics: Dict[str, Any]) -> bool:
        """Create and send a package containing system metrics.

        Args:
            metrics: System metrics to include in the package

        Returns:
            True if the operation was successful, False otherwise
        """
        try:
            # Generate a package ID
            description = f"System metrics from device {self.device_id}"
            package_id = self.sync_interface.create_package(
                device_id=self.device_id,
                package_type="metrics",
                description=description
            )
            
            # Create a properly named metrics file (needs to be named metrics.json for the transformer)
            # Create a temp directory to store our properly named files
            temp_dir = tempfile.mkdtemp()
            metrics_file_path = os.path.join(temp_dir, "metrics.json")

            # Write the data to the proper file
            with open(metrics_file_path, 'w') as metrics_file:
                json.dump(metrics, metrics_file)

            # Add the metrics file to the package
            success = self.sync_interface.add_file_to_package(
                package_id=package_id,
                file_path=metrics_file_path,
                file_type="metrics"
            )

            # Clean up the temporary files
            try:
                os.remove(metrics_file_path)
                os.rmdir(temp_dir)
            except Exception as e:
                logger.warning(f"Failed to remove temporary metrics files: {e}")
            
            # Finalize the package
            self.sync_interface.finalize_package(package_id)
            
            # Send the package
            return self.send_package(package_id)
            
        except Exception as e:
            logger.error(f"Failed to create and send metrics package: {e}")
            return False
    
    def add_prediction_image(self, image_path: str, prediction: str, confidence: float) -> bool:
        """Add a prediction image to the queue for synchronization.

        Args:
            image_path: Path to the image file
            prediction: Prediction label (e.g., "lego", "stone")
            confidence: Confidence score for the prediction (0-1)

        Returns:
            True if the image was added to the queue, False otherwise
        """
        if not self.enable_prediction_images:
            logger.debug(f"Prediction image transfer is disabled, not adding image {image_path}")
            return False

        try:
            if not os.path.exists(image_path):
                logger.warning(f"Image file not found: {image_path}")
                return False

            # Add to pending images queue
            self.pending_prediction_images.append({
                "path": image_path,
                "prediction": prediction,
                "confidence": confidence,
                "timestamp": time.time()
            })

            logger.debug(f"Added prediction image to sync queue: {image_path}")
            return True
        except Exception as e:
            logger.error(f"Failed to add prediction image: {str(e)}")
            return False

    def sync_prediction_images(self) -> List[Dict[str, Any]]:
        """Synchronize all pending prediction images.

        Returns:
            List of results for each image batch
        """
        if not self.enable_prediction_images or not self.pending_prediction_images:
            return []

        try:
            results = []
            max_images_per_batch = 10  # Limit number of images per batch

            # Group images by prediction class
            prediction_groups = {}
            for img in self.pending_prediction_images:
                pred = img["prediction"]
                if pred not in prediction_groups:
                    prediction_groups[pred] = []
                prediction_groups[pred].append(img)

            # Process each prediction group
            for prediction, images in prediction_groups.items():
                # Process images in batches
                for i in range(0, len(images), max_images_per_batch):
                    batch = images[i:i+max_images_per_batch]
                    result = self._create_and_send_prediction_images_package(prediction, batch)
                    results.append(result)

            # Identify successfully processed images
            processed_images = []
            for result in results:
                if result.get("status") == "success":
                    processed_images.extend(result.get("processed_images", []))

            # Remove processed images from the pending list
            self.pending_prediction_images = [img for img in self.pending_prediction_images
                                          if img["path"] not in processed_images]

            # Log results
            if processed_images:
                logger.info(f"Synchronized {len(processed_images)} prediction images")
            if self.pending_prediction_images:
                logger.warning(f"{len(self.pending_prediction_images)} prediction images still pending")

            return results
        except Exception as e:
            logger.error(f"Failed to sync prediction images: {str(e)}")
            return []

    def _create_and_send_prediction_images_package(self, prediction: str, images: List[Dict[str, Any]]) -> Dict[str, Any]:
        """Create and send a package containing prediction images.

        Args:
            prediction: Prediction class
            images: List of images with metadata

        Returns:
            Result dictionary
        """
        if not images:
            return {"status": "error", "message": "No images provided"}

        try:
            timestamp = datetime.now().strftime("%Y%m%d_%H%M%S")
            description = f"Prediction '{prediction}' images from device {self.device_id}"
            package_id = self.sync_interface.create_package(
                device_id=self.device_id,
                package_type="prediction_images",
                description=description
            )

            # Create metadata
            metadata = {
                "device_id": self.device_id,
                "timestamp": time.time(),
                "prediction": prediction,
                "image_count": len(images),
                "confidence_avg": sum(img["confidence"] for img in images) / len(images) if images else 0,
            }

            # Write metadata to a temp file
            temp_dir = tempfile.mkdtemp()
            metadata_file_path = os.path.join(temp_dir, "prediction_images.json")

            with open(metadata_file_path, 'w') as metadata_file:
                json.dump(metadata, metadata_file)

            # Add the metadata file to the package
            self.sync_interface.add_file_to_package(
                package_id=package_id,
                file_path=metadata_file_path,
                file_type="metadata"
            )

            # Add each image with a proper name
            processed_images = []
            for idx, img in enumerate(images):
                image_path = img["path"]
                if os.path.exists(image_path):
                    # Get image extension
                    _, ext = os.path.splitext(image_path)

                    # Create a meaningful image name
                    conf_str = f"{img['confidence']:.2f}".replace(".", "")
                    ts_str = datetime.fromtimestamp(img["timestamp"]).strftime("%H%M%S")
                    new_filename = f"{prediction}_{ts_str}_{conf_str}_{idx}{ext}"

                    # Create a temporary copy with the new name
                    temp_img_path = os.path.join(temp_dir, new_filename)
                    shutil.copy(image_path, temp_img_path)

                    # Add to package
                    success = self.sync_interface.add_file_to_package(
                        package_id=package_id,
                        file_path=temp_img_path,
                        file_type="image"
                    )

                    if success:
                        processed_images.append(img["path"])
                    else:
                        logger.warning(f"Failed to add image {image_path} to package")

            # Clean up temp directory
            try:
                shutil.rmtree(temp_dir)
            except Exception as e:
                logger.warning(f"Failed to clean up temp directory: {e}")

            # Finalize package
            self.sync_interface.finalize_package(package_id)

            # Send the package
            result = self.send_package(package_id)
            if result:
                logger.info(f"Successfully sent {len(processed_images)} prediction images to server")
                # Return list of image paths that were successfully processed
                return {
                    "status": "success",
                    "message": f"Successfully sent {len(processed_images)} prediction images",
                    "image_count": len(processed_images),
                    "prediction": prediction,
                    "processed_images": processed_images,
                    "package_type": "prediction_images",
                    "package_id": package_id
                }
            else:
                logger.error(f"Failed to send prediction images package to server")
                return {
                    "status": "error",
                    "message": "Failed to send prediction images package",
                    "prediction": prediction
                }

        except Exception as e:
            logger.error(f"Failed to create prediction images package: {str(e)}")
            return {"status": "error", "message": str(e)}

    def delete_transferred_images(self, image_paths: List[str]) -> Tuple[int, int]:
        """Delete images that have been successfully transferred to the server.

        Args:
            image_paths: List of image file paths to delete

        Returns:
            Tuple of (success_count, fail_count)
        """
        success_count = 0
        fail_count = 0

        for path in image_paths:
            try:
                if os.path.exists(path):
                    os.remove(path)
                    success_count += 1
                    logger.debug(f"Deleted transferred image: {path}")
                else:
                    logger.warning(f"Image file not found: {path}")
                    fail_count += 1
            except Exception as e:
                logger.error(f"Failed to delete image {path}: {e}")
                fail_count += 1

        logger.info(f"Deleted {success_count} transferred images, {fail_count} failed")
        return success_count, fail_count

    def close(self):
        """Close the connection and clean up resources."""
        try:
            self.client.close()
            logger.info("SyncClient closed")
        except Exception as e:
            logger.error(f"Error closing SyncClient: {e}")


# Test function
def main():
    """Test the ExtendedSyncClient functionality."""
    import argparse
    
    # Configure logging
    logging.basicConfig(level=logging.INFO)
    
    # Parse arguments
    parser = argparse.ArgumentParser(description='Test ExtendedSyncClient')
    parser.add_argument('--server', default='http://localhost:8000', help='TinySphere server URL')
    parser.add_argument('--key', default='tinylcm-demo-key', help='API key')
    parser.add_argument('--device', default=None, help='Device ID (defaults to hostname)')
    parser.add_argument('--sync-dir', default='./sync_data', help='Sync directory')
    
    args = parser.parse_args()
    
    # Generate default device ID if not provided
    if not args.device:
        import socket
        args.device = f"test-{socket.gethostname()}"
    
    # Create sync client
    client = ExtendedSyncClient(
        server_url=args.server,
        api_key=args.key,
        device_id=args.device,
        sync_dir=args.sync_dir
    )
    
    # Check connection
    connected = client.check_connection()
    print(f"Connection status: {'Connected' if connected else 'Disconnected'}")
    
    if not connected:
        print("Cannot continue without connection")
        return
    
    # Register device
    registered = client.register_device()
    print(f"Device registration: {'Successful' if registered else 'Failed'}")
    
    # Create and send a test metrics package
    test_metrics = {
        "cpu_usage": 25.5,
        "memory_usage": 512.3,
        "temperature": 45.2,
        "uptime": 3600,
        "timestamp": time.time()
    }
    
    metrics_sent = client.create_and_send_metrics_package(test_metrics)
    print(f"Metrics package sending: {'Successful' if metrics_sent else 'Failed'}")
    
    # Create and send a test drift event package
    test_drift_event = {
        "change_magnitude": 0.35,
        "window_size": 100,
        "baseline_mean": 0.85,
        "current_mean": 0.65,
        "threshold": 0.2
    }
    
    # Mock sample
    mock_sample = type('MockSample', (), {
        'sample_id': 'mock_123',
        'prediction': 'red_stone',
        'confidence': 0.75,
        'timestamp': time.time()
    })
    
    drift_sent = client.create_and_send_drift_event_package(
        detector_name="EWMAConfidenceMonitor",
        reason="Confidence drop detected",
        metrics=test_drift_event,
        sample=mock_sample
    )
    print(f"Drift event package sending: {'Successful' if drift_sent else 'Failed'}")
    
    # Sync all pending packages
    results = client.sync_all_pending_packages()
    print(f"Synced {len(results)} packages")
    
    # Close client
    client.close()

if __name__ == "__main__":
    main()