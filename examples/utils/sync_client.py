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
import platform
import socket
import tempfile
import time
from datetime import datetime
from pathlib import Path
from typing import Any, Dict, List, Optional, Tuple

import shutil

# Import geolocation module if available
try:
    from tinylcm.utils.geolocation import Geolocator
    GEOLOCATION_AVAILABLE = True
except ImportError:
    GEOLOCATION_AVAILABLE = False

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
        auto_register: bool = True,
        enable_geolocation: bool = False,
        geolocation_api_key: Optional[str] = None,
        geolocation_cache_ttl: int = 86400,
        geolocation_update_interval: int = 3600,
        geolocation_fallback_coordinates: List[float] = [0.0, 0.0]
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
            enable_geolocation: Whether to enable geolocation
            geolocation_api_key: API key for geolocation service (if needed)
            geolocation_cache_ttl: Time-to-live for geolocation cache in seconds
            geolocation_update_interval: Interval between geolocation updates in seconds
            geolocation_fallback_coordinates: Default coordinates to use if geolocation fails [lat, lon]
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
        
        # Geolocation attributes
        self.enable_geolocation = enable_geolocation
        self.geolocation_api_key = geolocation_api_key
        self.geolocation_cache_ttl = geolocation_cache_ttl
        self.geolocation_update_interval = geolocation_update_interval
        self.geolocation_fallback_coordinates = geolocation_fallback_coordinates
        self.geolocator = None
        self.location_cache = None
        self.last_location_update = 0
        
        # Initialize geolocator if enabled
        if self.enable_geolocation and GEOLOCATION_AVAILABLE:
            try:
                self.geolocator = Geolocator(
                    api_key=self.geolocation_api_key,
                    cache_ttl=self.geolocation_cache_ttl,
                    fallback_coordinates=self.geolocation_fallback_coordinates
                )
                logger.info(f"Geolocator initialized for device {device_id}")
            except Exception as e:
                logger.warning(f"Failed to initialize geolocator: {e}")
        elif self.enable_geolocation and not GEOLOCATION_AVAILABLE:
            logger.warning("Geolocation is enabled in config but the geolocation module is not available.")
            logger.warning("Install the tinylcm package with geolocation extras: pip install tinylcm[geolocation]")


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
                    
            # Update device information including geolocation if enabled
            if self.enable_geolocation and GEOLOCATION_AVAILABLE and self.geolocator:
                self.update_device_info()

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

            # Determine drift type from detector name
            drift_type = "unknown"
            detector_lower = detector_name.lower()
            if "confidence" in detector_lower:
                drift_type = "confidence"
            elif "distribution" in detector_lower:
                drift_type = "distribution"
            elif "feature" in detector_lower and not "knn" in detector_lower:
                drift_type = "feature"
            elif "knn" in detector_lower or "distance" in detector_lower:
                drift_type = "knn_distance"
            elif "outlier" in detector_lower:
                drift_type = "outlier"
                
            # Log the drift type for debugging
            logger.info(f"ExtendedSyncClient: Derived drift_type={drift_type} from detector_name={detector_name}")
                
            # Create the drift event data with explicit drift_type
            drift_data = {
                "timestamp": time.time(),
                "detector_name": detector_name,
                "reason": reason,
                "drift_type": drift_type,  # Add explicitly derived drift_type
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

                # Use the drift_type we already calculated above
                # No need to recalculate it here
                logger.info(f"Using drift_type={drift_type} for image metadata")
                
                # Create a metadata file for the image with the proper destination path
                metadata = {
                    "relative_path": rel_path,  # Store this for the server to reconstruct the path
                    "drift_type": drift_type,  # Use the converted drift_type
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
            
    def create_and_send_operational_logs_package(self, operational_logs_dir: str, session_id: Optional[str] = None) -> bool:
        """Create and send a package containing raw operational logs (JSONL files).
        
        This sends the raw operational log files to TinySphere to be stored in the 'data_logs' bucket.
        These files contain detailed inference data for each prediction, including KNN debug information.

        Args:
            operational_logs_dir: Directory containing operational log files
            session_id: Optional session ID to filter logs (if None, all logs are sent)

        Returns:
            True if the operation was successful, False otherwise
        """
        try:
            # Ensure the logs directory exists
            if not os.path.exists(operational_logs_dir):
                logger.warning(f"Operational logs directory not found: {operational_logs_dir}")
                return False
                
            # Find operational log files in the directory
            log_files = []
            for filename in os.listdir(operational_logs_dir):
                if filename.startswith("operational_log_") and filename.endswith(".jsonl"):
                    # Filter by session_id if provided
                    if session_id is None or session_id in filename:
                        log_files.append(os.path.join(operational_logs_dir, filename))
            
            if not log_files:
                logger.warning(f"No operational log files found in {operational_logs_dir}")
                return False
                
            logger.info(f"Found {len(log_files)} operational log files to send")
            
            # Generate a package ID
            description = f"Raw operational logs from device {self.device_id}"
            package_id = self.sync_interface.create_package(
                device_id=self.device_id,
                package_type="operational_logs",  # Special type for raw operational logs
                description=description
            )
            
            # Create metadata file with information about the logs
            temp_dir = tempfile.mkdtemp()
            metadata = {
                "device_id": self.device_id,
                "timestamp": time.time(),
                "session_id": session_id,
                "log_files": [os.path.basename(f) for f in log_files],
                "file_count": len(log_files),
                "target_bucket": "data-logs"  # Specify target bucket for the transformer
            }
            
            metadata_file_path = os.path.join(temp_dir, "logs_metadata.json")
            with open(metadata_file_path, 'w') as metadata_file:
                json.dump(metadata, metadata_file)
                
            # Add metadata file to package
            self.sync_interface.add_file_to_package(
                package_id=package_id,
                file_path=metadata_file_path,
                file_type="metadata"
            )
            
            # Add each log file to the package
            for log_file in log_files:
                self.sync_interface.add_file_to_package(
                    package_id=package_id,
                    file_path=log_file,
                    file_type="operational_log"
                )
                
            # Clean up temp directory
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
                logger.info(f"Successfully sent {len(log_files)} operational log files to server")
            else:
                logger.error(f"Failed to send operational logs package to server")
                
            return result
            
        except Exception as e:
            logger.error(f"Failed to create and send operational logs package: {e}")
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

    def _update_geolocation(self) -> Dict[str, Any]:
        """Update and retrieve geolocation information.
        
        Returns:
            Dictionary with location information (latitude, longitude, accuracy, source)
        """
        if not self.enable_geolocation or not GEOLOCATION_AVAILABLE or not self.geolocator:
            return {"latitude": 0.0, "longitude": 0.0, "accuracy": 0.0, "source": "disabled"}
            
        current_time = time.time()
        
        # Check if it's time to update location
        if (current_time - self.last_location_update) >= self.geolocation_update_interval:
            try:
                location = self.geolocator.get_location()
                self.location_cache = location
                self.last_location_update = current_time
                logger.info(f"Updated device geolocation: {location.get('latitude'):.6f}, {location.get('longitude'):.6f}")
                return location
            except Exception as e:
                logger.warning(f"Failed to update geolocation: {e}")
                # Fall back to cached location or defaults
        
        # Return cached location if available
        if self.location_cache:
            return self.location_cache
            
        # Use fallback coordinates if everything fails
        return {
            "latitude": self.geolocation_fallback_coordinates[0],
            "longitude": self.geolocation_fallback_coordinates[1],
            "accuracy": 0.0,
            "source": "fallback"
        }
        
    def _get_detailed_platform_info(self) -> Dict[str, str]:
        """Get detailed platform information including OS type, version and device model.
        
        Returns:
            Dictionary with detailed platform information
        """
        import os
        import re
        # Use imported platform module instead of importing as plt
        import platform
        
        # Initialize with defaults
        platform_info = {
            "os_type": "unknown",
            "os_version": "unknown",
            "device_model": "unknown"
        }
        
        try:
            # Get basic OS information
            system = platform.system().lower()
            platform_info["os_type"] = system
            
            if system == "linux":
                # For Linux, try to get distribution info
                try:
                    # Try os-release file first (modern Linux)
                    if os.path.exists('/etc/os-release'):
                        with open('/etc/os-release', 'r') as f:
                            for line in f:
                                if line.startswith('VERSION='):
                                    platform_info["os_version"] = line.split('=')[1].strip().strip('"\'')
                                elif line.startswith('ID='):
                                    distro_id = line.split('=')[1].strip().strip('"\'')
                                    platform_info["os_type"] = f"Linux-{distro_id}"
                except Exception as e:
                    logger.debug(f"Error reading os-release: {e}")
                
                # For Raspberry Pi, detect model
                try:
                    if os.path.exists('/proc/device-tree/model'):
                        with open('/proc/device-tree/model', 'r') as f:
                            model = f.read().strip('\x00')
                            if model:
                                platform_info["device_model"] = model
                                # Extract Pi model type
                                if "raspberry pi" in model.lower():
                                    # Extract Pi model number (e.g., "Pi 4" from "Raspberry Pi 4 Model B")
                                    pi_match = re.search(r'raspberry pi\s+(\w+)', model.lower())
                                    if pi_match:
                                        platform_info["device_model"] = f"Pi {pi_match.group(1)}"
                                        
                except Exception as e:
                    logger.debug(f"Error reading device model: {e}")
                
                # If os_version is still unknown, try lsb_release command
                if platform_info["os_version"] == "unknown":
                    try:
                        import subprocess
                        output = subprocess.check_output(['lsb_release', '-d'], universal_newlines=True)
                        platform_info["os_version"] = output.split(':')[1].strip()
                    except Exception:
                        # Fallback to platform.release()
                        platform_info["os_version"] = platform.release()
                        
            elif system == "darwin":  # macOS
                platform_info["os_type"] = "macOS"
                platform_info["os_version"] = platform.mac_ver()[0]
                # For Mac, get model identifier
                try:
                    import subprocess
                    output = subprocess.check_output(['sysctl', '-n', 'hw.model'], universal_newlines=True)
                    platform_info["device_model"] = output.strip()
                except Exception:
                    platform_info["device_model"] = "Mac"
                    
            elif system == "windows":
                platform_info["os_type"] = "Windows"
                platform_info["os_version"] = platform.version()
                # For Windows, get detailed version and model
                try:
                    import subprocess
                    # Get OS version
                    output = subprocess.check_output(['systeminfo'], universal_newlines=True)
                    for line in output.splitlines():
                        if 'OS Version:' in line:
                            platform_info["os_version"] = line.split(':')[1].strip()
                        elif 'System Model:' in line:
                            platform_info["device_model"] = line.split(':')[1].strip()
                except Exception:
                    pass
            
            # If we still don't have detailed info, use platform.platform()
            if platform_info["os_version"] == "unknown":
                platform_info["os_version"] = platform.version()
                
            if platform_info["device_model"] == "unknown":
                platform_info["device_model"] = platform.machine()
                
        except Exception as e:
            logger.warning(f"Error getting detailed platform info: {e}")
            # Fallback to simple platform info
            platform_info["os_type"] = platform.system()
            platform_info["os_version"] = platform.release()
            platform_info["device_model"] = platform.machine()
            
        return platform_info
        
    def _get_tinylcm_version(self) -> str:
        """Get TinyLCM version from the library or fallback to a default.
        
        Returns:
            TinyLCM version string
        """
        try:
            # First try to get from the tinylcm package
            import tinylcm
            if hasattr(tinylcm, "__version__"):
                return tinylcm.__version__
            
            # If not found in the package, try constants module
            from tinylcm.constants import VERSION
            return VERSION
        except (ImportError, AttributeError):
            # Fallback to hard-coded version as last resort
            return "0.2.0"

    def _get_device_info(self) -> Dict[str, Any]:
        """Get device information including hardware details and location.
        
        Returns:
            Dictionary with device information
        """
        try:
            # Get hostname
            hostname = socket.gethostname()
            
            # Get IP address
            try:
                s = socket.socket(socket.AF_INET, socket.SOCK_DGRAM)
                s.settimeout(0.1)
                s.connect(("8.8.8.8", 80))
                ip_address = s.getsockname()[0]
                s.close()
            except:
                ip_address = "127.0.0.1"  # Fallback
            
            # Get platform info with specific details
            platform_info = self._get_detailed_platform_info()
            
            # Use the already imported platform module
            # Basic device info with enhanced platform details
            device_info = {
                "device_id": self.device_id,
                "hostname": hostname,
                "ip_address": ip_address,
                "platform": platform_info["os_type"],
                "platform_version": platform_info["os_version"],
                "device_model": platform_info["device_model"],
                "python_version": platform.python_version(),
                "tinylcm_version": self._get_tinylcm_version(),
                "last_update": time.time()
            }
            
            # Add hardware info if psutil is available
            try:
                import psutil
                device_info["hw_info"] = {
                    "cpu_count": psutil.cpu_count(),
                    "memory_total": psutil.virtual_memory().total,
                    "disk_total": psutil.disk_usage("/").total
                }
            except ImportError:
                pass
                
            # Add geolocation data if enabled
            if self.enable_geolocation and GEOLOCATION_AVAILABLE and self.geolocator:
                location = self._update_geolocation()
                device_info["location"] = {
                    "latitude": location.get("latitude", 0.0),
                    "longitude": location.get("longitude", 0.0),
                    "accuracy": location.get("accuracy", 0.0),
                    "source": location.get("source", "unknown"),
                    "updated_at": self.last_location_update
                }
                
            return device_info
        except Exception as e:
            logger.error(f"Error getting device info: {e}")
            # Return minimal device info
            return {"device_id": self.device_id}
    
    def update_device_info(self) -> bool:
        """Update device information with the server.
        
        Returns:
            True if successful, False otherwise
        """
        try:
            # Get device info including geolocation
            device_info = self._get_device_info()
            
            # Rather than using the client's update_device_info which may use PATCH (not supported),
            # we'll directly use the registration endpoint which uses POST
            success = False
            
            if hasattr(self.client, 'connection_manager'):
                # Store geolocation in client for later use
                if self.enable_geolocation and 'location' in device_info:
                    location = device_info['location']
                    if hasattr(self.client, 'current_location'):
                        self.client.current_location = {
                            'latitude': location['latitude'],
                            'longitude': location['longitude'],
                            'accuracy': location['accuracy'],
                            'source': location['source']
                        }
                        self.client.last_geolocation_update = time.time()
                
                # Use POST to /devices/register endpoint instead of PATCH to update device info
                # Log the data we're sending to help debug
                logger.info(f"Sending device update with platform={device_info.get('platform')}, " + 
                            f"platform_version={device_info.get('platform_version')}, " +
                            f"device_model={device_info.get('device_model')}")
                
                # Log geolocation data if present
                if 'location' in device_info:
                    loc = device_info['location']
                    logger.info(f"Device update includes geolocation: " + 
                               f"lat={loc.get('latitude'):.6f}, lon={loc.get('longitude'):.6f}, " + 
                               f"source={loc.get('source')}, accuracy={loc.get('accuracy')}")
                
                update_data = {
                    "device_id": self.device_id,
                    "device_info": device_info,
                    "last_sync_time": time.time()
                }
                
                try:
                    response = self.client.connection_manager.execute_request(
                        method="POST", 
                        endpoint="devices/register", 
                        json=update_data
                    )
                    
                    if response.status_code == 200:
                        logger.info(f"Successfully updated device info with server via registration endpoint")
                        success = True
                    else:
                        error_msg = f"Update via registration failed: {response.status_code} - {response.text}"
                        logger.warning(error_msg)
                        success = False
                except Exception as e:
                    logger.warning(f"Error updating device info via registration: {e}")
                    success = False
            else:
                # Fallback if we don't have connection_manager access
                if hasattr(self.client, 'register_device'):
                    success = self.client.register_device()
                    if success:
                        logger.info(f"Updated device info via register_device method")
                    else:
                        logger.warning(f"Failed to update device info via register_device method")
                
            return success
        except Exception as e:
            logger.error(f"Error updating device info: {e}")
            return False
        
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
    parser.add_argument('--geolocation', action='store_true', help='Enable geolocation')
    parser.add_argument('--geo-key', default=None, help='Geolocation API key (if needed)')
    
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
        sync_dir=args.sync_dir,
        enable_geolocation=args.geolocation,
        geolocation_api_key=args.geo_key
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
    
    # Update device info with geolocation if enabled
    if args.geolocation:
        updated = client.update_device_info()
        print(f"Device info update with geolocation: {'Successful' if updated else 'Failed'}")
        
        # If successful, print the location
        if updated and client.location_cache:
            loc = client.location_cache
            print(f"Device location: {loc.get('latitude'):.6f}, {loc.get('longitude'):.6f}")
            print(f"Location source: {loc.get('source')}")
            print(f"Location accuracy: {loc.get('accuracy')}")
    
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