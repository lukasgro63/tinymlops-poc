"""
DriftEventsTransformer for TinySphere
------------------------------------
Processes packages containing drift event images from TinyLCM devices.
Organizes and stores images in MinIO with a structured hierarchy in the drift bucket.
"""

import json
import logging
import os
import tempfile
from datetime import datetime
from pathlib import Path
from typing import Any, Dict, List, Optional

import boto3
from botocore.client import Config

from tinysphere.importer.transformers.base import DataTransformer

logger = logging.getLogger(__name__)

class DriftEventsTransformer(DataTransformer):
    """
    Transformer for drift event image packages.
    Extracts drift event images from TinyLCM device packages and uploads them to MinIO/drift bucket.
    """
    
    def __init__(self):
        """Initialize the DriftEventsTransformer."""
        self.s3_client = self._init_s3_client()
        self.drift_bucket_name = "drift"
    
    def _init_s3_client(self):
        """Initialize the S3 client for MinIO."""
        return boto3.client(
            's3',
            endpoint_url=f"http://{os.environ.get('MINIO_ENDPOINT', 'minio:9000')}",
            aws_access_key_id=os.environ.get('MINIO_ACCESS_KEY', 'minioadmin'),
            aws_secret_access_key=os.environ.get('MINIO_SECRET_KEY', 'minioadmin'),
            config=Config(signature_version='s3v4'),
            region_name='us-east-1',  # This is ignored but required by boto3
            verify=False  # Disable SSL verification for local MinIO
        )
    
    def can_transform(self, package_type: str, files: List[Path]) -> bool:
        """
        Determine if this transformer can handle the given package.

        Args:
            package_type: Type of the package
            files: List of file paths in the package

        Returns:
            True if this transformer can handle the package, False otherwise
        """
        logger.info(f"DriftEventsTransformer.can_transform called with package_type: {package_type}")
        
        # Check for drift_event package type
        if package_type and "drift_event" in package_type.lower():
            # Check if there are image files in the package
            image_files = [f for f in files if f.suffix.lower() in ['.jpg', '.jpeg', '.png']]
            if image_files:
                logger.info(f"DriftEventsTransformer will handle package with type: {package_type} (found {len(image_files)} images)")
                return True
        
        # Also check for metadata files with target_bucket set to drift
        for file in files:
            if file.name.lower() == "image_metadata.json" or "image_metadata" in file.name.lower():
                try:
                    with open(file, 'r') as f:
                        metadata = json.load(f)
                        if metadata.get("target_bucket") == "drift":
                            logger.info(f"DriftEventsTransformer will handle package due to image metadata specifying drift bucket")
                            return True
                except Exception as e:
                    logger.warning(f"Error reading image metadata file: {e}")
        
        # Check for drift-related images by filename patterns
        image_files = []
        for file in files:
            if file.suffix.lower() in ['.jpg', '.jpeg', '.png']:
                filename = file.name.lower()
                # Look for drift-related patterns in filenames
                if any(pattern in filename for pattern in ["drift", "event", "anomaly", "change"]):
                    image_files.append(file)
                    logger.info(f"DriftEventsTransformer found drift-related image file: {file}")
                
        if image_files and len(image_files) > 0:
            # Only handle if we have drift event JSON files as well
            drift_json_files = [f for f in files if "drift" in f.name.lower() and f.suffix.lower() == ".json"]
            if drift_json_files:
                logger.info(f"DriftEventsTransformer found {len(image_files)} drift image files with drift JSON data")
                return True
        
        logger.info(f"DriftEventsTransformer cannot handle package of type: {package_type}")
        return False
        
    def transform(self, package_id: str, device_id: str, files: List[Path], metadata: Dict[str, Any]) -> Dict[str, Any]:
        """
        Transform drift event images package content.

        Args:
            package_id: ID of the package
            device_id: ID of the device that sent the package
            files: List of file paths in the package
            metadata: Package metadata

        Returns:
            Dictionary with transformation results
        """
        logger.info(f"DriftEventsTransformer.transform called for package: {package_id} from device: {device_id}")
        
        # Initialize result object
        result = {
            "transformer": "drift_events",
            "status": "success",
            "processed_files": [],
            "uploaded_images": []
        }
        
        # Identify image files
        image_files = [f for f in files if f.suffix.lower() in ['.jpg', '.jpeg', '.png']]
        
        if not image_files:
            result["status"] = "error"
            result["message"] = "No image files found in package"
            logger.warning(f"No image files found in package {package_id}")
            return result
        
        # Find drift event metadata and/or image metadata
        drift_metadata_files = [f for f in files if 
                              f.name.lower() == "drift_event.json" or 
                              "drift" in f.name.lower() and f.suffix.lower() == ".json"]
        
        image_metadata_files = [f for f in files if 
                             f.name.lower() == "image_metadata.json" or 
                             "image_metadata" in f.name.lower()]
        
        # Try to get drift type from metadata
        drift_type = "unknown"
        event_id = None

        # First try image metadata for direct info
        for meta_file in image_metadata_files:
            try:
                with open(meta_file, 'r') as f:
                    metadata_content = json.load(f)
                    if "drift_type" in metadata_content:
                        drift_type = metadata_content.get("drift_type", "unknown")
                    
                    # Check for relative path which might include drift type
                    rel_path = metadata_content.get("relative_path")
                    if rel_path and isinstance(rel_path, str):
                        path_parts = rel_path.split("/")
                        if len(path_parts) >= 2:
                            # Structure usually: device_id/drift_type/date/filename
                            drift_type = path_parts[1]
                    
                    logger.info(f"Found drift type in image metadata: {drift_type}")
                    break
            except Exception as e:
                logger.warning(f"Error reading image metadata file: {e}")
        
        # Then try drift event JSON
        if drift_type == "unknown" and drift_metadata_files:
            try:
                with open(drift_metadata_files[0], 'r') as f:
                    metadata_content = json.load(f)
                    
                    # Try to get drift type from detector name
                    detector_name = metadata_content.get("detector_name", "")
                    if detector_name:
                        detector_lower = detector_name.lower()
                        # Check for KNN distance monitor specifically
                        if "knn" in detector_lower and "distance" in detector_lower:
                            drift_type = "knn_distance"
                        else:
                            # Clean up detector name to create drift type
                            drift_type = detector_lower.replace("monitor", "").replace(" ", "_")
                        logger.info(f"Derived drift type from detector name: {drift_type}")
                    
                    # Extract event ID if available
                    if "event_id" in metadata_content:
                        event_id = metadata_content.get("event_id")
                        logger.info(f"Found event ID in drift metadata: {event_id}")
            except Exception as e:
                logger.warning(f"Error reading drift metadata file: {e}")
        
        # If we still don't have a drift type, check package description
        if drift_type == "unknown" and "description" in metadata:
            package_description = metadata.get("description", "")
            if "drift event from" in package_description.lower():
                parts = package_description.split("from ")[1].split(":")[0]
                drift_type = parts.lower().replace("monitor", "").strip().replace(" ", "_")
                logger.info(f"Extracted drift type from package description: {drift_type}")
        
        # Upload images to MinIO drift bucket
        uploaded_count = 0
        
        for img_file in image_files:
            try:
                # Create a key with hierarchical structure
                timestamp = datetime.now().strftime("%Y%m%d")
                
                # Extract original filename or use the existing one
                filename = img_file.name
                
                # If we have an event ID, include it in the path
                if event_id:
                    # For event images, we'll structure it as device_id/drift_type/date/event_id/filename
                    # This keeps the event images grouped together
                    s3_key = f"{device_id}/{drift_type}/{timestamp}/event_{event_id}/{filename}"
                else:
                    # Use standard structure: device_id/drift_type/date/filename
                    s3_key = f"{device_id}/{drift_type}/{timestamp}/{filename}"
                
                # Upload to MinIO's drift bucket
                self.s3_client.upload_file(
                    Filename=str(img_file),
                    Bucket=self.drift_bucket_name,
                    Key=s3_key
                )
                
                logger.info(f"Uploaded drift event image to MinIO drift bucket: {s3_key}")
                result["processed_files"].append(str(img_file))
                result["uploaded_images"].append(s3_key)
                uploaded_count += 1
                
            except Exception as e:
                logger.error(f"Error uploading drift image {img_file} to MinIO: {e}")
                result["errors"] = result.get("errors", [])
                result["errors"].append(f"Failed to upload {img_file}: {str(e)}")
        
        # Upload the drift metadata files as well for reference
        for meta_file in drift_metadata_files:
            try:
                # Create a structured path similar to the image path
                timestamp = datetime.now().strftime("%Y%m%d")
                
                # If we have an event ID, include it in the path
                if event_id:
                    s3_key = f"{device_id}/{drift_type}/{timestamp}/event_{event_id}/{meta_file.name}"
                else:
                    s3_key = f"{device_id}/{drift_type}/{timestamp}/{meta_file.name}"
                
                # Upload to MinIO
                self.s3_client.upload_file(
                    Filename=str(meta_file),
                    Bucket=self.drift_bucket_name,
                    Key=s3_key
                )
                
                logger.info(f"Uploaded drift metadata to MinIO: {s3_key}")
                result["processed_files"].append(str(meta_file))
                
            except Exception as e:
                logger.error(f"Error uploading drift metadata {meta_file} to MinIO: {e}")
                result["errors"] = result.get("errors", [])
                result["errors"].append(f"Failed to upload {meta_file}: {str(e)}")
        
        # Update result status
        if uploaded_count == 0:
            result["status"] = "error"
            result["message"] = "Failed to upload any drift event images"
        else:
            result["message"] = f"Successfully uploaded {uploaded_count} drift event images"
            result["upload_count"] = uploaded_count
            result["drift_type"] = drift_type
            if event_id:
                result["event_id"] = event_id
        
        return result