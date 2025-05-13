"""
PredictionImagesTransformer for TinySphere
-----------------------------------------
Processes image packages containing prediction images from TinyLCM devices.
Organizes and stores images in MinIO with a structured hierarchy.
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

class PredictionImagesTransformer(DataTransformer):
    """
    Transformer for prediction image packages.
    Extracts prediction images from TinyLCM device packages and uploads them to MinIO.
    """
    
    def __init__(self):
        """Initialize the PredictionImagesTransformer."""
        self.s3_client = self._init_s3_client()
    
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
        logger.info(f"PredictionImagesTransformer.can_transform called with package_type: {package_type}")

        # Skip packages that are drift-related
        if package_type and any(drift_term in package_type.lower() for drift_term in ["drift", "event"]):
            logger.info(f"PredictionImagesTransformer will NOT handle drift-related package type: {package_type}")
            return False

        # Check for prediction_images package type
        if package_type and "prediction_images" in package_type.lower():
            logger.info(f"PredictionImagesTransformer will handle package with type: {package_type}")
            return True

        # Check for image files but exclude drift-related packages
        image_files = []

        # Check if any drift-related files exist
        drift_files = [f for f in files if "drift" in f.name.lower() or "event" in f.name.lower()]
        if drift_files:
            logger.info(f"PredictionImagesTransformer will NOT handle package with drift-related files")
            return False

        for file in files:
            if file.suffix.lower() in ['.jpg', '.jpeg', '.png']:
                # Avoid drift event images
                if "drift" in str(file).lower() or "event" in str(file).lower():
                    continue

                image_files.append(file)
                logger.info(f"PredictionImagesTransformer found image file: {file}")

        if image_files:
            # Metadata file should be present to identify prediction type
            metadata_files = [f for f in files if f.name == "prediction_images.json" or "metadata" in f.name.lower()]

            # Skip if any metadata files mention target_bucket=drift
            for meta_file in metadata_files:
                try:
                    with open(meta_file, 'r') as f:
                        metadata = json.load(f)
                        if metadata.get("target_bucket") == "drift":
                            logger.info(f"PredictionImagesTransformer will NOT handle package with drift target_bucket")
                            return False
                except Exception:
                    pass

            if metadata_files:
                logger.info(f"PredictionImagesTransformer found {len(image_files)} image files with metadata")
                return True

        logger.info(f"PredictionImagesTransformer cannot handle package of type: {package_type}")
        return False
        
    def transform(self, package_id: str, device_id: str, files: List[Path], metadata: Dict[str, Any]) -> Dict[str, Any]:
        """
        Transform prediction images package content.

        Args:
            package_id: ID of the package
            device_id: ID of the device that sent the package
            files: List of file paths in the package
            metadata: Package metadata

        Returns:
            Dictionary with transformation results
        """
        logger.info(f"PredictionImagesTransformer.transform called for package: {package_id} from device: {device_id}")
        
        # Initialize result object
        result = {
            "transformer": "prediction_images",
            "status": "success",
            "processed_files": [],
            "uploaded_images": []
        }
        
        # Identify image files
        image_files = [f for f in files if f.suffix.lower() in ['.jpg', '.jpeg', '.png']]
        
        # Find metadata file
        metadata_files = [f for f in files if f.name == "prediction_images.json" or "metadata.json" in f.name.lower()]
        
        if not image_files:
            result["status"] = "error"
            result["message"] = "No image files found in package"
            logger.warning(f"No image files found in package {package_id}")
            return result
            
        # Try to get prediction type from metadata
        prediction_type = "unknown"
        if metadata_files:
            try:
                with open(metadata_files[0], 'r') as f:
                    metadata_content = json.load(f)
                    prediction_type = metadata_content.get("prediction", "unknown")
                    logger.info(f"Found prediction type in metadata: {prediction_type}")
            except Exception as e:
                logger.warning(f"Error reading metadata file: {e}")
        
        # Fall back to package metadata if no prediction type found
        if prediction_type == "unknown" and "package_type" in metadata:
            package_description = metadata.get("description", "")
            if "prediction '" in package_description and "'" in package_description.split("prediction '")[1]:
                prediction_type = package_description.split("prediction '")[1].split("'")[0]
                logger.info(f"Extracted prediction type from package description: {prediction_type}")
        
        # Try to determine prediction type from filenames if still unknown
        if prediction_type == "unknown" and image_files:
            # Check if filenames follow pattern like "lego_xxxxxx.jpg"
            for img_file in image_files:
                parts = img_file.stem.split('_')
                if len(parts) > 0 and parts[0] not in ["frame", "image", "img"]:
                    prediction_type = parts[0]
                    logger.info(f"Extracted prediction type from filename: {prediction_type}")
                    break
        
        # Upload images to MinIO
        bucket_name = "prediction-images"
        uploaded_count = 0
        
        for img_file in image_files:
            try:
                # Create a key with hierarchical structure
                timestamp = datetime.now().strftime("%Y%m%d")
                
                # Extract original filename or use the existing one
                filename = img_file.name
                
                # Construct the S3 key with the structure: device_id/prediction_type/timestamp/filename
                s3_key = f"{device_id}/{prediction_type}/{timestamp}/{filename}"
                
                # Upload to MinIO
                self.s3_client.upload_file(
                    Filename=str(img_file),
                    Bucket=bucket_name,
                    Key=s3_key
                )
                
                logger.info(f"Uploaded image to MinIO: {s3_key}")
                result["processed_files"].append(str(img_file))
                result["uploaded_images"].append(s3_key)
                uploaded_count += 1
                
            except Exception as e:
                logger.error(f"Error uploading image {img_file} to MinIO: {e}")
                result["errors"] = result.get("errors", [])
                result["errors"].append(f"Failed to upload {img_file}: {str(e)}")
        
        # Update result status
        if uploaded_count == 0:
            result["status"] = "error"
            result["message"] = "Failed to upload any images"
        else:
            result["message"] = f"Successfully uploaded {uploaded_count} images"
            result["upload_count"] = uploaded_count
            result["prediction_type"] = prediction_type
        
        return result