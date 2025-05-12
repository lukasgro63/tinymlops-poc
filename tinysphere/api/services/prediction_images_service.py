# api/services/prediction_images_service.py
import os
import logging
from datetime import datetime, timedelta
from typing import List, Dict, Any, Optional

import boto3
from botocore.client import Config

logger = logging.getLogger(__name__)

class PredictionImagesService:
    def __init__(self):
        """Initialize S3 client for MinIO access."""
        self.s3_client = self._init_s3_client()
        self.bucket_name = "prediction-images"
    
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
    
    def list_device_ids(self) -> List[str]:
        """List all device IDs that have prediction images."""
        try:
            # List 'folders' (prefixes) at the root level
            response = self.s3_client.list_objects_v2(
                Bucket=self.bucket_name,
                Delimiter='/'
            )
            
            # Extract device IDs from CommonPrefixes
            device_ids = []
            if 'CommonPrefixes' in response:
                for prefix in response['CommonPrefixes']:
                    device_id = prefix['Prefix'].rstrip('/')
                    device_ids.append(device_id)
            
            return device_ids
        except Exception as e:
            logger.error(f"Error listing device IDs: {e}")
            return []
    
    def list_prediction_types(self, device_id: str) -> List[str]:
        """List all prediction types for a specific device."""
        try:
            # List 'folders' (prefixes) at the device level
            response = self.s3_client.list_objects_v2(
                Bucket=self.bucket_name,
                Prefix=f"{device_id}/",
                Delimiter='/'
            )
            
            # Extract prediction types from CommonPrefixes
            prediction_types = []
            if 'CommonPrefixes' in response:
                for prefix in response['CommonPrefixes']:
                    pred_type = prefix['Prefix'].split('/')[1]
                    prediction_types.append(pred_type)
            
            return prediction_types
        except Exception as e:
            logger.error(f"Error listing prediction types for device {device_id}: {e}")
            return []
    
    def list_dates(self, device_id: str, prediction_type: str) -> List[str]:
        """List all dates with prediction images for a device and prediction type."""
        try:
            # List 'folders' (prefixes) at the prediction type level
            response = self.s3_client.list_objects_v2(
                Bucket=self.bucket_name,
                Prefix=f"{device_id}/{prediction_type}/",
                Delimiter='/'
            )
            
            # Extract dates from CommonPrefixes
            dates = []
            if 'CommonPrefixes' in response:
                for prefix in response['CommonPrefixes']:
                    date = prefix['Prefix'].split('/')[2]
                    dates.append(date)
            
            return dates
        except Exception as e:
            logger.error(f"Error listing dates for device {device_id} and type {prediction_type}: {e}")
            return []
    
    def list_images(
        self, 
        device_id: Optional[str] = None, 
        prediction_type: Optional[str] = None, 
        date: Optional[str] = None,
        limit: int = 100,
        offset: int = 0
    ) -> Dict[str, Any]:
        """
        List prediction images with optional filtering.
        
        Args:
            device_id: Filter by device ID
            prediction_type: Filter by prediction type
            date: Filter by date (YYYYMMDD format)
            limit: Maximum number of images to return
            offset: Number of images to skip
            
        Returns:
            Dictionary with list of images and total count
        """
        try:
            # Build prefix based on filters
            prefix = ""
            if device_id:
                prefix += f"{device_id}/"
                if prediction_type:
                    prefix += f"{prediction_type}/"
                    if date:
                        prefix += f"{date}/"
            
            # Execute the list operation
            response = self.s3_client.list_objects_v2(
                Bucket=self.bucket_name,
                Prefix=prefix
            )
            
            # Process results
            total_count = response.get('KeyCount', 0)
            
            # If it's a truncated response, we need to get the total count
            if response.get('IsTruncated', False):
                # For simplicity, we'll just report this as 'more than the current count'
                total_count = total_count + 1
            
            # Filter and process image objects
            images = []
            contents = response.get('Contents', [])
            
            # Apply manual pagination
            paginated_contents = contents[offset:offset+limit]
            
            for obj in paginated_contents:
                # Skip "directory" markers which end with /
                if obj['Key'].endswith('/'):
                    continue
                
                # Parse path components
                path_parts = obj['Key'].split('/')
                if len(path_parts) >= 4:  # Should have device_id/prediction_type/date/filename
                    img_device_id = path_parts[0]
                    img_prediction_type = path_parts[1]
                    img_date = path_parts[2]
                    img_filename = path_parts[3]
                    
                    # Generate pre-signed URL for temporary access
                    url = self.s3_client.generate_presigned_url(
                        'get_object',
                        Params={
                            'Bucket': self.bucket_name,
                            'Key': obj['Key']
                        },
                        ExpiresIn=3600  # URL valid for 1 hour
                    )
                    
                    # Add to results
                    images.append({
                        'key': obj['Key'],
                        'device_id': img_device_id,
                        'prediction_type': img_prediction_type,
                        'date': img_date,
                        'filename': img_filename,
                        'size': obj['Size'],
                        'last_modified': obj['LastModified'].isoformat(),
                        'url': url
                    })
            
            return {
                'total': total_count,
                'images': images
            }
        
        except Exception as e:
            logger.error(f"Error listing prediction images: {e}")
            return {
                'total': 0,
                'images': []
            }
    
    def get_image_url(self, image_key: str) -> Optional[str]:
        """
        Generate a pre-signed URL for accessing a specific image.
        
        Args:
            image_key: Full S3 key of the image
            
        Returns:
            Pre-signed URL for the image or None if error
        """
        try:
            # Generate pre-signed URL for temporary access
            url = self.s3_client.generate_presigned_url(
                'get_object',
                Params={
                    'Bucket': self.bucket_name,
                    'Key': image_key
                },
                ExpiresIn=3600  # URL valid for 1 hour
            )
            
            return url
        except Exception as e:
            logger.error(f"Error generating URL for image {image_key}: {e}")
            return None