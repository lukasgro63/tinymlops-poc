# api/services/drift_images_service.py
import os
import logging
import re
from datetime import datetime, timedelta
from typing import List, Dict, Any, Optional

import boto3
from botocore.client import Config

logger = logging.getLogger(__name__)

class DriftImagesService:
    def __init__(self):
        """Initialize S3 client for MinIO access."""
        self.s3_client = self._init_s3_client()
        self.bucket_name = "drift" # Using the single drift bucket
        # The endpoint URL that will be used for presigned URLs - accessible to browsers
        self.public_endpoint = "http://localhost:9000"

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
        """List all device IDs that have drift images."""
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
    
    def list_drift_types(self, device_id: str) -> List[str]:
        """List all drift types for a specific device."""
        try:
            # List 'folders' (prefixes) at the device level
            response = self.s3_client.list_objects_v2(
                Bucket=self.bucket_name,
                Prefix=f"{device_id}/",
                Delimiter='/'
            )
            
            # Extract drift types from CommonPrefixes
            drift_types = []
            if 'CommonPrefixes' in response:
                for prefix in response['CommonPrefixes']:
                    drift_type = prefix['Prefix'].split('/')[1]
                    
                    # Map from storage drift type to API drift type using normalized names
                    # The UI and API use detector names, but storage uses drift type names
                    if drift_type == "knn_distance":
                        # Map to the detector name used by the client
                        drift_types.append("KNNDistanceMonitor")
                    elif drift_type == "confidence":
                        drift_types.append("EWMAConfidenceMonitor")
                    elif drift_type == "feature":
                        drift_types.append("FeatureMonitor")
                    elif drift_type == "distribution":
                        drift_types.append("DistributionMonitor")
                    elif drift_type == "outlier":
                        drift_types.append("OutlierMonitor")
                    else:
                        # Fallback to the original name
                        drift_types.append(drift_type)
            
            return drift_types
        except Exception as e:
            logger.error(f"Error listing drift types for device {device_id}: {e}")
            return []
    
    def list_dates(self, device_id: str, drift_type: str) -> List[str]:
        """List all dates with drift images for a device and drift type."""
        try:
            # Convert from API drift type name to storage drift type
            storage_drift_type = self._convert_api_drift_type_to_storage(drift_type)
            logger.info(f"Converting API drift type {drift_type} to storage type {storage_drift_type}")
            
            # List 'folders' (prefixes) at the drift type level
            response = self.s3_client.list_objects_v2(
                Bucket=self.bucket_name,
                Prefix=f"{device_id}/{storage_drift_type}/",
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
            logger.error(f"Error listing dates for device {device_id} and type {drift_type}: {e}")
            return []
            
    def _convert_api_drift_type_to_storage(self, drift_type: str) -> str:
        """Convert from API drift type (detector name) to storage drift type."""
        # Map from API/UI detector name to storage drift type
        if drift_type == "KNNDistanceMonitor":
            return "knn_distance"
        elif drift_type == "EWMAConfidenceMonitor":
            return "confidence"
        elif drift_type == "FeatureMonitor":
            return "feature"
        elif drift_type == "DistributionMonitor":
            return "distribution"
        elif drift_type == "OutlierMonitor":
            return "outlier"
        # For "unknown" and any other types, keep as is
        return drift_type.lower()
        
    def _convert_storage_drift_type_to_api(self, drift_type: str) -> str:
        """Convert from storage drift type to API drift type (detector name)."""
        # Map from storage drift type to API/UI detector name
        drift_type_lower = drift_type.lower()
        if drift_type_lower == "knn_distance":
            return "KNNDistanceMonitor"
        elif drift_type_lower == "confidence":
            return "EWMAConfidenceMonitor"
        elif drift_type_lower == "feature":
            return "FeatureMonitor"
        elif drift_type_lower == "distribution":
            return "DistributionMonitor"
        elif drift_type_lower == "outlier":
            return "OutlierMonitor"
        # For "unknown" and any other types, return capitalized
        return drift_type.capitalize()
    
    def list_images(
        self, 
        device_id: Optional[str] = None, 
        drift_type: Optional[str] = None, 
        date: Optional[str] = None,
        limit: int = 100,
        offset: int = 0,
        sort_order: Optional[str] = "desc"  # Neuer Parameter für Sortierreihenfolge (asc oder desc)
    ) -> Dict[str, Any]:
        """
        List drift images with optional filtering.
        
        Args:
            device_id: Filter by device ID
            drift_type: Filter by drift type
            date: Filter by date (YYYYMMDD format)
            limit: Maximum number of images to return
            offset: Number of images to skip
            sort_order: Sort direction ('asc' or 'desc'), defaults to 'desc' (newest first)
            
        Returns:
            Dictionary with list of images and total count
        """
        try:
            # Build prefix based on filters
            prefix = ""
            if device_id:
                prefix += f"{device_id}/"
                if drift_type:
                    # Convert from API drift type name to storage drift type
                    storage_drift_type = self._convert_api_drift_type_to_storage(drift_type)
                    logger.info(f"For list_images: Converting API drift type {drift_type} to storage type {storage_drift_type}")
                    prefix += f"{storage_drift_type}/"
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
            
            # Sortieren der Ergebnisse nach last modified timestamp
            # Die S3-API gibt bereits ein 'LastModified' Feld für jedes Objekt zurück
            sortable_contents = []
            for obj in contents:
                # Skip "directory" markers which end with /
                if obj['Key'].endswith('/'):
                    continue
                
                # Skip non-image files (check if it's an image by extension)
                if not any(obj['Key'].lower().endswith(ext) for ext in ['.jpg', '.jpeg', '.png', '.gif']):
                    continue
                
                sortable_contents.append(obj)
            
            # Aktualisiere die Gesamtzahl basierend auf den tatsächlich verfügbaren Bildern
            total_count = len(sortable_contents)
            
            # Sortierung anwenden
            reverse = sort_order.lower() != 'asc'  # Bei desc (neueste zuerst) brauchen wir reverse=True
            sortable_contents.sort(key=lambda x: x['LastModified'], reverse=reverse)
            
            # Nach dem Sortieren die Pagination anwenden
            paginated_contents = sortable_contents[offset:offset+limit]
            
            for obj in paginated_contents:
                # Wir haben bereits Directory-Marker und Nicht-Bilddateien beim Sortieren übersprungen
                
                # Parse path components
                path_parts = obj['Key'].split('/')
                if len(path_parts) >= 4:  # Should have device_id/drift_type/date/filename
                    img_device_id = path_parts[0]
                    img_drift_type = path_parts[1]
                    img_date = path_parts[2]
                    img_filename = path_parts[3]
                    
                    # Use our API proxy URL instead of direct MinIO access
                    url = f"/api/drift-images/image/{obj['Key']}"
                    
                    # Get the event ID from metadata if available
                    event_id = None
                    if len(path_parts) > 4 and path_parts[3].startswith("event_"):
                        event_id = path_parts[3].replace("event_", "")
                    
                    # Convert from storage drift type to API drift type for consistent UI display
                    api_drift_type = self._convert_storage_drift_type_to_api(img_drift_type)
                    logger.info(f"Converting storage drift type {img_drift_type} to API type {api_drift_type}")
                    
                    # Add to results
                    images.append({
                        'key': obj['Key'],
                        'device_id': img_device_id,
                        'drift_type': api_drift_type,  # Use the API drift type
                        'date': img_date,
                        'filename': img_filename,
                        'size': obj['Size'],
                        'last_modified': obj['LastModified'].isoformat(),
                        'url': url,
                        'event_id': event_id
                    })
            
            return {
                'total': total_count,
                'images': images
            }
        
        except Exception as e:
            logger.error(f"Error listing drift images: {e}")
            return {
                'total': 0,
                'images': []
            }
    
    def get_image_url(self, image_key: str) -> Optional[str]:
        """
        Generate a URL for accessing a specific image.
        
        Args:
            image_key: Full S3 key of the image
            
        Returns:
            API proxy URL for the image or None if error
        """
        try:
            # Use our API proxy URL instead of direct MinIO access
            url = f"/api/drift-images/image/{image_key}"
            
            return url
        except Exception as e:
            logger.error(f"Error generating URL for image {image_key}: {e}")
            return None