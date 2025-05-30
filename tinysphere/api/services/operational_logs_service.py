# api/services/operational_logs_service.py
import os
import logging
from typing import List, Dict, Any, Optional

import boto3
from botocore.client import Config

logger = logging.getLogger(__name__)

class OperationalLogsService:
    def __init__(self):
        """Initialize S3 client for MinIO access."""
        self.s3_client = self._init_s3_client()
        self.bucket_name = "data-logs"
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
        """List all device IDs that have operational logs."""
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
    
    def list_session_ids(self, device_id: str) -> List[str]:
        """List all session IDs for a specific device."""
        try:
            # List 'folders' (prefixes) at the device level
            response = self.s3_client.list_objects_v2(
                Bucket=self.bucket_name,
                Prefix=f"{device_id}/",
                Delimiter='/'
            )
            
            # Extract session IDs from CommonPrefixes
            session_ids = []
            if 'CommonPrefixes' in response:
                for prefix in response['CommonPrefixes']:
                    session_id = prefix['Prefix'].split('/')[1]
                    session_ids.append(session_id)
            
            return session_ids
        except Exception as e:
            logger.error(f"Error listing session IDs for device {device_id}: {e}")
            return []
            
    def list_log_types(self, device_id: str) -> List[str]:
        """List all log types for a specific device.
        
        This extracts log types from filenames, assuming the format:
        device_id/session_id/log_type_*.log
        """
        try:
            # Get all logs for this device
            response = self.s3_client.list_objects_v2(
                Bucket=self.bucket_name,
                Prefix=f"{device_id}/"
            )
            
            # Extract log types from filenames
            log_types = set()
            if 'Contents' in response:
                for obj in response['Contents']:
                    # Skip directory markers
                    if obj['Key'].endswith('/'):
                        continue
                        
                    # Get filename from the key
                    filename = obj['Key'].split('/')[-1]
                    
                    # Extract log type from filename
                    # Assuming log files are named like: type_*.log or type-*.log
                    parts = filename.split('_', 1)
                    if len(parts) > 1:
                        log_type = parts[0]
                        log_types.add(log_type)
                    else:
                        # Try dash separator
                        parts = filename.split('-', 1)
                        if len(parts) > 1:
                            log_type = parts[0]
                            log_types.add(log_type)
            
            return sorted(list(log_types))
        except Exception as e:
            logger.error(f"Error listing log types for device {device_id}: {e}")
            return []
    
    def list_logs(
        self, 
        device_id: Optional[str] = None, 
        session_id: Optional[str] = None,
        log_type: Optional[str] = None,
        limit: int = 100,
        offset: int = 0,
        sort_order: Optional[str] = "desc",
        start_date: Optional[str] = None,
        end_date: Optional[str] = None
    ) -> Dict[str, Any]:
        """
        List operational logs with optional filtering.
        
        Args:
            device_id: Filter by device ID
            session_id: Filter by session ID
            log_type: Filter by log type
            limit: Maximum number of logs to return
            offset: Number of logs to skip
            sort_order: Sort direction ('asc' or 'desc'), defaults to 'desc' (newest first)
            start_date: Filter logs after this date (YYYY-MM-DD format)
            end_date: Filter logs before this date (YYYY-MM-DD format)
            
        Returns:
            Dictionary with list of logs and total count
        """
        try:
            # Build prefix based on filters
            prefix = ""
            if device_id:
                prefix += f"{device_id}/"
                if session_id:
                    prefix += f"{session_id}/"
            
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
            
            # Filter and process log objects
            logs = []
            contents = response.get('Contents', [])
            
            # For date filtering
            start_timestamp = None
            end_timestamp = None
            
            # Convert date strings to timestamps if provided
            import datetime
            if start_date:
                try:
                    start_datetime = datetime.datetime.strptime(start_date, '%Y-%m-%d').replace(tzinfo=datetime.timezone.utc)
                    start_timestamp = start_datetime.timestamp()
                except ValueError:
                    logger.warning(f"Invalid start_date format: {start_date}, expected YYYY-MM-DD")
            
            if end_date:
                try:
                    # Set to end of day (23:59:59) for end_date
                    end_datetime = datetime.datetime.strptime(end_date, '%Y-%m-%d')
                    end_datetime = end_datetime.replace(hour=23, minute=59, second=59, tzinfo=datetime.timezone.utc)
                    end_timestamp = end_datetime.timestamp()
                except ValueError:
                    logger.warning(f"Invalid end_date format: {end_date}, expected YYYY-MM-DD")
            
            # Sort the results by last modified timestamp
            sortable_contents = []
            for obj in contents:
                # Skip "directory" markers which end with /
                if obj['Key'].endswith('/'):
                    continue
                
                # If filtering by log_type, check if the filename matches
                if log_type:
                    filename = obj['Key'].split('/')[-1]
                    # Check if the filename starts with the log_type
                    if not (filename.startswith(f"{log_type}_") or filename.startswith(f"{log_type}-")):
                        continue
                
                # Apply date filters if provided
                if start_timestamp or end_timestamp:
                    obj_timestamp = obj['LastModified'].timestamp()
                    
                    if start_timestamp and obj_timestamp < start_timestamp:
                        continue  # Skip if before start date
                        
                    if end_timestamp and obj_timestamp > end_timestamp:
                        continue  # Skip if after end date
                
                sortable_contents.append(obj)
            
            # Update the total count based on the actually available logs
            total_count = len(sortable_contents)
            
            # Apply sorting
            reverse = sort_order.lower() != 'asc'  # For desc (newest first) we need reverse=True
            sortable_contents.sort(key=lambda x: x['LastModified'], reverse=reverse)
            
            # Apply pagination
            paginated_contents = sortable_contents[offset:offset+limit]
            
            for obj in paginated_contents:
                # We already skipped directory markers during sorting
                
                # Parse path components
                path_parts = obj['Key'].split('/')
                if len(path_parts) >= 3:  # Should have device_id/session_id/filename
                    log_device_id = path_parts[0]
                    log_session_id = path_parts[1]
                    log_filename = path_parts[2]
                    
                    # Use our API proxy URL instead of direct MinIO access
                    url = f"/api/operational-logs/log/{obj['Key']}"
                    
                    # Extract log type from filename for the frontend
                    extracted_log_type = None
                    if '_' in log_filename:
                        extracted_log_type = log_filename.split('_', 1)[0]
                    elif '-' in log_filename:
                        extracted_log_type = log_filename.split('-', 1)[0]
                    
                    # Determine if this is a consolidated log
                    is_consolidated = "operational_log_consolidated_" in log_filename
                    
                    # Add to results
                    logs.append({
                        'key': obj['Key'],
                        'device_id': log_device_id,
                        'session_id': log_session_id,
                        'filename': log_filename,
                        'size': obj['Size'],
                        'last_modified': obj['LastModified'].isoformat(),
                        'url': url,
                        'log_type': "consolidated" if is_consolidated else extracted_log_type,
                        'is_consolidated': is_consolidated
                    })
            
            return {
                'total': total_count,
                'logs': logs
            }
        
        except Exception as e:
            logger.error(f"Error listing operational logs: {e}")
            return {
                'total': 0,
                'logs': []
            }
    
    def get_log_url(self, log_key: str) -> Optional[str]:
        """
        Generate a pre-signed URL for accessing a specific log.
        
        Args:
            log_key: Full S3 key of the log
            
        Returns:
            Pre-signed URL for the log or None if error
        """
        try:
            # Use our API proxy URL instead of direct MinIO access
            url = f"/api/operational-logs/log/{log_key}"
            
            return url
        except Exception as e:
            logger.error(f"Error generating URL for log {log_key}: {e}")
            return None
            
    def delete_logs(self, device_id: str, session_id: Optional[str] = None) -> Dict[str, Any]:
        """
        Delete logs for a specific device and optionally a specific session.
        
        Args:
            device_id: The device ID to delete logs for
            session_id: Optional session ID to limit deletion to a specific session
            
        Returns:
            Dictionary with deletion results
        """
        try:
            # Build the prefix to list objects to delete
            prefix = f"{device_id}/"
            if session_id:
                prefix += f"{session_id}/"
                
            # List objects to delete
            response = self.s3_client.list_objects_v2(
                Bucket=self.bucket_name,
                Prefix=prefix
            )
            
            if 'Contents' not in response:
                return {
                    "status": "success",
                    "message": f"No logs found for {prefix}",
                    "deleted_count": 0
                }
            
            # Prepare deletion objects list
            objects_to_delete = [{"Key": obj["Key"]} for obj in response["Contents"]]
            
            # Delete the objects
            if objects_to_delete:
                delete_response = self.s3_client.delete_objects(
                    Bucket=self.bucket_name,
                    Delete={"Objects": objects_to_delete}
                )
                
                deleted_count = len(delete_response.get("Deleted", []))
                error_count = len(delete_response.get("Errors", []))
                
                if error_count > 0:
                    return {
                        "status": "partial",
                        "message": f"Deleted {deleted_count} logs, but encountered {error_count} errors",
                        "deleted_count": deleted_count,
                        "error_count": error_count,
                        "errors": delete_response.get("Errors", [])
                    }
                else:
                    return {
                        "status": "success",
                        "message": f"Successfully deleted {deleted_count} logs",
                        "deleted_count": deleted_count
                    }
            else:
                return {
                    "status": "success",
                    "message": "No logs to delete",
                    "deleted_count": 0
                }
                
        except Exception as e:
            logger.error(f"Error deleting logs for {device_id}/{session_id if session_id else ''}: {e}")
            return {
                "status": "error",
                "message": f"Failed to delete logs: {str(e)}",
                "deleted_count": 0
            }