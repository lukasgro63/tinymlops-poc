import json
import logging
import os
import boto3
from botocore.exceptions import ClientError
from pathlib import Path
from typing import Any, Dict, List

from tinysphere.importer.transformers.base import DataTransformer

class OperationalLogsTransformer(DataTransformer):
    """Transformer for raw operational log files.
    
    This transformer handles packages containing raw operational log files (JSONL)
    and stores them in the MinIO S3 'data_logs' bucket organized by device_id and session_id.
    """
    
    def __init__(self):
        # Initialize logger
        self.logger = logging.getLogger(f"{__name__}.{self.__class__.__name__}")
        
        # Configure S3 client for MinIO
        self.s3_client = self._initialize_s3_client()
        
        # Ensure the data-logs bucket exists
        self._ensure_bucket_exists("data-logs")
        
    def _initialize_s3_client(self):
        """Initialize the S3 client for MinIO."""
        try:
            # Get configuration from environment variables
            endpoint_url = os.environ.get("MINIO_ENDPOINT", "http://minio:9000")
            access_key = os.environ.get("MINIO_ACCESS_KEY", "minio")
            secret_key = os.environ.get("MINIO_SECRET_KEY", "minio123")
            
            # Create S3 client
            s3_client = boto3.client(
                's3',
                endpoint_url=endpoint_url,
                aws_access_key_id=access_key,
                aws_secret_access_key=secret_key,
                config=boto3.session.Config(signature_version='s3v4'),
                region_name='us-east-1'
            )
            
            self.logger.info(f"Initialized S3 client for MinIO at {endpoint_url}")
            return s3_client
            
        except Exception as e:
            self.logger.error(f"Failed to initialize S3 client: {str(e)}")
            return None
            
    def _ensure_bucket_exists(self, bucket_name: str):
        """Ensure that the bucket exists, creating it if necessary."""
        if not self.s3_client:
            self.logger.warning("S3 client not initialized, cannot ensure bucket exists")
            return
            
        try:
            # Check if bucket exists
            self.s3_client.head_bucket(Bucket=bucket_name)
            self.logger.info(f"Bucket '{bucket_name}' already exists")
        except ClientError as e:
            # If bucket doesn't exist, create it
            if e.response['Error']['Code'] == '404':
                try:
                    self.s3_client.create_bucket(Bucket=bucket_name)
                    self.logger.info(f"Created bucket '{bucket_name}'")
                except Exception as create_error:
                    self.logger.error(f"Failed to create bucket '{bucket_name}': {str(create_error)}")
            else:
                self.logger.error(f"Error checking bucket '{bucket_name}': {str(e)}")
    
    def can_transform(self, package_type: str, files: List[Path]) -> bool:
        """Check if this transformer can handle the package.
        
        Args:
            package_type: Type of the package
            files: List of files in the package
            
        Returns:
            True if this transformer can handle the package, False otherwise
        """
        # Check package type
        if package_type.lower() != "operational_logs":
            return False
            
        # Check for operational log files
        log_files = [f for f in files if f.name.startswith("operational_log_") and f.suffix == ".jsonl"]
        has_log_files = len(log_files) > 0
        
        # Check for metadata file
        metadata_files = [f for f in files if f.name.lower() == "logs_metadata.json"]
        has_metadata = len(metadata_files) > 0
        
        if has_log_files and has_metadata:
            self.logger.info(f"Found {len(log_files)} operational log files with metadata in package")
            return True
            
        return False
    
    def transform(self, package_id: str, device_id: str, files: List[Path], metadata: Dict[str, Any]) -> Dict[str, Any]:
        """Transform the package by storing operational logs in the MinIO data_logs bucket.
        
        Args:
            package_id: ID of the package
            device_id: ID of the device that sent the package
            files: List of files in the package
            metadata: Package metadata
            
        Returns:
            Dictionary with transformation results
        """
        if not self.s3_client:
            return {"status": "error", "message": "S3 client not initialized"}
            
        # Find metadata file and operational log files
        metadata_files = [f for f in files if f.name.lower() == "logs_metadata.json"]
        log_files = [f for f in files if f.name.startswith("operational_log_") and f.suffix == ".jsonl"]
        
        if not metadata_files:
            return {"status": "error", "message": "No logs metadata file found in package"}
            
        if not log_files:
            return {"status": "error", "message": "No operational log files found in package"}
            
        # Read metadata
        metadata_file = metadata_files[0]
        logs_metadata = {}
        
        try:
            with open(metadata_file, 'r') as f:
                logs_metadata = json.load(f)
        except Exception as e:
            self.logger.error(f"Failed to read logs metadata file: {str(e)}")
            return {"status": "error", "message": f"Failed to read logs metadata: {str(e)}"}
            
        # Get session ID from metadata or from log filenames
        session_id = logs_metadata.get("session_id")
        if not session_id and log_files:
            # Extract session ID from the first log file name
            # Format: operational_log_TIMESTAMP_SESSION_ID.jsonl
            filename = log_files[0].name
            parts = filename.split('_')
            if len(parts) >= 4:
                session_id = parts[3].split('.')[0]  # Remove .jsonl extension
                
        if not session_id:
            session_id = "unknown_session"
            
        # Upload log files to S3
        bucket_name = "data-logs"
        uploaded_files = []
        
        try:
            for log_file in log_files:
                # Construct S3 object key: device_id/session_id/filename
                object_key = f"{device_id}/{session_id}/{log_file.name}"
                
                # Upload file
                self.s3_client.upload_file(
                    Filename=str(log_file),
                    Bucket=bucket_name,
                    Key=object_key
                )
                
                uploaded_files.append(object_key)
                self.logger.info(f"Uploaded {log_file.name} to s3://{bucket_name}/{object_key}")
                
            # Create a success record in the database if needed
            # TODO: Implement database record creation if required
            
            return {
                "status": "success",
                "message": f"Uploaded {len(uploaded_files)} operational log files to MinIO",
                "bucket": bucket_name,
                "device_id": device_id,
                "session_id": session_id,
                "files": uploaded_files
            }
            
        except Exception as e:
            self.logger.error(f"Failed to upload operational logs: {str(e)}")
            return {"status": "error", "message": f"Failed to upload operational logs: {str(e)}"}