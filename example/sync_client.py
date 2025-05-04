#!/usr/bin/env python3
"""
Fixed version of SyncClient that works properly with TinySphere
"""

import json
import time
from pathlib import Path
from typing import Any, Dict, Optional, Union

import requests
from tinylcm.client.sync_client import SyncClient
from tinylcm.utils.versioning import calculate_file_hash


class SyncClient(SyncClient):
    """
    Fixed version of SyncClient that properly handles file uploads to TinySphere.
    This version removes Content-Type header from uploads to avoid issues with multipart form data.
    """
    
    def send_package(self, package_id: str) -> bool:
        """
        Send a package to the TinySphere server with fixed multipart form handling.
        
        Args:
            package_id (str): The ID of the package to send
            
        Returns:
            bool: True if package was sent successfully, False otherwise
        """
        self.logger.info(f"Preparing to send package: {package_id}")
        try:
            # Find the package
            packages = self.sync_interface.list_packages(filter_func=lambda p: p["package_id"] == package_id)
            if not packages:
                raise Exception(f"Package not found: {package_id}")
            
            package_meta = packages[0]
            package_dir = Path(self.sync_interface.packages_dir)
            package_files = list(package_dir.glob(f"{package_id}_*.tar.gz")) + \
                           list(package_dir.glob(f"{package_id}_*.zip")) + \
                           list(package_dir.glob(f"{package_id}_*.tar"))
            
            if not package_files:
                raise Exception(f"Package file not found for ID: {package_id}")
            
            # Get the most recent package file
            package_files.sort(key=lambda p: p.stat().st_mtime, reverse=True)
            package_file = package_files[0]
            self.logger.debug(f"Found package file: {package_file}")
            
            # Calculate file hash
            file_hash = calculate_file_hash(package_file)
            
            # Prepare metadata
            metadata = {
                'device_id': self.device_id,
                'package_id': package_id,
                'package_type': package_meta.get('package_type', 'unknown'),
                'hash': file_hash,
                'timestamp': time.time()
            }
            
            # Convert metadata to JSON string
            metadata_str = json.dumps(metadata)
            
            try:
                # Open the file for upload
                with open(package_file, 'rb') as file_obj:
                    # Create copy of headers without Content-Type
                    upload_headers = self.headers.copy()
                    if 'Content-Type' in upload_headers:
                        del upload_headers['Content-Type']
                    
                    # Construct the URL manually
                    url = f"{self.server_url}/api/packages/upload"
                    
                    # Send the request directly
                    response = requests.post(
                        url=url,
                        files={'package': (package_file.name, file_obj, 'application/octet-stream')},
                        data={'metadata': metadata_str},
                        headers=upload_headers
                    )
                    
                    if response.status_code == 200:
                        self.logger.info(f"Successfully sent package {package_id}")
                        self.sync_interface.mark_as_synced(
                            package_id=package_id,
                            sync_time=time.time(),
                            server_id=response.json().get('server_id', 'unknown'),
                            status="success"
                        )
                        return True
                    else:
                        error_msg = f"Package upload failed: {response.status_code} - {response.text}"
                        self.logger.error(error_msg)
                        self.sync_interface.mark_as_synced(
                            package_id=package_id,
                            sync_time=time.time(),
                            server_id="none", 
                            status="error"
                        )
                        return False
            except Exception as e:
                error_msg = f"Package upload request failed: {str(e)}"
                self.logger.error(error_msg)
                return False
        except Exception as e:
            self.logger.error(f"Error preparing package {package_id}: {str(e)}")
            return False