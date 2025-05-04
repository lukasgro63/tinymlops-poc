import json
import os
import platform
import re
import time
from pathlib import Path
from typing import Any, Dict, List, Optional, Union

import requests

from tinylcm.client.connection_manager import ConnectionManager
from tinylcm.client.sync_interface import SyncInterface
from tinylcm.utils.errors import ConnectionError, SyncError
from tinylcm.utils.logging import setup_logger
from tinylcm.utils.versioning import calculate_file_hash


class SyncClient:
    def __init__(self, server_url: str, api_key: str, device_id: str, sync_interface: Optional[SyncInterface] = None, sync_dir: Optional[Union[str, Path]] = None, max_retries: int = 3, connection_timeout: float = 300.0, auto_register: bool = True):
        if not self.validate_server_url(server_url):
            raise ValueError(f"Invalid server URL: {server_url}")
        self.logger = setup_logger(f"{__name__}.{self.__class__.__name__}")
        self.server_url = server_url.rstrip('/')
        self.api_key = api_key
        self.device_id = device_id
        self.auto_register = auto_register
        if sync_interface is None:
            if sync_dir is None:
                raise ValueError("Either sync_interface or sync_dir must be provided")
            self.sync_interface = SyncInterface(sync_dir=sync_dir)
        else:
            self.sync_interface = sync_interface
        self.headers = {
            "Authorization": f"Bearer {api_key}",
            "Content-Type": "application/json",
            "X-Device-ID": device_id
        }
        self.connection_manager = ConnectionManager(server_url=server_url, max_retries=max_retries, connection_timeout=connection_timeout, headers=self.headers)
        self.logger.info(f"Initialized sync client for server: {server_url}")
        if auto_register:
            try:
                self.register_device()
            except ConnectionError as e:
                self.logger.warning(f"Auto-registration failed: {str(e)}. Will retry on next connection.")
    
    @staticmethod
    def validate_server_url(url: str) -> bool:
        if not url:
            return False
        url_pattern = re.compile(r'^https?://.*')
        if not url_pattern.match(url):
            return False
        return True
    
    def _get_device_info(self) -> Dict[str, Any]:
        import socket
        try:
            hostname = socket.gethostname()
            ip_address = socket.gethostbyname(hostname)
        except Exception:
            hostname = "unknown"
            ip_address = "unknown"
        return {
            "device_id": self.device_id,
            "hostname": hostname,
            "ip_address": ip_address,
            "platform": platform.platform(),
            "python_version": platform.python_version(),
            "tinylcm_version": self._get_tinylcm_version()
        }
    
    def _get_tinylcm_version(self) -> str:
        try:
            import tinylcm
            return tinylcm.__version__
        except (ImportError, AttributeError):
            return "unknown"
    
    def register_device(self) -> bool:
        self.logger.info(f"Registering device {self.device_id} with server")
        registration_data = {
            "device_id": self.device_id,
            "device_info": self._get_device_info(),
            "registration_time": time.time()
        }
        try:
            response = self.connection_manager.execute_request(method="POST", endpoint="devices/register", json=registration_data)
            if response.status_code == 200:
                self.logger.info(f"Successfully registered device {self.device_id}")
                return True
            else:
                error_msg = f"Registration failed: {response.status_code} - {response.text}"
                self.logger.error(error_msg)
                raise ConnectionError(error_msg)
        except requests.RequestException as e:
            error_msg = f"Registration request failed: {str(e)}"
            self.logger.error(error_msg)
            raise ConnectionError(error_msg)
    
    def check_server_status(self) -> Dict[str, Any]:
        self.logger.debug("Checking server status")
        try:
            response = self.connection_manager.execute_request(method="GET", endpoint="status")
            if response.status_code == 200:
                return response.json()
            else:
                error_msg = f"Server status check failed: {response.status_code} - {response.text}"
                self.logger.error(error_msg)
                raise ConnectionError(error_msg)
        except requests.RequestException as e:
            error_msg = f"Server status request failed: {str(e)}"
            self.logger.error(error_msg)
            raise ConnectionError(error_msg)
    
    def send_package(self, package_id: str) -> bool:
        self.logger.info(f"Preparing to send package: {package_id}")
        try:
            # Find the package
            packages = self.sync_interface.list_packages(filter_func=lambda p: p["package_id"] == package_id)
            if not packages:
                raise SyncError(f"Package not found: {package_id}")
            
            package_meta = packages[0]
            package_dir = Path(self.sync_interface.packages_dir)
            package_files = list(package_dir.glob(f"{package_id}_*.tar.gz")) + list(package_dir.glob(f"{package_id}_*.zip")) + list(package_dir.glob(f"{package_id}_*.tar"))
            
            if not package_files:
                raise SyncError(f"Package file not found for ID: {package_id}")
            
            package_files.sort(key=lambda p: p.stat().st_mtime, reverse=True)
            package_file = package_files[0]
            self.logger.debug(f"Found package file: {package_file}")
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
                    # FIX: Use direct requests approach instead of ConnectionManager
                    # for file uploads to avoid Content-Type header issues
                    
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
                        raise ConnectionError(error_msg)
            except Exception as e:
                error_msg = f"Package upload request failed: {str(e)}"
                self.logger.error(error_msg)
                raise ConnectionError(error_msg)
        except SyncError as e:
            self.logger.error(f"Error preparing package {package_id}: {str(e)}")
            raise
    
    def sync_all_pending_packages(self) -> List[Dict[str, Any]]:
        self.logger.info("Synchronizing all pending packages")
        packages = self.sync_interface.list_packages(include_synced=False)
        if not packages:
            self.logger.info("No pending packages to synchronize")
            return []
        self.logger.info(f"Found {len(packages)} pending packages")
        results = []
        for package in packages:
            package_id = package["package_id"]
            result = {
                "package_id": package_id,
                "success": False,
                "error": None
            }
            try:
                success = self.send_package(package_id)
                result["success"] = success
            except Exception as e:
                self.logger.error(f"Failed to sync package {package_id}: {str(e)}")
                result["error"] = str(e)
            results.append(result)
        success_count = sum(1 for r in results if r["success"])
        self.logger.info(f"Sync complete: {success_count}/{len(results)} packages successful")
        return results
    
    def get_sync_status(self) -> Dict[str, Any]:
        all_packages = self.sync_interface.list_packages(include_synced=True)
        pending = sum(1 for p in all_packages if "sync_status" not in p)
        synced = sum(1 for p in all_packages if p.get("sync_status") == "success")
        failed = sum(1 for p in all_packages if p.get("sync_status") == "error")
        types = {}
        for package in all_packages:
            pkg_type = package.get("package_type", "unknown")
            if pkg_type not in types:
                types[pkg_type] = 0
            types[pkg_type] += 1
        return {
            "total_packages": len(all_packages),
            "pending_packages": pending,
            "synced_packages": synced,
            "failed_packages": failed,
            "package_types": types,
            "connection_status": self.connection_manager.connection_status,
            "last_connection_time": self.connection_manager.last_connection_time
        }
    
    def close(self) -> None:
        self.logger.info("Closing sync client")