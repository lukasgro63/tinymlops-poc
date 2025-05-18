import json
import os
import platform
import re
import shutil
import tempfile
import time
import uuid
from datetime import datetime
from pathlib import Path
from typing import Any, Dict, List, Optional, Tuple, Union

import requests

from tinylcm.client.connection_manager import ConnectionManager
from tinylcm.client.sync_interface import SyncInterface
from tinylcm.utils.errors import TinyLCMConnectionError, SyncError
from tinylcm.utils.logging import setup_logger
from tinylcm.utils.versioning import calculate_file_hash

# Optional import for quarantine support
try:
    from tinylcm.core.quarantine.buffer import QuarantineBuffer
    QUARANTINE_AVAILABLE = True
except ImportError:
    QUARANTINE_AVAILABLE = False

# Optional import for drift events support
try:
    from tinylcm.core.drift_detection import AutonomousDriftDetector
    DRIFT_DETECTION_AVAILABLE = True
except ImportError:
    DRIFT_DETECTION_AVAILABLE = False

# Optional import for adaptation tracking
try:
    from tinylcm.core.adaptation_tracker import AdaptationTracker
    ADAPTATION_TRACKER_AVAILABLE = True
except ImportError:
    ADAPTATION_TRACKER_AVAILABLE = False

# Optional import for state management and snapshots
try:
    from tinylcm.core.state_manager import AdaptiveStateManager
    STATE_MANAGER_AVAILABLE = True
except ImportError:
    STATE_MANAGER_AVAILABLE = False

# Optional import for geolocation
try:
    from tinylcm.utils.geolocation import Geolocator
    GEOLOCATION_AVAILABLE = True
except ImportError:
    GEOLOCATION_AVAILABLE = False


class SyncClient:
    def __init__(
        self,
        server_url: str,
        api_key: str,
        device_id: str,
        sync_interface: Optional[SyncInterface] = None,
        sync_dir: Optional[Union[str, Path]] = None,
        max_retries: int = 3,
        connection_timeout: float = 300.0,
        auto_register: bool = True,
        quarantine_buffer: Optional['QuarantineBuffer'] = None,
        drift_detectors: Optional[List[Any]] = None,
        adaptation_tracker: Optional['AdaptationTracker'] = None,
        state_manager: Optional['AdaptiveStateManager'] = None,
        enable_prediction_images: bool = False,
        enable_geolocation: bool = False,
        geolocation_api_key: Optional[str] = None,
        geolocation_update_interval: int = 3600,
        geolocation_fallback_coordinates: Optional[Tuple[float, float]] = None
    ):
        if not self.validate_server_url(server_url):
            raise ValueError(f"Invalid server URL: {server_url}")
        self.logger = setup_logger(f"{__name__}.{self.__class__.__name__}")
        self.server_url = server_url.rstrip('/')
        self.api_key = api_key
        self.device_id = device_id
        self.auto_register = auto_register
        self.enable_prediction_images = enable_prediction_images
        self.pending_prediction_images = []
        
        # Geolocation setup
        self.enable_geolocation = enable_geolocation
        self.geolocation_update_interval = geolocation_update_interval
        self.geolocation_fallback_coordinates = geolocation_fallback_coordinates
        self.last_geolocation_update = 0
        self.current_location = None
        
        # Initialize geolocator if enabled
        if enable_geolocation and GEOLOCATION_AVAILABLE:
            self.geolocator = Geolocator(
                api_key=geolocation_api_key, 
                fallback_coords=geolocation_fallback_coordinates
            )
            self.logger.info("Geolocation service initialized")
        else:
            self.geolocator = None
            if enable_geolocation and not GEOLOCATION_AVAILABLE:
                self.logger.warning("Geolocation requested but not available. Please install required packages.")

        # Set up sync interface
        if sync_interface is None:
            if sync_dir is None:
                raise ValueError("Either sync_interface or sync_dir must be provided")
            self.sync_interface = SyncInterface(sync_dir=sync_dir)
        else:
            self.sync_interface = sync_interface

        # Set up quarantine buffer reference
        self.quarantine_buffer = quarantine_buffer

        # Set up drift detectors reference
        self.drift_detectors = drift_detectors or []

        # Set up adaptation tracker reference
        self.adaptation_tracker = adaptation_tracker

        # Set up state manager reference
        self.state_manager = state_manager

        # Set up headers
        self.headers = {
            "Authorization": f"Bearer {api_key}",
            "Content-Type": "application/json",
            "X-Device-ID": device_id
        }

        # Set up connection manager
        self.connection_manager = ConnectionManager(
            server_url=server_url,
            max_retries=max_retries,
            connection_timeout=connection_timeout,
            headers=self.headers
        )

        # Log image transfer status
        image_transfer_status = "enabled" if enable_prediction_images else "disabled"
        geo_status = "enabled" if enable_geolocation and GEOLOCATION_AVAILABLE else "disabled"
        self.logger.info(f"Initialized sync client for server: {server_url} (prediction image transfer: {image_transfer_status}, geolocation: {geo_status})")

        # Auto-register if needed
        if auto_register:
            try:
                self.register_device()
            except TinyLCMConnectionError as e:
                self.logger.warning(f"Auto-registration failed: {str(e)}. Will retry on next connection.")
    
    @staticmethod
    def validate_server_url(url: str) -> bool:
        if not url:
            return False
        url_pattern = re.compile(r'^https?://.*')
        if not url_pattern.match(url):
            return False
        return True
    
    def _update_geolocation(self, force: bool = False) -> Dict[str, Union[float, str]]:
        """
        Update device geolocation information if enabled.
        
        Args:
            force: If True, update even if interval hasn't passed
            
        Returns:
            Dict with location data
        """
        # Return cached location if not time to update yet
        current_time = time.time()
        if not force and self.current_location and (current_time - self.last_geolocation_update) < self.geolocation_update_interval:
            return self.current_location
            
        # Return empty location if geolocator not available
        if not self.enable_geolocation or not GEOLOCATION_AVAILABLE or not self.geolocator:
            return {
                "latitude": 0.0,
                "longitude": 0.0,
                "accuracy": 0.0,
                "source": "disabled"
            }
            
        # Get location from geolocator
        try:
            location = self.geolocator.get_location()
            self.current_location = location
            self.last_geolocation_update = current_time
            self.logger.debug(f"Updated geolocation: {location['latitude']}, {location['longitude']} (source: {location['source']})")
            return location
        except Exception as e:
            self.logger.warning(f"Geolocation update failed: {str(e)}")
            if self.current_location:
                return self.current_location
            return {
                "latitude": 0.0,
                "longitude": 0.0,
                "accuracy": 0.0,
                "source": "error"
            }
    
    def _get_device_info(self) -> Dict[str, Any]:
        import socket
        try:
            hostname = socket.gethostname()
            ip_address = socket.gethostbyname(hostname)
        except Exception:
            hostname = "unknown"
            ip_address = "unknown"
            
        # Get platform info with specific details
        platform_info = self._get_detailed_platform_info()
            
        # Get device info with basic details
        device_info = {
            "device_id": self.device_id,
            "hostname": hostname,
            "ip_address": ip_address,
            "platform": platform_info["os_type"],
            "platform_version": platform_info["os_version"],
            "device_model": platform_info["device_model"],
            "python_version": platform.python_version(),
            "tinylcm_version": self._get_tinylcm_version()
        }
        
        # Add geolocation data if enabled
        if self.enable_geolocation and GEOLOCATION_AVAILABLE and self.geolocator:
            location = self._update_geolocation()
            device_info["location"] = {
                "latitude": location.get("latitude", 0.0),
                "longitude": location.get("longitude", 0.0),
                "accuracy": location.get("accuracy", 0.0),
                "source": location.get("source", "unknown")
            }
            
        return device_info
    
    def _get_tinylcm_version(self) -> str:
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
            # This should never return "N/A" or "unknown"
            return "0.2.0"
            
    def _get_detailed_platform_info(self) -> Dict[str, str]:
        """Get detailed platform information including OS type, version and device model.
        
        Returns:
            Dictionary with detailed platform information
        """
        import os
        import re
        
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
                    self.logger.debug(f"Error reading os-release: {e}")
                
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
                    self.logger.debug(f"Error reading device model: {e}")
                
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
            self.logger.warning(f"Error getting detailed platform info: {e}")
            # Fallback to simple platform info
            platform_info["os_type"] = platform.system()
            platform_info["os_version"] = platform.release()
            platform_info["device_model"] = platform.machine()
            
        return platform_info
    
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
                raise TinyLCMConnectionError(error_msg)
        except requests.RequestException as e:
            error_msg = f"Registration request failed: {str(e)}"
            self.logger.error(error_msg)
            raise TinyLCMConnectionError(error_msg)
    
    def update_device_info(self) -> bool:
        """
        Update device information on the server, including geolocation if enabled.
        
        Returns:
            True if successful, False otherwise
        """
        self.logger.info(f"Updating device information for {self.device_id}")
        update_data = {
            "device_id": self.device_id,
            "device_info": self._get_device_info(),
            "last_sync_time": time.time()
        }
        try:
            response = self.connection_manager.execute_request(
                method="PATCH", 
                endpoint=f"devices/{self.device_id}", 
                json=update_data
            )
            if response.status_code == 200:
                self.logger.info(f"Successfully updated device information for {self.device_id}")
                return True
            else:
                error_msg = f"Update failed: {response.status_code} - {response.text}"
                self.logger.error(error_msg)
                return False
        except requests.RequestException as e:
            error_msg = f"Update request failed: {str(e)}"
            self.logger.error(error_msg)
            return False
    
    def check_server_status(self) -> Dict[str, Any]:
        self.logger.debug("Checking server status")
        try:
            response = self.connection_manager.execute_request(method="GET", endpoint="status")
            if response.status_code == 200:
                return response.json()
            else:
                error_msg = f"Server status check failed: {response.status_code} - {response.text}"
                self.logger.error(error_msg)
                raise TinyLCMConnectionError(error_msg)
        except requests.RequestException as e:
            error_msg = f"Server status request failed: {str(e)}"
            self.logger.error(error_msg)
            raise TinyLCMConnectionError(error_msg)
    
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
                        raise TinyLCMConnectionError(error_msg)
            except Exception as e:
                error_msg = f"Package upload request failed: {str(e)}"
                self.logger.error(error_msg)
                raise TinyLCMConnectionError(error_msg)
        except SyncError as e:
            self.logger.error(f"Error preparing package {package_id}: {str(e)}")
            raise
    
    def sync_all_pending_packages(self) -> List[Dict[str, Any]]:
        """Synchronize all pending packages and prediction images if enabled.

        Returns:
            List of results for each package
        """
        self.logger.info("Synchronizing all pending packages")

        # First sync prediction images if enabled
        if self.enable_prediction_images and self.pending_prediction_images:
            self.logger.info(f"Syncing {len(self.pending_prediction_images)} pending prediction images")
            self._sync_prediction_images()

        # Then sync all other pending packages
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
    
    def send_quarantine_samples(self, max_samples: int = 50) -> Dict[str, Any]:
        """Send quarantined samples to the server for validation.
        
        Args:
            max_samples: Maximum number of samples to send in one batch
            
        Returns:
            Dictionary with operation results
        """
        if not QUARANTINE_AVAILABLE or self.quarantine_buffer is None:
            self.logger.warning("Quarantine buffer not available or not configured")
            return {
                "success": False,
                "error": "Quarantine buffer not available",
                "samples_sent": 0
            }
        
        # Get samples that need to be synced
        samples_to_sync = self.quarantine_buffer.get_samples_for_sync()
        
        # Limit to max_samples
        if len(samples_to_sync) > max_samples:
            samples_to_sync = samples_to_sync[:max_samples]
        
        if not samples_to_sync:
            self.logger.info("No quarantined samples to sync")
            return {
                "success": True,
                "samples_sent": 0
            }
        
        self.logger.info(f"Syncing {len(samples_to_sync)} quarantined samples")
        
        try:
            # Send samples to server
            response = self.connection_manager.execute_request(
                method="POST",
                endpoint="samples/validate",
                json={
                    "device_id": self.device_id,
                    "samples": samples_to_sync
                }
            )
            
            if response.status_code == 200:
                # Get the sample IDs that were successfully synced
                response_data = response.json()
                sample_ids = [sample["sample_id"] for sample in samples_to_sync]
                
                # Mark samples as synced
                self.quarantine_buffer.mark_as_synced(sample_ids)
                
                self.logger.info(f"Successfully synced {len(sample_ids)} quarantined samples")
                return {
                    "success": True,
                    "samples_sent": len(sample_ids),
                    "server_response": response_data
                }
            else:
                error_msg = f"Failed to sync quarantine samples: {response.status_code} - {response.text}"
                self.logger.error(error_msg)
                return {
                    "success": False,
                    "error": error_msg,
                    "samples_sent": 0
                }
        except Exception as e:
            error_msg = f"Error syncing quarantine samples: {str(e)}"
            self.logger.error(error_msg)
            return {
                "success": False,
                "error": error_msg,
                "samples_sent": 0
            }
            
    def get_validation_results(self) -> List[Dict[str, Any]]:
        """Get validation results for previously sent quarantined samples.
        
        Returns:
            List of validation results from the server
        """
        if not QUARANTINE_AVAILABLE or self.quarantine_buffer is None:
            self.logger.warning("Quarantine buffer not available or not configured")
            return []
        
        try:
            # Request validation results from server
            response = self.connection_manager.execute_request(
                method="GET",
                endpoint=f"samples/validation-results/{self.device_id}"
            )
            
            if response.status_code == 200:
                validation_results = response.json().get("results", [])
                
                # Process validation results
                if validation_results:
                    self.quarantine_buffer.process_validation_results(validation_results)
                    self.logger.info(f"Processed {len(validation_results)} validation results")
                
                return validation_results
            else:
                self.logger.error(f"Failed to get validation results: {response.status_code} - {response.text}")
                return []
        except Exception as e:
            self.logger.error(f"Error getting validation results: {str(e)}")
            return []
    
    def sync_quarantine(self) -> Dict[str, Any]:
        """Synchronize quarantine buffer with the server.
        
        This method performs a full synchronization cycle:
        1. Send pending samples to the server
        2. Fetch validation results from the server
        
        Returns:
            Dictionary with operation results
        """
        if not QUARANTINE_AVAILABLE or self.quarantine_buffer is None:
            self.logger.warning("Quarantine buffer not available or not configured")
            return {
                "success": False,
                "error": "Quarantine buffer not available"
            }
        
        # Step 1: Send samples
        send_result = self.send_quarantine_samples()
        
        # Step 2: Get validation results
        validation_results = self.get_validation_results()
        
        return {
            "success": send_result.get("success", False),
            "samples_sent": send_result.get("samples_sent", 0),
            "validation_results_received": len(validation_results),
            "server_error": send_result.get("error")
        }
    
    def report_drift_event(self, drift_data: Dict[str, Any]) -> Dict[str, Any]:
        """Report a drift event to the TinySphere server.
        
        Args:
            drift_data: Drift event data including type, score, detector name, 
                        and optional samples
        
        Returns:
            Dictionary with operation results
        """
        if not DRIFT_DETECTION_AVAILABLE:
            self.logger.warning("Drift detection not available")
            return {
                "success": False,
                "error": "Drift detection not available"
            }
        
        # Ensure required fields
        if "drift_type" not in drift_data:
            drift_data["drift_type"] = "unknown"
        
        # Add device ID
        drift_data["device_id"] = self.device_id
        
        # Add event_id if not present
        if "event_id" not in drift_data:
            drift_data["event_id"] = str(uuid.uuid4())
            
        # Add timestamp if not present
        if "timestamp" not in drift_data:
            drift_data["timestamp"] = time.time()
            
        self.logger.info(f"Reporting drift event: {drift_data['drift_type']} with ID {drift_data['event_id']}")
        
        try:
            # Send drift event to server
            response = self.connection_manager.execute_request(
                method="POST",
                endpoint="drift/events",
                json=drift_data
            )
            
            if response.status_code == 200:
                response_data = response.json()
                self.logger.info(f"Successfully reported drift event {drift_data['event_id']}")
                return {
                    "success": True,
                    "event_id": drift_data["event_id"],
                    "server_response": response_data
                }
            else:
                error_msg = f"Failed to report drift event: {response.status_code} - {response.text}"
                self.logger.error(error_msg)
                return {
                    "success": False,
                    "error": error_msg
                }
        except Exception as e:
            error_msg = f"Error reporting drift event: {str(e)}"
            self.logger.error(error_msg)
            return {
                "success": False,
                "error": error_msg
            }
    
    def get_drift_validations(self) -> List[Dict[str, Any]]:
        """Get pending drift validations from the server.
        
        Returns:
            List of validation results from the server
        """
        if not DRIFT_DETECTION_AVAILABLE:
            self.logger.warning("Drift detection not available")
            return []
        
        try:
            # Request pending validations from server
            response = self.connection_manager.execute_request(
                method="GET",
                endpoint=f"drift/validations/pending?device_id={self.device_id}"
            )
            
            if response.status_code == 200:
                validations = response.json()
                
                if validations:
                    self.logger.info(f"Received {len(validations)} drift validations")
                    
                    # Acknowledge validations
                    if validations:
                        validation_ids = [v["validation_id"] for v in validations]
                        self._acknowledge_validations(validation_ids)
                
                return validations
            else:
                self.logger.error(f"Failed to get drift validations: {response.status_code} - {response.text}")
                return []
        except Exception as e:
            self.logger.error(f"Error getting drift validations: {str(e)}")
            return []
    
    def _acknowledge_validations(self, validation_ids: List[str]) -> bool:
        """Acknowledge drift validations.
        
        Args:
            validation_ids: List of validation IDs to acknowledge
            
        Returns:
            True if successful, False otherwise
        """
        if not validation_ids:
            return True
            
        try:
            response = self.connection_manager.execute_request(
                method="POST",
                endpoint="drift/validations/acknowledge",
                json={"validation_ids": validation_ids}
            )
            
            if response.status_code == 200:
                self.logger.info(f"Acknowledged {len(validation_ids)} drift validations")
                return True
            else:
                self.logger.error(f"Failed to acknowledge validations: {response.status_code} - {response.text}")
                return False
        except Exception as e:
            self.logger.error(f"Error acknowledging validations: {str(e)}")
            return False
    
    def send_adaptation_logs(self, max_logs: int = 50) -> Dict[str, Any]:
        """Send adaptation logs to the TinySphere server.
        
        Args:
            max_logs: Maximum number of logs to send in one batch
            
        Returns:
            Dictionary with operation results
        """
        if not ADAPTATION_TRACKER_AVAILABLE or self.adaptation_tracker is None:
            self.logger.warning("Adaptation tracker not available or not configured")
            return {
                "success": False,
                "error": "Adaptation tracker not available",
                "logs_sent": 0
            }
        
        # Get logs that need to be synced
        logs_to_sync = self.adaptation_tracker.get_logs_for_sync(max_logs)
        
        if not logs_to_sync:
            self.logger.info("No adaptation logs to sync")
            return {
                "success": True,
                "logs_sent": 0
            }
        
        self.logger.info(f"Syncing {len(logs_to_sync)} adaptation logs")
        
        try:
            # Send logs to server
            response = self.connection_manager.execute_request(
                method="POST",
                endpoint="adaptation/logs",
                json={
                    "device_id": self.device_id,
                    "adaptation_logs": logs_to_sync
                }
            )
            
            if response.status_code == 200:
                # Get the log IDs that were successfully synced
                response_data = response.json()
                log_ids = [log["log_id"] for log in logs_to_sync]
                
                # Mark logs as synced
                self.adaptation_tracker.mark_as_synced(log_ids)
                
                self.logger.info(f"Successfully synced {len(log_ids)} adaptation logs")
                return {
                    "success": True,
                    "logs_sent": len(log_ids),
                    "server_response": response_data
                }
            else:
                error_msg = f"Failed to sync adaptation logs: {response.status_code} - {response.text}"
                self.logger.error(error_msg)
                return {
                    "success": False,
                    "error": error_msg,
                    "logs_sent": 0
                }
        except Exception as e:
            error_msg = f"Error syncing adaptation logs: {str(e)}"
            self.logger.error(error_msg)
            return {
                "success": False,
                "error": error_msg,
                "logs_sent": 0
            }
    
    def send_snapshot_metadata(self, max_snapshots: int = 20) -> Dict[str, Any]:
        """Send snapshot metadata to the TinySphere server.
        
        Args:
            max_snapshots: Maximum number of snapshot metadata to send in one batch
            
        Returns:
            Dictionary with operation results
        """
        if not STATE_MANAGER_AVAILABLE or self.state_manager is None:
            self.logger.warning("State manager not available or not configured")
            return {
                "success": False,
                "error": "State manager not available",
                "snapshots_sent": 0
            }
        
        # Get snapshots that need to be synced
        snapshots_to_sync = self.state_manager.get_snapshots_for_sync(max_snapshots)
        
        if not snapshots_to_sync:
            self.logger.info("No snapshot metadata to sync")
            return {
                "success": True,
                "snapshots_sent": 0
            }
        
        self.logger.info(f"Syncing metadata for {len(snapshots_to_sync)} snapshots")
        
        try:
            # Send snapshot metadata to server
            response = self.connection_manager.execute_request(
                method="POST",
                endpoint="adaptation/snapshots",
                json={
                    "device_id": self.device_id,
                    "snapshots": snapshots_to_sync
                }
            )
            
            if response.status_code == 200:
                # Get the snapshot IDs that were successfully synced
                response_data = response.json()
                snapshot_ids = [snapshot["snapshot_id"] for snapshot in snapshots_to_sync]
                
                # Mark snapshots as synced
                self.state_manager.mark_snapshots_as_synced(snapshot_ids)
                
                self.logger.info(f"Successfully synced metadata for {len(snapshot_ids)} snapshots")
                return {
                    "success": True,
                    "snapshots_sent": len(snapshot_ids),
                    "server_response": response_data
                }
            else:
                error_msg = f"Failed to sync snapshot metadata: {response.status_code} - {response.text}"
                self.logger.error(error_msg)
                return {
                    "success": False,
                    "error": error_msg,
                    "snapshots_sent": 0
                }
        except Exception as e:
            error_msg = f"Error syncing snapshot metadata: {str(e)}"
            self.logger.error(error_msg)
            return {
                "success": False,
                "error": error_msg,
                "snapshots_sent": 0
            }
            
    def get_adaptation_feedback(self) -> List[Dict[str, Any]]:
        """Get adaptation feedback from the server.
        
        This includes validated labels and potential rollback commands.
        
        Returns:
            List of adaptation feedback items from the server
        """
        try:
            # Request adaptation feedback from server
            response = self.connection_manager.execute_request(
                method="GET",
                endpoint=f"adaptation/feedback/{self.device_id}"
            )
            
            if response.status_code == 200:
                feedback_items = response.json().get("feedback", [])
                
                if feedback_items:
                    self.logger.info(f"Received {len(feedback_items)} adaptation feedback items")
                    
                    # Acknowledge feedback
                    if feedback_items:
                        feedback_ids = [item["feedback_id"] for item in feedback_items]
                        self._acknowledge_adaptation_feedback(feedback_ids)
                
                return feedback_items
            else:
                self.logger.error(f"Failed to get adaptation feedback: {response.status_code} - {response.text}")
                return []
        except Exception as e:
            self.logger.error(f"Error getting adaptation feedback: {str(e)}")
            return []
    
    def _acknowledge_adaptation_feedback(self, feedback_ids: List[str]) -> bool:
        """Acknowledge adaptation feedback items.
        
        Args:
            feedback_ids: List of feedback IDs to acknowledge
            
        Returns:
            True if successful, False otherwise
        """
        if not feedback_ids:
            return True
            
        try:
            response = self.connection_manager.execute_request(
                method="POST",
                endpoint="adaptation/feedback/acknowledge",
                json={"feedback_ids": feedback_ids}
            )
            
            if response.status_code == 200:
                self.logger.info(f"Acknowledged {len(feedback_ids)} adaptation feedback items")
                return True
            else:
                self.logger.error(f"Failed to acknowledge adaptation feedback: {response.status_code} - {response.text}")
                return False
        except Exception as e:
            self.logger.error(f"Error acknowledging adaptation feedback: {str(e)}")
            return False
            
    def process_adaptation_feedback(self, feedback_items: List[Dict[str, Any]]) -> Dict[str, Any]:
        """Process adaptation feedback from the server.
        
        This handles both validated labels and rollback commands.
        
        Args:
            feedback_items: List of feedback items from the server
            
        Returns:
            Dictionary with processing results
        """
        if not feedback_items:
            return {
                "success": True,
                "validated_labels_processed": 0,
                "rollbacks_processed": 0
            }
            
        validated_labels = []
        rollback_commands = []
        
        # Categorize feedback items
        for item in feedback_items:
            if item.get("feedback_type") == "validated_label":
                validated_labels.append(item)
            elif item.get("feedback_type") == "rollback":
                rollback_commands.append(item)
                
        # Process validated labels
        validated_labels_processed = 0
        for label_item in validated_labels:
            try:
                if STATE_MANAGER_AVAILABLE and self.state_manager is not None and ADAPTATION_TRACKER_AVAILABLE and self.adaptation_tracker is not None:
                    # Extract information
                    sample_id = label_item.get("sample_id")
                    features = label_item.get("features")
                    validated_label = label_item.get("validated_label")
                    
                    if sample_id and features and validated_label:
                        # Use signal handler to update with validated label
                        self.adaptation_tracker.log_event({
                            "event_type": "external_validation",
                            "sample_id": sample_id,
                            "validated_label": validated_label,
                            "timestamp": time.time()
                        })
                        
                        # Signal is expected to be handled by AdaptivePipeline
                        validated_labels_processed += 1
                    else:
                        self.logger.warning("Invalid validated label format in feedback")
            except Exception as e:
                self.logger.error(f"Error processing validated label: {str(e)}")
                
        # Process rollback commands
        rollbacks_processed = 0
        for rollback_item in rollback_commands:
            try:
                if STATE_MANAGER_AVAILABLE and self.state_manager is not None:
                    # Extract information
                    snapshot_id = rollback_item.get("snapshot_id")
                    reason = rollback_item.get("reason", "Server-initiated rollback")
                    
                    if snapshot_id:
                        # Perform the rollback
                        self.state_manager.load_snapshot(snapshot_id)
                        
                        # Log the rollback
                        if ADAPTATION_TRACKER_AVAILABLE and self.adaptation_tracker is not None:
                            self.adaptation_tracker.log_event({
                                "event_type": "external_rollback",
                                "snapshot_id": snapshot_id,
                                "reason": reason,
                                "timestamp": time.time()
                            })
                            
                        rollbacks_processed += 1
                    else:
                        self.logger.warning("Invalid rollback format in feedback")
            except Exception as e:
                self.logger.error(f"Error processing rollback command: {str(e)}")
                
        return {
            "success": True,
            "validated_labels_processed": validated_labels_processed,
            "rollbacks_processed": rollbacks_processed
        }
        
    def sync_adaptation(self) -> Dict[str, Any]:
        """Synchronize adaptation components with the server.
        
        This method performs a full synchronization cycle:
        1. Send adaptation logs to the server
        2. Send snapshot metadata to the server
        3. Fetch adaptation feedback from the server
        4. Process the feedback
        
        Returns:
            Dictionary with operation results
        """
        results = {
            "success": True,
            "logs_sent": 0,
            "snapshots_sent": 0,
            "feedback_received": 0,
            "validated_labels_processed": 0,
            "rollbacks_processed": 0
        }
        
        # Step 1: Send adaptation logs
        if ADAPTATION_TRACKER_AVAILABLE and self.adaptation_tracker is not None:
            logs_result = self.send_adaptation_logs()
            results["logs_sent"] = logs_result.get("logs_sent", 0)
            if not logs_result.get("success", False):
                results["success"] = False
                results["logs_error"] = logs_result.get("error")
        
        # Step 2: Send snapshot metadata
        if STATE_MANAGER_AVAILABLE and self.state_manager is not None:
            snapshots_result = self.send_snapshot_metadata()
            results["snapshots_sent"] = snapshots_result.get("snapshots_sent", 0)
            if not snapshots_result.get("success", False):
                results["success"] = False
                results["snapshots_error"] = snapshots_result.get("error")
        
        # Step 3: Get adaptation feedback
        feedback_items = self.get_adaptation_feedback()
        results["feedback_received"] = len(feedback_items)
        
        # Step 4: Process feedback
        if feedback_items:
            process_result = self.process_adaptation_feedback(feedback_items)
            results["validated_labels_processed"] = process_result.get("validated_labels_processed", 0)
            results["rollbacks_processed"] = process_result.get("rollbacks_processed", 0)
            if not process_result.get("success", False):
                results["success"] = False
                results["process_error"] = process_result.get("error")
                
        return results
    
    def sync_drift_events(self) -> Dict[str, Any]:
        """Synchronize drift events with the server.
        
        This performs a full synchronization:
        1. For each detector, collect drift events and report them
        2. Retrieve validation results from the server
        
        Returns:
            Dictionary with operation results
        """
        if not DRIFT_DETECTION_AVAILABLE or not self.drift_detectors:
            self.logger.warning("Drift detection not available or no detectors configured")
            return {
                "success": False,
                "error": "Drift detection not available or no detectors configured"
            }
            
        # Collect drift events from detectors
        events_sent = 0
        events_failed = 0
        
        for detector in self.drift_detectors:
            if not hasattr(detector, "get_drift_events"):
                self.logger.warning(f"Detector {detector.__class__.__name__} does not support get_drift_events method")
                continue
                
            try:
                # Get drift events from detector
                drift_events = detector.get_drift_events()
                
                for event in drift_events:
                    result = self.report_drift_event(event)
                    if result.get("success", False):
                        events_sent += 1
                    else:
                        events_failed += 1
            except Exception as e:
                self.logger.error(f"Error collecting drift events from {detector.__class__.__name__}: {str(e)}")
                events_failed += 1
        
        # Get validations from server
        validations = self.get_drift_validations()
        
        # Process validations if we have handlers to do so
        validations_processed = 0
        for detector in self.drift_detectors:
            if hasattr(detector, "process_validations"):
                try:
                    processed = detector.process_validations(validations)
                    validations_processed += processed
                except Exception as e:
                    self.logger.error(f"Error processing validations with {detector.__class__.__name__}: {str(e)}")
        
        return {
            "success": events_failed == 0,
            "events_sent": events_sent,
            "events_failed": events_failed,
            "validations_received": len(validations),
            "validations_processed": validations_processed
        }
    
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
            self.logger.debug(f"Prediction image transfer is disabled, not adding image {image_path}")
            return False

        try:
            if not os.path.exists(image_path):
                self.logger.warning(f"Image file not found: {image_path}")
                return False

            # Add to pending images queue
            self.pending_prediction_images.append({
                "path": image_path,
                "prediction": prediction,
                "confidence": confidence,
                "timestamp": time.time()
            })

            self.logger.debug(f"Added prediction image to sync queue: {image_path}")
            return True
        except Exception as e:
            self.logger.error(f"Failed to add prediction image: {str(e)}")
            return False

    def _sync_prediction_images(self) -> List[Dict[str, Any]]:
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

            # Clear the pending list if all were processed successfully
            processed_images = []
            for result in results:
                if result.get("status") == "success":
                    processed_images.extend(result.get("processed_images", []))

            # Remove processed images from the pending list
            self.pending_prediction_images = [img for img in self.pending_prediction_images
                                           if img["path"] not in processed_images]

            # Log results
            if processed_images:
                self.logger.info(f"Synchronized {len(processed_images)} prediction images")
            if self.pending_prediction_images:
                self.logger.warning(f"{len(self.pending_prediction_images)} prediction images still pending")

            return results
        except Exception as e:
            self.logger.error(f"Failed to sync prediction images: {str(e)}")
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
                        self.logger.warning(f"Failed to add image {image_path} to package")

            # Clean up temp directory
            try:
                shutil.rmtree(temp_dir)
            except Exception as e:
                self.logger.warning(f"Failed to clean up temp directory: {e}")

            # Finalize package
            self.sync_interface.finalize_package(package_id)

            # Send the package
            result = self.send_package(package_id)
            if result:
                self.logger.info(f"Successfully sent {len(processed_images)} prediction images to server")
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
                self.logger.error(f"Failed to send prediction images package to server")
                return {
                    "status": "error",
                    "message": "Failed to send prediction images package",
                    "prediction": prediction
                }

        except Exception as e:
            self.logger.error(f"Failed to create prediction images package: {str(e)}")
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
                    self.logger.debug(f"Deleted transferred image: {path}")
                else:
                    self.logger.warning(f"Image file not found: {path}")
                    fail_count += 1
            except Exception as e:
                self.logger.error(f"Failed to delete image {path}: {e}")
                fail_count += 1

        self.logger.info(f"Deleted {success_count} transferred images, {fail_count} failed")
        return success_count, fail_count

    def sync_all(self) -> Dict[str, Any]:
        """Perform a comprehensive synchronization of all components with the server.

        This method synchronizes:
        1. Device info update (including geolocation if enabled)
        2. Quarantine buffer (samples and validation results)
        3. Drift events and validations
        4. Adaptation logs, snapshots, and feedback
        5. Prediction images (if enabled)
        6. Pending packages

        Returns:
            Dictionary with comprehensive sync results
        """
        self.logger.info("Starting comprehensive synchronization with server")

        results = {
            "success": True,
            "components": {}
        }
        
        # Step 0: Update device information including geolocation
        device_update_result = self.update_device_info()
        results["components"]["device_info"] = {
            "success": device_update_result,
            "geolocation_enabled": self.enable_geolocation and GEOLOCATION_AVAILABLE
        }

        # Step 1: Sync quarantine data
        if QUARANTINE_AVAILABLE and self.quarantine_buffer is not None:
            quarantine_result = self.sync_quarantine()
            results["components"]["quarantine"] = quarantine_result
            if not quarantine_result.get("success", False) and quarantine_result.get("error") != "Quarantine buffer not available":
                results["success"] = False

        # Step 2: Sync drift events
        if DRIFT_DETECTION_AVAILABLE and self.drift_detectors:
            drift_result = self.sync_drift_events()
            results["components"]["drift"] = drift_result
            if not drift_result.get("success", False) and drift_result.get("error") != "Drift detection not available or no detectors configured":
                results["success"] = False

        # Step 3: Sync adaptation components
        if (ADAPTATION_TRACKER_AVAILABLE and self.adaptation_tracker is not None) or \
           (STATE_MANAGER_AVAILABLE and self.state_manager is not None):
            adaptation_result = self.sync_adaptation()
            results["components"]["adaptation"] = adaptation_result
            if not adaptation_result.get("success", False):
                results["success"] = False

        # Step 4: Sync prediction images
        if self.enable_prediction_images and self.pending_prediction_images:
            image_results = self._sync_prediction_images()
            if image_results:
                success_count = sum(1 for r in image_results if r.get("status") == "success")
                total_images = sum(r.get("image_count", 0) for r in image_results if r.get("status") == "success")
                results["components"]["prediction_images"] = {
                    "success": success_count == len(image_results),
                    "total_batches": len(image_results),
                    "successful_batches": success_count,
                    "total_images": total_images
                }
                if success_count < len(image_results):
                    results["success"] = False

        # Step 5: Sync pending packages
        package_results = self.sync_all_pending_packages()
        if package_results:
            success_count = sum(1 for r in package_results if r.get("success", False))
            results["components"]["packages"] = {
                "success": success_count == len(package_results),
                "total": len(package_results),
                "successful": success_count,
                "failed": len(package_results) - success_count,
                "details": package_results
            }
            if success_count < len(package_results):
                results["success"] = False

        # Compile summary
        components_synced = len(results["components"])
        successful_components = sum(1 for c in results["components"].values() if c.get("success", False))

        results["summary"] = {
            "components_synced": components_synced,
            "successful_components": successful_components,
            "timestamp": time.time()
        }

        self.logger.info(f"Comprehensive synchronization completed: {successful_components}/{components_synced} components successful")

        return results

    def close(self) -> None:
        """Close the sync client and release resources."""
        self.logger.info("Closing sync client")