import json
import os
import platform
import re
import time
import uuid
from pathlib import Path
from typing import Any, Dict, List, Optional, Union

import requests

from tinylcm.client.connection_manager import ConnectionManager
from tinylcm.client.sync_interface import SyncInterface
from tinylcm.utils.errors import ConnectionError, SyncError
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
        drift_detectors: Optional[List[Any]] = None
    ):
        if not self.validate_server_url(server_url):
            raise ValueError(f"Invalid server URL: {server_url}")
        self.logger = setup_logger(f"{__name__}.{self.__class__.__name__}")
        self.server_url = server_url.rstrip('/')
        self.api_key = api_key
        self.device_id = device_id
        self.auto_register = auto_register
        
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
        
        self.logger.info(f"Initialized sync client for server: {server_url}")
        
        # Auto-register if needed
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
    
    def close(self) -> None:
        """Close the sync client and release resources."""
        self.logger.info("Closing sync client")