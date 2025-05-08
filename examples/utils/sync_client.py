#!/usr/bin/env python3
"""
SyncClient
---------
TinySphere synchronization client for the TinyLCM application.
Extends the base SyncClient with application-specific functionality.
"""

import json
import logging
import os
import time
from pathlib import Path
from typing import Any, Dict, List, Optional

from tinylcm.client.sync_client import SyncClient as TinyLCMSyncClient
from tinylcm.client.sync_interface import SyncInterface, SyncPackage
from tinylcm.core.data_structures import FeatureSample
from tinylcm.utils.errors import TinyLCMConnectionError, SyncError

logger = logging.getLogger(__name__)


class ExtendedSyncClient:
    """Extended SyncClient for the TinyLCM application."""
    
    def __init__(
        self,
        server_url: str,
        device_id: str,
        api_key: str = "tinylcm-demo-key",
        sync_dir: str = "./sync_data",
        sync_interval_seconds: int = 30,
        max_retries: int = 3,
        auto_register: bool = True
    ):
        """Initialize the extended SyncClient.
        
        Args:
            server_url: URL of the TinySphere server
            device_id: Unique device identifier
            api_key: API key for authentication
            sync_dir: Directory for storing sync data
            sync_interval_seconds: Interval between sync attempts
            max_retries: Maximum number of connection retry attempts
            auto_register: Whether to automatically register the device
        """
        # Create sync directory if it doesn't exist
        os.makedirs(sync_dir, exist_ok=True)
        
        # Create sync interface
        self.sync_interface = SyncInterface(sync_dir=sync_dir)
        
        # Custom headers for API key authentication
        self.headers = {
            "X-API-Key": api_key,
            "Content-Type": "application/json"
        }
        
        # Initialize the core TinyLCM SyncClient
        self.client = TinyLCMSyncClient(
            server_url=server_url,
            api_key=api_key,
            device_id=device_id,
            sync_interface=self.sync_interface,
            max_retries=max_retries,
            auto_register=auto_register
        )
        
        self.device_id = device_id
        self.sync_interval_seconds = sync_interval_seconds
        self.sync_dir = sync_dir
        self.last_sync_time = 0
        self._running = False
        
        logger.info(f"Extended SyncClient initialized for device {device_id}")
    
    def check_connection(self) -> bool:
        """Check connection to the TinySphere server.
        
        Returns:
            True if connected, False otherwise
        """
        try:
            status = self.client.check_server_status()
            logger.info(f"Server connection status: {status}")
            return True
        except TinyLCMConnectionError as e:
            logger.warning(f"Server connection check failed: {e}")
            return False
    
    def register_device(self) -> bool:
        """Register the device with TinySphere.
        
        Returns:
            True if registration was successful, False otherwise
        """
        try:
            return self.client.register_device()
        except TinyLCMConnectionError as e:
            logger.error(f"Device registration failed: {e}")
            return False
    
    def send_package(self, package_id: str) -> bool:
        """Send a specific package to TinySphere.
        
        Args:
            package_id: ID of the package to send
            
        Returns:
            True if sending was successful, False otherwise
        """
        try:
            return self.client.send_package(package_id)
        except (TinyLCMConnectionError, SyncError) as e:
            logger.error(f"Failed to send package {package_id}: {e}")
            return False
    
    def sync_all_pending_packages(self) -> List[Dict[str, Any]]:
        """Synchronize all pending packages.
        
        Returns:
            List of results for each package
        """
        try:
            results = self.client.sync_all_pending_packages()
            self.last_sync_time = time.time()
            return results
        except Exception as e:
            logger.error(f"Failed to sync pending packages: {e}")
            return []
    
    def should_sync(self) -> bool:
        """Check if it's time for a sync operation.
        
        Returns:
            True if it's time to sync, False otherwise
        """
        return time.time() - self.last_sync_time >= self.sync_interval_seconds
    
    def create_and_send_drift_event_package(
        self,
        detector_name: str,
        reason: str,
        metrics: Dict[str, Any],
        sample: Optional[FeatureSample] = None,
        image_path: Optional[str] = None
    ) -> bool:
        """Create and send a package containing drift event information.
        
        Args:
            detector_name: Name of the drift detector that triggered the event
            reason: Reason for the drift event
            metrics: Metrics associated with the drift event
            sample: FeatureSample that triggered the drift event
            image_path: Path to the image that triggered the drift event (optional)
            
        Returns:
            True if the operation was successful, False otherwise
        """
        try:
            # Generate a package ID
            package_id = self.sync_interface.create_package(
                package_type="drift_event",
                metadata={
                    "device_id": self.device_id,
                    "timestamp": time.time(),
                    "detector_name": detector_name,
                    "reason": reason
                }
            )
            
            # Create the drift event data
            drift_data = {
                "timestamp": time.time(),
                "detector_name": detector_name,
                "reason": reason,
                "metrics": metrics,
                "sample": {
                    "sample_id": sample.sample_id if sample else None,
                    "prediction": sample.prediction if sample else None,
                    "confidence": float(sample.metadata.get("confidence", 0.0)) if sample else None,
                    "timestamp": sample.timestamp if sample else None
                }
            }
            
            # Add drift event data to the package
            self.sync_interface.add_json_to_package(
                package_id=package_id,
                json_data=drift_data,
                path="drift_event.json"
            )
            
            # Add image to the package if provided
            if image_path and os.path.exists(image_path):
                with open(image_path, 'rb') as file:
                    self.sync_interface.add_file_to_package(
                        package_id=package_id,
                        file_path=image_path,
                        target_path="image.jpg"
                    )
            
            # Finalize the package
            self.sync_interface.finalize_package(package_id)
            
            # Send the package
            return self.send_package(package_id)
            
        except Exception as e:
            logger.error(f"Failed to create and send drift event package: {e}")
            return False
    
    def create_and_send_metrics_package(self, metrics: Dict[str, Any]) -> bool:
        """Create and send a package containing system metrics.
        
        Args:
            metrics: System metrics to include in the package
            
        Returns:
            True if the operation was successful, False otherwise
        """
        try:
            # Generate a package ID
            package_id = self.sync_interface.create_package(
                package_type="metrics",
                metadata={
                    "device_id": self.device_id,
                    "timestamp": time.time(),
                    "metrics_type": "system"
                }
            )
            
            # Add metrics data to the package
            self.sync_interface.add_json_to_package(
                package_id=package_id,
                json_data=metrics,
                path="metrics.json"
            )
            
            # Finalize the package
            self.sync_interface.finalize_package(package_id)
            
            # Send the package
            return self.send_package(package_id)
            
        except Exception as e:
            logger.error(f"Failed to create and send metrics package: {e}")
            return False
    
    def close(self):
        """Close the connection and clean up resources."""
        try:
            self.client.close()
            logger.info("SyncClient closed")
        except Exception as e:
            logger.error(f"Error closing SyncClient: {e}")


# Test function
def main():
    """Test the ExtendedSyncClient functionality."""
    import argparse
    
    # Configure logging
    logging.basicConfig(level=logging.INFO)
    
    # Parse arguments
    parser = argparse.ArgumentParser(description='Test ExtendedSyncClient')
    parser.add_argument('--server', default='http://localhost:8000', help='TinySphere server URL')
    parser.add_argument('--key', default='tinylcm-demo-key', help='API key')
    parser.add_argument('--device', default=None, help='Device ID (defaults to hostname)')
    parser.add_argument('--sync-dir', default='./sync_data', help='Sync directory')
    
    args = parser.parse_args()
    
    # Generate default device ID if not provided
    if not args.device:
        import socket
        args.device = f"test-{socket.gethostname()}"
    
    # Create sync client
    client = ExtendedSyncClient(
        server_url=args.server,
        api_key=args.key,
        device_id=args.device,
        sync_dir=args.sync_dir
    )
    
    # Check connection
    connected = client.check_connection()
    print(f"Connection status: {'Connected' if connected else 'Disconnected'}")
    
    if not connected:
        print("Cannot continue without connection")
        return
    
    # Register device
    registered = client.register_device()
    print(f"Device registration: {'Successful' if registered else 'Failed'}")
    
    # Create and send a test metrics package
    test_metrics = {
        "cpu_usage": 25.5,
        "memory_usage": 512.3,
        "temperature": 45.2,
        "uptime": 3600,
        "timestamp": time.time()
    }
    
    metrics_sent = client.create_and_send_metrics_package(test_metrics)
    print(f"Metrics package sending: {'Successful' if metrics_sent else 'Failed'}")
    
    # Create and send a test drift event package
    test_drift_event = {
        "change_magnitude": 0.35,
        "window_size": 100,
        "baseline_mean": 0.85,
        "current_mean": 0.65,
        "threshold": 0.2
    }
    
    # Mock sample
    mock_sample = type('MockSample', (), {
        'sample_id': 'mock_123',
        'prediction': 'red_stone',
        'confidence': 0.75,
        'timestamp': time.time()
    })
    
    drift_sent = client.create_and_send_drift_event_package(
        detector_name="EWMAConfidenceMonitor",
        reason="Confidence drop detected",
        metrics=test_drift_event,
        sample=mock_sample
    )
    print(f"Drift event package sending: {'Successful' if drift_sent else 'Failed'}")
    
    # Sync all pending packages
    results = client.sync_all_pending_packages()
    print(f"Synced {len(results)} packages")
    
    # Close client
    client.close()

if __name__ == "__main__":
    main()