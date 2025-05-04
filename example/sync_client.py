#!/usr/bin/env python3
"""
SyncClient
---------
TinySphere synchronization client for the Stone Detector application.
Extends the TinyLCM SyncClient with application-specific functionality.
"""

import logging
import time
from typing import Dict, Any, List, Optional

from tinylcm.client.sync_client import SyncClient as TinyLCMSyncClient
from tinylcm.client.sync_interface import SyncInterface
from tinylcm.utils.errors import ConnectionError, SyncError

logger = logging.getLogger(__name__)

class SyncClient:
    """Extended SyncClient for the Stone Detector application."""
    
    def __init__(self, server_url: str, api_key: str, device_id: str, 
                 sync_interface: Optional[SyncInterface] = None,
                 max_retries: int = 3, auto_register: bool = True):
        """Initialize the SyncClient.
        
        Args:
            server_url: URL of the TinySphere server
            api_key: API key for authentication
            device_id: Unique device identifier
            sync_interface: SyncInterface instance (optional)
            max_retries: Maximum number of connection retry attempts
            auto_register: Whether to automatically register the device
        """
        # Initialize the core TinyLCM SyncClient
        self.client = TinyLCMSyncClient(
            server_url=server_url,
            api_key=api_key,
            device_id=device_id,
            sync_interface=sync_interface,
            max_retries=max_retries,
            auto_register=auto_register
        )
        
        self.device_id = device_id
        self.sync_interface = sync_interface or self.client.sync_interface
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
        except ConnectionError as e:
            logger.warning(f"Server connection check failed: {e}")
            return False
    
    def register_device(self) -> bool:
        """Register the device with TinySphere.
        
        Returns:
            True if registration was successful, False otherwise
        """
        try:
            return self.client.register_device()
        except ConnectionError as e:
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
        except (ConnectionError, SyncError) as e:
            logger.error(f"Failed to send package {package_id}: {e}")
            return False
    
    def sync_all_pending_packages(self) -> List[Dict[str, Any]]:
        """Synchronize all pending packages.
        
        Returns:
            List of results for each package
        """
        try:
            return self.client.sync_all_pending_packages()
        except Exception as e:
            logger.error(f"Failed to sync pending packages: {e}")
            return []
    
    def get_status(self) -> Dict[str, Any]:
        """Get synchronization status.
        
        Returns:
            Dictionary with sync status information
        """
        try:
            return self.client.get_sync_status()
        except Exception as e:
            logger.error(f"Failed to get sync status: {e}")
            return {
                "error": str(e),
                "connected": False,
                "total_packages": 0,
                "pending_packages": 0,
                "synced_packages": 0,
                "failed_packages": 0
            }
    
    def create_and_send_metrics_package(self, metrics: Dict[str, Any]) -> bool:
        """Create and send a package containing system metrics.
        
        Args:
            metrics: System metrics to include in the package
            
        Returns:
            True if the operation was successful, False otherwise
        """
        try:
            # Create metrics package
            package_id = self.sync_interface.create_package(
                package_type="metrics",
                metadata={
                    "device_id": self.device_id,
                    "timestamp": time.time(),
                    "metrics_type": "system"
                }
            )
            
            # Add metrics data
            self.sync_interface.add_json_to_package(
                package_id=package_id,
                json_data=metrics,
                path="metrics.json"
            )
            
            # Finalize package
            self.sync_interface.finalize_package(package_id)
            
            # Send package
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
    """Test the SyncClient functionality."""
    import argparse
    import json
    
    # Configure logging
    logging.basicConfig(level=logging.INFO)
    
    # Parse arguments
    parser = argparse.ArgumentParser(description='Test SyncClient')
    parser.add_argument('--server', required=True, help='TinySphere server URL')
    parser.add_argument('--key', required=True, help='API key')
    parser.add_argument('--device', default=None, help='Device ID (defaults to hostname)')
    parser.add_argument('--sync-dir', default='./sync_data', help='Sync directory')
    
    args = parser.parse_args()
    
    # Generate default device ID if not provided
    if not args.device:
        import socket
        args.device = f"test-{socket.gethostname()}"
    
    # Create sync interface
    sync_interface = SyncInterface(sync_dir=args.sync_dir)
    
    # Create sync client
    client = SyncClient(
        server_url=args.server,
        api_key=args.key,
        device_id=args.device,
        sync_interface=sync_interface
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
    
    # Create a test package
    test_data = {
        "timestamp": time.time(),
        "message": "Test package from SyncClient",
        "device": args.device
    }
    
    package_id = sync_interface.create_package(
        package_type="test",
        metadata={
            "device_id": args.device,
            "test": True
        }
    )
    
    # Add data to package
    sync_interface.add_json_to_package(
        package_id=package_id,
        json_data=test_data,
        path="test_data.json"
    )
    
    # Finalize package
    sync_interface.finalize_package(package_id)
    
    # Send package
    sent = client.send_package(package_id)
    print(f"Package {package_id} sending: {'Successful' if sent else 'Failed'}")
    
    # Get status
    status = client.get_status()
    print(f"Sync status: {json.dumps(status, indent=2)}")
    
    # Close client
    client.close()

if __name__ == "__main__":
    main()