#!/usr/bin/env python3
"""
Device ID Manager
----------------
Handles the creation and retrieval of a unique device ID for the Raspberry Pi device.
Uses hardware information to create a consistent identifier.
"""

import hashlib
import logging
import os
import re
import socket
import uuid
from pathlib import Path

logger = logging.getLogger(__name__)

class DeviceIDManager:
    """Manages the device ID for the TinyLCM application."""
    
    def __init__(self, device_id_file="device_id.txt", generate_if_missing=True):
        """Initialize the DeviceIDManager
        
        Args:
            device_id_file: Path to the file storing the device ID
            generate_if_missing: Whether to generate a new device ID if none exists
        """
        self.device_id_file = device_id_file
        self.generate_if_missing = generate_if_missing
        logger.debug(f"DeviceIDManager initialized with file: {device_id_file}")
    
    def get_device_id(self) -> str:
        """Get the device ID, generating it if it doesn't exist."""
        # Try to load existing device ID
        if os.path.exists(self.device_id_file):
            try:
                with open(self.device_id_file, 'r') as f:
                    device_id = f.read().strip()
                    if device_id:
                        logger.debug(f"Loaded existing device ID: {device_id}")
                        return device_id
            except Exception as e:
                logger.error(f"Error reading device ID file: {e}")
        
        # Generate new device ID if needed
        if self.generate_if_missing:
            device_id = self._generate_device_id()
            try:
                with open(self.device_id_file, 'w') as f:
                    f.write(device_id)
                logger.info(f"Generated and saved new device ID: {device_id}")
                return device_id
            except Exception as e:
                logger.error(f"Error saving device ID: {e}")
                return device_id
        
        # Return a temporary ID if we couldn't load or generate
        logger.warning("Using temporary device ID")
        return f"temp-id-{uuid.uuid4().hex[:8]}"
    
    def _generate_device_id(self) -> str:
        """Generate a unique device ID based on hardware information."""
        # Get Raspberry Pi serial number if available
        serial = self._get_raspberry_pi_serial()
        
        # Get MAC address of the first network interface
        mac = self._get_mac_address()
        
        # Get hostname
        hostname = socket.gethostname()
        
        # Generate unique ID by hashing the combined values
        combined = f"{serial}:{mac}:{hostname}"
        return f"pi-{hashlib.md5(combined.encode()).hexdigest()[:12]}"
    
    def _get_raspberry_pi_serial(self) -> str:
        """Get the Raspberry Pi serial number from /proc/cpuinfo."""
        try:
            # Try to get serial from cpuinfo
            if os.path.exists('/proc/cpuinfo'):
                with open('/proc/cpuinfo', 'r') as f:
                    cpuinfo = f.read()
                    match = re.search(r'Serial\s+:\s+(\w+)', cpuinfo)
                    if match:
                        return match.group(1)
            
            # Alternative: try to get from device tree
            if os.path.exists('/sys/firmware/devicetree/base/serial-number'):
                with open('/sys/firmware/devicetree/base/serial-number', 'rb') as f:
                    return f.read().strip(b'\x00').decode('utf-8')
        except Exception as e:
            logger.warning(f"Error getting Raspberry Pi serial number: {e}")
        
        return "unknown-serial"
    
    def _get_mac_address(self) -> str:
        """Get the MAC address of the first non-loopback network interface."""
        try:
            # Try to get MAC from /sys/class/net
            for interface in os.listdir('/sys/class/net/'):
                if interface != 'lo':  # Skip loopback
                    mac_path = f'/sys/class/net/{interface}/address'
                    if os.path.exists(mac_path):
                        with open(mac_path, 'r') as f:
                            return f.read().strip()
        except Exception as e:
            logger.warning(f"Error getting MAC address: {e}")
        
        return "unknown-mac"


# Test function
def main():
    """Test the DeviceIDManager functionality."""
    logging.basicConfig(level=logging.DEBUG)
    manager = DeviceIDManager()
    device_id = manager.get_device_id()
    print(f"Device ID: {device_id}")

if __name__ == "__main__":
    main()