#!/usr/bin/env python3
"""
Device ID Manager
-----------------
Manages the device ID for this edge device, creating a persistent ID if one doesn't exist.
"""

import os
import socket
import uuid
from pathlib import Path


class DeviceIDManager:
    def __init__(self, id_file_path="device_id.txt"):
        """
        Initialize the device ID manager.
        
        Args:
            id_file_path (str): Path to the file where the device ID is stored
        """
        self.id_file_path = Path(id_file_path)
    
    def get_device_id(self):
        """
        Get the device ID. If no ID exists, generate a new one.
        
        Returns:
            str: The device ID
        """
        # Check if ID file exists
        if self.id_file_path.exists():
            with open(self.id_file_path, 'r') as f:
                device_id = f.read().strip()
                if device_id:
                    return device_id
        
        # Generate new device ID
        hostname = socket.gethostname()
        mac = uuid.getnode()
        
        # Create unique ID based on hostname and MAC address
        device_id = f"pi-{hostname}-{mac:x}"[:32]  # Limit to 32 chars
        
        # Save the device ID
        with open(self.id_file_path, 'w') as f:
            f.write(device_id)
        
        return device_id
    
    def set_device_id(self, device_id):
        """
        Manually set the device ID.
        
        Args:
            device_id (str): The device ID to set
        """
        with open(self.id_file_path, 'w') as f:
            f.write(device_id)