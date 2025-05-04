# tinylcm/client/__init__.py
"""
Client components for TinyLCM.

This module provides components for edge devices to communicate
with central TinyLCM servers:

- ConnectionManager: Handles connection establishment and maintenance
- SyncClient: Manages synchronization of data packages with the server
- SyncInterface: Prepares and packages data for synchronization
"""

from tinylcm.client.connection_manager import ConnectionManager
from tinylcm.client.sync_client import SyncClient
from tinylcm.client.sync_interface import SyncInterface, SyncPackage

__all__ = [
    "ConnectionManager",
    "SyncClient",
    "SyncInterface",
    "SyncPackage"
]