#!/usr/bin/env python3
"""
TinySphere Connection Test
-------------------------
Simple script to test the connection to TinySphere server before starting the application.
"""

import json
import logging
import sys
from pathlib import Path

import requests

# Configure basic logging
logging.basicConfig(level=logging.INFO, format="%(message)s")
logger = logging.getLogger("tinysphere_test")

def load_config(config_path="config.json"):
    """Load configuration from JSON file"""
    try:
        with open(config_path, 'r') as f:
            return json.load(f)
    except Exception as e:
        logger.error(f"❌ Failed to load config: {e}")
        return None

def test_connection(server_url, api_key=None, device_id=None, timeout=5):
    """Test connection to TinySphere server"""
    try:
        # Prepare headers
        headers = {}
        if api_key:
            headers["Authorization"] = f"Bearer {api_key}"
            if device_id:
                headers["X-Device-ID"] = device_id
        
        # Make a simple GET request to the status endpoint
        status_url = f"{server_url}/api/status"
        response = requests.get(status_url, headers=headers, timeout=timeout)
        
        # Check response
        if response.status_code == 200:
            return True, "✅ Server is online and responding"
        else:
            return False, f"❌ Server responded with status code {response.status_code}: {response.text}"
    
    except requests.exceptions.ConnectTimeout:
        return False, "❌ Connection timed out. Server might be down or unreachable."
    except requests.exceptions.ConnectionError:
        return False, "❌ Connection error. Server might be down or network might be unreachable."
    except Exception as e:
        return False, f"❌ Unexpected error: {str(e)}"

def main():
    # Load configuration
    config = load_config()
    if not config:
        logger.error("❌ Could not load configuration")
        return 1
    
    # Extract TinySphere settings
    server_url = config.get("tinysphere", {}).get("server_url")
    api_key = config.get("tinysphere", {}).get("api_key")
    device_id = config.get("tinysphere", {}).get("device_id")
    
    if not server_url:
        logger.error("❌ Server URL not found in configuration")
        return 1
    
    # Test connection
    success, message = test_connection(server_url, api_key, device_id)
    logger.info(message)
    
    return 0 if success else 1

if __name__ == "__main__":
    sys.exit(main())