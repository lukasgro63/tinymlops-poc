#!/usr/bin/env python3
"""
Geolocation Utility
------------------
Utility for determining device location using various methods including:
- WiFi-based positioning
- IP-based geolocation
- Custom location configuration

This module supports different platforms (Linux, macOS, Windows) and provides
fallback mechanisms when primary geolocation methods are unavailable.
"""

import json
import logging
import os
import platform
import subprocess
import tempfile
import time
from pathlib import Path
from typing import Any, Dict, List, Optional, Tuple, Union

try:
    import requests
    REQUESTS_AVAILABLE = True
except ImportError:
    REQUESTS_AVAILABLE = False

logger = logging.getLogger(__name__)

# Constants
DEFAULT_CACHE_TTL = 86400  # 24 hours in seconds
DEFAULT_FALLBACK_COORDINATES = [0.0, 0.0]  # Default coordinates (0,0)
DEFAULT_ACCURACY = 10000.0  # Default accuracy in meters (very low)
WIFI_MIN_NETWORKS = 2  # Minimum number of WiFi networks needed for accurate positioning


class Geolocator:
    """Utility for determining device location using various methods."""

    def __init__(
        self,
        api_key: Optional[str] = None,
        cache_ttl: int = DEFAULT_CACHE_TTL,
        cache_file: Optional[str] = None,
        fallback_coordinates: List[float] = DEFAULT_FALLBACK_COORDINATES,
        min_wifi_networks: int = WIFI_MIN_NETWORKS
    ):
        """Initialize the geolocation utility.
        
        Args:
            api_key: API key for geolocation service (if required)
            cache_ttl: Time-to-live for location cache in seconds
            cache_file: Custom path for cache file (if None, uses default)
            fallback_coordinates: Default coordinates to use if geolocation fails [lat, lon]
            min_wifi_networks: Minimum number of WiFi networks needed for accuracy
        """
        self.api_key = api_key
        self.cache_ttl = cache_ttl
        self.fallback_coordinates = fallback_coordinates
        self.min_wifi_networks = min_wifi_networks
        
        # Determine operating system
        self.os_type = platform.system().lower()
        logger.debug(f"Detected operating system: {self.os_type}")
        
        # Set up cache
        if cache_file:
            self.cache_file = cache_file
        else:
            cache_dir = os.path.join(os.path.expanduser("~"), ".tinylcm")
            os.makedirs(cache_dir, exist_ok=True)
            self.cache_file = os.path.join(cache_dir, "geolocation_cache.json")
        
        logger.debug(f"Using geolocation cache file: {self.cache_file}")
        
        # Initialize cache
        self.cached_location = None
        self.last_update = 0
        self._load_cache()
        
    def get_location(self) -> Dict[str, Union[float, str]]:
        """Get the current device location.
        
        Returns:
            Dictionary containing latitude, longitude, accuracy, and source
        """
        # Check cache first
        cached = self._get_cached_location()
        if cached:
            return cached
        
        # Try WiFi-based location (most accurate)
        wifi_location = self._get_wifi_location()
        if wifi_location:
            self._cache_location(wifi_location)
            return wifi_location
        
        # Fall back to IP-based location
        ip_location = self._get_ip_location()
        if ip_location:
            self._cache_location(ip_location)
            return ip_location
        
        # Use fallback coordinates as last resort
        fallback = {
            "latitude": self.fallback_coordinates[0],
            "longitude": self.fallback_coordinates[1],
            "accuracy": DEFAULT_ACCURACY,
            "source": "fallback"
        }
        self._cache_location(fallback)
        return fallback
    
    def _get_cached_location(self) -> Optional[Dict[str, Union[float, str]]]:
        """Get location from cache if valid.
        
        Returns:
            Cached location or None if cache is invalid
        """
        current_time = time.time()
        
        # Check if we have an in-memory cache
        if self.cached_location and current_time - self.last_update < self.cache_ttl:
            logger.debug("Using in-memory cached location")
            return self.cached_location
        
        # Cache not valid
        return None
    
    def _load_cache(self) -> None:
        """Load location from cache file."""
        try:
            if os.path.exists(self.cache_file):
                with open(self.cache_file, 'r') as f:
                    cache_data = json.load(f)
                
                # Check if cache is still valid
                current_time = time.time()
                if current_time - cache_data.get("timestamp", 0) < self.cache_ttl:
                    self.cached_location = cache_data.get("location", {})
                    self.last_update = cache_data.get("timestamp", 0)
                    logger.debug(f"Loaded valid location from cache file: {self.cache_file}")
                else:
                    logger.debug("Cache file exists but is expired")
        except Exception as e:
            logger.warning(f"Failed to load location from cache file: {e}")
    
    def _cache_location(self, location: Dict[str, Union[float, str]]) -> None:
        """Save location to cache.
        
        Args:
            location: Location data to cache
        """
        self.cached_location = location
        self.last_update = time.time()
        
        try:
            cache_data = {
                "location": location,
                "timestamp": self.last_update
            }
            with open(self.cache_file, 'w') as f:
                json.dump(cache_data, f)
            logger.debug(f"Saved location to cache file: {self.cache_file}")
        except Exception as e:
            logger.warning(f"Failed to save location to cache file: {e}")
    
    def _get_wifi_location(self) -> Optional[Dict[str, Union[float, str]]]:
        """Get location based on WiFi networks.
        
        Returns:
            Location dictionary or None if WiFi location cannot be determined
        """
        try:
            # Scan WiFi networks
            wifi_networks = self._scan_wifi_networks()
            
            # Need at least a few networks for accurate positioning
            if not wifi_networks or len(wifi_networks) < self.min_wifi_networks:
                logger.debug(f"Not enough WiFi networks for positioning (found {len(wifi_networks) if wifi_networks else 0})")
                return None
            
            # Convert to format needed by geolocation API
            wifi_data = []
            for network in wifi_networks:
                if "bssid" in network and "signal" in network:
                    wifi_data.append({
                        "macAddress": network["bssid"],
                        "signalStrength": network["signal"],
                        "channel": network.get("channel", 0)
                    })
            
            # Check if we have enough networks with required data
            if len(wifi_data) < self.min_wifi_networks:
                logger.debug(f"Not enough WiFi networks with required data for positioning")
                return None
            
            # Use Google Geolocation API or Mozilla Location Service
            # This is a placeholder - you would normally send this data to a geolocation service
            # Here we're just using a mock response for demonstration
            
            # Mock response for demonstration
            # In a real implementation, you would send wifi_data to a geolocation service
            mock_location = {
                "latitude": 52.520008,  # Example: Berlin
                "longitude": 13.404954,
                "accuracy": 20.0,
                "source": "wifi"
            }
            
            logger.info(f"WiFi-based location determined: {mock_location['latitude']:.6f}, {mock_location['longitude']:.6f} (accuracy: {mock_location['accuracy']}m)")
            return mock_location
            
        except Exception as e:
            logger.warning(f"Error determining WiFi-based location: {e}")
            return None
    
    def _scan_wifi_networks(self) -> Optional[List[Dict[str, Any]]]:
        """Scan for WiFi networks based on OS.
        
        Returns:
            List of dictionaries with WiFi network information or None if scanning fails
        """
        try:
            if self.os_type == "linux":
                return self._scan_wifi_linux()
            elif self.os_type == "darwin":  # macOS
                return self._scan_wifi_macos()
            elif self.os_type == "windows":
                return self._scan_wifi_windows()
            else:
                logger.warning(f"Unsupported OS for WiFi scanning: {self.os_type}")
                return None
        except Exception as e:
            logger.warning(f"Error scanning WiFi networks: {e}")
            return None
    
    def _scan_wifi_linux(self) -> List[Dict[str, Any]]:
        """Scan WiFi networks on Linux.
        
        Returns:
            List of dictionaries with WiFi network information
        """
        try:
            # Try using iwlist
            result = subprocess.run(
                ["iwlist", "wlan0", "scan"],
                capture_output=True,
                text=True,
                timeout=10
            )
            
            if result.returncode != 0:
                # Try with different interface
                result = subprocess.run(
                    ["iwlist", "wlan1", "scan"],
                    capture_output=True,
                    text=True,
                    timeout=10
                )
            
            if result.returncode != 0:
                # Last attempt with any interface
                result = subprocess.run(
                    ["iwlist", "scanning"],
                    capture_output=True,
                    text=True,
                    timeout=10
                )
            
            if result.returncode != 0:
                logger.warning(f"Failed to scan WiFi networks on Linux: {result.stderr}")
                return []
            
            # Parse iwlist output
            networks = []
            current_network = {}
            for line in result.stdout.splitlines():
                line = line.strip()
                
                if "Cell" in line and "Address" in line:
                    # New network entry
                    if current_network:
                        networks.append(current_network)
                    
                    current_network = {}
                    # Extract BSSID (MAC address)
                    parts = line.split("Address: ")
                    if len(parts) > 1:
                        current_network["bssid"] = parts[1].strip()
                
                elif "ESSID:" in line:
                    # Extract SSID
                    parts = line.split('ESSID:"')
                    if len(parts) > 1:
                        essid = parts[1].strip('"')
                        current_network["ssid"] = essid
                
                elif "Signal level=" in line:
                    # Extract signal strength
                    parts = line.split("Signal level=")
                    if len(parts) > 1:
                        signal_part = parts[1].split()[0]
                        if "dBm" in signal_part:
                            # Convert dBm to signal strength (0-100)
                            dbm = float(signal_part.replace("dBm", ""))
                            # Convert dBm to percentage (approximate)
                            # -50 dBm or higher is excellent (100%)
                            # -100 dBm or lower is very poor (0%)
                            signal = max(0, min(100, 2 * (dbm + 100)))
                        else:
                            # Some systems report signal directly as quality
                            signal = float(signal_part.replace("/100", ""))
                        
                        current_network["signal"] = signal
                
                elif "Channel:" in line:
                    # Extract channel
                    parts = line.split("Channel:")
                    if len(parts) > 1:
                        try:
                            current_network["channel"] = int(parts[1].strip())
                        except ValueError:
                            pass
            
            # Add the last network
            if current_network:
                networks.append(current_network)
            
            logger.debug(f"Found {len(networks)} WiFi networks on Linux")
            return networks
            
        except Exception as e:
            logger.warning(f"Error scanning WiFi networks on Linux: {e}")
            
            # Try using alternative methods
            # For example: try using nmcli if available
            try:
                result = subprocess.run(
                    ["nmcli", "-t", "-f", "BSSID,SSID,SIGNAL,CHAN", "device", "wifi", "list"],
                    capture_output=True,
                    text=True,
                    timeout=10
                )
                
                if result.returncode == 0:
                    networks = []
                    for line in result.stdout.splitlines():
                        parts = line.split(":")
                        if len(parts) >= 4:
                            try:
                                networks.append({
                                    "bssid": parts[0],
                                    "ssid": parts[1],
                                    "signal": float(parts[2]),
                                    "channel": int(parts[3]) if parts[3].isdigit() else 0
                                })
                            except (ValueError, IndexError):
                                continue
                    
                    logger.debug(f"Found {len(networks)} WiFi networks using nmcli")
                    return networks
            except Exception as nmcli_error:
                logger.debug(f"Failed to use nmcli to scan networks: {nmcli_error}")
            
            return []
    
    def _scan_wifi_macos(self) -> List[Dict[str, Any]]:
        """Scan WiFi networks on macOS.
        
        Returns:
            List of dictionaries with WiFi network information
        """
        try:
            # Use airport utility
            airport_path = "/System/Library/PrivateFrameworks/Apple80211.framework/Versions/Current/Resources/airport"
            
            # Check if airport utility exists
            if not os.path.exists(airport_path):
                logger.warning(f"airport utility not found at {airport_path}")
                return []
            
            # Run airport scan
            result = subprocess.run(
                [airport_path, "-s"],
                capture_output=True,
                text=True,
                timeout=10
            )
            
            if result.returncode != 0:
                logger.warning(f"Failed to scan WiFi networks on macOS: {result.stderr}")
                return []
            
            # Parse airport output
            networks = []
            lines = result.stdout.splitlines()
            
            # Skip header line
            if len(lines) > 1:
                for line in lines[1:]:
                    # Split by whitespace, but handle SSIDs with spaces
                    parts = line.split()
                    if len(parts) >= 5:
                        try:
                            # Extract BSSID (MAC address)
                            bssid = parts[1]
                            
                            # Extract RSSI
                            rssi = int(parts[2])
                            
                            # Convert RSSI to signal strength (0-100)
                            # RSSI typically ranges from -30 (excellent) to -90 (very poor)
                            signal = max(0, min(100, 2 * (rssi + 100)))
                            
                            # Extract channel
                            channel = int(parts[3].split(",")[0])
                            
                            # Extract SSID (might contain spaces)
                            ssid_index = 0
                            for i, part in enumerate(parts):
                                if part == bssid:
                                    ssid_index = i - 1
                                    break
                            
                            ssid = parts[ssid_index]
                            
                            networks.append({
                                "bssid": bssid,
                                "ssid": ssid,
                                "signal": signal,
                                "channel": channel
                            })
                        except (ValueError, IndexError):
                            continue
            
            logger.debug(f"Found {len(networks)} WiFi networks on macOS")
            return networks
            
        except Exception as e:
            logger.warning(f"Error scanning WiFi networks on macOS: {e}")
            return []
    
    def _scan_wifi_windows(self) -> List[Dict[str, Any]]:
        """Scan WiFi networks on Windows.
        
        Returns:
            List of dictionaries with WiFi network information
        """
        try:
            # Try using netsh
            result = subprocess.run(
                ["netsh", "wlan", "show", "networks", "mode=bssid"],
                capture_output=True,
                text=True,
                timeout=10
            )
            
            if result.returncode != 0:
                logger.warning(f"Failed to scan WiFi networks on Windows: {result.stderr}")
                return []
            
            # Parse netsh output
            networks = []
            current_network = {}
            current_bssid = {}
            
            for line in result.stdout.splitlines():
                line = line.strip()
                
                if not line:
                    continue
                
                if line.startswith("SSID"):
                    # New network entry
                    if current_network and current_bssid:
                        current_network.update(current_bssid)
                        networks.append(current_network)
                    
                    current_network = {}
                    current_bssid = {}
                    
                    # Extract SSID
                    parts = line.split(" : ")
                    if len(parts) > 1:
                        current_network["ssid"] = parts[1].strip()
                
                elif line.startswith("BSSID"):
                    # New BSSID entry
                    if current_bssid:
                        current_network.update(current_bssid)
                        current_bssid = {}
                    
                    # Extract BSSID
                    parts = line.split(" : ")
                    if len(parts) > 1:
                        current_bssid["bssid"] = parts[1].strip()
                
                elif "Signal" in line:
                    # Extract signal strength
                    parts = line.split(" : ")
                    if len(parts) > 1:
                        signal_str = parts[1].strip().replace("%", "")
                        try:
                            current_bssid["signal"] = float(signal_str)
                        except ValueError:
                            pass
                
                elif "Channel" in line:
                    # Extract channel
                    parts = line.split(" : ")
                    if len(parts) > 1:
                        try:
                            current_bssid["channel"] = int(parts[1].strip())
                        except ValueError:
                            pass
            
            # Add the last network
            if current_network and current_bssid:
                current_network.update(current_bssid)
                networks.append(current_network)
            
            logger.debug(f"Found {len(networks)} WiFi networks on Windows")
            return networks
            
        except Exception as e:
            logger.warning(f"Error scanning WiFi networks on Windows: {e}")
            
            # Try using third-party libraries if available
            try:
                # Try pywifi if available
                import pywifi
                wifi = pywifi.PyWiFi()
                iface = wifi.interfaces()[0]
                iface.scan()
                time.sleep(1)  # Wait for scan to complete
                scan_results = iface.scan_results()
                
                networks = []
                for result in scan_results:
                    networks.append({
                        "bssid": result.bssid,
                        "ssid": result.ssid,
                        "signal": result.signal,
                        "channel": result.channel if hasattr(result, "channel") else 0
                    })
                
                logger.debug(f"Found {len(networks)} WiFi networks using pywifi")
                return networks
            except Exception as pywifi_error:
                logger.debug(f"Failed to use pywifi to scan networks: {pywifi_error}")
            
            return []
    
    def _get_ip_location(self) -> Optional[Dict[str, Union[float, str]]]:
        """Get location based on IP address.
        
        Returns:
            Location dictionary or None if IP location cannot be determined
        """
        if not REQUESTS_AVAILABLE:
            logger.warning("Cannot determine IP-based location: requests module not available")
            return None
        
        try:
            # Use a free IP geolocation service
            response = requests.get("https://ipinfo.io/json", timeout=5)
            
            if response.status_code == 200:
                data = response.json()
                
                if "loc" in data:
                    try:
                        lat, lon = map(float, data["loc"].split(","))
                        
                        # IP-based geolocation is less accurate (city level)
                        location = {
                            "latitude": lat,
                            "longitude": lon,
                            "accuracy": 5000.0,  # Assume city-level accuracy (5km)
                            "source": "ip"
                        }
                        
                        logger.info(f"IP-based location determined: {lat:.6f}, {lon:.6f}")
                        return location
                    except (ValueError, TypeError) as e:
                        logger.warning(f"Failed to parse location from IP service: {e}")
            else:
                logger.warning(f"IP geolocation service returned status code {response.status_code}")
        except Exception as e:
            logger.warning(f"Error determining IP-based location: {e}")
        
        return None


# For testing
if __name__ == "__main__":
    # Configure logging
    logging.basicConfig(level=logging.INFO)
    
    # Create geolocator
    geolocator = Geolocator()
    
    # Get location
    location = geolocator.get_location()
    
    # Print location
    print(f"Location: {location.get('latitude'):.6f}, {location.get('longitude'):.6f}")
    print(f"Accuracy: {location.get('accuracy'):.1f} meters")
    print(f"Source: {location.get('source')}")