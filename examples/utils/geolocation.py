#!/usr/bin/env python3
"""
Geolocation Utility
------------------
A simple wrapper for the TinyLCM geolocation module, to be used in example applications.
"""

import logging
import os
import time
from typing import Any, Dict, List, Optional, Tuple

# Try to import the main TinyLCM geolocation module
try:
    from tinylcm.utils.geolocation import Geolocator as TinyLCMGeolocator
    TINYLCM_GEOLOCATION_AVAILABLE = True
except ImportError:
    TINYLCM_GEOLOCATION_AVAILABLE = False

logger = logging.getLogger(__name__)

class SimpleGeolocator:
    """A simplified geolocation utility for use in example applications."""
    
    def __init__(
        self,
        api_key: Optional[str] = None,
        cache_ttl: int = 86400,
        fallback_coordinates: List[float] = [0.0, 0.0]
    ):
        """Initialize the geolocation utility.
        
        Args:
            api_key: Optional API key for geolocation service
            cache_ttl: Time-to-live for location cache in seconds
            fallback_coordinates: Default coordinates to use if geolocation fails [lat, lon]
        """
        self.api_key = api_key
        self.cache_ttl = cache_ttl
        self.fallback_coordinates = fallback_coordinates
        self.cache_file = os.path.expanduser("~/.tinylcm_geolocation_cache.json")
        self.cached_location = None
        self.last_update = 0
        
        # If TinyLCM geolocation is available, use it
        self.geolocator = None
        if TINYLCM_GEOLOCATION_AVAILABLE:
            try:
                self.geolocator = TinyLCMGeolocator(
                    api_key=api_key,
                    cache_ttl=cache_ttl,
                    fallback_coordinates=fallback_coordinates
                )
                logger.info("Initialized TinyLCM Geolocator")
            except Exception as e:
                logger.warning(f"Failed to initialize TinyLCM Geolocator: {e}")
        else:
            logger.warning("TinyLCM geolocation module not available, using simple implementation")
            
    def get_location(self) -> Dict[str, Any]:
        """Get the current device location.
        
        Returns:
            Dictionary containing latitude, longitude, accuracy, and source
        """
        # If TinyLCM geolocation is available, delegate to it
        if self.geolocator:
            return self.geolocator.get_location()
            
        # Otherwise, use simple implementation
        current_time = time.time()
        
        # Check if cache has expired
        if self.cached_location is not None and current_time - self.last_update < self.cache_ttl:
            return self.cached_location
            
        # Try to load from cache file
        try:
            if os.path.exists(self.cache_file):
                import json
                with open(self.cache_file, 'r') as f:
                    cache_data = json.load(f)
                if current_time - cache_data.get("timestamp", 0) < self.cache_ttl:
                    self.cached_location = cache_data.get("location", {})
                    self.last_update = cache_data.get("timestamp", 0)
                    return self.cached_location
        except Exception as e:
            logger.warning(f"Failed to load location from cache: {e}")
            
        # Try to get location from IP-based service (fallback method)
        try:
            import requests
            response = requests.get("https://ipinfo.io/json", timeout=5)
            if response.status_code == 200:
                data = response.json()
                if "loc" in data:
                    try:
                        lat, lon = map(float, data["loc"].split(","))
                        location = {
                            "latitude": lat,
                            "longitude": lon,
                            "accuracy": 5000.0,  # IP-based location is not very accurate (5km)
                            "source": "ip"
                        }
                        
                        # Save to cache
                        self._save_to_cache(location)
                        
                        return location
                    except Exception as e:
                        logger.warning(f"Failed to parse location from IP service: {e}")
        except Exception as e:
            logger.warning(f"Failed to get location from IP service: {e}")
            
        # Use fallback coordinates
        fallback = {
            "latitude": self.fallback_coordinates[0],
            "longitude": self.fallback_coordinates[1],
            "accuracy": 10000.0,  # Very low accuracy
            "source": "fallback"
        }
        
        return fallback
        
    def _save_to_cache(self, location: Dict[str, Any]) -> None:
        """Save location to cache.
        
        Args:
            location: Location data to cache
        """
        self.cached_location = location
        self.last_update = time.time()
        
        try:
            import json
            cache_data = {
                "location": location,
                "timestamp": self.last_update
            }
            with open(self.cache_file, 'w') as f:
                json.dump(cache_data, f)
        except Exception as e:
            logger.warning(f"Failed to save location to cache: {e}")
            
# Simple usage example
if __name__ == "__main__":
    logging.basicConfig(level=logging.INFO)
    
    print("Initializing geolocation...")
    geo = SimpleGeolocator()
    
    print("Getting location...")
    location = geo.get_location()
    
    print(f"Device location: {location.get('latitude'):.6f}, {location.get('longitude'):.6f}")
    print(f"Location source: {location.get('source')}")
    print(f"Location accuracy: {location.get('accuracy')}")