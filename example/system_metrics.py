#!/usr/bin/env python3
"""
System Metrics Collector
-----------------------
Collects and logs system metrics such as CPU, memory, disk usage, and temperature 
for Raspberry Pi monitoring.
"""

import asyncio
import json
import logging
import os
import platform
import time
from datetime import datetime
from typing import Dict, Any, Optional

# Import DATA_TYPE constants from tinylcm
try:
    from tinylcm.constants import DATA_TYPE_JSON
except ImportError:
    # Fallback if import fails
    DATA_TYPE_JSON = "json"

import psutil

# Try to import Raspberry Pi specific modules
try:
    import gpiozero
    HAS_GPIOZERO = True
except ImportError:
    HAS_GPIOZERO = False

logger = logging.getLogger(__name__)

class SystemMetricsCollector:
    """Collects system metrics from the Raspberry Pi."""
    
    def __init__(self, interval: int = 60):
        """Initialize the metrics collector.
        
        Args:
            interval: Collection interval in seconds
        """
        self.interval = interval
        self.is_running = False
        self.stop_event = asyncio.Event()
    
    async def start(self, data_logger=None):
        """Start collecting metrics at regular intervals.
        
        Args:
            data_logger: Optional DataLogger to log metrics
        """
        self.is_running = True
        try:
            while not self.stop_event.is_set():
                metrics = self.collect_metrics()
                
                # Log metrics if data_logger is provided
                if data_logger and metrics:
                    try:
                        self._log_metrics(data_logger, metrics)
                    except Exception as e:
                        logger.error(f"Error logging metrics: {e}")
                
                # Log metrics to console
                logger.debug(f"System metrics: {json.dumps(metrics, indent=2)}")
                
                # Wait for the next interval
                try:
                    await asyncio.wait_for(self.stop_event.wait(), self.interval)
                except asyncio.TimeoutError:
                    # This is expected when the timeout expires before the event is set
                    pass
        
        except asyncio.CancelledError:
            logger.info("Metrics collector task cancelled")
            raise
        except Exception as e:
            logger.error(f"Error in metrics collection: {e}")
        finally:
            self.is_running = False
    
    def stop(self):
        """Stop the metrics collection."""
        self.stop_event.set()
    
    def collect_metrics(self) -> Dict[str, Any]:
        """Collect system metrics.
        
        Returns:
            Dictionary containing system metrics
        """
        metrics = {
            "timestamp": datetime.now().isoformat(),
            "system": self._get_system_info(),
            "cpu": self._get_cpu_metrics(),
            "memory": self._get_memory_metrics(),
            "disk": self._get_disk_metrics(),
            "network": self._get_network_metrics()
        }
        
        # Add Raspberry Pi specific metrics if available
        if HAS_GPIOZERO:
            metrics["temperature"] = self._get_temperature_metrics()
        
        return metrics
    
    def _get_system_info(self) -> Dict[str, Any]:
        """Get basic system information."""
        return {
            "platform": platform.platform(),
            "system": platform.system(),
            "node": platform.node(),
            "release": platform.release(),
            "machine": platform.machine(),
            "processor": platform.processor(),
            "uptime_seconds": time.time() - psutil.boot_time()
        }
    
    def _get_cpu_metrics(self) -> Dict[str, Any]:
        """Get CPU usage metrics."""
        return {
            "percent": psutil.cpu_percent(interval=0.5),
            "count": psutil.cpu_count(),
            "frequency": self._safe_dict(psutil.cpu_freq()),
            "load_avg": os.getloadavg() if hasattr(os, 'getloadavg') else None
        }
    
    def _get_memory_metrics(self) -> Dict[str, Any]:
        """Get memory usage metrics."""
        mem = psutil.virtual_memory()
        swap = psutil.swap_memory()
        
        return {
            "total_mb": mem.total / (1024 * 1024),
            "available_mb": mem.available / (1024 * 1024),
            "used_mb": mem.used / (1024 * 1024),
            "percent": mem.percent,
            "swap_total_mb": swap.total / (1024 * 1024),
            "swap_used_mb": swap.used / (1024 * 1024),
            "swap_percent": swap.percent
        }
    
    def _get_disk_metrics(self) -> Dict[str, Any]:
        """Get disk usage metrics."""
        metrics = {}
        
        # Get overall disk usage for root partition
        root = psutil.disk_usage('/')
        metrics["root"] = {
            "total_gb": root.total / (1024**3),
            "used_gb": root.used / (1024**3),
            "free_gb": root.free / (1024**3),
            "percent": root.percent
        }
        
        # Get I/O stats for all disks
        try:
            disk_io = psutil.disk_io_counters(perdisk=True)
            metrics["io"] = {disk: self._safe_dict(stats) for disk, stats in disk_io.items()}
        except Exception as e:
            logger.warning(f"Could not collect disk I/O metrics: {e}")
        
        return metrics
    
    def _get_network_metrics(self) -> Dict[str, Any]:
        """Get network usage metrics."""
        try:
            net_io = psutil.net_io_counters(pernic=True)
            return {
                nic: {
                    "bytes_sent": stats.bytes_sent,
                    "bytes_recv": stats.bytes_recv,
                    "packets_sent": stats.packets_sent,
                    "packets_recv": stats.packets_recv,
                    "errin": stats.errin if hasattr(stats, "errin") else None,
                    "errout": stats.errout if hasattr(stats, "errout") else None,
                    "dropin": stats.dropin if hasattr(stats, "dropin") else None,
                    "dropout": stats.dropout if hasattr(stats, "dropout") else None
                }
                for nic, stats in net_io.items()
                if nic != 'lo'  # Skip loopback interface
            }
        except Exception as e:
            logger.warning(f"Could not collect network metrics: {e}")
            return {}
    
    def _get_temperature_metrics(self) -> Dict[str, Any]:
        """Get temperature metrics (Raspberry Pi specific)."""
        metrics = {}
        
        try:
            # Get CPU temperature
            cpu_temp = gpiozero.CPUTemperature().temperature
            metrics["cpu_celsius"] = cpu_temp
        except Exception as e:
            logger.warning(f"Could not get CPU temperature: {e}")
        
        return metrics
    
    def _safe_dict(self, obj) -> Dict[str, Any]:
        """Safely convert a named tuple or object to a dictionary.
        
        Args:
            obj: Object to convert to dictionary
            
        Returns:
            Dictionary representation of the object, or None if conversion fails
        """
        if obj is None:
            return None
        try:
            if hasattr(obj, "_asdict"):
                return obj._asdict()
            elif hasattr(obj, "__dict__"):
                return obj.__dict__
            else:
                return {k: getattr(obj, k) for k in dir(obj) 
                        if not k.startswith('_') and not callable(getattr(obj, k))}
        except Exception:
            return None
    
    def _log_metrics(self, data_logger, metrics: Dict[str, Any]):
        """Log metrics using the TinyLCM DataLogger.
        
        Args:
            data_logger: TinyLCM DataLogger instance
            metrics: Metrics to log
        """
        # Log metrics as metadata
        data_logger.log_data(
            input_data=json.dumps(metrics),
            input_type=DATA_TYPE_JSON,
            metadata={
                "type": "system_metrics",
                "timestamp": metrics["timestamp"],
                "device": platform.node()
            }
        )


# Test function
async def main():
    """Test the SystemMetricsCollector functionality."""
    logging.basicConfig(level=logging.INFO)
    
    # Create metrics collector with a short interval for testing
    collector = SystemMetricsCollector(interval=5)
    
    print("Starting metrics collection (press Ctrl+C to stop)...")
    try:
        # Collect metrics for 15 seconds
        collection_task = asyncio.create_task(collector.start())
        await asyncio.sleep(15)
        collector.stop()
        await collection_task
    except asyncio.CancelledError:
        pass
    
    print("Metrics collection stopped")

if __name__ == "__main__":
    asyncio.run(main())