#!/usr/bin/env python3
"""
System Metrics Collector
-----------------------
Collects system metrics like CPU and memory usage asynchronously.
"""

import asyncio
import logging
import time

import psutil

logger = logging.getLogger("stone_detector.metrics")

class SystemMetricsCollector:
    def __init__(self, interval=60):
        """
        Initialize the system metrics collector.
        
        Args:
            interval (int): Collection interval in seconds
        """
        self.interval = interval
        self.running = False
        self.last_cpu_percent = 0.0
        self.last_memory_percent = 0.0
        
    async def start(self, data_logger=None):
        """Start collecting system metrics asynchronously"""
        self.running = True
        self.data_logger = data_logger
        
        logger.info("System metrics collector started")
        
        while self.running:
            try:
                # Collect metrics asynchronously
                cpu_percent = await self._get_cpu_percent_async()
                memory_info = psutil.virtual_memory()
                memory_percent = memory_info.percent
                
                # Update cached values
                self.last_cpu_percent = cpu_percent
                self.last_memory_percent = memory_percent
                
                # Log metrics if data logger is available
                if self.data_logger:
                    await self._log_metrics_async(cpu_percent, memory_percent)
                
                # Wait for the next interval
                await asyncio.sleep(self.interval)
                
            except asyncio.CancelledError:
                break
            except Exception as e:
                logger.error(f"Error collecting system metrics: {e}")
                await asyncio.sleep(self.interval)
    
    async def _get_cpu_percent_async(self):
        """Get CPU usage percentage asynchronously"""
        loop = asyncio.get_event_loop()
        return await loop.run_in_executor(None, psutil.cpu_percent, 1.0)
    
    async def _log_metrics_async(self, cpu_percent, memory_percent):
        """Log metrics asynchronously"""
        if self.data_logger:
            loop = asyncio.get_event_loop()
            metrics_data = {
                "cpu_usage": cpu_percent,
                "memory_usage": memory_percent,
                "timestamp": time.time()
            }
            
            # Log using TinyLCM DataLogger in executor to avoid blocking
            await loop.run_in_executor(
                None,
                self.data_logger.log_data,
                metrics_data,
                "json",
                None,  # prediction
                None,  # confidence
                None,  # label
                {"type": "system_metrics"}
            )
    
    def stop(self):
        """Stop collecting system metrics"""
        self.running = False
        logger.info("System metrics collector stopped")
    
    def get_current_metrics(self):
        """Get the most recent metrics values"""
        return {
            "cpu_percent": self.last_cpu_percent,
            "memory_percent": self.last_memory_percent
        }