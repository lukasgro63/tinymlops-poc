from collections import Counter, deque
import time
from typing import Any, Dict, List, Optional

import numpy as np

from tinylcm.interfaces.monitoring import MetricsProvider
from tinylcm.utils.logging import setup_logger
from tinylcm.utils.metrics import MetricsCalculator

logger = setup_logger(__name__)

class OperationalMetricsCollector(MetricsProvider):
    """Collects and calculates operational metrics.
    
    Focuses on operational aspects including:
    - Inference latency (time for processing)
    - System metrics (CPU, memory, etc.)
    - Confidence scores (if available)
    - Prediction distribution
    - System uptime and stability
    """
    
    def __init__(self, window_size: int = 1000):
        """Initialize the operational metrics collector.
        
        Args:
            window_size: Maximum number of entries to keep in the sliding window
        """
        self.window_size = window_size
        self.latency_window = deque(maxlen=window_size)
        self.confidence_window = deque(maxlen=window_size)
        self.prediction_counts = Counter()
        self.system_metrics_window = deque(maxlen=window_size)
        self.total_inferences = 0
        self.start_time = time.time()
        self.logger = setup_logger(f"{__name__}.{self.__class__.__name__}")
    
    def add_record(self, record: Dict[str, Any]) -> None:
        """Add an inference record to the metrics collection.
        
        Args:
            record: Dictionary containing inference data (prediction, confidence, latency)
        """
        self.total_inferences += 1
        
        # Track confidence if available
        if "confidence" in record and record["confidence"] is not None:
            self.confidence_window.append(record["confidence"])
        
        # Track latency if available
        if "latency_ms" in record and record["latency_ms"] is not None:
            self.latency_window.append(record["latency_ms"])
        
        # Track prediction distribution
        if "prediction" in record:
            self.prediction_counts[record["prediction"]] += 1
            
        # Track system metrics if available
        if "system_metrics" in record and record["system_metrics"] is not None:
            self.system_metrics_window.append(record["system_metrics"])
    
    def add_system_metrics(self, metrics: Dict[str, Any]) -> None:
        """Add system metrics (CPU, memory, etc.) directly.
        
        Args:
            metrics: Dictionary containing system metrics
        """
        self.system_metrics_window.append(metrics)
    
    def get_metrics(self) -> Dict[str, Any]:
        """Get current metrics summary.
        
        Returns:
            Dictionary containing operational metrics
        """
        metrics = {
            "total_inferences": self.total_inferences,
            "prediction_distribution": dict(self.prediction_counts),
            "uptime_seconds": time.time() - self.start_time
        }
        
        # Add latency stats if available
        if self.latency_window:
            metrics["latency"] = MetricsCalculator.calculate_latency_stats(list(self.latency_window))
        
        # Add confidence stats if available
        if self.confidence_window:
            metrics["confidence"] = MetricsCalculator.calculate_confidence_stats(list(self.confidence_window))
        
        # Add system metrics if available
        if self.system_metrics_window:
            # Calculate averages for system metrics
            system_metrics = {}
            for metric_key in self.system_metrics_window[0].keys():
                values = [m.get(metric_key) for m in self.system_metrics_window if metric_key in m]
                values = [v for v in values if v is not None]  # Filter out None values
                if values:
                    system_metrics[metric_key] = {
                        "current": values[-1],
                        "avg": sum(values) / len(values),
                        "max": max(values)
                    }
            
            metrics["system"] = system_metrics
        
        return metrics
    
    def get_statistical_context(self) -> Dict[str, Any]:
        """Get statistical context of metrics for detailed analysis.
        
        Returns:
            Dictionary containing detailed statistical metrics
        """
        context = {
            "prediction_distribution": dict(self.prediction_counts),
            "uptime_seconds": time.time() - self.start_time,
        }
        
        # Add confidence stats
        if self.confidence_window:
            confidence_array = np.array(list(self.confidence_window))
            context["confidence_stats"] = {
                "mean": float(np.mean(confidence_array)),
                "std": float(np.std(confidence_array)),
                "min": float(np.min(confidence_array)),
                "max": float(np.max(confidence_array)),
                "p25": float(np.percentile(confidence_array, 25)),
                "p50": float(np.percentile(confidence_array, 50)),
                "p75": float(np.percentile(confidence_array, 75)),
                "p95": float(np.percentile(confidence_array, 95))
            }
        
        # Add latency stats
        if self.latency_window:
            latency_array = np.array(list(self.latency_window))
            context["latency_stats"] = {
                "mean": float(np.mean(latency_array)),
                "std": float(np.std(latency_array)),
                "min": float(np.min(latency_array)),
                "max": float(np.max(latency_array)),
                "p50": float(np.percentile(latency_array, 50)),
                "p95": float(np.percentile(latency_array, 95)),
                "p99": float(np.percentile(latency_array, 99))
            }
            
            # Calculate throughput (inferences per second)
            if np.mean(latency_array) > 0:
                throughput = 1000.0 / np.mean(latency_array)  # Convert ms to inferences/sec
                context["throughput"] = {
                    "inferences_per_second": float(throughput)
                }
        
        # Add detailed system metrics stats if available
        if self.system_metrics_window:
            system_stats = {}
            
            # Get all unique metric keys
            all_keys = set()
            for metrics in self.system_metrics_window:
                all_keys.update(metrics.keys())
            
            # Calculate stats for each metric
            for key in all_keys:
                values = [metrics.get(key) for metrics in self.system_metrics_window if key in metrics]
                values = [v for v in values if v is not None]  # Filter out None values
                
                if not values:
                    continue
                    
                if all(isinstance(v, (int, float)) for v in values):
                    # Numeric metrics
                    values_array = np.array(values)
                    system_stats[key] = {
                        "mean": float(np.mean(values_array)),
                        "std": float(np.std(values_array)),
                        "min": float(np.min(values_array)),
                        "max": float(np.max(values_array)),
                        "current": float(values[-1])
                    }
                else:
                    # Non-numeric metrics (e.g., strings)
                    system_stats[key] = {
                        "current": values[-1],
                        "values": list(set(values))  # Unique values
                    }
            
            context["system_stats"] = system_stats
        
        return context
    
    def reset(self) -> None:
        """Reset all collected metrics."""
        self.latency_window.clear()
        self.confidence_window.clear()
        self.prediction_counts = Counter()
        self.system_metrics_window.clear()
        self.total_inferences = 0
        self.start_time = time.time()
        self.logger.info("Operational metrics collector reset")
        
    def get_latency_percentiles(self) -> Dict[str, float]:
        """Get latency percentiles.
        
        Returns:
            Dictionary with latency percentiles (p50, p95, p99)
        """
        if not self.latency_window:
            return {
                "p50_ms": 0.0,
                "p95_ms": 0.0, 
                "p99_ms": 0.0
            }
            
        latency_array = np.array(list(self.latency_window))
        return {
            "p50_ms": float(np.percentile(latency_array, 50)),
            "p95_ms": float(np.percentile(latency_array, 95)),
            "p99_ms": float(np.percentile(latency_array, 99))
        }
        
    def get_throughput(self) -> Optional[float]:
        """Calculate throughput in inferences per second.
        
        Returns:
            Throughput value (inferences/sec) or None if no latency data
        """
        if not self.latency_window:
            return None
            
        latency_array = np.array(list(self.latency_window))
        if np.mean(latency_array) > 0:
            return 1000.0 / np.mean(latency_array)  # Convert ms to inferences/sec
        return None
        
    def get_uptime(self) -> float:
        """Get system uptime in seconds.
        
        Returns:
            Uptime in seconds
        """
        return time.time() - self.start_time
        
    def get_system_metrics_summary(self) -> Dict[str, Any]:
        """Get summary of system metrics.
        
        Returns:
            Dictionary of system metrics summaries
        """
        if not self.system_metrics_window:
            return {}
            
        # Get all unique metric keys
        all_keys = set()
        for metrics in self.system_metrics_window:
            all_keys.update(metrics.keys())
        
        # Calculate latest and average for each metric
        summary = {}
        for key in all_keys:
            values = [metrics.get(key) for metrics in self.system_metrics_window if key in metrics]
            values = [v for v in values if v is not None]  # Filter out None values
            
            if not values:
                continue
                
            if all(isinstance(v, (int, float)) for v in values):
                # Numeric metrics
                summary[key] = {
                    "current": values[-1],
                    "avg": sum(values) / len(values),
                    "max": max(values)
                }
            else:
                # Non-numeric metrics
                summary[key] = {
                    "current": values[-1]
                }
        
        return summary