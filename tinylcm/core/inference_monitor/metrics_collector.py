from collections import Counter, deque
from typing import Any, Dict, List, Optional

import numpy as np

from tinylcm.interfaces.monitoring import MetricsProvider
from tinylcm.utils.logging import setup_logger
from tinylcm.utils.metrics import MetricsCalculator

logger = setup_logger(__name__)

class InferenceMetricsCollector(MetricsProvider):
    """Collects and calculates basic inference metrics.
    
    Focuses specifically on key performance indicators:
    - Latency (inference time)
    - Confidence scores (if available)
    - Prediction distribution
    
    Accuracy tracking is no longer done here as it's handled by the AdaptationTracker.
    """
    
    def __init__(self, window_size: int = 1000):
        """Initialize the metrics collector.
        
        Args:
            window_size: Maximum number of entries to keep in the sliding window
        """
        self.window_size = window_size
        self.latency_window = deque(maxlen=window_size)
        self.confidence_window = deque(maxlen=window_size)
        self.prediction_counts = Counter()
        self.total_inferences = 0
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
    
    def get_metrics(self) -> Dict[str, Any]:
        """Get current metrics summary.
        
        Returns:
            Dictionary containing inference metrics
        """
        metrics = {
            "total_inferences": self.total_inferences,
            "prediction_distribution": dict(self.prediction_counts)
        }
        
        # Add latency stats if available
        if self.latency_window:
            metrics["latency"] = MetricsCalculator.calculate_latency_stats(list(self.latency_window))
        
        # Add confidence stats if available
        if self.confidence_window:
            metrics["confidence"] = MetricsCalculator.calculate_confidence_stats(list(self.confidence_window))
        
        return metrics
    
    def get_statistical_context(self) -> Dict[str, Any]:
        """Get statistical context of metrics for detailed analysis.
        
        Returns:
            Dictionary containing detailed statistical metrics
        """
        context = {
            "prediction_distribution": dict(self.prediction_counts)
        }
        
        # Add confidence stats
        if self.confidence_window:
            confidence_array = np.array(list(self.confidence_window))
            context["confidence_stats"] = {
                "mean": float(np.mean(confidence_array)),
                "std": float(np.std(confidence_array)),
                "min": float(np.min(confidence_array)),
                "max": float(np.max(confidence_array))
            }
        
        # Add latency stats
        if self.latency_window:
            latency_array = np.array(list(self.latency_window))
            context["latency_stats"] = {
                "mean": float(np.mean(latency_array)),
                "std": float(np.std(latency_array)),
                "min": float(np.min(latency_array)),
                "max": float(np.max(latency_array))
            }
            
            # Calculate throughput (inferences per second)
            if np.mean(latency_array) > 0:
                throughput = 1000.0 / np.mean(latency_array)  # Convert ms to inferences/sec
                context["throughput"] = {
                    "inferences_per_second": float(throughput)
                }
        
        return context
    
    def reset(self) -> None:
        """Reset all collected metrics."""
        self.latency_window.clear()
        self.confidence_window.clear()
        self.prediction_counts = Counter()
        self.total_inferences = 0
        self.logger.info("Metrics collector reset")
        
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