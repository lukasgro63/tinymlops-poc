"""Monitoring of inference performance metrics.

This module provides components for tracking inference metrics such as:
- Inference latency (processing time per inference)
- Confidence scores (when available from the classifier)
- Prediction distribution
- Throughput (inferences per second)

The components use non-blocking I/O operations to minimize performance impact,
especially important on resource-constrained devices.

Note: Anomaly detection has been removed as it's now handled by the adaptive
components (AdaptationTracker and adaptive handlers in the core module).
"""

from tinylcm.core.inference_monitor.monitor import InferenceMonitor
from tinylcm.core.inference_monitor.metrics_collector import InferenceMetricsCollector

__all__ = [
    "InferenceMonitor",
    "InferenceMetricsCollector",
]