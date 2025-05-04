"""Interfaces for TinyLCM components."""

from tinylcm.interfaces.storage import (
    StorageBackend,
    ModelStorageBackend,
    StreamingStorageBackend
)
from tinylcm.interfaces.monitoring import (
    AnomalyDetector,
    MetricsProvider,
    MetricsConsumer,
    DataStreamProcessor
)
from tinylcm.interfaces.registry import Registry

__all__ = [
    # Storage interfaces
    "StorageBackend",
    "ModelStorageBackend",
    "StreamingStorageBackend",
    
    # Monitoring interfaces
    "AnomalyDetector",
    "MetricsProvider",
    "MetricsConsumer",
    "DataStreamProcessor",
    
    # Registry
    "Registry"
]