# tinylcm/__init__.py
"""
TinyLCM - Lightweight lifecycle management for ML models on edge devices.
"""

__version__ = "0.1.0"

from tinylcm.client.sync_interface import SyncInterface, SyncPackage
from tinylcm.core.data_logger import DataLogger
from tinylcm.core.drift_detector import DriftDetector
from tinylcm.core.inference_monitor import InferenceMonitor

# Direkte Imports aus den Submodulen
from tinylcm.core.model_manager import ModelManager
from tinylcm.core.training_tracker import TrainingTracker

__all__ = [
    "__version__",
    "ModelManager",
    "DataLogger",
    "TrainingTracker",
    "InferenceMonitor",
    "DriftDetector",
    "SyncInterface",
    "SyncPackage"
]