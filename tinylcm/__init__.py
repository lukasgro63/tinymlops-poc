# tinylcm/__init__.py
"""
TinyLCM - Lightweight lifecycle management for ML models on edge devices.
"""

__version__ = "0.1.0"

from tinylcm.client.sync_interface import SyncInterface, SyncPackage
from tinylcm.core.data_logger import DataLogger
from tinylcm.core.inference_monitor import InferenceMonitor

# Import the new core adaptive components
from tinylcm.core.pipeline import AdaptivePipeline
from tinylcm.core.state_manager import StateManager
from tinylcm.core.adaptation_tracker import AdaptationTracker
from tinylcm.core.data_structures import FeatureSample, AdaptationEvent

# Import specific components
from tinylcm.core.classifiers import LightweightKNN
from tinylcm.core.feature_extractors import TFLiteFeatureExtractor
from tinylcm.core.handlers import PassiveHandler, ActiveHandler, HybridHandler

__all__ = [
    "__version__",
    # New Adaptive Components
    "AdaptivePipeline",
    "StateManager",
    "AdaptationTracker",
    "FeatureSample",
    "AdaptationEvent",
    "LightweightKNN",
    "TFLiteFeatureExtractor",
    "PassiveHandler",
    "ActiveHandler",
    "HybridHandler",
    
    # Retained components
    "DataLogger",
    "InferenceMonitor",
    "SyncInterface",
    "SyncPackage"
]