# tinylcm/__init__.py
"""
TinyLCM - Lightweight lifecycle management for ML models on edge devices.

TinyLCM provides a comprehensive framework for autonomous and adaptive machine learning
on resource-constrained devices, with a focus on:

1. Autonomous drift detection without requiring ground truth labels
2. On-device adaptation using heuristic approaches
3. Opportunistic server validation and feedback
4. Lightweight state management with rollback capabilities
5. Comprehensive adaptation tracking and operational monitoring

For detailed documentation, see the README and examples.
"""

__version__ = "0.2.0"

# Client components
from tinylcm.client.sync_interface import SyncInterface, SyncPackage
from tinylcm.client.sync_client import SyncClient
from tinylcm.client.connection_manager import ConnectionManager

# Core pipeline components
from tinylcm.core.pipeline import InferencePipeline, AdaptivePipeline
from tinylcm.core.state_manager import AdaptiveStateManager
from tinylcm.core.adaptation_tracker import AdaptationTracker
from tinylcm.core.data_structures import FeatureSample, AdaptationEvent, AdaptiveState

# Data handling components
from tinylcm.core.data_logger import DataLogger
from tinylcm.core.operational_monitor.monitor import OperationalMonitor

# Drift detection and adaptation
from tinylcm.core.drift_detection import AutonomousDriftDetector
from tinylcm.core.quarantine.buffer import QuarantineBuffer
from tinylcm.core.heuristics.adapter import HeuristicAdapter

# Specific implementation components
from tinylcm.core.classifiers import LightweightKNN
from tinylcm.core.feature_extractors import TFLiteFeatureExtractor
from tinylcm.core.handlers import PassiveHandler, ActiveHandler, HybridHandler

__all__ = [
    "__version__",
    
    # Pipeline components
    "InferencePipeline",
    "AdaptivePipeline",
    "AdaptiveStateManager",
    "AdaptationTracker",
    
    # Data structures
    "FeatureSample",
    "AdaptationEvent",
    "AdaptiveState",
    
    # Classification and feature extraction
    "LightweightKNN",
    "TFLiteFeatureExtractor",
    
    # Adaptation and drift detection
    "PassiveHandler",
    "ActiveHandler",
    "HybridHandler",
    "AutonomousDriftDetector",
    "QuarantineBuffer",
    "HeuristicAdapter",
    
    # Data and monitoring
    "DataLogger",
    "OperationalMonitor",
    
    # Client components
    "SyncClient",
    "SyncInterface",
    "SyncPackage",
    "ConnectionManager"
]