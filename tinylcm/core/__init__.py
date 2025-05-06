"""Core components for the TinyLCM library with adaptive learning capabilities.

This module provides the main functional components of TinyLCM with a focus on adaptive ML:
- InferencePipeline: Base pipeline for monitoring and inference
- AdaptivePipeline: Extended pipeline with adaptation capabilities
- AdaptiveStateManager: For managing and persisting adaptive model states
- AdaptationTracker: For tracking adaptation events and model changes
- OperationalMonitor: For monitoring operational aspects and performance

The adaptive learning architecture follows a modular and configurable approach
with a clear separation between monitoring, inference, and adaptation capabilities.
"""

# Main pipeline and management components
from tinylcm.core.pipeline import InferencePipeline, AdaptivePipeline
from tinylcm.core.state_manager import AdaptiveStateManager
from tinylcm.core.adaptation_tracker import AdaptationTracker
from tinylcm.core.data_structures import FeatureSample, AdaptationEvent, AdaptiveState

# Feature extraction components
from tinylcm.core.feature_extractors import BaseFeatureExtractor, NullFeatureExtractor, PreprocessingFeatureExtractor, TFLiteFeatureExtractor

# Classification components
from tinylcm.core.classifiers import AdaptiveClassifier, LightweightKNN

# Adaptation strategy handlers
from tinylcm.core.handlers import (
    AdaptiveHandler,
    PassiveHandler,
    ActiveHandler,
    HybridHandler
)

# Drift detection components
from tinylcm.core.drift_detection import (
    AutonomousDriftDetector,
    AccuracyCUSUM,
    CUSUMState
)

# Quarantine and heuristic adaptation
from tinylcm.core.quarantine.buffer import QuarantineBuffer
from tinylcm.core.heuristics.adapter import HeuristicAdapter

# Operational monitoring
from tinylcm.core.operational_monitor.monitor import OperationalMonitor
from tinylcm.core.operational_monitor.metrics_collector import OperationalMetricsCollector

# Data logging
from tinylcm.core.data_logger import DataLogger

__all__ = [
    # Core pipeline components
    "InferencePipeline",
    "AdaptivePipeline",
    "AdaptiveStateManager",
    "AdaptationTracker",
    "FeatureSample",
    "AdaptationEvent",
    "AdaptiveState",
    
    # Feature extraction
    "BaseFeatureExtractor",
    "NullFeatureExtractor",
    "PreprocessingFeatureExtractor",
    "TFLiteFeatureExtractor",
    
    # Classification
    "AdaptiveClassifier",
    "LightweightKNN",
    
    # Adaptation handlers
    "AdaptiveHandler",
    "PassiveHandler",
    "ActiveHandler",
    "HybridHandler",
    
    # Drift detection
    "AutonomousDriftDetector",
    "AccuracyCUSUM",
    "CUSUMState",
    
    # Quarantine and heuristic adaptation
    "QuarantineBuffer",
    "HeuristicAdapter",
    
    # Operational monitoring
    "OperationalMonitor",
    "OperationalMetricsCollector",
    
    # Data logging
    "DataLogger",
]
