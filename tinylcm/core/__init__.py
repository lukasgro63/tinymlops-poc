"""Core components for the TinyLCM library with adaptive learning capabilities.

This module provides the main functional components of TinyLCM with a focus on adaptive ML:
- AdaptivePipeline: Integration of feature extraction, classification, and adaptation
- StateManager: For managing and persisting adaptive model states
- AdaptationTracker: For tracking adaptation events and model changes

The adaptive learning architecture replaces the previous ModelManager, TrainingTracker, 
and DriftDetector components with a more integrated approach to online learning.
"""

# Main pipeline and management components
from tinylcm.core.pipeline import AdaptivePipeline
from tinylcm.core.state_manager import StateManager
from tinylcm.core.adaptation_tracker import AdaptationTracker
from tinylcm.core.data_structures import FeatureSample

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
from tinylcm.core.drift_detection import AccuracyCUSUM, CUSUMState

# Retain DataLogger for backward compatibility and new feature logging
from tinylcm.core.data_logger import DataLogger

# Retain InferenceMonitor for basic inference monitoring
from tinylcm.core.inference_monitor import (
    InferenceMonitor,
    InferenceMetricsCollector
)

__all__ = [
    # Core adaptive components
    "AdaptivePipeline",
    "StateManager",
    "AdaptationTracker",
    "FeatureSample",
    
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
    "AccuracyCUSUM",
    "CUSUMState",
    
    # Retained for backward compatibility
    "DataLogger",
    "InferenceMonitor",
    "InferenceMetricsCollector",
]
