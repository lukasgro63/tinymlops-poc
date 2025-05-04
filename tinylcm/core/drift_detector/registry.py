from typing import List

from tinylcm.interfaces.monitoring import DriftDetector as IDriftDetector
from tinylcm.interfaces.registry import Registry

from .detectors.distribution import DistributionDriftDetector
from .detectors.feature import FeatureStatisticsDriftDetector
from .detectors.confidence import ConfidenceDriftDetector
from .detectors.prediction import PredictionFrequencyDriftDetector
from .detectors.composite import CompositeDriftDetector

drift_detector_registry = Registry(IDriftDetector)

# Register standard detectors
drift_detector_registry.register("distribution", DistributionDriftDetector)
drift_detector_registry.register("feature", FeatureStatisticsDriftDetector)
drift_detector_registry.register("confidence", ConfidenceDriftDetector)
drift_detector_registry.register("prediction", PredictionFrequencyDriftDetector)
drift_detector_registry.register("composite", CompositeDriftDetector)