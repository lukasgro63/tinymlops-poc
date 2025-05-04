# Feature extractors for adaptive learning

from tinylcm.core.feature_extractors.base import FeatureExtractor
from tinylcm.core.feature_extractors.tflite import TFLiteFeatureExtractor

__all__ = ['FeatureExtractor', 'TFLiteFeatureExtractor']
