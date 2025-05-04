import pytest

from tinylcm.core.drift_detector.registry import drift_detector_registry
from tinylcm.interfaces.monitoring import DriftDetector as IDriftDetector

def test_registry_has_standard_detectors():
    registered = drift_detector_registry.list_registered()
    assert "distribution" in registered
    assert "feature" in registered
    assert "confidence" in registered
    assert "prediction" in registered
    assert "composite" in registered

def test_registry_get():
    detector_cls = drift_detector_registry.get("distribution")
    assert isinstance(detector_cls, type)
    assert issubclass(detector_cls, IDriftDetector)

def test_registry_create():
    detector = drift_detector_registry.create("distribution", threshold=0.15)
    assert detector.threshold == 0.15
    assert hasattr(detector, "check_for_drift")

def test_registry_invalid_name():
    with pytest.raises(KeyError):
        drift_detector_registry.get("nonexistent_detector")