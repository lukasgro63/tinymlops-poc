import pytest

from tinylcm.core.drift_detector.detectors.composite import CompositeDriftDetector
from tinylcm.core.drift_detector.detectors.distribution import DistributionDriftDetector
from tinylcm.core.drift_detector.detectors.confidence import ConfidenceDriftDetector


def test_composite_detector_initialization(temp_dir):
    dist_detector = DistributionDriftDetector(threshold=0.2)
    conf_detector = ConfidenceDriftDetector(threshold=0.15)
    detector = CompositeDriftDetector(
        storage_dir=temp_dir,
        detectors=[dist_detector, conf_detector]
    )
    assert len(detector.detectors) == 2
    assert detector.detectors[0] == dist_detector
    assert detector.detectors[1] == conf_detector


def test_composite_create_reference(temp_dir, sample_distribution_data):
    dist_detector = DistributionDriftDetector()
    conf_detector = ConfidenceDriftDetector()
    detector = CompositeDriftDetector(
        storage_dir=temp_dir,
        detectors=[dist_detector, conf_detector]
    )
    detector.create_reference_distribution(sample_distribution_data)
    assert dist_detector.reference_distribution is not None
    assert conf_detector.reference_distribution is not None


def test_composite_check_drift(temp_dir):
    dist_detector = DistributionDriftDetector(window_size=8, threshold=0.2)
    dist_detector.reference_distribution = {
        "class_distribution": {"A": 0.5, "B": 0.5}
    }
    
    conf_detector = ConfidenceDriftDetector(window_size=8, threshold=0.2)
    conf_detector.reference_distribution = {
        "confidence_stats": {"mean": 0.9, "std": 0.05}
    }
    
    detector = CompositeDriftDetector(
        storage_dir=temp_dir,
        detectors=[dist_detector, conf_detector],
        window_size=8
    )
    detector.reference_distribution = {}
    
    for _ in range(5):
        detector.current_window.append({
            "prediction": "B",
            "confidence": 0.6
        })
    
    result = detector.check_for_drift()
    
    assert result["drift_detected"] is True


def test_composite_reset(temp_dir):
    dist_detector = DistributionDriftDetector()
    conf_detector = ConfidenceDriftDetector()
    dist_detector.current_window.append({"prediction": "A"})
    conf_detector.current_window.append({"confidence": 0.8})
    detector = CompositeDriftDetector(
        storage_dir=temp_dir,
        detectors=[dist_detector, conf_detector]
    )
    detector.current_window.append({"prediction": "B"})
    detector.reset()
    assert len(detector.current_window) == 0
    assert len(dist_detector.current_window) == 0
    assert len(conf_detector.current_window) == 0