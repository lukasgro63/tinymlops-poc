import pytest

from tinylcm.core.drift_detector.detectors.confidence import ConfidenceDriftDetector


def test_confidence_detector_process_reference(sample_confidence_data):
    detector = ConfidenceDriftDetector()
    
    processed = detector._process_reference_data(sample_confidence_data)
    
    assert "confidence_stats" in processed
    
    stats = processed["confidence_stats"]
    assert "mean" in stats
    assert "std" in stats
    assert "min" in stats
    assert "max" in stats
    assert "median" in stats


def test_confidence_detector_calculate_drift(temp_dir):
    detector = ConfidenceDriftDetector(storage_dir=temp_dir, threshold=0.15)
    
    detector.reference_distribution = {
        "confidence_stats": {
            "mean": 0.9,
            "std": 0.05,
            "min": 0.8,
            "max": 0.98,
            "median": 0.91
        }
    }
    
    for value in [0.6, 0.65, 0.7, 0.68, 0.72]:
        detector.current_window.append({
            "confidence": value
        })
    
    result = detector._calculate_drift()
    
    assert "drift_detected" in result
    assert "drift_type" in result
    assert "confidence_stats" in result
    
    assert result["drift_detected"] is True


def test_confidence_detector_no_drift_on_similar(temp_dir):
    detector = ConfidenceDriftDetector(storage_dir=temp_dir, threshold=0.1)
    
    detector.reference_distribution = {
        "confidence_stats": {
            "mean": 0.9,
            "std": 0.05,
            "min": 0.8,
            "max": 0.98,
            "median": 0.91
        }
    }
    
    for value in [0.88, 0.92, 0.91, 0.89, 0.93]:
        detector.current_window.append({
            "confidence": value
        })
    
    result = detector._calculate_drift()
    
    assert result["drift_detected"] is False


def test_relative_difference_calculation(temp_dir):
    detector = ConfidenceDriftDetector(storage_dir=temp_dir, threshold=0.2)
    
    detector.reference_distribution = {
        "confidence_stats": {
            "mean": 0.8,
            "std": 0.1,
            "min": 0.6,
            "max": 0.95
        }
    }
    
    current_mean = 0.56
    for _ in range(10):
        detector.current_window.append({
            "confidence": current_mean
        })
    
    result = detector._calculate_drift()
    
    assert "confidence_stats" in result
    assert "relative_difference" in result["confidence_stats"]
    assert result["confidence_stats"]["relative_difference"] == pytest.approx(0.3, abs=0.05)
    
    assert result["drift_detected"] is True