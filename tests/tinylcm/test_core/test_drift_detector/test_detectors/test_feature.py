import pytest
import numpy as np

from tinylcm.core.drift_detector.detectors.feature import FeatureStatisticsDriftDetector

def test_feature_detector_process_reference(sample_feature_data):
    detector = FeatureStatisticsDriftDetector()
    
    processed = detector._process_reference_data(sample_feature_data)
    
    assert "feature_stats" in processed
    
    stats = processed["feature_stats"]
    assert "size" in stats
    assert "weight" in stats
    
    size_stats = stats["size"]
    assert "mean" in size_stats
    assert "std" in size_stats
    assert "min" in size_stats
    assert "max" in size_stats
    assert "median" in size_stats

def test_feature_detector_calculate_drift(temp_dir):
    detector = FeatureStatisticsDriftDetector(storage_dir=temp_dir)
    
    detector.reference_distribution = {
        "feature_stats": {
            "size": {
                "mean": 1.2,
                "std": 0.3,
                "min": 0.8,
                "max": 1.6,
                "median": 1.25
            }
        }
    }
    
    for _ in range(10):
        detector.current_window.append({
            "features": {
                "size": 1.8
            }
        })
    
    result = detector._calculate_drift()
    
    assert "drift_detected" in result
    assert "drift_type" in result
    assert "feature_drifts" in result
    
    assert "size" in result["feature_drifts"]
    assert result["feature_drifts"]["size"]["drift_detected"] is True

def test_feature_detector_no_drift_on_similar(temp_dir):
    detector = FeatureStatisticsDriftDetector(storage_dir=temp_dir)
    
    detector.reference_distribution = {
        "feature_stats": {
            "size": {
                "mean": 1.2,
                "std": 0.3,
                "min": 0.8,
                "max": 1.6,
                "median": 1.25
            }
        }
    }
    
    for value in [1.1, 1.2, 1.3, 1.0, 1.4]:
        detector.current_window.append({
            "features": {
                "size": value
            }
        })
    
    result = detector._calculate_drift()
    
    if "feature_drifts" in result and "size" in result["feature_drifts"]:
        assert result["feature_drifts"]["size"]["drift_detected"] is False
    assert result["drift_detected"] is False