"""Tests for the StatisticalAnomalyDetector."""

import pytest
import numpy as np

from tinylcm.core.inference_monitor.anomaly_detectors import StatisticalAnomalyDetector

def test_statistical_detector_init():
    """Test that the StatisticalAnomalyDetector initializes correctly."""
    detector = StatisticalAnomalyDetector(
        confidence_z_threshold=-1.5,
        latency_z_threshold=1.5
    )
    
    assert detector.confidence_z_threshold == -1.5
    assert detector.latency_z_threshold == 1.5

def test_confidence_z_score():
    """Test that the detector identifies statistically low confidence."""
    detector = StatisticalAnomalyDetector(confidence_z_threshold=-1.0)
    
    # Create a context with confidence stats
    context = {
        "confidence_stats": {
            "mean": 0.8,
            "std": 0.1,
            "min": 0.6,
            "max": 0.95
        }
    }
    
    # Record with confidence significantly below mean (z-score = -2.0)
    record = {"confidence": 0.6, "prediction": "cat"}
    
    is_anomaly, reasons = detector.check_for_anomalies(record, context)
    
    assert is_anomaly is True
    assert len(reasons) == 1
    assert "Statistically low confidence" in reasons[0]
    
    # Record with confidence just slightly below mean (z-score = -0.5)
    record = {"confidence": 0.75, "prediction": "cat"}
    
    is_anomaly, reasons = detector.check_for_anomalies(record, context)
    
    assert is_anomaly is False
    assert len(reasons) == 0

def test_latency_z_score():
    """Test that the detector identifies statistically high latency."""
    detector = StatisticalAnomalyDetector(latency_z_threshold=1.0)
    
    # Create a context with latency stats
    context = {
        "latency_stats": {
            "mean": 50.0,
            "std": 10.0,
            "min": 30.0,
            "max": 80.0
        }
    }
    
    # Record with latency significantly above mean (z-score = 2.0)
    record = {"latency_ms": 70.0, "prediction": "cat"}
    
    is_anomaly, reasons = detector.check_for_anomalies(record, context)
    
    assert is_anomaly is True
    assert len(reasons) == 1
    assert "Statistically high latency" in reasons[0]
    
    # Record with latency just slightly above mean (z-score = 0.5)
    record = {"latency_ms": 55.0, "prediction": "cat"}
    
    is_anomaly, reasons = detector.check_for_anomalies(record, context)
    
    assert is_anomaly is False
    assert len(reasons) == 0

def test_rare_prediction():
    """Test that the detector identifies rare predictions."""
    detector = StatisticalAnomalyDetector()
    
    # Create a context with prediction distribution
    context = {
        "prediction_distribution": {
            "cat": 90,
            "dog": 80,
            "bird": 30,
            "fish": 2  # Very rare
        }
    }
    
    # Record with a rare prediction
    record = {"prediction": "fish"}
    
    is_anomaly, reasons = detector.check_for_anomalies(record, context)
    
    assert is_anomaly is True
    assert len(reasons) == 1
    assert "Rare prediction" in reasons[0]
    
    # Record with a common prediction
    record = {"prediction": "cat"}
    
    is_anomaly, reasons = detector.check_for_anomalies(record, context)
    
    assert is_anomaly is False
    assert len(reasons) == 0

def test_no_stats_in_context():
    """Test that the detector handles missing stats in context."""
    detector = StatisticalAnomalyDetector()
    
    # Empty context
    context = {}
    
    # Record with data
    record = {
        "confidence": 0.7,
        "latency_ms": 50.0,
        "prediction": "cat"
    }
    
    is_anomaly, reasons = detector.check_for_anomalies(record, context)
    
    # Should not find anomalies when no context stats are available
    assert is_anomaly is False
    assert len(reasons) == 0

def test_zero_std_in_stats():
    """Test that the detector handles zero standard deviation in stats."""
    detector = StatisticalAnomalyDetector()
    
    # Context with zero std
    context = {
        "confidence_stats": {
            "mean": 0.8,
            "std": 0.0,  # Zero std
            "min": 0.8,
            "max": 0.8
        },
        "latency_stats": {
            "mean": 50.0,
            "std": 0.0,  # Zero std
            "min": 50.0,
            "max": 50.0
        }
    }
    
    # Record with different values
    record = {
        "confidence": 0.7,
        "latency_ms": 60.0,
        "prediction": "cat"
    }
    
    # Should not crash when std is zero
    is_anomaly, reasons = detector.check_for_anomalies(record, context)
    
    # The result doesn't matter as much as not crashing
    assert isinstance(is_anomaly, bool)