"""Tests for the CompositeAnomalyDetector."""

import pytest
from unittest.mock import MagicMock

from tinylcm.core.inference_monitor.anomaly_detectors import (
    CompositeAnomalyDetector,
    ThresholdAnomalyDetector,
    StatisticalAnomalyDetector
)

def test_composite_detector_init():
    """Test that the CompositeAnomalyDetector initializes correctly."""
    detector1 = ThresholdAnomalyDetector()
    detector2 = StatisticalAnomalyDetector()
    
    composite = CompositeAnomalyDetector([detector1, detector2])
    
    assert len(composite.detectors) == 2
    assert composite.detectors[0] == detector1
    assert composite.detectors[1] == detector2

def test_no_anomalies():
    """Test that the composite detector returns no anomalies when no detectors find any."""
    # Create mock detectors
    detector1 = MagicMock()
    detector1.check_for_anomalies.return_value = (False, [])
    
    detector2 = MagicMock()
    detector2.check_for_anomalies.return_value = (False, [])
    
    composite = CompositeAnomalyDetector([detector1, detector2])
    
    record = {"prediction": "cat"}
    context = {}
    
    is_anomaly, reasons = composite.check_for_anomalies(record, context)
    
    assert is_anomaly is False
    assert len(reasons) == 0
    
    # Verify both detectors were called
    detector1.check_for_anomalies.assert_called_once_with(record, context)
    detector2.check_for_anomalies.assert_called_once_with(record, context)

def test_one_detector_finds_anomaly():
    """Test that the composite detector returns anomalies if any detector finds them."""
    # Create mock detectors
    detector1 = MagicMock()
    detector1.check_for_anomalies.return_value = (False, [])
    
    detector2 = MagicMock()
    detector2.check_for_anomalies.return_value = (True, ["Test anomaly"])
    
    composite = CompositeAnomalyDetector([detector1, detector2])
    
    record = {"prediction": "cat"}
    context = {}
    
    is_anomaly, reasons = composite.check_for_anomalies(record, context)
    
    assert is_anomaly is True
    assert len(reasons) == 1
    assert reasons[0] == "Test anomaly"

def test_multiple_detectors_find_anomalies():
    """Test that the composite detector combines all reasons from multiple detectors."""
    # Create mock detectors
    detector1 = MagicMock()
    detector1.check_for_anomalies.return_value = (True, ["Anomaly 1"])
    
    detector2 = MagicMock()
    detector2.check_for_anomalies.return_value = (True, ["Anomaly 2", "Anomaly 3"])
    
    composite = CompositeAnomalyDetector([detector1, detector2])
    
    record = {"prediction": "cat"}
    context = {}
    
    is_anomaly, reasons = composite.check_for_anomalies(record, context)
    
    assert is_anomaly is True
    assert len(reasons) == 3
    assert "Anomaly 1" in reasons
    assert "Anomaly 2" in reasons
    assert "Anomaly 3" in reasons

def test_real_detectors():
    """Test that the composite detector works with real detector instances."""
    # Create real detectors
    threshold = ThresholdAnomalyDetector(confidence_threshold=0.5)
    statistical = StatisticalAnomalyDetector(confidence_z_threshold=-1.0)
    
    composite = CompositeAnomalyDetector([threshold, statistical])
    
    # Record with low confidence and context with statistics
    record = {"confidence": 0.4, "prediction": "cat"}
    context = {
        "confidence_stats": {
            "mean": 0.8,
            "std": 0.1,
            "min": 0.6,
            "max": 0.95
        }
    }
    
    is_anomaly, reasons = composite.check_for_anomalies(record, context)
    
    assert is_anomaly is True
    assert len(reasons) == 2  # Both detectors should find an anomaly
    assert any("Low confidence" in reason for reason in reasons)
    assert any("Statistically low confidence" in reason for reason in reasons)