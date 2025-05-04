import pytest

from tinylcm.core.inference_monitor.anomaly_detectors import ThresholdAnomalyDetector

def test_threshold_detector_init():
    detector = ThresholdAnomalyDetector(
        confidence_threshold=0.5,
        latency_threshold_ms=50.0
    )
    
    assert detector.confidence_threshold == 0.5
    assert detector.latency_threshold_ms == 50.0

def test_confidence_threshold():
    detector = ThresholdAnomalyDetector(confidence_threshold=0.5)
    
    record = {"confidence": 0.4, "prediction": "cat"}
    context = {}
    
    is_anomaly, reasons = detector.check_for_anomalies(record, context)
    
    assert is_anomaly is True
    assert len(reasons) == 1
    assert "Low confidence" in reasons[0]
    
    record = {"confidence": 0.6, "prediction": "cat"}
    
    is_anomaly, reasons = detector.check_for_anomalies(record, context)
    
    assert is_anomaly is False
    assert len(reasons) == 0

def test_latency_threshold():
    detector = ThresholdAnomalyDetector(latency_threshold_ms=50.0)
    
    record = {"latency_ms": 60.0, "prediction": "cat"}
    context = {}
    
    is_anomaly, reasons = detector.check_for_anomalies(record, context)
    
    assert is_anomaly is True
    assert len(reasons) == 1
    assert "High latency" in reasons[0]
    
    record = {"latency_ms": 40.0, "prediction": "cat"}
    
    is_anomaly, reasons = detector.check_for_anomalies(record, context)
    
    assert is_anomaly is False
    assert len(reasons) == 0

def test_ground_truth_check():
    detector = ThresholdAnomalyDetector()
    
    record = {
        "prediction": "cat",
        "ground_truth": "dog"
    }
    context = {}
    
    is_anomaly, reasons = detector.check_for_anomalies(record, context)
    
    assert is_anomaly is True
    assert len(reasons) == 1
    assert "Incorrect prediction" in reasons[0]
    
    record = {
        "prediction": "cat",
        "ground_truth": "cat"
    }
    
    is_anomaly, reasons = detector.check_for_anomalies(record, context)
    
    assert is_anomaly is False
    assert len(reasons) == 0

def test_multiple_anomalies():
    detector = ThresholdAnomalyDetector(
        confidence_threshold=0.5,
        latency_threshold_ms=50.0
    )
    
    record = {
        "confidence": 0.4,
        "latency_ms": 60.0,
        "prediction": "cat",
        "ground_truth": "dog"
    }
    context = {}
    
    is_anomaly, reasons = detector.check_for_anomalies(record, context)
    
    assert is_anomaly is True
    assert len(reasons) == 3
    assert any("confidence" in reason.lower() for reason in reasons)
    assert any("latency" in reason.lower() for reason in reasons)
    assert any("prediction" in reason.lower() for reason in reasons)