"""Tests for the InferenceMonitor class."""

import json
import os
import time
from pathlib import Path

import pytest

from tinylcm.core.inference_monitor import InferenceMonitor

def test_inference_monitor_init(tmp_path):
    """Test that the InferenceMonitor initializes correctly."""
    monitor = InferenceMonitor(storage_dir=tmp_path)
    
    assert monitor.storage_dir == tmp_path
    assert isinstance(monitor.session_id, str)
    assert monitor._current_log_file is None
    assert monitor._record_count == 0

def test_track_inference_creates_record(tmp_path):
    """Test that track_inference creates and returns a record."""
    monitor = InferenceMonitor(storage_dir=tmp_path)
    
    record = monitor.track_inference(
        input_id="test1",
        prediction="cat",
        confidence=0.9,
        latency_ms=12.5
    )
    
    assert record["input_id"] == "test1"
    assert record["prediction"] == "cat"
    assert record["confidence"] == 0.9
    assert record["latency_ms"] == 12.5
    assert "timestamp" in record
    assert record["session_id"] == monitor.session_id

def test_track_inference_writes_to_file(tmp_path):
    """Test that track_inference writes records to file."""
    monitor = InferenceMonitor(storage_dir=tmp_path)
    
    monitor.track_inference(
        input_id="test1",
        prediction="cat",
        confidence=0.9,
        latency_ms=12.5
    )
    
    log_files = list(tmp_path.glob("*.jsonl"))
    assert len(log_files) == 1
    
    with open(log_files[0], "r") as f:
        content = f.read()
        assert "test1" in content
        assert "cat" in content

def test_track_inference_detects_anomalies(tmp_path):
    """Test that track_inference detects anomalies."""
    monitor = InferenceMonitor(
        storage_dir=tmp_path,
        anomaly_detector=None  # Will use default composite detector
    )
    
    # Add a record with good values first
    monitor.track_inference(
        input_id="good",
        prediction="cat",
        confidence=0.9,
        latency_ms=12.5
    )
    
    # Now add one with low confidence
    record = monitor.track_inference(
        input_id="low_conf",
        prediction="cat",
        confidence=0.2,  # Below default threshold of 0.3
        latency_ms=12.5
    )
    
    assert "anomaly" in record
    assert record["anomaly"] is True
    assert "anomaly_reasons" in record
    assert any("confidence" in reason.lower() for reason in record["anomaly_reasons"])

def test_get_current_metrics(tmp_path, mock_inference_data):
    """Test that get_current_metrics returns the correct metrics."""
    monitor = InferenceMonitor(storage_dir=tmp_path)
    
    # Add some records
    for record in mock_inference_data:
        monitor.track_inference(**record)
    
    metrics = monitor.get_current_metrics()
    
    assert "total_inferences" in metrics
    assert metrics["total_inferences"] == len(mock_inference_data)
    assert "prediction_distribution" in metrics
    assert "cat" in metrics["prediction_distribution"]
    assert "dog" in metrics["prediction_distribution"]
    
    # Should have latency stats
    assert "latency" in metrics
    assert "mean_ms" in metrics["latency"]
    
    # Should have confidence stats
    assert "confidence" in metrics
    assert "mean" in metrics["confidence"]
    
    # Should have accuracy (since we have ground truth)
    assert "accuracy" in metrics
    assert 0 <= metrics["accuracy"] <= 1.0

def test_export_metrics_json(tmp_path, mock_inference_data):
    """Test that export_metrics exports to JSON correctly."""
    monitor = InferenceMonitor(storage_dir=tmp_path)
    
    # Add some records
    for record in mock_inference_data:
        monitor.track_inference(**record)
    
    file_path = monitor.export_metrics(format="json")
    
    assert os.path.exists(file_path)
    with open(file_path, "r") as f:
        metrics = json.load(f)
    
    assert "total_inferences" in metrics
    assert metrics["total_inferences"] == len(mock_inference_data)

def test_export_metrics_csv(tmp_path, mock_inference_data):
    """Test that export_metrics exports to CSV correctly."""
    monitor = InferenceMonitor(storage_dir=tmp_path)
    
    # Add some records
    for record in mock_inference_data:
        monitor.track_inference(**record)
    
    file_path = monitor.export_metrics(format="csv")
    
    assert os.path.exists(file_path)
    
    with open(file_path, "r") as f:
        header = f.readline().strip()
    
    assert "total_inferences" in header
    assert "timestamp" in header

def test_context_manager(tmp_path):
    """Test that the InferenceMonitor works as a context manager."""
    with InferenceMonitor(storage_dir=tmp_path) as monitor:
        monitor.track_inference(
            input_id="test1",
            prediction="cat",
            confidence=0.9,
            latency_ms=12.5
        )
    
    # After exiting the context manager, metrics should be exported
    metric_files = list(tmp_path.glob("metrics_*.json"))
    assert len(metric_files) == 1