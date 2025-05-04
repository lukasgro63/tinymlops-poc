import pytest
import numpy as np
from collections import Counter

from tinylcm.core.inference_monitor import InferenceMetricsCollector

def test_metrics_collector_init():
    collector = InferenceMetricsCollector(window_size=100)
    assert collector.window_size == 100
    assert len(collector.latency_window) == 0
    assert len(collector.confidence_window) == 0
    assert collector.total_inferences == 0
    assert collector.ground_truth_correct == 0
    assert collector.ground_truth_total == 0

def test_add_record():
    collector = InferenceMetricsCollector()
    record = {
        "confidence": 0.9,
        "latency_ms": 12.5,
        "prediction": "cat",
        "ground_truth": "cat"
    }
    collector.add_record(record)
    assert collector.total_inferences == 1
    assert len(collector.confidence_window) == 1
    assert collector.confidence_window[0] == 0.9
    assert len(collector.latency_window) == 1
    assert collector.latency_window[0] == 12.5
    assert collector.prediction_counts["cat"] == 1
    assert collector.ground_truth_total == 1
    assert collector.ground_truth_correct == 1

def test_add_record_no_ground_truth():
    collector = InferenceMetricsCollector()
    record = {
        "confidence": 0.9,
        "latency_ms": 12.5,
        "prediction": "cat"
    }
    collector.add_record(record)
    assert collector.total_inferences == 1
    assert collector.ground_truth_total == 0
    assert collector.ground_truth_correct == 0

def test_add_record_incorrect_prediction():
    collector = InferenceMetricsCollector()
    record = {
        "confidence": 0.9,
        "latency_ms": 12.5,
        "prediction": "cat",
        "ground_truth": "dog"
    }
    collector.add_record(record)
    assert collector.total_inferences == 1
    assert collector.ground_truth_total == 1
    assert collector.ground_truth_correct == 0

def test_get_metrics():
    collector = InferenceMetricsCollector()
    records = [
        {
            "confidence": 0.9,
            "latency_ms": 12.5,
            "prediction": "cat",
            "ground_truth": "cat"
        },
        {
            "confidence": 0.8,
            "latency_ms": 13.2,
            "prediction": "dog",
            "ground_truth": "dog"
        },
        {
            "confidence": 0.7,
            "latency_ms": 11.8,
            "prediction": "cat",
            "ground_truth": "dog"
        }
    ]
    for record in records:
        collector.add_record(record)
    metrics = collector.get_metrics()
    assert metrics["total_inferences"] == 3
    assert "prediction_distribution" in metrics
    assert metrics["prediction_distribution"]["cat"] == 2
    assert metrics["prediction_distribution"]["dog"] == 1
    assert "latency" in metrics
    assert "mean_ms" in metrics["latency"]
    assert abs(metrics["latency"]["mean_ms"] - np.mean([12.5, 13.2, 11.8])) < 0.001
    assert "confidence" in metrics
    assert "mean" in metrics["confidence"]
    assert abs(metrics["confidence"]["mean"] - np.mean([0.9, 0.8, 0.7])) < 0.001
    assert "accuracy" in metrics
    assert metrics["accuracy"] == 2/3

def test_get_statistical_context():
    collector = InferenceMetricsCollector()
    records = [
        {
            "confidence": 0.9,
            "latency_ms": 12.5,
            "prediction": "cat",
            "ground_truth": "cat"
        },
        {
            "confidence": 0.8,
            "latency_ms": 13.2,
            "prediction": "dog",
            "ground_truth": "dog"
        },
        {
            "confidence": 0.7,
            "latency_ms": 11.8,
            "prediction": "cat",
            "ground_truth": "dog"
        }
    ]
    for record in records:
        collector.add_record(record)
    context = collector.get_statistical_context()
    assert "prediction_distribution" in context
    assert context["prediction_distribution"]["cat"] == 2
    assert context["prediction_distribution"]["dog"] == 1
    assert "confidence_stats" in context
    assert "mean" in context["confidence_stats"]
    assert abs(context["confidence_stats"]["mean"] - np.mean([0.9, 0.8, 0.7])) < 0.001
    assert "latency_stats" in context
    assert "mean" in context["latency_stats"]
    assert abs(context["latency_stats"]["mean"] - np.mean([12.5, 13.2, 11.8])) < 0.001

def test_reset():
    collector = InferenceMetricsCollector()
    record = {
        "confidence": 0.9,
        "latency_ms": 12.5,
        "prediction": "cat",
        "ground_truth": "cat"
    }
    collector.add_record(record)
    collector.reset()
    assert collector.total_inferences == 0
    assert len(collector.confidence_window) == 0
    assert len(collector.latency_window) == 0
    assert len(collector.prediction_counts) == 0
    assert collector.ground_truth_total == 0
    assert collector.ground_truth_correct == 0