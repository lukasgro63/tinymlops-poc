import time
import pytest
from pathlib import Path

from tinylcm.core.drift_detector.base import DriftDetector

def test_drift_detector_initialization(temp_dir):
    detector = DriftDetector(storage_dir=temp_dir, window_size=50, threshold=0.2)
    assert Path(temp_dir).exists()
    assert (Path(temp_dir) / "reference").exists()
    assert (Path(temp_dir) / "snapshots").exists()
    assert detector.window_size == 50
    assert detector.threshold == 0.2
    assert detector.session_id is not None
    assert len(detector.current_window) == 0
    assert detector.reference_distribution is None

def test_register_drift_callback():
    detector = DriftDetector()
    callback_data = []
    def test_callback(data):
        callback_data.append(data)
    detector.register_drift_callback(test_callback)
    assert len(detector.drift_callbacks) == 1
    assert detector.drift_callbacks[0] == test_callback

def test_save_current_state(temp_dir):
    detector = DriftDetector(storage_dir=temp_dir)
    detector.current_window.append({"prediction": "cat", "confidence": 0.9})
    detector.current_window.append({"prediction": "dog", "confidence": 0.8})
    state_path = detector._save_current_state()
    assert Path(state_path).exists()
    import json
    with open(state_path, 'r') as f:
        state = json.load(f)
    assert state["session_id"] == detector.session_id
    assert len(state["current_window"]) == 2

def test_create_reference_distribution(temp_dir, sample_distribution_data):
    detector = DriftDetector(storage_dir=temp_dir)
    detector.create_reference_distribution(sample_distribution_data)
    assert detector.reference_distribution is not None
    assert detector.reference_metadata is not None
    reference_files = list(Path(temp_dir).glob("reference/reference_*.json"))
    assert len(reference_files) > 0

def test_reset():
    detector = DriftDetector()
    detector.current_window.append({"prediction": "cat"})
    detector.last_check_time = time.time()
    detector.reset()
    assert len(detector.current_window) == 0
    assert detector.last_check_time == 0