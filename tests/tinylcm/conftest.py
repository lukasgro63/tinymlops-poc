"""Configuration for pytest fixtures."""

import json
import os
import shutil
import tempfile
from pathlib import Path

import pytest


@pytest.fixture
def temp_dir():
    """Create a temporary directory for tests."""
    dir_path = tempfile.mkdtemp()
    yield dir_path
    shutil.rmtree(dir_path)


@pytest.fixture
def mock_model_file(temp_dir):
    """Create a mock model file for testing."""
    model_dir = os.path.join(temp_dir, "models")
    os.makedirs(model_dir, exist_ok=True)

    model_path = os.path.join(model_dir, "test_model.json")
    model_content = {"weights": [1.0, 2.0, 3.0], "layers": [10, 5, 1]}

    with open(model_path, "w", encoding="utf-8") as f:
        json.dump(model_content, f)

    return model_path


@pytest.fixture
def mock_inference_data():
    """Create mock inference data for testing."""
    return [
        {
            "input_id": "test1",
            "prediction": "cat",
            "confidence": 0.9,
            "latency_ms": 12.5,
            "ground_truth": "cat",
            "timestamp": 1633000000
        },
        {
            "input_id": "test2",
            "prediction": "dog",
            "confidence": 0.8,
            "latency_ms": 13.2,
            "ground_truth": "dog",
            "timestamp": 1633000001
        },
        {
            "input_id": "test3",
            "prediction": "cat",
            "confidence": 0.7,
            "latency_ms": 11.8,
            "ground_truth": "dog",  # Incorrect prediction
            "timestamp": 1633000002
        }
    ]
