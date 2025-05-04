import json
import numpy as np
import os
import pytest
import tempfile
import shutil
from pathlib import Path

@pytest.fixture
def temp_dir():
    dir_path = tempfile.mkdtemp()
    yield dir_path
    shutil.rmtree(dir_path)

@pytest.fixture
def sample_distribution_data():
    return {
        "predictions": ["cat", "dog", "cat", "bird", "cat", "dog", "fish", 
                        "cat", "dog", "cat", "cat", "dog", "bird"]
    }

@pytest.fixture
def reference_distribution():
    return {
        "class_distribution": {
            "cat": 0.5,
            "dog": 0.3,
            "bird": 0.15,
            "fish": 0.05
        }
    }

@pytest.fixture
def current_distribution():
    return {
        "class_distribution": {
            "cat": 0.3,
            "dog": 0.5,
            "bird": 0.1,
            "fish": 0.1
        }
    }

@pytest.fixture
def sample_feature_data():
    return {
        "features": [
            {
                "name": "size",
                "values": [1.2, 1.5, 0.8, 1.1, 1.3, 1.6, 0.9, 1.0, 1.4]
            },
            {
                "name": "weight",
                "values": [2.1, 2.3, 1.8, 2.0, 2.2, 2.5, 1.7, 1.9, 2.4]
            }
        ]
    }

@pytest.fixture
def sample_confidence_data():
    return {
        "confidences": [0.9, 0.85, 0.92, 0.78, 0.95, 0.88, 0.91, 0.83, 0.94]
    }

@pytest.fixture
def sample_record():
    return {
        "input_id": "test123",
        "prediction": "cat",
        "confidence": 0.85,
        "timestamp": 1633000000,
        "features": {
            "size": 1.2,
            "weight": 2.1
        }
    }