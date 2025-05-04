"""Test fixtures for model manager tests."""

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
def storage_dir(temp_dir):
    """Create a directory structure for model manager."""
    storage_path = Path(temp_dir) / "models"
    models_dir = storage_path / "models"
    metadata_dir = storage_path / "metadata"
    
    models_dir.mkdir(parents=True, exist_ok=True)
    metadata_dir.mkdir(parents=True, exist_ok=True)
    
    yield storage_path
    

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
def mock_model_binary(temp_dir):
    """Create a mock binary model file for testing."""
    model_dir = os.path.join(temp_dir, "models")
    os.makedirs(model_dir, exist_ok=True)
    
    model_path = os.path.join(model_dir, "test_model.bin")
    
    # Create a simple binary file with some bytes
    with open(model_path, "wb") as f:
        f.write(b"\x00\x01\x02\x03\x04\x05")
    
    return model_path


@pytest.fixture
def sample_metadata():
    """Create sample model metadata."""
    return {
        "model_id": "test123",
        "model_format": "json",
        "version": "v_1",
        "description": "Test model",
        "tags": ["test", "sample"],
        "metrics": {"accuracy": 0.95, "f1": 0.94},
        "timestamp": 1633000000,
        "is_active": False,
        "md5_hash": "abcdef123456",
        "filename": "test_model.json"
    }