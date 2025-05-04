# Gemeinsame Test-Fixtures für das Testmodul test_data_logger
"""Fixtures for data logger tests."""
import os
import pytest
import tempfile
import shutil
from pathlib import Path


@pytest.fixture
def temp_dir():
    """Create a temporary directory for tests."""
    dir_path = tempfile.mkdtemp()
    yield dir_path
    shutil.rmtree(dir_path)


@pytest.fixture
def sample_text_data():
    """Sample text data for testing."""
    return "This is test data content."


@pytest.fixture
def sample_json_data():
    """Sample JSON data for testing."""
    return {"key1": "value1", "key2": 42, "nested": {"inner": "value"}}


@pytest.fixture
def sample_image_data():
    """Sample image data (bytes) for testing."""
    # Simple 1x1 black pixel in PNG format
    return b'\x89PNG\r\n\x1a\n\x00\x00\x00\rIHDR\x00\x00\x00\x01\x00\x00\x00\x01\x08\x00\x00\x00\x00:~\x9bU\x00\x00\x00\nIDATx\x9cc`\x00\x00\x00\x02\x00\x01H\xaf\xa4q\x00\x00\x00\x00IEND\xaeB`\x82'