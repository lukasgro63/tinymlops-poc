# Tests für TextDataStorage
"""Tests for TextDataStorage."""
import os
import pytest
from pathlib import Path

from tinylcm.core.data_logger.storage.text import TextDataStorage
from tinylcm.constants import DATA_TYPE_TEXT


def test_text_storage_store(temp_dir, sample_text_data):
    """Test storing text data."""
    storage = TextDataStorage()
    entry_id = "test123"
    
    rel_path = storage.store(
        data=sample_text_data,
        data_type=DATA_TYPE_TEXT,
        entry_id=entry_id,
        storage_dir=temp_dir
    )
    
    # Check that the file exists
    full_path = Path(temp_dir) / rel_path
    assert full_path.exists()
    
    # Check content
    with open(full_path, "r", encoding="utf-8") as f:
        content = f.read()
    
    assert content == sample_text_data


def test_text_storage_load(temp_dir, sample_text_data):
    """Test loading text data."""
    # First store the data
    storage = TextDataStorage()
    entry_id = "test456"
    
    rel_path = storage.store(
        data=sample_text_data,
        data_type=DATA_TYPE_TEXT,
        entry_id=entry_id,
        storage_dir=temp_dir
    )
    
    full_path = Path(temp_dir) / rel_path
    
    # Now load it
    loaded_data = storage.load(full_path)
    
    assert loaded_data == sample_text_data


def test_text_storage_invalid_data():
    """Test storing invalid data type."""
    storage = TextDataStorage()
    
    with pytest.raises(TypeError):
        storage.store(
            data=42,  # Not a string
            data_type=DATA_TYPE_TEXT,
            entry_id="invalid",
            storage_dir="/tmp"
        )


def test_text_storage_file_not_found():
    """Test loading from non-existent file."""
    storage = TextDataStorage()
    
    with pytest.raises(FileNotFoundError):
        storage.load("/path/does/not/exist.txt")