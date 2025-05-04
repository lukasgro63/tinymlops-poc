# Tests für JSONDataStorage
"""Tests for JSONDataStorage."""
import json
import pytest
from pathlib import Path

from tinylcm.core.data_logger.storage.json import JSONDataStorage
from tinylcm.constants import DATA_TYPE_JSON


def test_json_storage_store(temp_dir, sample_json_data):
    """Test storing JSON data."""
    storage = JSONDataStorage()
    entry_id = "json123"
    
    rel_path = storage.store(
        data=sample_json_data,
        data_type=DATA_TYPE_JSON,
        entry_id=entry_id,
        storage_dir=temp_dir
    )
    
    # Check that the file exists
    full_path = Path(temp_dir) / rel_path
    assert full_path.exists()
    
    # Check content
    with open(full_path, "r", encoding="utf-8") as f:
        loaded_data = json.load(f)
    
    assert loaded_data == sample_json_data


def test_json_storage_load(temp_dir, sample_json_data):
    """Test loading JSON data."""
    # First store the data
    storage = JSONDataStorage()
    entry_id = "json456"
    
    rel_path = storage.store(
        data=sample_json_data,
        data_type=DATA_TYPE_JSON,
        entry_id=entry_id,
        storage_dir=temp_dir
    )
    
    full_path = Path(temp_dir) / rel_path
    
    # Now load it
    loaded_data = storage.load(full_path)
    
    assert loaded_data == sample_json_data