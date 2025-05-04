"""Tests for JSONFileMetadataManager."""
import json
import pytest
from pathlib import Path

from tinylcm.core.data_logger.metadata.json import JSONFileMetadataManager


def test_json_metadata_save(temp_dir):
    """Test saving metadata as JSON."""
    manager = JSONFileMetadataManager()
    entry = {
        "entry_id": "meta123",
        "timestamp": 1633000000,
        "input_type": "text",
        "filename": "text/meta123.txt"
    }
    
    manager.save_metadata(entry, temp_dir)
    
    # Check that the file exists
    meta_path = Path(temp_dir) / f"{entry['entry_id']}.json"
    assert meta_path.exists()
    
    # Check content
    with open(meta_path, "r", encoding="utf-8") as f:
        loaded_data = json.load(f)
    
    assert loaded_data == entry


def test_json_metadata_load(temp_dir):
    """Test loading metadata from JSON."""
    manager = JSONFileMetadataManager()
    entry = {
        "entry_id": "meta456",
        "timestamp": 1633000001,
        "input_type": "image",
        "filename": "images/meta456.jpg"
    }
    
    # Save first
    manager.save_metadata(entry, temp_dir)
    
    # Now load
    loaded_entry = manager.load_metadata(entry["entry_id"], temp_dir)
    
    assert loaded_entry == entry


def test_json_metadata_update(temp_dir):
    """Test updating metadata."""
    manager = JSONFileMetadataManager()
    entry_id = "meta789"
    
    # Create initial metadata
    entry = {
        "entry_id": entry_id,
        "timestamp": 1633000002,
        "input_type": "json",
        "filename": "json/meta789.json"
    }
    
    manager.save_metadata(entry, temp_dir)
    
    # Update with new values
    updates = {
        "prediction": "cat",
        "confidence": 0.95
    }
    
    result = manager.update_metadata(entry_id, updates, temp_dir)
    assert result is True
    
    # Load and check updated values
    loaded_entry = manager.load_metadata(entry_id, temp_dir)
    assert loaded_entry["prediction"] == "cat"
    assert loaded_entry["confidence"] == 0.95
    
    # Original values should still be there
    assert loaded_entry["timestamp"] == entry["timestamp"]
    assert loaded_entry["input_type"] == entry["input_type"]


def test_json_metadata_list(temp_dir):
    """Test listing metadata."""
    manager = JSONFileMetadataManager()
    
    # Create multiple entries
    entries = [
        {
            "entry_id": f"list{i}",
            "timestamp": 1633000000 + i,
            "input_type": "text" if i % 2 == 0 else "image"
        }
        for i in range(5)
    ]
    
    for entry in entries:
        manager.save_metadata(entry, temp_dir)
    
    # List all entries
    all_entries = manager.list_metadata(temp_dir)
    assert len(all_entries) == 5
    
    # List with filter
    text_entries = manager.list_metadata(
        temp_dir,
        filter_func=lambda e: e["input_type"] == "text"
    )
    assert len(text_entries) == 3  # Entries 0, 2, 4