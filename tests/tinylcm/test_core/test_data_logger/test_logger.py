"""Tests for DataLogger."""
import os
import json
import time
import pytest
from pathlib import Path

from tinylcm.core.data_logger.logger import DataLogger
from tinylcm.constants import DATA_TYPE_TEXT, DATA_TYPE_JSON, DATA_TYPE_IMAGE


@pytest.fixture
def data_logger(temp_dir):
    """Create a DataLogger instance for testing."""
    logger = DataLogger(storage_dir=temp_dir)
    yield logger
    logger.close()


def test_logger_initialization(temp_dir):
    """Test DataLogger initialization."""
    logger = DataLogger(storage_dir=temp_dir)
    
    # Check directories are created
    assert Path(temp_dir).exists()
    assert (Path(temp_dir) / "metadata").exists()
    assert (Path(temp_dir) / "images").exists()
    assert (Path(temp_dir) / "text").exists()
    assert (Path(temp_dir) / "json").exists()
    
    # Check initial state
    assert logger.session_id is not None
    assert isinstance(logger.metadata_buffer, list)
    assert len(logger.metadata_buffer) == 0


def test_log_text_data(data_logger, sample_text_data):
    """Test logging text data."""
    entry_id = data_logger.log_data(
        input_data=sample_text_data,
        input_type=DATA_TYPE_TEXT,
        prediction="example",
        confidence=0.9,
        label="example_label"
    )
    
    # Check that entry ID is returned
    assert entry_id is not None
    
    # Check that the entry is in the buffer
    assert len(data_logger.metadata_buffer) == 1
    assert data_logger.metadata_buffer[0]["entry_id"] == entry_id
    assert data_logger.metadata_buffer[0]["input_type"] == DATA_TYPE_TEXT
    assert data_logger.metadata_buffer[0]["prediction"] == "example"
    assert data_logger.metadata_buffer[0]["confidence"] == 0.9
    assert data_logger.metadata_buffer[0]["label"] == "example_label"


def test_log_image_convenience(data_logger, sample_image_data):
    """Test convenience method for logging images."""
    entry_id = data_logger.log_image(
        image_data=sample_image_data,
        prediction="cat",
        confidence=0.85
    )
    
    # Check that entry ID is returned
    assert entry_id is not None
    
    # Check that the entry is in the buffer
    assert len(data_logger.metadata_buffer) == 1
    assert data_logger.metadata_buffer[0]["entry_id"] == entry_id
    assert data_logger.metadata_buffer[0]["input_type"] == DATA_TYPE_IMAGE
    assert data_logger.metadata_buffer[0]["prediction"] == "cat"
    assert data_logger.metadata_buffer[0]["confidence"] == 0.85


def test_log_prediction(data_logger, sample_text_data):
    """Test adding a prediction to an existing entry."""
    # First log data without prediction
    entry_id = data_logger.log_data(
        input_data=sample_text_data,
        input_type=DATA_TYPE_TEXT
    )
    
    # Then add a prediction
    result = data_logger.log_prediction(
        input_id=entry_id,
        prediction="new_prediction",
        confidence=0.75
    )
    
    # Check success
    assert result is True
    
    # Check that prediction was added
    assert data_logger.metadata_buffer[0]["prediction"] == "new_prediction"
    assert data_logger.metadata_buffer[0]["confidence"] == 0.75


def test_get_entry(data_logger, sample_text_data):
    """Test getting an entry's metadata."""
    # Log some data
    entry_id = data_logger.log_data(
        input_data=sample_text_data,
        input_type=DATA_TYPE_TEXT,
        prediction="test"
    )
    
    # Get the entry
    entry = data_logger.get_entry(entry_id)
    
    # Check entry data
    assert entry["entry_id"] == entry_id
    assert entry["input_type"] == DATA_TYPE_TEXT
    assert entry["prediction"] == "test"
    assert "timestamp" in entry
    assert "filename" in entry


def test_query_entries(data_logger):
    """Test querying entries with filters."""
    # Log multiple entries with different types
    data_logger.log_data("text1", DATA_TYPE_TEXT, prediction="a")
    data_logger.log_data("text2", DATA_TYPE_TEXT, prediction="b")
    data_logger.log_data({"json": True}, DATA_TYPE_JSON, prediction="a")
    
    # Query by input type
    text_entries = data_logger.query_entries(input_type=DATA_TYPE_TEXT)
    assert len(text_entries) == 2
    
    # Query by prediction
    a_entries = data_logger.query_entries(prediction="a")
    assert len(a_entries) == 2
    
    # Query by both
    specific_entries = data_logger.query_entries(
        input_type=DATA_TYPE_TEXT,
        prediction="a"
    )
    assert len(specific_entries) == 1


def test_count_entries(data_logger):
    """Test counting entries with filters."""
    # Log multiple entries
    data_logger.log_data("text1", DATA_TYPE_TEXT, prediction="a")
    data_logger.log_data("text2", DATA_TYPE_TEXT, prediction="b")
    data_logger.log_data({"json": True}, DATA_TYPE_JSON, prediction="a")
    
    # Count all entries
    count = data_logger.count_entries()
    assert count == 3
    
    # Count with filter
    filtered_count = data_logger.count_entries(prediction="a")
    assert filtered_count == 2


def test_export_to_csv(data_logger, temp_dir):
    """Test exporting entries to CSV."""
    # Log some entries
    for i in range(5):
        data_logger.log_data(
            f"text{i}",
            DATA_TYPE_TEXT,
            prediction=f"pred{i}",
            confidence=0.5 + i/10
        )
    
    # Force write to disk
    data_logger._write_metadata_buffer()
    
    # Export to CSV
    csv_path = data_logger.export_to_csv()
    
    # Check that file exists and has content
    assert Path(csv_path).exists()
    
    # Basic check of CSV content (should have header + 5 entries)
    with open(csv_path, 'r', encoding='utf-8') as f:
        lines = f.readlines()
    
    assert len(lines) == 6  # Header + 5 entries