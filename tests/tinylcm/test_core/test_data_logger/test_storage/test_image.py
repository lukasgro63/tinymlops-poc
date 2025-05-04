"""Tests for ImageDataStorage."""
import pytest
from pathlib import Path

from tinylcm.core.data_logger.storage.image import ImageDataStorage
from tinylcm.constants import DATA_TYPE_IMAGE, FILE_FORMAT_JPEG


def test_image_storage_store(temp_dir, sample_image_data):
    """Test storing image data."""
    storage = ImageDataStorage()
    entry_id = "img123"
    
    rel_path = storage.store(
        data=sample_image_data,
        data_type=DATA_TYPE_IMAGE,
        entry_id=entry_id,
        storage_dir=temp_dir
    )
    
    # Check that the file exists
    full_path = Path(temp_dir) / rel_path
    assert full_path.exists()
    
    # Check content
    with open(full_path, "rb") as f:
        content = f.read()
    
    assert content == sample_image_data


def test_image_storage_load(temp_dir, sample_image_data):
    """Test loading image data."""
    # First store the data
    storage = ImageDataStorage()
    entry_id = "img456"
    
    rel_path = storage.store(
        data=sample_image_data,
        data_type=DATA_TYPE_IMAGE,
        entry_id=entry_id,
        storage_dir=temp_dir
    )
    
    full_path = Path(temp_dir) / rel_path
    
    # Now load it
    loaded_data = storage.load(full_path)
    
    assert loaded_data == sample_image_data


def test_image_storage_custom_format(temp_dir, sample_image_data):
    """Test storing image with custom format."""
    format = "png"
    storage = ImageDataStorage(format=format)
    entry_id = "img789"
    
    rel_path = storage.store(
        data=sample_image_data,
        data_type=DATA_TYPE_IMAGE,
        entry_id=entry_id,
        storage_dir=temp_dir
    )
    
    # Check correct extension
    assert rel_path.endswith(f".{format}")
    
    # Check that the file exists
    full_path = Path(temp_dir) / rel_path
    assert full_path.exists()


def test_image_storage_invalid_data():
    """Test storing invalid data type."""
    storage = ImageDataStorage()
    
    with pytest.raises(TypeError):
        storage.store(
            data="not bytes",  # Not bytes
            data_type=DATA_TYPE_IMAGE,
            entry_id="invalid",
            storage_dir="/tmp"
        )