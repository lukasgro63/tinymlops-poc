"""Tests for DataStorageFactory."""
import pytest

from tinylcm.core.data_logger.storage.factory import DataStorageFactory
from tinylcm.core.data_logger.storage.text import TextDataStorage
from tinylcm.core.data_logger.storage.json import JSONDataStorage
from tinylcm.core.data_logger.storage.image import ImageDataStorage
from tinylcm.constants import (
    DATA_TYPE_TEXT,
    DATA_TYPE_JSON,
    DATA_TYPE_IMAGE,
    DATA_TYPE_SENSOR,
    FILE_FORMAT_JPEG
)


def test_factory_creates_text_storage():
    """Test factory creates TextDataStorage."""
    storage = DataStorageFactory.create_storage(DATA_TYPE_TEXT)
    assert isinstance(storage, TextDataStorage)


def test_factory_creates_json_storage():
    """Test factory creates JSONDataStorage."""
    storage = DataStorageFactory.create_storage(DATA_TYPE_JSON)
    assert isinstance(storage, JSONDataStorage)


def test_factory_creates_image_storage():
    """Test factory creates ImageDataStorage."""
    storage = DataStorageFactory.create_storage(DATA_TYPE_IMAGE)
    assert isinstance(storage, ImageDataStorage)


def test_factory_creates_image_storage_with_format():
    """Test factory creates ImageDataStorage with custom format."""
    format = "png"
    storage = DataStorageFactory.create_storage(DATA_TYPE_IMAGE, image_format=format)
    assert isinstance(storage, ImageDataStorage)
    assert storage.format == format


def test_factory_uses_json_for_sensor_data():
    """Test factory returns JSONDataStorage for sensor data."""
    storage = DataStorageFactory.create_storage(DATA_TYPE_SENSOR)
    assert isinstance(storage, JSONDataStorage)


def test_factory_invalid_data_type():
    """Test factory raises error for invalid data type."""
    with pytest.raises(ValueError):
        DataStorageFactory.create_storage("invalid_type")