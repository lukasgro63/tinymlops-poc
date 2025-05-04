from typing import Any, Dict

from .base import DataStorageStrategy
from .text import TextDataStorage
from .json import JSONDataStorage
from .image import ImageDataStorage
from tinylcm.constants import (
    DATA_TYPE_TEXT,
    DATA_TYPE_JSON,
    DATA_TYPE_IMAGE,
    DATA_TYPE_SENSOR,
    FILE_FORMAT_JPEG,
)


class DataStorageFactory:
    @staticmethod
    def create_storage(data_type: str, **kwargs) -> DataStorageStrategy:
        if data_type == DATA_TYPE_TEXT:
            return TextDataStorage()
        elif data_type == DATA_TYPE_JSON:
            return JSONDataStorage()
        elif data_type == DATA_TYPE_IMAGE:
            format = kwargs.get("image_format", FILE_FORMAT_JPEG)
            return ImageDataStorage(format=format)
        elif data_type == DATA_TYPE_SENSOR:
            return JSONDataStorage()
        else:
            raise ValueError(f"Unsupported data type: {data_type}")