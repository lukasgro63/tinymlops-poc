"""Storage strategies for data logging."""

from .base import DataStorageStrategy
from .text import TextDataStorage
from .json import JSONDataStorage
from .image import ImageDataStorage
from .factory import DataStorageFactory

__all__ = [
    "DataStorageStrategy",
    "TextDataStorage",
    "JSONDataStorage",
    "ImageDataStorage",
    "DataStorageFactory"
]