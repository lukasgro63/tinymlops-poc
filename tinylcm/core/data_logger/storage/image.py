"""Storage strategy for image data."""

from pathlib import Path
from typing import Any, Dict, Optional, Union

from .base import DataStorageStrategy
from tinylcm.constants import FILE_FORMAT_JPEG
from tinylcm.utils.file_utils import ensure_dir


class ImageDataStorage(DataStorageStrategy):
    def __init__(self, format: str = FILE_FORMAT_JPEG):
        self.format = format

    def store(
        self,
        data: bytes,
        data_type: str,
        entry_id: str,
        storage_dir: Union[str, Path],
        metadata: Optional[Dict[str, Any]] = None
    ) -> str:
        if not isinstance(data, bytes):
            raise TypeError(f"Expected bytes data for image storage, got {type(data)}")

        relative_path = f"images/{entry_id}.{self.format}"
        full_path = Path(storage_dir) / relative_path

        ensure_dir(full_path.parent)

        with open(full_path, "wb") as f:
            f.write(data)

        return relative_path

    def load(
        self,
        file_path: Union[str, Path]
    ) -> bytes:
        path_obj = Path(file_path)

        if not path_obj.exists():
            raise FileNotFoundError(f"Image file not found: {file_path}")

        with open(path_obj, "rb") as f:
            return f.read()