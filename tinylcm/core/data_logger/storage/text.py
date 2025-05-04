"""Storage strategy for text data."""

from pathlib import Path
from typing import Any, Dict, Optional, Union

from .base import DataStorageStrategy
from tinylcm.utils.file_utils import ensure_dir


class TextDataStorage(DataStorageStrategy):
    def store(
        self,
        data: str,
        data_type: str,
        entry_id: str,
        storage_dir: Union[str, Path],
        metadata: Optional[Dict[str, Any]] = None
    ) -> str:
        if not isinstance(data, str):
            raise TypeError(f"Expected string data for text storage, got {type(data)}")

        relative_path = f"text/{entry_id}.txt"
        full_path = Path(storage_dir) / relative_path

        ensure_dir(full_path.parent)

        with open(full_path, "w", encoding="utf-8") as f:
            f.write(data)

        return relative_path

    def load(
        self,
        file_path: Union[str, Path]
    ) -> str:
        path_obj = Path(file_path)

        if not path_obj.exists():
            raise FileNotFoundError(f"Text file not found: {file_path}")

        with open(path_obj, "r", encoding="utf-8") as f:
            return f.read()