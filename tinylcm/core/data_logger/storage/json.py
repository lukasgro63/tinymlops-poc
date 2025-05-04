"""Storage strategy for JSON data."""

from pathlib import Path
from typing import Any, Dict, Optional, Union

from .base import DataStorageStrategy
from tinylcm.utils.file_utils import ensure_dir, load_json, save_json


class JSONDataStorage(DataStorageStrategy):
    def store(
        self,
        data: Dict[str, Any],
        data_type: str,
        entry_id: str,
        storage_dir: Union[str, Path],
        metadata: Optional[Dict[str, Any]] = None
    ) -> str:
        relative_path = f"json/{entry_id}.json"
        full_path = Path(storage_dir) / relative_path

        ensure_dir(full_path.parent)

        save_json(data, full_path)

        return relative_path

    def load(
        self,
        file_path: Union[str, Path]
    ) -> Dict[str, Any]:
        return load_json(file_path)