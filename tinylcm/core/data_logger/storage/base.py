"""Base storage strategy for data logging."""

from abc import ABC, abstractmethod
from pathlib import Path
from typing import Any, Dict, Optional, Union


class DataStorageStrategy(ABC):
    def store(
        self,
        data: Any,
        data_type: str,
        entry_id: str,
        storage_dir: Union[str, Path],
        metadata: Optional[Dict[str, Any]] = None
    ) -> str:
        pass

    def load(
        self,
        file_path: Union[str, Path]
    ) -> Any:
        pass