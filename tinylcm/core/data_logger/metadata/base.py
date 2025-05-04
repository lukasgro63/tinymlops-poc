"""Base metadata manager for data logging."""

from abc import ABC, abstractmethod
from pathlib import Path
from typing import Any, Callable, Dict, List, Optional, Union


class DataEntryMetadataManager(ABC):
    def save_metadata(
        self,
        entry: Dict[str, Any],
        metadata_dir: Union[str, Path]
    ) -> None:
        pass

    def load_metadata(
        self,
        entry_id: str,
        metadata_dir: Union[str, Path]
    ) -> Dict[str, Any]:
        pass

    def update_metadata(
        self,
        entry_id: str,
        updates: Dict[str, Any],
        metadata_dir: Union[str, Path]
    ) -> bool:
        pass

    def list_metadata(
        self,
        metadata_dir: Union[str, Path],
        filter_func: Optional[Callable[[Dict[str, Any]], bool]] = None
    ) -> List[Dict[str, Any]]:
        pass