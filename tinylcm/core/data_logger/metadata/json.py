"""JSON file implementation of metadata manager."""

import logging
from pathlib import Path
from typing import Any, Callable, Dict, List, Optional, Union

from .base import DataEntryMetadataManager
from tinylcm.utils.file_utils import load_json, save_json
from tinylcm.utils.logging import setup_logger


class JSONFileMetadataManager(DataEntryMetadataManager):
    def __init__(self):
        self.logger = setup_logger(f"{__name__}.{self.__class__.__name__}")

    def save_metadata(
        self,
        entry: Dict[str, Any],
        metadata_dir: Union[str, Path]
    ) -> None:
        entry_id = entry["entry_id"]
        metadata_path = Path(metadata_dir) / f"{entry_id}.json"
        save_json(entry, metadata_path)

    def load_metadata(
        self,
        entry_id: str,
        metadata_dir: Union[str, Path]
    ) -> Dict[str, Any]:
        metadata_path = Path(metadata_dir) / f"{entry_id}.json"

        if not metadata_path.exists():
            raise ValueError(f"Entry metadata not found: {entry_id}")

        return load_json(metadata_path)

    def update_metadata(
        self,
        entry_id: str,
        updates: Dict[str, Any],
        metadata_dir: Union[str, Path]
    ) -> bool:
        try:
            metadata_path = Path(metadata_dir) / f"{entry_id}.json"

            if not metadata_path.exists():
                return False

            metadata = load_json(metadata_path)
            metadata.update(updates)
            save_json(metadata, metadata_path)

            return True

        except Exception as e:
            self.logger.error(f"Error updating metadata for entry {entry_id}: {str(e)}")
            return False

    def list_metadata(
        self,
        metadata_dir: Union[str, Path],
        filter_func: Optional[Callable[[Dict[str, Any]], bool]] = None
    ) -> List[Dict[str, Any]]:
        metadata_list = []
        metadata_dir_path = Path(metadata_dir)

        if not metadata_dir_path.exists():
            return []

        for metadata_file in metadata_dir_path.glob("*.json"):
            try:
                metadata = load_json(metadata_file)

                if filter_func is None or filter_func(metadata):
                    metadata_list.append(metadata)
            except Exception as e:
                self.logger.error(f"Error loading metadata from {metadata_file}: {str(e)}")

        return metadata_list