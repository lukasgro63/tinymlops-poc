from pathlib import Path
from typing import Any, Dict, List, Optional, Union
import csv
import json
import time
import uuid

from tinylcm.constants import (
    DATA_TYPE_IMAGE,
    DATA_TYPE_JSON,
    DATA_TYPE_SENSOR,
    DATA_TYPE_TEXT,
    DEFAULT_BUFFER_SIZE,
    DEFAULT_DATA_DIR,
    FILE_FORMAT_JPEG,
)
from tinylcm.utils.config import Config, get_config
from tinylcm.utils.file_utils import ensure_dir

from .metadata.json import JSONFileMetadataManager
from .storage.factory import DataStorageFactory


class DataLogger:
    def __init__(
        self,
        storage_dir: Optional[Union[str, Path]] = None,
        buffer_size: Optional[int] = None,
        metadata_manager: Optional[Any] = None,
        config: Optional[Config] = None
    ):
        self.config = config or get_config()
        component_config = self.config.get_component_config("data_logger")
        self.storage_dir = Path(storage_dir or component_config.get("storage_dir", DEFAULT_DATA_DIR))
        self.buffer_size = buffer_size or component_config.get("buffer_size", DEFAULT_BUFFER_SIZE)
        self.image_format = component_config.get("image_format", FILE_FORMAT_JPEG)
        ensure_dir(self.storage_dir)
        ensure_dir(self.storage_dir / "metadata")
        ensure_dir(self.storage_dir / "images")
        ensure_dir(self.storage_dir / "text")
        ensure_dir(self.storage_dir / "json")
        self.session_id = str(uuid.uuid4())
        self.metadata_buffer: List[Dict[str, Any]] = []
        self.metadata_manager = metadata_manager or JSONFileMetadataManager()
        self.storage_factory = DataStorageFactory()

    def log_data(
        self,
        input_data: Any,
        input_type: str,
        prediction: Optional[str] = None,
        confidence: Optional[float] = None,
        label: Optional[str] = None,
        metadata: Optional[Dict[str, Any]] = None
    ) -> str:
        entry_id = str(uuid.uuid4())
        storage = self.storage_factory.create_storage(
            input_type,
            image_format=self.image_format
        )
        try:
            relative_path = storage.store(
                data=input_data,
                data_type=input_type,
                entry_id=entry_id,
                storage_dir=self.storage_dir,
                metadata=metadata
            )
            entry = {
                "entry_id": entry_id,
                "timestamp": time.time(),
                "input_type": input_type,
                "filename": relative_path,
                "session_id": self.session_id,
                "metadata": metadata or {}
            }
            if prediction is not None:
                entry["prediction"] = prediction
            if confidence is not None:
                entry["confidence"] = confidence
            if label is not None:
                entry["label"] = label
            self.metadata_buffer.append(entry)
            if len(self.metadata_buffer) >= self.buffer_size:
                self._write_metadata_buffer()
            return entry_id
        except Exception as e:
            print(f"Error logging data: {str(e)}")
            raise

    def log_image(
        self,
        image_data: bytes,
        prediction: Optional[str] = None,
        confidence: Optional[float] = None,
        label: Optional[str] = None,
        metadata: Optional[Dict[str, Any]] = None
    ) -> str:
        return self.log_data(
            input_data=image_data,
            input_type=DATA_TYPE_IMAGE,
            prediction=prediction,
            confidence=confidence,
            label=label,
            metadata=metadata
        )

    def log_prediction(
        self,
        input_id: str,
        prediction: str,
        confidence: Optional[float] = None
    ) -> bool:
        for entry in self.metadata_buffer:
            if entry["entry_id"] == input_id:
                entry["prediction"] = prediction
                if confidence is not None:
                    entry["confidence"] = confidence
                return True
        updates = {"prediction": prediction}
        if confidence is not None:
            updates["confidence"] = confidence
        return self.metadata_manager.update_metadata(
            entry_id=input_id,
            updates=updates,
            metadata_dir=self.storage_dir / "metadata"
        )

    def get_entry(self, entry_id: str) -> Dict[str, Any]:
        for entry in self.metadata_buffer:
            if entry["entry_id"] == entry_id:
                return entry.copy()
        return self.metadata_manager.load_metadata(
            entry_id=entry_id,
            metadata_dir=self.storage_dir / "metadata"
        )

    def get_data_file(self, entry_id: str) -> Optional[str]:
        try:
            entry = self.get_entry(entry_id)
            relative_path = entry.get("filename")
            if not relative_path:
                return None
            full_path = self.storage_dir / relative_path
            if not full_path.exists():
                return None
            return str(full_path)
        except Exception as e:
            print(f"Error getting data file for entry {entry_id}: {str(e)}")
            return None

    def _write_metadata_buffer(self) -> None:
        if not self.metadata_buffer:
            return
        for entry in self.metadata_buffer:
            try:
                self.metadata_manager.save_metadata(
                    entry=entry,
                    metadata_dir=self.storage_dir / "metadata"
                )
            except Exception as e:
                print(f"Error writing metadata for entry {entry.get('entry_id')}: {str(e)}")
        self.metadata_buffer.clear()

    def query_entries(
        self,
        start_time: Optional[float] = None,
        end_time: Optional[float] = None,
        input_type: Optional[str] = None,
        prediction: Optional[str] = None,
        label: Optional[str] = None,
        session_id: Optional[str] = None,
        limit: Optional[int] = None
    ) -> List[Dict[str, Any]]:
        matching_buffer_entries = []
        for entry in self.metadata_buffer:
            if self._matches_filter(entry, start_time, end_time, input_type, prediction, label, session_id):
                matching_buffer_entries.append(entry.copy())
        self._write_metadata_buffer()
        def filter_func(entry: Dict[str, Any]) -> bool:
            return self._matches_filter(entry, start_time, end_time, input_type, prediction, label, session_id)
        disk_entries = self.metadata_manager.list_metadata(
            metadata_dir=self.storage_dir / "metadata",
            filter_func=filter_func
        )
        disk_entry_ids = {entry.get("entry_id") for entry in disk_entries}
        combined_entries = disk_entries.copy()
        for entry in matching_buffer_entries:
            if entry.get("entry_id") not in disk_entry_ids:
                combined_entries.append(entry)
        combined_entries.sort(key=lambda e: e.get("timestamp", 0))
        if limit is not None and len(combined_entries) > limit:
            combined_entries = combined_entries[:limit]
        return combined_entries

    def _matches_filter(
        self,
        entry: Dict[str, Any],
        start_time: Optional[float] = None,
        end_time: Optional[float] = None,
        input_type: Optional[str] = None,
        prediction: Optional[str] = None,
        label: Optional[str] = None,
        session_id: Optional[str] = None
    ) -> bool:
        effective_timestamp = entry.get("timestamp", 0)
        if "metadata" in entry and isinstance(entry["metadata"], dict):
            if "time_override" in entry["metadata"]:
                effective_timestamp = entry["metadata"]["time_override"]
        if start_time is not None and effective_timestamp < start_time:
            return False
        if end_time is not None and effective_timestamp > end_time:
            return False
        if input_type is not None and entry.get("input_type") != input_type:
            return False
        if prediction is not None and entry.get("prediction") != prediction:
            return False
        if label is not None and entry.get("label") != label:
            return False
        if session_id is not None and entry.get("session_id") != session_id:
            return False
        return True

    def count_entries(
        self,
        start_time: Optional[float] = None,
        end_time: Optional[float] = None,
        input_type: Optional[str] = None,
        prediction: Optional[str] = None,
        label: Optional[str] = None,
        session_id: Optional[str] = None
    ) -> int:
        entries = self.query_entries(
            start_time=start_time,
            end_time=end_time,
            input_type=input_type,
            prediction=prediction,
            label=label,
            session_id=session_id
        )
        return len(entries)

    def export_to_csv(
        self,
        output_path: Optional[Union[str, Path]] = None,
        filter_func: Optional[callable] = None
    ) -> str:
        self._write_metadata_buffer()
        entries = self.metadata_manager.list_metadata(
            metadata_dir=self.storage_dir / "metadata",
            filter_func=filter_func
        )
        if not entries:
            return ""
        if output_path is None:
            timestamp = int(time.time())
            output_path = self.storage_dir / f"export_{timestamp}.csv"
        else:
            output_path = Path(output_path)
        ensure_dir(output_path.parent)
        all_fields = set()
        for entry in entries:
            all_fields.update(entry.keys())
        fields = sorted(list(all_fields))
        with open(output_path, "w", newline="", encoding='utf-8') as f:
            writer = csv.DictWriter(f, fieldnames=fields)
            writer.writeheader()
            for entry in entries:
                row = {field: entry.get(field, "") for field in fields}
                writer.writerow(row)
        return str(output_path)

    def close(self) -> None:
        self._write_metadata_buffer()
        try:
            timestamp = int(time.time())
            summary_path = self.storage_dir / f"logger_summary_{timestamp}.json"
            summary = {
                "timestamp": timestamp,
                "session_id": self.session_id,
                "total_entries": self.count_entries(),
                "storage_dir": str(self.storage_dir),
                "entry_types": {}
            }
            try:
                for input_type in ["text", "image", "json", "sensor"]:
                    count = self.count_entries(input_type=input_type)
                    if count > 0:
                        summary["entry_types"][input_type] = count
            except Exception:
                pass
            with open(summary_path, "w", encoding='utf-8') as f:
                json.dump(summary, f, indent=2)
        except Exception as e:
            print(f"Warning: Error creating logger summary: {str(e)}")
        self.session_id = None