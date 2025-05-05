from pathlib import Path
from typing import Any, Dict, List, Optional, Union
import csv
import json
import queue
import threading
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
    """DataLogger for collecting and storing input data with non-blocking I/O.
    
    This class provides functionality to log different types of data (images, text, etc.)
    and track metadata about them. It uses a background thread for disk I/O operations
    to avoid blocking the main application thread.
    
    All file I/O operations are performed in a separate worker thread to ensure
    they don't block the main application thread.
    """
    def __init__(
        self,
        storage_dir: Optional[Union[str, Path]] = None,
        buffer_size: Optional[int] = None,
        metadata_manager: Optional[Any] = None,
        config: Optional[Config] = None,
        worker_thread_daemon: bool = True
    ):
        self.config = config or get_config()
        component_config = self.config.get_component_config("data_logger")
        self.storage_dir = Path(storage_dir or component_config.get("storage_dir", DEFAULT_DATA_DIR))
        self.buffer_size = buffer_size or component_config.get("buffer_size", DEFAULT_BUFFER_SIZE)
        self.image_format = component_config.get("image_format", FILE_FORMAT_JPEG)
        
        # Create necessary directories
        ensure_dir(self.storage_dir)
        ensure_dir(self.storage_dir / "metadata")
        ensure_dir(self.storage_dir / "images")
        ensure_dir(self.storage_dir / "text")
        ensure_dir(self.storage_dir / "json")
        
        self.session_id = str(uuid.uuid4())
        self.metadata_buffer: List[Dict[str, Any]] = []
        self.metadata_manager = metadata_manager or JSONFileMetadataManager()
        self.storage_factory = DataStorageFactory()
        
        # Initialize worker thread for non-blocking I/O operations
        self._task_queue = queue.Queue()
        self._stop_event = threading.Event()
        self._worker_thread_daemon = worker_thread_daemon
        self._worker_thread = None
        self._results = {}  # To store results of tasks for retrieval
        self._results_lock = threading.Lock()
        self._start_worker_thread()
    
    def _start_worker_thread(self) -> None:
        """Starts the worker thread that processes the task queue."""
        if self._worker_thread is not None and self._worker_thread.is_alive():
            print("Warning: Worker thread already running")
            return
        
        self._stop_event.clear()
        self._worker_thread = threading.Thread(
            target=self._worker_loop,
            daemon=self._worker_thread_daemon
        )
        self._worker_thread.start()
        print(f"Started data logger worker thread")
    
    def _worker_loop(self) -> None:
        """Main worker loop that processes tasks in the queue."""
        while not self._stop_event.is_set():
            try:
                # Wait for a task with a timeout to allow checking the stop event
                try:
                    task = self._task_queue.get(timeout=0.1)
                except queue.Empty:
                    continue
                
                # Process the task
                try:
                    task_type = task.get("type")
                    task_id = task.get("id")
                    
                    if task_type == "store_data":
                        result = self._store_data_internal(**task.get("args", {}))
                    elif task_type == "update_metadata":
                        result = self._update_metadata_internal(**task.get("args", {}))
                    elif task_type == "write_buffer":
                        result = self._write_metadata_buffer_internal()
                    elif task_type == "export_csv":
                        result = self._export_to_csv_internal(**task.get("args", {}))
                    else:
                        print(f"Warning: Unknown task type: {task_type}")
                        result = None
                    
                    # Store result if task_id was provided
                    if task_id is not None:
                        with self._results_lock:
                            self._results[task_id] = {
                                "result": result,
                                "timestamp": time.time()
                            }
                    
                except Exception as e:
                    print(f"Error processing task: {str(e)}")
                    if task_id is not None:
                        with self._results_lock:
                            self._results[task_id] = {
                                "error": str(e),
                                "timestamp": time.time()
                            }
                finally:
                    self._task_queue.task_done()
                    
            except Exception as e:
                print(f"Unexpected error in worker thread: {str(e)}")
                # Sleep briefly to avoid tight loop in case of recurring errors
                time.sleep(0.1)
        
        print("Worker thread stopping")

    def _store_data_internal(
        self,
        input_data: Any,
        input_type: str,
        entry_id: str,
        metadata: Optional[Dict[str, Any]] = None
    ) -> str:
        """Internal method to store data (called by worker thread).
        
        Args:
            input_data: The data to store
            input_type: Type of data (image, text, etc.)
            entry_id: Unique ID for this entry
            metadata: Additional metadata
            
        Returns:
            Relative path to the stored data
        """
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
            return relative_path
        except Exception as e:
            print(f"Error storing data: {str(e)}")
            raise
    
    def log_data(
        self,
        input_data: Any,
        input_type: str,
        prediction: Optional[str] = None,
        confidence: Optional[float] = None,
        label: Optional[str] = None,
        metadata: Optional[Dict[str, Any]] = None
    ) -> str:
        """Log data with optional prediction and metadata (non-blocking).
        
        This method is non-blocking. The data storage operation will be performed
        in a background thread.
        
        Args:
            input_data: The data to log (bytes, string, etc.)
            input_type: Type of data (image, text, json, etc.)
            prediction: Optional model prediction
            confidence: Optional confidence score
            label: Optional ground truth label
            metadata: Additional metadata
            
        Returns:
            Unique entry ID for this logged data
        """
        entry_id = str(uuid.uuid4())
        
        # Create entry for metadata buffer
        entry = {
            "entry_id": entry_id,
            "timestamp": time.time(),
            "input_type": input_type,
            "session_id": self.session_id,
            "metadata": metadata or {}
        }
        
        # Add optional fields
        if prediction is not None:
            entry["prediction"] = prediction
        if confidence is not None:
            entry["confidence"] = confidence
        if label is not None:
            entry["label"] = label
            
        # Queue the data storage operation
        self._task_queue.put({
            "type": "store_data",
            "id": f"store_{entry_id}",
            "args": {
                "input_data": input_data,
                "input_type": input_type,
                "entry_id": entry_id,
                "metadata": metadata
            }
        })
        
        # Add to buffer (filename will be updated later if/when available)
        self.metadata_buffer.append(entry)
        
        # Check if buffer needs to be written
        if len(self.metadata_buffer) >= self.buffer_size:
            self._task_queue.put({
                "type": "write_buffer",
                "id": None  # No need to track result
            })
            
        return entry_id

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
        
    def log_sample(
        self,
        input_data: Any,
        prediction: Optional[str] = None,
        confidence: Optional[float] = None,
        label: Optional[str] = None,
        metadata: Optional[Dict[str, Any]] = None
    ) -> str:
        """Log a sample (alias for log_data with autodetection of type).
        
        This is a convenience method that tries to determine the input type
        based on the input data, then calls log_data with the appropriate type.
        
        Args:
            input_data: The data to log
            prediction: Optional model prediction
            confidence: Optional confidence score
            label: Optional ground truth label
            metadata: Additional metadata
            
        Returns:
            Unique entry ID for this logged sample
        """
        # Try to determine input type based on data
        if isinstance(input_data, bytes) and len(input_data) > 0:
            # Check for common image headers
            if input_data.startswith(b'\xff\xd8\xff'):  # JPEG
                input_type = DATA_TYPE_IMAGE
            elif input_data.startswith(b'\x89PNG\r\n\x1a\n'):  # PNG
                input_type = DATA_TYPE_IMAGE
            elif input_data.startswith(b'GIF8'):  # GIF
                input_type = DATA_TYPE_IMAGE
            else:
                # Default to binary data (treat as image)
                input_type = DATA_TYPE_IMAGE
        elif isinstance(input_data, dict):
            input_type = DATA_TYPE_JSON
        elif isinstance(input_data, (str, list)):
            input_type = DATA_TYPE_TEXT
        else:
            # Default to JSON for other types
            input_type = DATA_TYPE_JSON
            
        return self.log_data(
            input_data=input_data,
            input_type=input_type,
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
        """Queue a task to write the metadata buffer to disk (non-blocking)."""
        if not self.metadata_buffer:
            return
        
        # Take a copy of the current buffer to process
        current_buffer = self.metadata_buffer.copy()
        
        # Clear the buffer now since we have a copy
        self.metadata_buffer = []
        
        # Queue a task to write the buffer
        self._task_queue.put({
            "type": "write_buffer",
            "args": {"buffer": current_buffer}
        })
    
    def _write_metadata_buffer_internal(self, buffer=None) -> None:
        """Internal method to write metadata buffer to disk (called by worker thread)."""
        buffer_to_write = buffer or self.metadata_buffer
        
        if not buffer_to_write:
            return
            
        for entry in buffer_to_write:
            try:
                self.metadata_manager.save_metadata(
                    entry=entry,
                    metadata_dir=self.storage_dir / "metadata"
                )
            except Exception as e:
                print(f"Error writing metadata for entry {entry.get('entry_id')}: {str(e)}")
        
        # Only clear the main buffer if we're not using a provided buffer
        if buffer is None:
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

    def _export_to_csv_internal(
        self,
        output_path: Optional[Union[str, Path]] = None,
        filter_func: Optional[callable] = None
    ) -> str:
        """Internal method to export entries to CSV (called by worker thread)."""
        # This already runs in the worker thread, so we can perform I/O directly
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

    def export_to_csv(
        self,
        output_path: Optional[Union[str, Path]] = None,
        filter_func: Optional[callable] = None,
        blocking: bool = False,
        timeout: float = 10.0
    ) -> str:
        """Export entries to CSV.
        
        By default, this method is non-blocking and the CSV export will be performed in a 
        background thread. It first ensures any pending metadata is saved.
        
        If blocking=True, it will wait for the export to complete before returning.
        
        Args:
            output_path: Optional custom path for the CSV file
            filter_func: Optional function to filter entries
            blocking: Whether to wait for the export to complete (default: False)
            timeout: Timeout in seconds when in blocking mode (default: 10.0)
            
        Returns:
            The path where the CSV will be saved
        """
        # First ensure any pending metadata is written
        self._write_metadata_buffer()
        
        # Determine the output path
        if output_path is None:
            timestamp = int(time.time())
            output_path = self.storage_dir / f"export_{timestamp}.csv"
        else:
            output_path = Path(output_path)
        
        # Create parent directory if needed
        ensure_dir(output_path.parent)
        
        # If blocking, export synchronously without using the queue
        if blocking:
            try:
                # Create empty file immediately to signal the file exists
                with open(output_path, "w") as f:
                    f.write("")  # Create empty file
                
                # Then perform actual export
                self._export_to_csv_internal(output_path=output_path, filter_func=filter_func)
                return str(output_path)
            except Exception as e:
                print(f"Error in blocking CSV export: {e}")
                return str(output_path)
            
        # Queue the export task for non-blocking operation
        task_id = f"export_{int(time.time())}"
        
        # Create empty file immediately to signal the file exists
        try:
            with open(output_path, "w") as f:
                f.write("")  # Create empty file
        except Exception as e:
            print(f"Error creating empty CSV file: {e}")
        
        # Queue the actual export task
        self._task_queue.put({
            "type": "export_csv",
            "id": task_id,
            "args": {
                "output_path": output_path,
                "filter_func": filter_func
            }
        })
        
        # Return the expected path
        return str(output_path)

    def close(self) -> None:
        """Close the logger and clean up resources.
        
        This method will block until all pending operations are completed.
        """
        if self._stop_event.is_set():
            return
            
        print("Closing data logger and stopping worker thread")
        
        # First, write any pending metadata
        if self.metadata_buffer:
            # Do this synchronously to ensure it completes before shutdown
            self._write_metadata_buffer_internal()
        
        # Wait for queue to empty
        if self._task_queue.qsize() > 0:
            print(f"Waiting for {self._task_queue.qsize()} pending tasks to be processed")
            self._task_queue.join()
        
        # Create final summary (synchronously)
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
                # Import and use TinyLCMJSONEncoder to handle numpy arrays
                from tinylcm.utils.file_utils import TinyLCMJSONEncoder
                json.dump(summary, f, indent=2, cls=TinyLCMJSONEncoder)
        except Exception as e:
            print(f"Warning: Error creating logger summary: {str(e)}")
        
        # Stop the worker thread
        self._stop_event.set()
        
        # Wait for worker thread to exit (with timeout)
        if self._worker_thread and self._worker_thread.is_alive():
            self._worker_thread.join(timeout=5.0)
            if self._worker_thread.is_alive():
                print("Warning: Worker thread did not exit cleanly within timeout")
                
        self.session_id = None