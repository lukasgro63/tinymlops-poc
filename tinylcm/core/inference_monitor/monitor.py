import json
import os
import threading
import queue
import time
import uuid
from pathlib import Path
from typing import Any, Dict, List, Optional, Union

from tinylcm.constants import DEFAULT_INFERENCE_DIR, DEFAULT_LOG_INTERVAL, DEFAULT_MEMORY_ENTRIES
from tinylcm.core.inference_monitor.metrics_collector import InferenceMetricsCollector
from tinylcm.utils.config import Config, get_config
from tinylcm.utils.errors import MonitoringError
from tinylcm.utils.file_utils import ensure_dir, save_json
from tinylcm.utils.logging import setup_logger

logger = setup_logger(__name__)

class InferenceMonitor:
    """Monitors model inference latency and confidence metrics using non-blocking I/O.
    
    This class collects core inference metrics (latency, confidence) without any anomaly
    detection logic, as drift and performance changes are now handled by the adaptive
    components (AdaptationTracker and adaptive handlers).
    
    The class implements a non-blocking interface where logging and I/O operations
    run in a background thread to avoid impacting inference performance, especially
    on resource-constrained devices.
    
    The primary metrics tracked are:
    - Inference latency (time for a single pipeline.process call)
    - Prediction confidence (if provided by the classifier)
    - Prediction distribution (frequency of each predicted class)
    """
    def __init__(
        self,
        storage_dir: Optional[Union[str, Path]] = None,
        memory_window_size: Optional[int] = None,
        log_interval: Optional[int] = None,
        config: Optional[Config] = None,
        worker_thread_daemon: bool = True
    ):
        """Initialize the inference monitor.
        
        Args:
            storage_dir: Directory to store inference logs
            memory_window_size: Maximum number of inference records to keep in memory
            log_interval: Number of records before starting a new log file
            config: Configuration object
            worker_thread_daemon: Whether the worker thread should be a daemon thread
        """
        self.config = config or get_config()
        component_config = self.config.get_component_config("inference_monitor")
        self.logger = setup_logger(f"{__name__}.{self.__class__.__name__}")
        self.storage_dir = Path(storage_dir or component_config.get("storage_dir", DEFAULT_INFERENCE_DIR))
        self.memory_window_size = memory_window_size or component_config.get("memory_window_size", DEFAULT_MEMORY_ENTRIES)
        self.log_interval = log_interval or component_config.get("log_interval", DEFAULT_LOG_INTERVAL)
        ensure_dir(self.storage_dir)
        self.session_id = str(uuid.uuid4())
        self.logger.info(f"Starting inference monitor session: {self.session_id}")
        self.metrics_collector = InferenceMetricsCollector(window_size=self.memory_window_size)
        self._current_log_file = None
        self._record_count = 0
        
        # Initialize worker thread for non-blocking I/O operations
        self._record_queue = queue.Queue()
        self._worker_thread = None
        self._stop_event = threading.Event()
        self._worker_thread_daemon = worker_thread_daemon
        self._start_worker_thread()

    def _start_worker_thread(self) -> None:
        """Starts the worker thread that processes the record queue."""
        if self._worker_thread is not None and self._worker_thread.is_alive():
            self.logger.warning("Worker thread already running")
            return
        
        self._stop_event.clear()
        self._worker_thread = threading.Thread(
            target=self._worker_loop,
            daemon=self._worker_thread_daemon
        )
        self._worker_thread.start()
        self.logger.debug("Started inference monitor worker thread")
    
    def _worker_loop(self) -> None:
        """Main worker loop that processes records and tasks in the queue."""
        while not self._stop_event.is_set():
            try:
                # Wait for a record or task with a timeout to allow checking the stop event
                try:
                    item = self._record_queue.get(timeout=0.1)
                except queue.Empty:
                    continue
                
                # Process the item
                try:
                    # Check if this is an export task
                    if isinstance(item, dict) and item.get("_export_task"):
                        self._export_metrics_internal(item)
                    else:
                        # Regular record
                        self._write_record_internal(item)
                except Exception as e:
                    self.logger.error(f"Error processing queue item: {str(e)}")
                finally:
                    self._record_queue.task_done()
                    
            except Exception as e:
                self.logger.error(f"Unexpected error in worker thread: {str(e)}")
                # Sleep briefly to avoid tight loop in case of recurring errors
                time.sleep(0.1)
        
        self.logger.debug("Worker thread stopping")
    
    def track_inference(
        self,
        input_id: str,
        prediction: str,
        confidence: Optional[float] = None,
        latency_ms: Optional[float] = None,
        metadata: Optional[Dict[str, Any]] = None,
        timestamp: Optional[float] = None
    ) -> Dict[str, Any]:
        """Tracks an inference event with core metrics (non-blocking).
        
        This method is non-blocking. The record will be processed and written to
        disk in a background thread.
        
        Args:
            input_id: Unique identifier for the input
            prediction: Model prediction (label/class/value)
            confidence: Confidence score (0-1)
            latency_ms: Inference latency in milliseconds
            metadata: Additional data to store with the record
            timestamp: Optional timestamp (defaults to current time)
            
        Returns:
            The inference record dictionary
        """
        record = {
            "input_id": input_id,
            "prediction": prediction,
            "timestamp": timestamp if timestamp is not None else time.time(),
            "session_id": self.session_id,
            "metadata": metadata or {}
        }
        
        # Only include optional fields if they are provided
        if confidence is not None:
            record["confidence"] = confidence
        
        if latency_ms is not None:
            record["latency_ms"] = latency_ms
        
        # Update metrics (this is fast and runs in the current thread)
        self.metrics_collector.add_record(record)
        
        # Queue the record for writing by the worker thread
        self._record_queue.put(record)
        
        return record

    def _write_record(self, record: Dict[str, Any]) -> None:
        """Queue a record to be written asynchronously.
        
        This method is non-blocking and delegates the actual writing to the worker thread.
        
        Args:
            record: The record to write
        """
        self._record_queue.put(record)
    
    def _write_record_internal(self, record: Dict[str, Any]) -> None:
        """Actually write the record to disk (called by worker thread).
        
        Args:
            record: The record to write
        
        Raises:
            MonitoringError: If writing to the log file fails
        """
        self._record_count += 1
        if self._current_log_file is None:
            timestamp = int(time.time())
            log_path = self.storage_dir / f"inference_log_{timestamp}_{self.session_id}.jsonl"
            ensure_dir(log_path.parent)
            self._current_log_file = log_path
            self.logger.info(f"Created new log file: {log_path}")
        
        try:
            with open(self._current_log_file, "a", encoding="utf-8") as f:
                f.write(json.dumps(record))
                f.write("\n")
        except Exception as e:
            self.logger.error(f"Error writing record to log file: {str(e)}")
            raise MonitoringError(f"Failed to write record to log file: {str(e)}")
            
        if self._record_count >= self.log_interval:
            self._current_log_file = None
            self._record_count = 0
            self.logger.debug(f"Log interval reached ({self.log_interval} records). Next record will start a new log file.")

    def get_current_metrics(self) -> Dict[str, Any]:
        """Get the current metrics collected by the monitor.
        
        This method is thread-safe and can be called from any thread.
        
        Returns:
            Dictionary of current metrics (latency, confidence, prediction distribution)
        """
        metrics = self.metrics_collector.get_metrics()
        metrics["session_id"] = self.session_id
        return metrics

    def export_metrics(self, format: str = "json") -> str:
        """Export current metrics to a file (non-blocking).
        
        This method is non-blocking and delegates the export operation to the worker thread.
        
        Args:
            format: Output format ("json" or "csv")
            
        Returns:
            The expected path where metrics will be saved
        """
        metrics = self.get_current_metrics()
        timestamp = int(time.time())
        metrics["timestamp"] = timestamp
        
        # Determine file path
        if format.lower() == "json":
            filename = f"metrics_{timestamp}_{self.session_id}.json"
        elif format.lower() == "csv":
            filename = f"metrics_{timestamp}_{self.session_id}.csv"
        else:
            raise ValueError(f"Unsupported export format: {format}")
            
        file_path = self.storage_dir / filename
        
        # Create export task for worker thread
        self._record_queue.put({
            "_export_task": True,
            "format": format,
            "metrics": metrics,
            "file_path": file_path
        })
        
        self.logger.info(f"Queued metrics export to {file_path}")
        return str(file_path)

    def _export_metrics_internal(self, task: Dict[str, Any]) -> None:
        """Actually export metrics to a file (called by worker thread).
        
        Args:
            task: Export task details
        """
        format = task["format"]
        metrics = task["metrics"]
        file_path = task["file_path"]
        
        try:
            if format.lower() == "json":
                save_json(metrics, file_path)
            elif format.lower() == "csv":
                flat_metrics = self._flatten_metrics(metrics)
                import csv
                with open(file_path, 'w', newline='', encoding='utf-8') as csvfile:
                    writer = csv.DictWriter(csvfile, fieldnames=flat_metrics.keys())
                    writer.writeheader()
                    writer.writerow(flat_metrics)
            self.logger.info(f"Exported metrics to {file_path}")
        except Exception as e:
            self.logger.error(f"Error exporting metrics to {file_path}: {str(e)}")

    def _flatten_metrics(self, metrics: Dict[str, Any], prefix: str = "") -> Dict[str, Any]:
        """Flatten a nested metrics dictionary for CSV export.
        
        Args:
            metrics: Nested metrics dictionary
            prefix: Prefix for nested keys
            
        Returns:
            Flattened metrics dictionary
        """
        flattened = {}
        for key, value in metrics.items():
            if isinstance(value, dict):
                nested_flat = self._flatten_metrics(value, f"{prefix}{key}_")
                flattened.update(nested_flat)
            else:
                flattened[f"{prefix}{key}"] = value
        return flattened

    def close(self) -> None:
        """Close the monitor and clean up resources.
        
        This method stops the worker thread and exports final metrics. It will
        block until the worker thread has processed all queued records and
        finished any pending exports.
        """
        if not self._stop_event.is_set():
            self.logger.info("Closing inference monitor and stopping worker thread")
            
            # Export final metrics before stopping
            try:
                metrics = self.get_current_metrics()
                timestamp = int(time.time())
                metrics["timestamp"] = timestamp
                filename = f"final_metrics_{timestamp}_{self.session_id}.json"
                file_path = self.storage_dir / filename
                
                # Do this synchronously to ensure it completes before shutdown
                save_json(metrics, file_path)
                self.logger.info(f"Exported final metrics to {file_path}")
            except Exception as e:
                self.logger.error(f"Error exporting final metrics: {str(e)}")
            
            # Wait for queue to empty
            if self._record_queue.qsize() > 0:
                self.logger.info(f"Waiting for {self._record_queue.qsize()} pending records to be processed")
                self._record_queue.join()
            
            # Stop the worker thread
            self._stop_event.set()
            
            # Wait for worker thread to exit (with timeout)
            if self._worker_thread and self._worker_thread.is_alive():
                self._worker_thread.join(timeout=5.0)
                if self._worker_thread.is_alive():
                    self.logger.warning("Worker thread did not exit cleanly within timeout")
            
            self.logger.info(f"Closed inference monitor session: {self.session_id}")
    
    def reset_metrics(self) -> None:
        """Reset all collected metrics."""
        self.metrics_collector.reset()
        self.logger.info("Reset inference metrics")
    
    def __enter__(self):
        """Enter context manager protocol."""
        return self
    
    def __exit__(self, exc_type, exc_val, exc_tb):
        """Exit context manager protocol."""
        self.close()
        return False