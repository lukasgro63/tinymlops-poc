import json
import os
import threading
import queue
import time
import uuid
import psutil
from pathlib import Path
from typing import Any, Dict, List, Optional, Union, Callable

from tinylcm.constants import DEFAULT_OPERATION_DIR, DEFAULT_LOG_INTERVAL, DEFAULT_MEMORY_ENTRIES
from tinylcm.core.operational_monitor.metrics_collector import OperationalMetricsCollector
from tinylcm.utils.config import Config, get_config
from tinylcm.utils.errors import MonitoringError
from tinylcm.utils.file_utils import ensure_dir, save_json
from tinylcm.utils.logging import setup_logger

logger = setup_logger(__name__)

class OperationalMonitor:
    """Monitors operational aspects of the TinyLCM framework with non-blocking I/O.
    
    This class collects and tracks operational metrics including:
    - Inference latency and throughput
    - System resource usage (CPU, memory)
    - Confidence scores
    - Prediction distributions
    - System uptime and stability
    
    The class implements a non-blocking interface where logging and I/O operations
    run in a background thread to avoid impacting inference performance, especially
    on resource-constrained devices.
    
    This class replaces the older InferenceMonitor with expanded capabilities for
    monitoring operational aspects beyond just inference performance.
    """
    def __init__(
        self,
        storage_dir: Optional[Union[str, Path]] = None,
        memory_window_size: Optional[int] = None,
        log_interval: Optional[int] = None,
        config: Optional[Config] = None,
        worker_thread_daemon: bool = True,
        collect_system_metrics: bool = True,
        system_metrics_interval: float = 10.0
    ):
        """Initialize the operational monitor.
        
        Args:
            storage_dir: Directory to store logs
            memory_window_size: Maximum number of records to keep in memory
            log_interval: Number of records before starting a new log file
            config: Configuration object
            worker_thread_daemon: Whether the worker thread should be a daemon thread
            collect_system_metrics: Whether to collect system metrics automatically
            system_metrics_interval: Interval (in seconds) for collecting system metrics
        """
        self.config = config or get_config()
        component_config = self.config.get_component_config("operational_monitor")
        self.logger = setup_logger(f"{__name__}.{self.__class__.__name__}")
        self.storage_dir = Path(storage_dir or component_config.get("storage_dir", DEFAULT_OPERATION_DIR))
        self.memory_window_size = memory_window_size or component_config.get("memory_window_size", DEFAULT_MEMORY_ENTRIES)
        self.log_interval = log_interval or component_config.get("log_interval", DEFAULT_LOG_INTERVAL)
        ensure_dir(self.storage_dir)
        self.session_id = str(uuid.uuid4())
        self.logger.info(f"Starting operational monitor session: {self.session_id}")
        self.metrics_collector = OperationalMetricsCollector(window_size=self.memory_window_size)
        self._current_log_file = None
        self._record_count = 0
        
        # Initialize worker thread for non-blocking I/O operations
        self._record_queue = queue.Queue()
        self._worker_thread = None
        self._stop_event = threading.Event()
        self._worker_thread_daemon = worker_thread_daemon
        self._start_worker_thread()
        
        # System metrics collection
        self.collect_system_metrics = collect_system_metrics
        self.system_metrics_interval = system_metrics_interval
        self._system_metrics_thread = None
        
        if self.collect_system_metrics:
            self._start_system_metrics_collection()
        
        # Register metadata about this session
        self._register_session_metadata()

    def _register_session_metadata(self) -> None:
        """Register initial metadata about this monitoring session."""
        metadata_file = self.storage_dir / f"session_{self.session_id}_metadata.json"
        
        # Collect system information
        import platform as plt
        system_info = {
            "platform": plt.system(),
            "platform_release": plt.release(),
            "platform_version": plt.version(),
            "platform_machine": plt.machine(),
            "python_version": os.sys.version,
            "start_time": time.time(),
            "start_time_iso": time.strftime("%Y-%m-%d %H:%M:%S", time.localtime()),
            "hostname": os.uname().nodename if hasattr(os, 'uname') else 'unknown',
            "session_id": self.session_id
        }
        
        # Try to collect CPU and memory information
        try:
            import psutil
            memory = psutil.virtual_memory()
            system_info["cpu_count"] = psutil.cpu_count()
            system_info["memory_total_mb"] = memory.total / (1024 * 1024)
            system_info["cpu_freq_mhz"] = psutil.cpu_freq().current if psutil.cpu_freq() else None
        except (ImportError, AttributeError):
            self.logger.debug("psutil not available, skipping detailed system info collection")
        
        # Queue the metadata save operation
        self._record_queue.put(("save_metadata", metadata_file, system_info))

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
        self.logger.debug("Started operational monitor worker thread")
    
    def _start_system_metrics_collection(self) -> None:
        """Start background thread for collecting system metrics."""
        if self._system_metrics_thread is not None and self._system_metrics_thread.is_alive():
            self.logger.warning("System metrics thread already running")
            return
        
        self._system_metrics_thread = threading.Thread(
            target=self._system_metrics_loop,
            daemon=True
        )
        self._system_metrics_thread.start()
        self.logger.debug(f"Started system metrics collection thread (interval: {self.system_metrics_interval}s)")
    
    def _system_metrics_loop(self) -> None:
        """Background thread for periodically collecting system metrics."""
        while not self._stop_event.is_set():
            try:
                # Collect system metrics
                metrics = self._collect_system_metrics()
                
                # Add to metrics collector
                if metrics:
                    self.metrics_collector.add_system_metrics(metrics)
                    
                    # Log system metrics periodically
                    self._record_queue.put(("system_metrics", metrics))
                
                # Sleep until next collection
                time.sleep(self.system_metrics_interval)
                
            except Exception as e:
                self.logger.error(f"Error collecting system metrics: {str(e)}")
                time.sleep(self.system_metrics_interval)
    
    def _collect_system_metrics(self) -> Dict[str, Any]:
        """Collect system metrics (CPU, memory, etc.)."""
        metrics = {}
        
        try:
            # CPU
            metrics["cpu_percent"] = psutil.cpu_percent(interval=0.1)
            
            # Memory
            memory = psutil.virtual_memory()
            metrics["memory_percent"] = memory.percent
            metrics["memory_used_mb"] = memory.used / (1024 * 1024)
            metrics["memory_available_mb"] = memory.available / (1024 * 1024)
            
            # Disk
            disk = psutil.disk_usage('/')
            metrics["disk_percent"] = disk.percent
            metrics["disk_free_gb"] = disk.free / (1024 * 1024 * 1024)
            
            # Process info
            process = psutil.Process()
            metrics["process_cpu_percent"] = process.cpu_percent(interval=0.1)
            metrics["process_memory_mb"] = process.memory_info().rss / (1024 * 1024)
            metrics["process_threads"] = process.num_threads()
            
            # Network (optional, can be disabled if needed)
            # net = psutil.net_io_counters()
            # metrics["net_sent_mb"] = net.bytes_sent / (1024 * 1024)
            # metrics["net_recv_mb"] = net.bytes_recv / (1024 * 1024)
            
            # Battery (if available)
            if hasattr(psutil, 'sensors_battery'):
                battery = psutil.sensors_battery()
                if battery:
                    metrics["battery_percent"] = battery.percent
                    metrics["battery_power_plugged"] = battery.power_plugged
            
            # Add timestamp
            metrics["timestamp"] = time.time()
            
        except (ImportError, AttributeError, OSError) as e:
            self.logger.debug(f"Could not collect some system metrics: {str(e)}")
        
        return metrics
    
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
                    task_type = item[0]
                    
                    if task_type == "record":
                        self._write_record_internal(item[1])
                    elif task_type == "system_metrics":
                        self._write_system_metrics_internal(item[1])
                    elif task_type == "export":
                        self._export_metrics_internal(item[1], item[2], item[3])
                    elif task_type == "save_metadata":
                        self._save_metadata_internal(item[1], item[2])
                    else:
                        self.logger.warning(f"Unknown task type: {task_type}")
                        
                except Exception as e:
                    self.logger.error(f"Error processing queue item: {str(e)}")
                finally:
                    self._record_queue.task_done()
                    
            except Exception as e:
                self.logger.error(f"Unexpected error in worker thread: {str(e)}")
                # Sleep briefly to avoid tight loop in case of recurring errors
                time.sleep(0.1)
        
        self.logger.debug("Worker thread stopping")
    
    def track_operation(
        self,
        operation_id: str,
        operation_type: str,
        result: Any,
        success: bool = True,
        latency_ms: Optional[float] = None,
        confidence: Optional[float] = None,
        metadata: Optional[Dict[str, Any]] = None,
        timestamp: Optional[float] = None
    ) -> Dict[str, Any]:
        """Tracks an operational event with metrics (non-blocking).
        
        This method is non-blocking. The record will be processed and written to
        disk in a background thread.
        
        Args:
            operation_id: Unique identifier for this operation
            operation_type: Type of operation (e.g., "inference", "adaptation", "drift_check")
            result: Operation result or output
            success: Whether the operation was successful
            latency_ms: Operation latency in milliseconds
            confidence: Confidence score if applicable (0-1)
            metadata: Additional data to store with the record
            timestamp: Optional timestamp (defaults to current time)
            
        Returns:
            The operation record dictionary
        """
        record = {
            "operation_id": operation_id,
            "operation_type": operation_type,
            "result": result,
            "success": success,
            "timestamp": timestamp if timestamp is not None else time.time(),
            "session_id": self.session_id,
            "metadata": metadata or {}
        }
        
        # Only include optional fields if they are provided
        if confidence is not None:
            record["confidence"] = confidence
        
        if latency_ms is not None:
            record["latency_ms"] = latency_ms
            
        # For inference operations, track as prediction for metrics
        if operation_type == "inference" and result is not None:
            # Update metrics (this is fast and runs in the current thread)
            inference_record = {
                "prediction": result,
                "confidence": confidence,
                "latency_ms": latency_ms
            }
            self.metrics_collector.add_record(inference_record)
        
        # Queue the record for writing by the worker thread
        self._record_queue.put(("record", record))
        
        return record
    
    def track_inference(
        self,
        input_id: str,
        prediction: Any,
        confidence: Optional[float] = None,
        latency_ms: Optional[float] = None,
        metadata: Optional[Dict[str, Any]] = None,
        timestamp: Optional[float] = None,
        features: Optional[Any] = None
    ) -> Dict[str, Any]:
        """Tracks an inference event with core metrics (non-blocking).
        
        This is a convenience method equivalent to track_operation with operation_type="inference".
        
        Args:
            input_id: Unique identifier for the input
            prediction: Model prediction (label/class/value)
            confidence: Confidence score (0-1)
            latency_ms: Inference latency in milliseconds
            metadata: Additional data to store with the record
            timestamp: Optional timestamp (defaults to current time)
            features: Optional feature vector used for prediction
            
        Returns:
            The inference record dictionary
        """
        # Ensure metadata is a dictionary
        if metadata is None:
            metadata = {}
            
        record = {
            "input_id": input_id,
            "prediction": prediction,
            "timestamp": timestamp if timestamp is not None else time.time(),
            "session_id": self.session_id,
            "metadata": metadata
        }
        
        # Only include optional fields if they are provided
        if confidence is not None:
            record["confidence"] = confidence
        
        if latency_ms is not None:
            record["latency_ms"] = latency_ms
            
        # Don't include features in default record for storage efficiency
        # But retain them for drift detection if needed
        if features is not None:
            record["features"] = features
        
        # Update metrics (this is fast and runs in the current thread)
        self.metrics_collector.add_record(record)
        
        # Don't include features in stored record to save space
        # Clone record without features for storage
        storage_record = record.copy()
        if "features" in storage_record:
            del storage_record["features"]
        
        # Queue the record for writing by the worker thread
        self._record_queue.put(("record", storage_record))
        
        return record
    
    def track_system_metrics(self, metrics: Dict[str, Any]) -> None:
        """Track custom system metrics.
        
        This can be used to track custom system metrics not collected automatically.
        
        Args:
            metrics: Dictionary of system metrics to track
        """
        # Add to metrics collector
        self.metrics_collector.add_system_metrics(metrics)
        
        # Queue for storage
        self._record_queue.put(("system_metrics", metrics))
    
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
            log_path = self.storage_dir / f"operational_log_{timestamp}_{self.session_id}.jsonl"
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
    
    def _write_system_metrics_internal(self, metrics: Dict[str, Any]) -> None:
        """Write system metrics to a separate log file.
        
        Args:
            metrics: System metrics to write
        """
        # Use a dedicated file for system metrics
        metrics_file = self.storage_dir / f"system_metrics_{self.session_id}.jsonl"
        
        try:
            with open(metrics_file, "a", encoding="utf-8") as f:
                f.write(json.dumps(metrics))
                f.write("\n")
        except Exception as e:
            self.logger.error(f"Error writing system metrics to log file: {str(e)}")
    
    def _save_metadata_internal(self, file_path: Path, metadata: Dict[str, Any]) -> None:
        """Save session metadata to a file.
        
        Args:
            file_path: Path to save metadata to
            metadata: Session metadata
        """
        try:
            with open(file_path, "w", encoding="utf-8") as f:
                json.dump(metadata, f, indent=2)
            self.logger.debug(f"Saved session metadata to {file_path}")
        except Exception as e:
            self.logger.error(f"Error saving session metadata: {str(e)}")

    def get_current_metrics(self) -> Dict[str, Any]:
        """Get the current metrics collected by the monitor.
        
        This method is thread-safe and can be called from any thread.
        
        Returns:
            Dictionary of current metrics (latency, confidence, prediction distribution, system)
        """
        metrics = self.metrics_collector.get_metrics()
        metrics["session_id"] = self.session_id
        return metrics

    def export_metrics(
        self, 
        format: str = "json", 
        include_system_metrics: bool = True,
        callback: Optional[Callable[[str], None]] = None
    ) -> str:
        """Export current metrics to a file (non-blocking).
        
        Args:
            format: Output format ("json" or "csv")
            include_system_metrics: Whether to include system metrics
            callback: Optional callback to call with the exported file path
            
        Returns:
            The path where metrics will be saved
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
        
        # Ensure the parent directory exists
        ensure_dir(self.storage_dir)
        
        # Create empty file immediately
        try:
            with open(file_path, "w") as f:
                f.write("")  # Create empty file
        except Exception as e:
            self.logger.error(f"Error creating empty metrics file: {e}")
            
        # Create export task for worker thread
        self._record_queue.put(("export", format, metrics, file_path))
        
        self.logger.info(f"Queued metrics export to {file_path}")
        
        # Call callback if provided
        if callback:
            callback(str(file_path))
            
        return str(file_path)

    def _export_metrics_internal(
        self, 
        format: str, 
        metrics: Dict[str, Any],
        file_path: Path
    ) -> None:
        """Actually export metrics to a file (called by worker thread).
        
        Args:
            format: Output format ("json" or "csv")
            metrics: Metrics to export
            file_path: Path to export to
        """
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
            self.logger.info("Closing operational monitor and stopping worker threads")
            
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
            
            # Wait for system metrics thread to exit (with timeout)
            if self._system_metrics_thread and self._system_metrics_thread.is_alive():
                self._system_metrics_thread.join(timeout=2.0)
                if self._system_metrics_thread.is_alive():
                    self.logger.warning("System metrics thread did not exit cleanly within timeout")
            
            self.logger.info(f"Closed operational monitor session: {self.session_id}")
    
    def reset_metrics(self) -> None:
        """Reset all collected metrics."""
        self.metrics_collector.reset()
        self.logger.info("Reset operational metrics")
    
    def get_latency_percentiles(self) -> Dict[str, float]:
        """Get latency percentiles.
        
        Returns:
            Dictionary with latency percentiles (p50, p95, p99)
        """
        return self.metrics_collector.get_latency_percentiles()
    
    def get_throughput(self) -> Optional[float]:
        """Get estimated throughput in inferences per second.
        
        Returns:
            Throughput or None if no data available
        """
        return self.metrics_collector.get_throughput()
    
    def get_system_metrics(self) -> Dict[str, Any]:
        """Get current system metrics.
        
        Returns:
            Dictionary of system metrics
        """
        return self.metrics_collector.get_system_metrics_summary()
    
    def get_uptime(self) -> float:
        """Get operational uptime in seconds.
        
        Returns:
            Uptime in seconds
        """
        return self.metrics_collector.get_uptime()
    
    def __enter__(self):
        """Enter context manager protocol."""
        return self
    
    def __exit__(self, exc_type, exc_val, exc_tb):
        """Exit context manager protocol."""
        self.close()
        return False