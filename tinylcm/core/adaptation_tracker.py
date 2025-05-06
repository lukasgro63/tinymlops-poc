import os
import json
import time
import csv
import glob
import threading
import queue
import atexit
from typing import Any, Dict, List, Optional, Union, Tuple, Callable
from pathlib import Path
from collections import deque

from tinylcm.core.data_structures import AdaptationEvent
from tinylcm.utils.logging import setup_logger
from tinylcm.utils.file_utils import ensure_directory_exists, TinyLCMJSONEncoder

logger = setup_logger(__name__)


class AdaptationTracker:
    """Tracks adaptation events and metrics over time.
    
    This class is responsible for recording adaptation events, metrics,
    and other information about the adaptation process. It replaces the
    functionality of the old TrainingTracker with a focus on adaptive learning.
    
    The logging operations are performed in a background thread to avoid blocking
    the main application thread, especially important for resource-constrained
    devices.
    """
    
    def __init__(
        self,
        log_dir: str = "./adaptation_logs",
        max_events_in_memory: int = 100,
        auto_create_dir: bool = True,
        export_format: str = "json",
        queue_size: int = 100,
        worker_count: int = 1
    ):
        """Initialize the adaptation tracker.
        
        Args:
            log_dir: Directory to store adaptation logs
            max_events_in_memory: Maximum number of events to keep in memory
            auto_create_dir: Whether to automatically create the log directory
            export_format: Format for exporting logs ("json" or "csv")
            queue_size: Size of the internal task queue for background operations
            worker_count: Number of worker threads to spawn (usually 1 is sufficient)
        """
        self.log_dir = log_dir
        self.max_events_in_memory = max_events_in_memory
        self.export_format = export_format
        
        # Create log directory if it doesn't exist
        if auto_create_dir:
            ensure_directory_exists(log_dir)
        
        # Initialize session
        self.session_id = f"session_{int(time.time())}"
        self.session_start_time = time.time()
        self.session_metadata = {}
        
        # Initialize events and metrics
        self.events = deque(maxlen=max_events_in_memory)
        self.metrics = {}
        
        # Create session directory
        self.session_dir = os.path.join(log_dir, self.session_id)
        if auto_create_dir:
            ensure_directory_exists(self.session_dir)
        
        # Create log files
        self.events_file = os.path.join(self.session_dir, "events.json")
        self.metrics_file = os.path.join(self.session_dir, "metrics.json")
        self.session_file = os.path.join(self.session_dir, "session.json")
        
        # Initialize log files with empty data
        with open(self.events_file, 'w') as f:
            json.dump([], f, cls=TinyLCMJSONEncoder)
        
        with open(self.metrics_file, 'w') as f:
            json.dump({}, f, cls=TinyLCMJSONEncoder)
        
        # Set up background worker thread and queue for non-blocking I/O
        self.task_queue = queue.Queue(maxsize=queue_size)
        self.workers = []
        self.stop_event = threading.Event()
        
        # Start worker threads
        for _ in range(worker_count):
            worker = threading.Thread(target=self._worker_loop, daemon=True)
            worker.start()
            self.workers.append(worker)
        
        # Register cleanup handler for graceful shutdown
        atexit.register(self.join)
        
        logger.debug(f"Initialized AdaptationTracker with log_dir={log_dir}, worker_count={worker_count}")
    
    def start_session(self, metadata: Optional[Dict[str, Any]] = None) -> str:
        """Start a new tracking session.
        
        Args:
            metadata: Optional metadata for the session
            
        Returns:
            ID of the session
        """
        # Update session metadata
        self.session_metadata = metadata or {}
        self.session_metadata.update({
            "session_id": self.session_id,
            "session_start_time": self.session_start_time,
            "session_start_time_iso": time.strftime("%Y-%m-%d %H:%M:%S", time.localtime(self.session_start_time))
        })
        
        # Queue session info save
        self.task_queue.put(("save_session", self.session_metadata))
        
        logger.info(f"Started adaptation tracking session: {self.session_id}")
        
        return self.session_id
    
    def log_adaptation_event(
        self,
        event: AdaptationEvent,
        callback: Optional[Callable[[bool], None]] = None
    ) -> None:
        """Log an adaptation event (non-blocking).
        
        This method queues the event logging operation to be performed in a
        background thread and returns immediately.
        
        Args:
            event: The adaptation event to log
            callback: Optional callback function to call when logging is complete
        """
        # Add event to in-memory queue
        self.events.append(event)
        
        # Convert to dictionary for serialization
        event_dict = event.to_dict()
        
        # Queue event logging
        self.task_queue.put(("log_event", event_dict, callback))
        
        logger.debug(f"Queued adaptation event: {event.event_type} (drift: {event.drift_detected})")
        
    def log_drift_detection(
        self,
        detector_id: str,
        detector_type: str,
        metric_value: float,
        threshold: float,
        sample_id: Optional[str] = None,
        additional_info: Optional[Dict[str, Any]] = None
    ) -> None:
        """Log a drift detection event (non-blocking).
        
        Args:
            detector_id: Identifier of the detector that detected the drift
            detector_type: Type of the detector (e.g., "confidence", "distribution")
            metric_value: Value of the monitored metric that triggered the detection
            threshold: Threshold value that was exceeded
            sample_id: ID of the sample that triggered the detection (if available)
            additional_info: Additional information about the detection
        """
        event = AdaptationEvent(
            event_type="drift_detection",
            drift_detected=True,
            metadata={
                "detector_id": detector_id,
                "detector_type": detector_type,
                "metric_value": metric_value,
                "threshold": threshold,
                "sample_id": sample_id,
                **(additional_info or {})
            }
        )
        self.log_adaptation_event(event)
        
    def log_quarantine(
        self,
        sample_id: str,
        reason: str,
        prediction: Any,
        confidence: Optional[float] = None,
        detector_id: Optional[str] = None
    ) -> None:
        """Log a sample quarantine event (non-blocking).
        
        Args:
            sample_id: ID of the quarantined sample
            reason: Reason for quarantining the sample
            prediction: Original prediction for the sample
            confidence: Confidence score of the prediction (if available)
            detector_id: ID of the detector that triggered the quarantine (if applicable)
        """
        metadata = {
            "sample_id": sample_id,
            "reason": reason,
            "prediction": prediction
        }
        
        if confidence is not None:
            metadata["confidence"] = confidence
        
        if detector_id is not None:
            metadata["detector_id"] = detector_id
        
        event = AdaptationEvent(
            event_type="sample_quarantined",
            drift_detected=True,
            metadata=metadata
        )
        self.log_adaptation_event(event)
    
    def log_heuristic_adaptation(
        self,
        samples: List[str],
        original_label: Any,
        new_label: Any,
        pre_snapshot_id: Optional[str] = None,
        cluster_size: Optional[int] = None,
        variance: Optional[float] = None
    ) -> None:
        """Log a heuristic adaptation event (non-blocking).
        
        Args:
            samples: List of sample IDs used for the adaptation
            original_label: Original label before adaptation
            new_label: New label assigned by the heuristic
            pre_snapshot_id: ID of the snapshot created before adaptation (if available)
            cluster_size: Size of the cluster used for adaptation (if applicable)
            variance: Variance/dispersion of the cluster (if applicable)
        """
        metadata = {
            "samples": samples,
            "original_label": original_label,
            "new_label": new_label
        }
        
        if pre_snapshot_id is not None:
            metadata["pre_snapshot_id"] = pre_snapshot_id
        
        if cluster_size is not None:
            metadata["cluster_size"] = cluster_size
        
        if variance is not None:
            metadata["variance"] = variance
        
        event = AdaptationEvent(
            event_type="heuristic_adaptation",
            samples_added=len(samples),
            metadata=metadata
        )
        self.log_adaptation_event(event)
    
    def log_server_validation(
        self,
        sample_id: str,
        heuristic_label: Optional[Any],
        validated_label: Any,
        is_correct: bool,
        confidence: Optional[float] = None
    ) -> None:
        """Log a server validation event (non-blocking).
        
        Args:
            sample_id: ID of the validated sample
            heuristic_label: Label assigned by the heuristic (if applicable)
            validated_label: Validated label from the server
            is_correct: Whether the original or heuristic prediction was correct
            confidence: Server confidence in the validation (if available)
        """
        metadata = {
            "sample_id": sample_id,
            "validated_label": validated_label,
            "is_correct": is_correct
        }
        
        if heuristic_label is not None:
            metadata["heuristic_label"] = heuristic_label
        
        if confidence is not None:
            metadata["confidence"] = confidence
        
        event = AdaptationEvent(
            event_type="server_validation",
            metadata=metadata
        )
        self.log_adaptation_event(event)
    
    def log_rollback(
        self,
        snapshot_id: str,
        reason: str,
        samples_affected: Optional[List[str]] = None,
        triggered_by: Optional[str] = None
    ) -> None:
        """Log a rollback event (non-blocking).
        
        Args:
            snapshot_id: ID of the snapshot that was restored
            reason: Reason for the rollback
            samples_affected: List of sample IDs affected by the rollback (if applicable)
            triggered_by: Entity that triggered the rollback (e.g., "server", "local")
        """
        metadata = {
            "snapshot_id": snapshot_id,
            "reason": reason
        }
        
        if samples_affected is not None:
            metadata["samples_affected"] = samples_affected
            metadata["samples_count"] = len(samples_affected)
        
        if triggered_by is not None:
            metadata["triggered_by"] = triggered_by
        
        event = AdaptationEvent(
            event_type="rollback",
            metadata=metadata
        )
        self.log_adaptation_event(event)
        
    def log_accuracy_metric(
        self,
        metric_name: str,
        value: float,
        batch_size: Optional[int] = None,
        window_size: Optional[int] = None,
        labels: Optional[List[Any]] = None,
        metadata: Optional[Dict[str, Any]] = None
    ) -> None:
        """Log an accuracy metric event (non-blocking).
        
        Args:
            metric_name: Name of the metric (e.g., "accuracy", "f1_score")
            value: Value of the metric
            batch_size: Size of the batch used to calculate the metric (if applicable)
            window_size: Size of the rolling window (if applicable)
            labels: List of labels included in the calculation (if applicable)
            metadata: Additional metadata about the metric calculation
        """
        metric_data = {
            "metric_name": metric_name,
            "value": value
        }
        
        if batch_size is not None:
            metric_data["batch_size"] = batch_size
        
        if window_size is not None:
            metric_data["window_size"] = window_size
        
        if labels is not None:
            metric_data["labels"] = labels
        
        if metadata is not None:
            metric_data.update(metadata)
        
        event = AdaptationEvent(
            event_type="accuracy_metric",
            performance_after={"accuracy": value} if metric_name == "accuracy" else {metric_name: value},
            metadata=metric_data
        )
        self.log_adaptation_event(event)
        
    def log_snapshot_creation(
        self,
        snapshot_id: str,
        reason: str,
        metadata: Optional[Dict[str, Any]] = None
    ) -> None:
        """Log a snapshot creation event (non-blocking).
        
        This method is designed to be called by the AdaptiveStateManager when
        a new snapshot is created.
        
        Args:
            snapshot_id: ID of the created snapshot
            reason: Reason for creating the snapshot
            metadata: Additional metadata about the snapshot
        """
        event_metadata = {
            "snapshot_id": snapshot_id,
            "reason": reason,
            "operation": "creation"
        }
        
        if metadata is not None:
            event_metadata.update(metadata)
        
        event = AdaptationEvent(
            event_type="snapshot",
            metadata=event_metadata
        )
        self.log_adaptation_event(event)
    
    def log_snapshot_loading(
        self,
        snapshot_id: str,
        reason: str,
        success: bool,
        metadata: Optional[Dict[str, Any]] = None
    ) -> None:
        """Log a snapshot loading event (non-blocking).
        
        This method is designed to be called by the AdaptiveStateManager when
        a snapshot is loaded.
        
        Args:
            snapshot_id: ID of the loaded snapshot
            reason: Reason for loading the snapshot
            success: Whether the snapshot was loaded successfully
            metadata: Additional metadata about the snapshot loading
        """
        event_metadata = {
            "snapshot_id": snapshot_id,
            "reason": reason,
            "operation": "loading",
            "success": success
        }
        
        if metadata is not None:
            event_metadata.update(metadata)
        
        event = AdaptationEvent(
            event_type="snapshot",
            metadata=event_metadata
        )
        self.log_adaptation_event(event)
    
    def log_metrics(
        self,
        metrics: Dict[str, Union[float, int, str]],
        timestamp: Optional[float] = None,
        callback: Optional[Callable[[bool], None]] = None
    ) -> None:
        """Log metrics at a specific timestamp (non-blocking).
        
        This method queues the metrics logging operation to be performed in a
        background thread and returns immediately.
        
        Args:
            metrics: Dictionary of metrics to log
            timestamp: Optional timestamp, defaults to current time
            callback: Optional callback function to call when logging is complete
        """
        if timestamp is None:
            timestamp = time.time()
        
        # Format timestamp as string for key
        timestamp_str = str(timestamp)
        
        # Update in-memory metrics
        self.metrics[timestamp_str] = metrics
        
        # Add timestamp to metrics
        metrics_with_time = metrics.copy()
        metrics_with_time["timestamp"] = timestamp
        
        # Queue metrics logging
        self.task_queue.put(("log_metrics", timestamp_str, metrics_with_time, callback))
        
        logger.debug(f"Queued metrics at {time.strftime('%H:%M:%S', time.localtime(timestamp))}")
    
    def export_to_mlflow(
        self,
        mlflow_dir: str,
        callback: Optional[Callable[[str], None]] = None
    ) -> None:
        """Export adaptation logs to MLflow format (non-blocking).
        
        This method queues the export operation to be performed in a background
        thread and returns immediately.
        
        Args:
            mlflow_dir: Directory to export MLflow data to
            callback: Optional callback function to call when export is complete
        """
        # Queue export operation
        self.task_queue.put(("export_mlflow", mlflow_dir, callback))
        logger.debug(f"Queued export to MLflow format: {mlflow_dir}")
    
    def get_events(self) -> List[AdaptationEvent]:
        """Get all adaptation events.
        
        Returns:
            List of adaptation events
        """
        return list(self.events)
    
    def get_metrics(self) -> Dict[str, Dict[str, Any]]:
        """Get all metrics.
        
        Returns:
            Dictionary of metrics indexed by timestamp
        """
        return self.metrics
    
    def get_session_info(self) -> Dict[str, Any]:
        """Get session information.
        
        Returns:
            Dictionary of session information
        """
        return self.session_metadata
    
    def get_drift_statistics(self) -> Dict[str, Any]:
        """Calculate drift statistics from adaptation events.
        
        Returns:
            Dictionary of drift statistics
        """
        # Count drift events
        drift_events = [e for e in self.events if e.drift_detected]
        
        # Count events by type
        event_types = {}
        for event in self.events:
            if event.event_type not in event_types:
                event_types[event.event_type] = 0
            event_types[event.event_type] += 1
        
        # Calculate time between drifts
        drift_intervals = []
        last_drift_time = None
        
        for event in drift_events:
            if last_drift_time is not None:
                interval = event.timestamp - last_drift_time
                drift_intervals.append(interval)
            
            last_drift_time = event.timestamp
        
        # Calculate statistics
        stats = {
            "total_events": len(self.events),
            "drift_events": len(drift_events),
            "event_types": event_types,
            "drift_percentage": (len(drift_events) / len(self.events)) if self.events else 0,
            "avg_drift_interval": sum(drift_intervals) / len(drift_intervals) if drift_intervals else None,
            "min_drift_interval": min(drift_intervals) if drift_intervals else None,
            "max_drift_interval": max(drift_intervals) if drift_intervals else None,
        }
        
        return stats
    
    def _worker_loop(self) -> None:
        """Worker thread loop to process tasks from the queue."""
        while not self.stop_event.is_set():
            try:
                # Get task from queue (block for 0.1 seconds then check stop event)
                try:
                    task = self.task_queue.get(timeout=0.1)
                except queue.Empty:
                    continue
                
                # Process task based on type
                task_type = task[0]
                
                if task_type == "save_session":
                    _, metadata = task
                    self._perform_save_session(metadata)
                
                elif task_type == "log_event":
                    _, event_dict, callback = task
                    success = self._perform_log_event(event_dict)
                    if callback:
                        callback(success)
                
                elif task_type == "log_metrics":
                    _, timestamp_str, metrics_with_time, callback = task
                    success = self._perform_log_metrics(timestamp_str, metrics_with_time)
                    if callback:
                        callback(success)
                
                elif task_type == "export_mlflow":
                    _, mlflow_dir, callback = task
                    try:
                        export_path = self._perform_export_mlflow(mlflow_dir)
                        if callback:
                            callback(export_path)
                    except Exception as e:
                        logger.error(f"Error exporting to MLflow: {str(e)}")
                        if callback:
                            callback(None)
                
                # Mark task as done
                self.task_queue.task_done()
                
            except Exception as e:
                logger.error(f"Error in AdaptationTracker worker thread: {str(e)}")
    
    def _perform_save_session(self, metadata: Dict[str, Any]) -> bool:
        """Internal method to save session information."""
        try:
            with open(self.session_file, 'w') as f:
                json.dump(metadata, f, indent=2, cls=TinyLCMJSONEncoder)
            return True
        except Exception as e:
            logger.error(f"Error saving session info: {str(e)}")
            return False
    
    def _perform_log_event(self, event_dict: Dict[str, Any]) -> bool:
        """Internal method to log an adaptation event."""
        try:
            # Read existing events
            with open(self.events_file, 'r+') as f:
                try:
                    events = json.load(f)
                except json.JSONDecodeError:
                    events = []
                
                # Append new event
                events.append(event_dict)
                
                # Seek to beginning and write updated events
                f.seek(0)
                f.truncate()
                json.dump(events, f, indent=2, cls=TinyLCMJSONEncoder)
            
            return True
        except Exception as e:
            logger.error(f"Error logging adaptation event: {str(e)}")
            return False
    
    def _perform_log_metrics(
        self,
        timestamp_str: str,
        metrics_with_time: Dict[str, Any]
    ) -> bool:
        """Internal method to log metrics."""
        try:
            # Read existing metrics
            with open(self.metrics_file, 'r+') as f:
                try:
                    all_metrics = json.load(f)
                except json.JSONDecodeError:
                    all_metrics = {}
                
                # Add new metrics
                all_metrics[timestamp_str] = metrics_with_time
                
                # Seek to beginning and write updated metrics
                f.seek(0)
                f.truncate()
                json.dump(all_metrics, f, indent=2, cls=TinyLCMJSONEncoder)
            
            return True
        except Exception as e:
            logger.error(f"Error logging metrics: {str(e)}")
            return False
    
    def _perform_export_mlflow(self, mlflow_dir: str) -> str:
        """Internal method to export logs to MLflow format."""
        # Check for PyYAML
        if yaml is None:
            raise ImportError("PyYAML is required for MLflow export")
        
        # Create MLflow run directory
        run_id = f"run_{self.session_id}"
        run_dir = os.path.join(mlflow_dir, run_id)
        ensure_directory_exists(run_dir)
        
        # Export session info
        with open(os.path.join(run_dir, "meta.yaml"), 'w') as f:
            # Convert session metadata to MLflow format
            mlflow_meta = {
                "artifact_uri": f"file://{os.path.abspath(run_dir)}/artifacts",
                "end_time": int(time.time() * 1000),  # MLflow uses milliseconds
                "experiment_id": "0",  # Default experiment
                "lifecycle_stage": "active",
                "run_id": run_id,
                "start_time": int(self.session_start_time * 1000),
                "status": "FINISHED",
                "user_id": os.getenv("USER", "unknown")
            }
            
            # Add session metadata as tags
            mlflow_meta["tags"] = {}
            for key, value in self.session_metadata.items():
                if isinstance(value, (str, int, float, bool)):
                    mlflow_meta["tags"][key] = str(value)
            
            f.write(yaml.dump(mlflow_meta, default_flow_style=False))
        
        # Export metrics
        metrics_dir = os.path.join(run_dir, "metrics")
        ensure_directory_exists(metrics_dir)
        
        # Collect all metrics
        all_metrics = {}
        for timestamp_str, metrics in self.metrics.items():
            for key, value in metrics.items():
                if isinstance(value, (int, float)):
                    if key not in all_metrics:
                        all_metrics[key] = []
                    
                    all_metrics[key].append((float(timestamp_str), value))
        
        # Write each metric to a separate file
        for metric_name, values in all_metrics.items():
            with open(os.path.join(metrics_dir, f"{metric_name}.csv"), 'w', newline='') as f:
                writer = csv.writer(f)
                for timestamp, value in values:
                    # MLflow format: timestamp, value, step
                    writer.writerow([timestamp * 1000, value, 0])
        
        # Export adaptation events as artifacts
        artifacts_dir = os.path.join(run_dir, "artifacts")
        ensure_directory_exists(artifacts_dir)
        
        # Copy events file
        with open(os.path.join(artifacts_dir, "adaptation_events.json"), 'w') as f:
            events_list = [event.to_dict() if hasattr(event, "to_dict") else event for event in self.events]
            json.dump(events_list, f, indent=2, cls=TinyLCMJSONEncoder)
        
        logger.info(f"Exported adaptation logs to MLflow format: {run_dir}")
        
        return run_dir
    
    def join(self, timeout: Optional[float] = None) -> None:
        """Wait for all pending tasks to complete and shutdown worker threads.
        
        This method is called automatically when the application exits, but can
        also be called manually to ensure all tasks are completed before shutdown.
        
        Args:
            timeout: Maximum time to wait for tasks to complete (None = wait forever)
        """
        # Wait for all tasks to complete
        try:
            self.task_queue.join()
        except Exception as e:
            logger.warning(f"Error joining task queue: {str(e)}")
        
        # Signal workers to stop
        self.stop_event.set()
        
        # Wait for workers to finish
        for worker in self.workers:
            worker.join(timeout=timeout)
        
        logger.debug("AdaptationTracker worker threads stopped")
    
    def get_queue_size(self) -> int:
        """Get the current number of tasks in the queue.
        
        Returns:
            Number of tasks in the queue
        """
        return self.task_queue.qsize()
    
    def is_idle(self) -> bool:
        """Check if the AdaptationTracker is idle (no pending tasks).
        
        Returns:
            True if no tasks are pending, False otherwise
        """
        return self.task_queue.empty()
    
    @classmethod
    def list_sessions(cls, log_dir: str) -> List[Dict[str, Any]]:
        """List all available sessions.
        
        Args:
            log_dir: Directory containing session logs
            
        Returns:
            List of session information dictionaries
        """
        # Get all session directories
        session_dirs = [d for d in glob.glob(os.path.join(log_dir, "session_*")) if os.path.isdir(d)]
        
        sessions = []
        
        for session_dir in session_dirs:
            session_file = os.path.join(session_dir, "session.json")
            
            if os.path.exists(session_file):
                try:
                    with open(session_file, 'r') as f:
                        session_info = json.load(f)
                    
                    # Add session directory
                    session_info["session_dir"] = session_dir
                    
                    sessions.append(session_info)
                except Exception as e:
                    logger.warning(f"Error loading session from {session_file}: {str(e)}")
        
        # Sort by start time (newest first)
        sessions.sort(key=lambda x: x.get("session_start_time", 0), reverse=True)
        
        return sessions
    
    @classmethod
    def load_session(cls, session_dir: str) -> 'AdaptationTracker':
        """Load an existing session.
        
        Args:
            session_dir: Directory containing session data
            
        Returns:
            AdaptationTracker instance with loaded data
        """
        # Create tracker with existing directory
        tracker = cls(log_dir=os.path.dirname(session_dir), auto_create_dir=False)
        
        # Set session info
        tracker.session_dir = session_dir
        tracker.session_id = os.path.basename(session_dir)
        
        # Load session metadata
        session_file = os.path.join(session_dir, "session.json")
        if os.path.exists(session_file):
            with open(session_file, 'r') as f:
                tracker.session_metadata = json.load(f)
                tracker.session_start_time = tracker.session_metadata.get("session_start_time", time.time())
        
        # Set file paths
        tracker.events_file = os.path.join(session_dir, "events.json")
        tracker.metrics_file = os.path.join(session_dir, "metrics.json")
        
        # Load events
        if os.path.exists(tracker.events_file):
            with open(tracker.events_file, 'r') as f:
                try:
                    events_data = json.load(f)
                    for event_dict in events_data:
                        event = AdaptationEvent.from_dict(event_dict)
                        tracker.events.append(event)
                except json.JSONDecodeError:
                    logger.warning(f"Error loading events from {tracker.events_file}")
        
        # Load metrics
        if os.path.exists(tracker.metrics_file):
            with open(tracker.metrics_file, 'r') as f:
                try:
                    tracker.metrics = json.load(f)
                except json.JSONDecodeError:
                    logger.warning(f"Error loading metrics from {tracker.metrics_file}")
        
        logger.info(f"Loaded adaptation tracking session: {tracker.session_id}")
        
        return tracker

try:
    import yaml
except ImportError:
    yaml = None
    logger.warning("PyYAML not installed. Export to MLflow format will not be available.")