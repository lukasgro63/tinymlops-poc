import os
import json
import time
import glob
import shutil
import threading
import queue
import atexit
from typing import Any, Dict, List, Optional, Union, Tuple, Callable
from pathlib import Path
import uuid

from tinylcm.core.data_structures import AdaptiveState
from tinylcm.core.base import AdaptiveComponent
from tinylcm.utils.logging import setup_logger
from tinylcm.utils.file_utils import ensure_directory_exists

# Optional imports for quarantine and heuristic support
try:
    from tinylcm.core.quarantine.buffer import QuarantineBuffer
    QUARANTINE_AVAILABLE = True
except ImportError:
    QUARANTINE_AVAILABLE = False

try:
    from tinylcm.core.heuristics.adapter import HeuristicAdapter
    HEURISTICS_AVAILABLE = True
except ImportError:
    HEURISTICS_AVAILABLE = False

try:
    from tinylcm.core.drift_detection.base import DriftDetector, AutonomousDriftDetector
    AUTONOMOUS_DETECTORS_AVAILABLE = True
except ImportError:
    AUTONOMOUS_DETECTORS_AVAILABLE = False

logger = setup_logger(__name__)


class StateManager:
    """Manages the persistence and loading of adaptive model states.
    
    This class is responsible for saving and loading the state of adaptive
    components, such as classifiers, feature extractors, and handlers. It
    replaces the functionality of the old ModelManager with a focus on
    adaptive learning.
    
    The save operations are performed in a background thread to avoid blocking
    the main application thread, especially important for resource-constrained
    devices.
    """
    
    def __init__(
        self,
        storage_dir: str = "./adaptive_states",
        max_states: int = 10,
        auto_create_dir: bool = True,
        queue_size: int = 100,
        worker_count: int = 1
    ):
        """Initialize the state manager.
        
        Args:
            storage_dir: Directory to store state snapshots
            max_states: Maximum number of state snapshots to keep
            auto_create_dir: Whether to automatically create the storage directory
            queue_size: Size of the internal task queue for background operations
            worker_count: Number of worker threads to spawn (usually 1 is sufficient)
        """
        self.storage_dir = storage_dir
        self.max_states = max_states
        
        if auto_create_dir:
            ensure_directory_exists(storage_dir)
        
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
        
        logger.debug(f"Initialized StateManager with storage_dir={storage_dir}, worker_count={worker_count}")
    
    def save_state(
        self,
        classifier: AdaptiveComponent,
        handler: AdaptiveComponent,
        extractor: Optional[AdaptiveComponent] = None,
        samples: Optional[List[Dict[str, Any]]] = None,
        metadata: Optional[Dict[str, Any]] = None,
        state_id: Optional[str] = None,
        callback: Optional[Callable[[str], None]] = None,
        quarantine_buffer: Optional['QuarantineBuffer'] = None,
        heuristic_adapter: Optional['HeuristicAdapter'] = None,
        autonomous_detectors: Optional[List['AutonomousDriftDetector']] = None
    ) -> str:
        """Save the current state of adaptive components (non-blocking).
        
        This method queues the save operation to be performed in a background
        thread and returns immediately.
        
        Args:
            classifier: Adaptive classifier component
            handler: Adaptive handler component
            extractor: Optional feature extractor component
            samples: Optional list of samples to include
            metadata: Optional metadata to include with the state
            state_id: Optional ID for the state, will be generated if not provided
            callback: Optional callback function to call when save is complete
            quarantine_buffer: Optional quarantine buffer to include in the state
            heuristic_adapter: Optional heuristic adapter to include in the state
            autonomous_detectors: Optional list of autonomous drift detectors
            
        Returns:
            ID of the state being saved
        """
        # Generate state ID if not provided
        if state_id is None:
            state_id = f"state_{int(time.time())}_{str(uuid.uuid4())[:8]}"
        
        # Get states from extensions
        quarantine_state = {}
        if QUARANTINE_AVAILABLE and quarantine_buffer is not None:
            try:
                quarantine_state = quarantine_buffer.get_statistics()
            except Exception as e:
                logger.warning(f"Error getting quarantine state: {str(e)}")
        
        heuristic_state = {}
        if HEURISTICS_AVAILABLE and heuristic_adapter is not None:
            try:
                heuristic_state = heuristic_adapter.get_statistics()
            except Exception as e:
                logger.warning(f"Error getting heuristic state: {str(e)}")
        
        # Get autonomous detector states
        detector_states = {}
        if AUTONOMOUS_DETECTORS_AVAILABLE and autonomous_detectors:
            for i, detector in enumerate(autonomous_detectors):
                try:
                    detector_name = detector.__class__.__name__
                    detector_states[detector_name] = detector.get_state()
                except Exception as e:
                    logger.warning(f"Error getting state for detector {i}: {str(e)}")
        
        # Prepare extended metadata
        extended_metadata = metadata or {}
        extended_metadata.update({
            "has_quarantine_state": bool(quarantine_state),
            "has_heuristic_state": bool(heuristic_state),
            "autonomous_detectors": list(detector_states.keys()) if detector_states else []
        })
        
        # Create state object
        state = AdaptiveState(
            classifier_state=classifier.get_state(),
            handler_state=handler.get_state(),
            extractor_state=extractor.get_state() if extractor is not None else {},
            samples=samples or [],
            creation_timestamp=time.time(),
            metadata=extended_metadata
        )
        
        # Add extension states to the state object
        if quarantine_state:
            state.metadata["quarantine_state"] = quarantine_state
            
        if heuristic_state:
            state.metadata["heuristic_state"] = heuristic_state
            
        if detector_states:
            state.metadata["detector_states"] = detector_states
        
        # Queue the save task
        self.task_queue.put(("save", state, state_id, callback))
        
        logger.debug(f"Queued save operation for state: {state_id}")
        
        return state_id
    
    def load_state(
        self,
        state_id: str,
        classifier: AdaptiveComponent,
        handler: AdaptiveComponent,
        extractor: Optional[AdaptiveComponent] = None,
        quarantine_buffer: Optional['QuarantineBuffer'] = None,
        heuristic_adapter: Optional['HeuristicAdapter'] = None,
        autonomous_detectors: Optional[List['AutonomousDriftDetector']] = None
    ) -> Dict[str, Any]:
        """Load a saved state into adaptive components.
        
        This method is blocking because loading needs to happen synchronously
        to ensure the components are properly initialized before use.
        
        Args:
            state_id: ID of the state to load
            classifier: Adaptive classifier component to load state into
            handler: Adaptive handler component to load state into
            extractor: Optional feature extractor component to load state into
            quarantine_buffer: Optional quarantine buffer to restore
            heuristic_adapter: Optional heuristic adapter to restore
            autonomous_detectors: Optional list of autonomous drift detectors
            
        Returns:
            Metadata from the loaded state
        """
        # Construct file path
        file_path = os.path.join(self.storage_dir, f"{state_id}.json")
        
        # Check if file exists
        if not os.path.exists(file_path):
            raise FileNotFoundError(f"State not found: {file_path}")
        
        # Load state from file
        with open(file_path, 'r') as f:
            state_dict = json.load(f)
        
        # Convert to AdaptiveState object
        state = AdaptiveState.from_dict(state_dict)
        
        # Load state into components
        classifier.set_state(state.classifier_state)
        handler.set_state(state.handler_state)
        
        if extractor is not None and state.extractor_state:
            extractor.set_state(state.extractor_state)
        
        # Load state into autonomous detectors if available
        if AUTONOMOUS_DETECTORS_AVAILABLE and autonomous_detectors and "detector_states" in state.metadata:
            detector_states = state.metadata.get("detector_states", {})
            for detector in autonomous_detectors:
                detector_name = detector.__class__.__name__
                if detector_name in detector_states:
                    try:
                        detector.set_state(detector_states[detector_name])
                        logger.debug(f"Restored state for detector {detector_name}")
                    except Exception as e:
                        logger.warning(f"Error restoring state for detector {detector_name}: {str(e)}")
        
        # We don't restore full quarantine and heuristic states as they might be
        # very large and complex. Instead, we just log information about them.
        if QUARANTINE_AVAILABLE and quarantine_buffer is not None and "quarantine_state" in state.metadata:
            quarantine_state = state.metadata.get("quarantine_state", {})
            logger.info(f"Found quarantine state with {quarantine_state.get('total_samples', 0)} samples")
            
        if HEURISTICS_AVAILABLE and heuristic_adapter is not None and "heuristic_state" in state.metadata:
            heuristic_state = state.metadata.get("heuristic_state", {})
            logger.info(f"Found heuristic state with {heuristic_state.get('total_adaptations', 0)} adaptations")
            
            # Update known classes in heuristic adapter
            if "created_classes" in heuristic_state:
                created_classes = set(heuristic_state.get("created_classes", []))
                known_classes = set(heuristic_state.get("known_classes", []))
                if created_classes or known_classes:
                    heuristic_adapter.created_classes = created_classes
                    heuristic_adapter.known_classes = known_classes
                    logger.debug(f"Restored {len(created_classes)} created classes and {len(known_classes)} known classes")
        
        logger.info(f"Loaded adaptive state from {file_path}")
        
        # Return metadata
        return state.metadata
    
    def list_states(self) -> List[Dict[str, Any]]:
        """List all available states.
        
        Returns:
            List of state information dictionaries
        """
        # Get all state files
        state_files = glob.glob(os.path.join(self.storage_dir, "*.json"))
        
        states = []
        
        for file_path in state_files:
            try:
                # Load state metadata
                with open(file_path, 'r') as f:
                    state_dict = json.load(f)
                
                state = AdaptiveState.from_dict(state_dict)
                
                # Extract file name as state ID
                state_id = os.path.splitext(os.path.basename(file_path))[0]
                
                # Add state information
                states.append({
                    "state_id": state_id,
                    "creation_timestamp": state.creation_timestamp,
                    "version": state.version,
                    "metadata": state.metadata,
                    "sample_count": len(state.samples)
                })
            except Exception as e:
                logger.warning(f"Error loading state from {file_path}: {str(e)}")
        
        # Sort by creation timestamp (newest first)
        states.sort(key=lambda x: x["creation_timestamp"], reverse=True)
        
        return states
    
    def delete_state(
        self,
        state_id: str,
        blocking: bool = False,
        callback: Optional[Callable[[bool], None]] = None
    ) -> bool:
        """Delete a saved state.
        
        Args:
            state_id: ID of the state to delete
            blocking: Whether to perform the deletion synchronously
            callback: Optional callback function to call when delete is complete
            
        Returns:
            True if the operation was scheduled or executed, False if the state was not found
        """
        # Construct file path
        file_path = os.path.join(self.storage_dir, f"{state_id}.json")
        
        # Check if file exists
        if not os.path.exists(file_path):
            logger.warning(f"State not found: {file_path}")
            if callback:
                callback(False)
            return False
        
        if blocking:
            # Perform the deletion synchronously
            os.remove(file_path)
            logger.info(f"Deleted state: {state_id}")
            if callback:
                callback(True)
            return True
        else:
            # Queue the deletion task
            self.task_queue.put(("delete", state_id, callback))
            logger.debug(f"Queued delete operation for state: {state_id}")
            return True
    
    def cleanup_old_states(
        self,
        blocking: bool = False,
        callback: Optional[Callable[[], None]] = None
    ) -> None:
        """Clean up old states if there are more than max_states.
        
        Args:
            blocking: Whether to perform the cleanup synchronously
            callback: Optional callback function to call when cleanup is complete
        """
        if blocking:
            self._perform_cleanup()
            if callback:
                callback()
        else:
            # Queue the cleanup task
            self.task_queue.put(("cleanup", callback))
            logger.debug("Queued cleanup operation")
    
    def _perform_cleanup(self) -> None:
        """Internal method to perform the cleanup operation."""
        # List all states
        states = self.list_states()
        
        # If there are more states than max_states, delete the oldest ones
        if len(states) > self.max_states:
            # Sort by creation timestamp (oldest first)
            states.sort(key=lambda x: x["creation_timestamp"])
            
            # Delete oldest states
            for state in states[:len(states) - self.max_states]:
                self.delete_state(state["state_id"], blocking=True)
    
    def get_latest_state_id(self) -> Optional[str]:
        """Get the ID of the most recent state.
        
        Returns:
            ID of the most recent state, or None if no states exist
        """
        states = self.list_states()
        
        if not states:
            return None
        
        # States are already sorted by creation timestamp (newest first)
        return states[0]["state_id"]
    
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
                
                if task_type == "save":
                    _, state, state_id, callback = task
                    self._perform_save(state, state_id, callback)
                
                elif task_type == "delete":
                    _, state_id, callback = task
                    success = self._perform_delete(state_id)
                    if callback:
                        callback(success)
                
                elif task_type == "cleanup":
                    _, callback = task
                    self._perform_cleanup()
                    if callback:
                        callback()
                
                # Mark task as done
                self.task_queue.task_done()
                
            except Exception as e:
                logger.error(f"Error in StateManager worker thread: {str(e)}")
    
    def _perform_save(
        self,
        state: AdaptiveState,
        state_id: str,
        callback: Optional[Callable[[str], None]] = None
    ) -> None:
        """Internal method to perform the save operation."""
        try:
            # Construct file path
            file_path = os.path.join(self.storage_dir, f"{state_id}.json")
            
            # Save state to file
            with open(file_path, 'w') as f:
                json.dump(state.to_dict(), f, indent=2)
            
            logger.info(f"Saved adaptive state to {file_path}")
            
            # Clean up old states
            self._perform_cleanup()
            
            if callback:
                callback(state_id)
                
        except Exception as e:
            logger.error(f"Error saving state {state_id}: {str(e)}")
            
            # If callback is provided, call it with None to indicate failure
            if callback:
                callback(None)
    
    def _perform_delete(self, state_id: str) -> bool:
        """Internal method to perform the delete operation."""
        try:
            # Construct file path
            file_path = os.path.join(self.storage_dir, f"{state_id}.json")
            
            # Delete file
            os.remove(file_path)
            
            logger.info(f"Deleted state: {state_id}")
            
            return True
        except Exception as e:
            logger.error(f"Error deleting state {state_id}: {str(e)}")
            return False
    
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
        
        logger.debug("StateManager worker threads stopped")
    
    def get_queue_size(self) -> int:
        """Get the current number of tasks in the queue.
        
        Returns:
            Number of tasks in the queue
        """
        return self.task_queue.qsize()
    
    def is_idle(self) -> bool:
        """Check if the StateManager is idle (no pending tasks).
        
        Returns:
            True if no tasks are pending, False otherwise
        """
        return self.task_queue.empty()