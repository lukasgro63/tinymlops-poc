"""
AdaptiveStateManager module for TinyLCM.

This module handles persistence and versioning of model states, allowing
for snapshot creation and rollbacks in case of problematic adaptations.
It uses non-blocking I/O for saving states to minimize impact on the
main application thread.
"""

import os
import json
import time
import threading
import queue
import copy
from typing import Dict, Any, List, Optional, Callable, Union
from pathlib import Path
import uuid

from tinylcm.utils.logging import setup_logger
from tinylcm.utils.file_utils import ensure_directory_exists
from tinylcm.core.adaptation_tracker import AdaptationTracker

logger = setup_logger(__name__)


class AdaptiveStateManager:
    """Manages state persistence and versioning for adaptive components.
    
    This class is responsible for:
    1. Saving and loading states of adaptive components (classifier, detectors, etc.)
    2. Creating versioned snapshots of states for rollback capabilities
    3. Managing non-blocking I/O to minimize impact on the main thread
    
    It provides a lightweight versioning system that enables rolling back
    to previous states when problematic adaptations are detected.
    """
    
    def __init__(
        self,
        storage_dir: str = "./states",
        max_snapshots: int = 10,
        worker_thread_daemon: bool = True,
        adaptation_tracker: Optional[AdaptationTracker] = None
    ):
        """Initialize the state manager.
        
        Args:
            storage_dir: Directory to store state files
            max_snapshots: Maximum number of snapshots to keep
            worker_thread_daemon: Whether worker thread should be daemon
            adaptation_tracker: Optional AdaptationTracker for logging events
        """
        self.storage_dir = Path(storage_dir)
        self.max_snapshots = max_snapshots
        self.adaptation_tracker = adaptation_tracker
        
        # Create storage directory
        ensure_directory_exists(self.storage_dir)
        ensure_directory_exists(self.storage_dir / "snapshots")
        
        # Session identifier
        self.session_id = str(uuid.uuid4())[:8]
        
        # Track snapshots
        self.snapshots = []
        self._load_snapshot_registry()
        
        # Track the current state
        self.current_state_components = {}
        self.current_state_path = None
        
        # Worker thread for non-blocking I/O
        self._task_queue = queue.Queue()
        self._stop_event = threading.Event()
        self._worker_thread = threading.Thread(
            target=self._worker_loop,
            daemon=worker_thread_daemon
        )
        self._worker_thread.start()
        
        logger.debug(f"AdaptiveStateManager initialized with storage_dir={storage_dir}")
    
    def save_current_state(
        self,
        component_states: Dict[str, Any],
        sync: bool = False
    ) -> bool:
        """Save the current state of components.
        
        Args:
            component_states: Dictionary mapping component names to their states
            sync: Whether to save synchronously (blocking) or asynchronously
            
        Returns:
            True if successful, False if failed (only for sync mode)
        """
        # Update current state components
        self.current_state_components = copy.deepcopy(component_states)
        
        # Create state file path
        timestamp = int(time.time())
        state_file = self.storage_dir / f"state_{self.session_id}_{timestamp}.json"
        self.current_state_path = state_file
        
        # Prepare state object
        state_obj = {
            "timestamp": timestamp,
            "session_id": self.session_id,
            "components": component_states
        }
        
        if sync:
            # Synchronous (blocking) save
            try:
                with open(state_file, 'w') as f:
                    json.dump(state_obj, f, indent=2)
                
                logger.debug(f"Saved current state to {state_file}")
                return True
            except Exception as e:
                logger.error(f"Error saving state: {str(e)}")
                return False
        else:
            # Asynchronous (non-blocking) save
            self._task_queue.put(("save_state", state_obj, state_file))
            return True
    
    def load_state(
        self,
        state_file: Optional[Union[str, Path]] = None
    ) -> Optional[Dict[str, Any]]:
        """Load a state from disk (blocking operation).
        
        Args:
            state_file: Path to state file (uses most recent if None)
            
        Returns:
            Loaded state dictionary or None if failed
        """
        try:
            # If no state file provided, use most recent
            if state_file is None:
                if self.current_state_path is not None:
                    state_file = self.current_state_path
                else:
                    # Find most recent state file
                    state_files = list(self.storage_dir.glob("state_*.json"))
                    if not state_files:
                        logger.warning("No state files found")
                        return None
                    
                    # Sort by modification time (newest first)
                    state_files.sort(key=lambda x: x.stat().st_mtime, reverse=True)
                    state_file = state_files[0]
            
            # Convert string to Path if needed
            if isinstance(state_file, str):
                state_file = Path(state_file)
            
            # Load state file
            with open(state_file, 'r') as f:
                state = json.load(f)
            
            # Update current state path
            self.current_state_path = state_file
            self.current_state_components = state.get("components", {})
            
            logger.debug(f"Loaded state from {state_file}")
            return state.get("components", {})
            
        except Exception as e:
            logger.error(f"Error loading state: {str(e)}")
            return None
    
    def create_snapshot(
        self,
        reason: str,
        metadata: Optional[Dict[str, Any]] = None
    ) -> Optional[str]:
        """Create a snapshot of the current state for rollback.
        
        Args:
            reason: Reason for creating this snapshot
            metadata: Additional metadata to include
            
        Returns:
            Snapshot ID if successful, None if failed
        """
        # Generate snapshot ID
        snapshot_id = str(uuid.uuid4())
        
        # Create snapshot timestamp
        timestamp = int(time.time())
        
        # Create snapshot file path
        snapshot_file = self.storage_dir / "snapshots" / f"snapshot_{snapshot_id}.json"
        
        # Prepare snapshot object
        snapshot_obj = {
            "id": snapshot_id,
            "timestamp": timestamp,
            "reason": reason,
            "metadata": metadata or {},
            "components": self.current_state_components
        }
        
        try:
            # Save snapshot synchronously (snapshots are critical)
            with open(snapshot_file, 'w') as f:
                json.dump(snapshot_obj, f, indent=2)
            
            # Update snapshot registry
            self.snapshots.append({
                "id": snapshot_id,
                "timestamp": timestamp,
                "reason": reason,
                "file": str(snapshot_file)
            })
            
            # Maintain max snapshots limit
            if len(self.snapshots) > self.max_snapshots:
                # Remove oldest snapshot
                oldest = self.snapshots.pop(0)
                try:
                    os.remove(oldest["file"])
                    logger.debug(f"Removed old snapshot: {oldest['file']}")
                except:
                    pass
            
            # Save snapshot registry
            self._save_snapshot_registry()
            
            # Log the snapshot creation event if we have a tracker
            if self.adaptation_tracker is not None:
                self.adaptation_tracker.log_snapshot_creation(
                    snapshot_id=snapshot_id,
                    reason=reason,
                    metadata=metadata
                )
            
            logger.info(f"Created snapshot {snapshot_id} ({reason})")
            return snapshot_id
            
        except Exception as e:
            logger.error(f"Error creating snapshot: {str(e)}")
            return None
    
    def load_snapshot(
        self,
        snapshot_id: str,
        reason: str = "Rollback"
    ) -> Optional[Dict[str, Any]]:
        """Load a snapshot and restore state from it.
        
        Args:
            snapshot_id: ID of the snapshot to load
            reason: Reason for loading this snapshot
            
        Returns:
            Restored component states if successful, None if failed
        """
        # Find snapshot in registry
        snapshot_info = None
        for snapshot in self.snapshots:
            if snapshot["id"] == snapshot_id:
                snapshot_info = snapshot
                break
        
        if snapshot_info is None:
            logger.error(f"Snapshot {snapshot_id} not found")
            
            # Log the failed attempt if we have a tracker
            if self.adaptation_tracker is not None:
                self.adaptation_tracker.log_snapshot_loading(
                    snapshot_id=snapshot_id,
                    reason=reason,
                    success=False,
                    metadata={"error": "Snapshot not found"}
                )
            
            return None
        
        # Load snapshot file
        try:
            with open(snapshot_info["file"], 'r') as f:
                snapshot_data = json.load(f)
            
            # Extract component states
            component_states = snapshot_data.get("components", {})
            
            # Update current state
            self.current_state_components = copy.deepcopy(component_states)
            
            # Log the successful snapshot loading if we have a tracker
            if self.adaptation_tracker is not None:
                self.adaptation_tracker.log_snapshot_loading(
                    snapshot_id=snapshot_id,
                    reason=reason,
                    success=True,
                    metadata={
                        "original_timestamp": snapshot_info.get("timestamp"),
                        "original_reason": snapshot_info.get("reason")
                    }
                )
            
            logger.info(f"Loaded snapshot {snapshot_id} from {snapshot_info['file']}")
            
            return component_states
            
        except Exception as e:
            error_message = str(e)
            logger.error(f"Error loading snapshot: {error_message}")
            
            # Log the failed attempt if we have a tracker
            if self.adaptation_tracker is not None:
                self.adaptation_tracker.log_snapshot_loading(
                    snapshot_id=snapshot_id,
                    reason=reason,
                    success=False,
                    metadata={"error": error_message}
                )
            
            return None
    
    def list_snapshots(self) -> List[Dict[str, Any]]:
        """List all available snapshots.
        
        Returns:
            List of snapshot metadata
        """
        return copy.deepcopy(self.snapshots)
    
    def _load_snapshot_registry(self) -> None:
        """Load the snapshot registry from disk."""
        registry_path = self.storage_dir / "snapshot_registry.json"
        
        if registry_path.exists():
            try:
                with open(registry_path, 'r') as f:
                    self.snapshots = json.load(f)
                logger.debug(f"Loaded {len(self.snapshots)} snapshots from registry")
            except Exception as e:
                logger.error(f"Error loading snapshot registry: {str(e)}")
                self.snapshots = []
        else:
            self.snapshots = []
    
    def _save_snapshot_registry(self) -> None:
        """Save the snapshot registry to disk."""
        registry_path = self.storage_dir / "snapshot_registry.json"
        
        try:
            with open(registry_path, 'w') as f:
                json.dump(self.snapshots, f, indent=2)
        except Exception as e:
            logger.error(f"Error saving snapshot registry: {str(e)}")
    
    def _worker_loop(self) -> None:
        """Worker thread for non-blocking I/O operations."""
        while not self._stop_event.is_set():
            try:
                # Get task with timeout
                try:
                    task = self._task_queue.get(timeout=0.5)
                except queue.Empty:
                    continue
                
                # Process task
                task_type = task[0]
                
                if task_type == "save_state":
                    # Unpack task
                    _, state_obj, state_file = task
                    
                    try:
                        with open(state_file, 'w') as f:
                            json.dump(state_obj, f, indent=2)
                        logger.debug(f"Asynchronously saved state to {state_file}")
                    except Exception as e:
                        logger.error(f"Error in async state save: {str(e)}")
                
                # Mark task as done
                self._task_queue.task_done()
                
            except Exception as e:
                logger.error(f"Error in state manager worker: {str(e)}")
    
    def cleanup(self) -> None:
        """Cleanup resources before shutdown."""
        # Signal worker thread to stop
        self._stop_event.set()
        
        # Wait for pending tasks (with timeout)
        if self._task_queue.qsize() > 0:
            logger.debug(f"Waiting for {self._task_queue.qsize()} pending tasks...")
            try:
                self._task_queue.join()
                logger.debug("All pending tasks completed")
            except:
                logger.warning("Timeout waiting for tasks to complete")
        
        logger.debug("AdaptiveStateManager cleaned up")