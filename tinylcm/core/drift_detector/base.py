import json
import time
import uuid
from collections import deque
from pathlib import Path
from typing import Any, Callable, Dict, List, Optional, Union

from tinylcm.interfaces.monitoring import DriftDetector as IDriftDetector
from tinylcm.utils.config import Config, get_config
from tinylcm.utils.file_utils import ensure_dir, save_json
from tinylcm.utils.logging import setup_logger
from tinylcm.utils.file_utils import ensure_dir, load_json, save_json

class DriftDetector(IDriftDetector):
    def __init__(
        self,
        storage_dir: Optional[Union[str, Path]] = None,
        window_size: int = 100,
        threshold: float = 0.1,
        config: Optional[Config] = None
    ):
        self.config = config or get_config()
        component_config = self.config.get_component_config("drift_detector")
        
        self.logger = setup_logger(f"{__name__}.{self.__class__.__name__}")
        
        self.storage_dir = Path(storage_dir or component_config.get("storage_dir"))
        self.reference_dir = ensure_dir(self.storage_dir / "reference")
        self.snapshots_dir = ensure_dir(self.storage_dir / "snapshots")
        
        self.window_size = window_size
        self.threshold = threshold
        self.session_id = str(uuid.uuid4())
        
        self.current_window = deque(maxlen=window_size)
        self.reference_distribution = None
        self.reference_metadata = {}
        self.last_check_time = 0
        
        self.drift_callbacks: List[Callable[[Dict[str, Any]], None]] = []
        
        self.logger.info(f"Initialized drift detector with session ID: {self.session_id}")
        self.logger.info(f"Using window size: {window_size}, threshold: {threshold}")
    
    def register_drift_callback(self, callback: Callable[[Dict[str, Any]], None]) -> None:
        self.drift_callbacks.append(callback)
        self.logger.debug(f"Registered drift callback: {callback.__name__ if hasattr(callback, '__name__') else 'anonymous'}")
    
    def create_reference_distribution(self, data: Dict[str, Any]) -> None:
        self.reference_distribution = self._process_reference_data(data)
        self.reference_metadata = {
            "timestamp": time.time(),
            "session_id": self.session_id,
            "sample_count": self._get_sample_count(data),
            "created_at": time.time(),
            "window_size": self.window_size,
            "threshold": self.threshold
        }
        
        self._save_reference_distribution()
        
        self.logger.info(f"Created reference distribution from {self._get_sample_count(data)} samples")
    
    def _process_reference_data(self, data: Dict[str, Any]) -> Dict[str, Any]:
        return data
    
    def _get_sample_count(self, data: Dict[str, Any]) -> int:
        for key in ["predictions", "confidences", "features"]:
            if key in data and isinstance(data[key], list):
                return len(data[key])
        
        return 0
    
    def _save_reference_distribution(self) -> str:
        if self.reference_distribution is None:
            self.logger.warning("Attempted to save None reference distribution")
            return ""
        
        timestamp = int(time.time())
        file_path = self.reference_dir / f"reference_{timestamp}_{self.session_id}.json"
        
        data = {
            "distribution": self.reference_distribution,
            "metadata": self.reference_metadata
        }
        
        save_json(data, file_path)
        self.logger.debug(f"Saved reference distribution to {file_path}")
        
        return str(file_path)
    
    def _load_reference_distribution(self, file_path: Optional[Union[str, Path]] = None) -> bool:
        if file_path is None:
            reference_files = list(self.reference_dir.glob("reference_*.json"))
            if not reference_files:
                self.logger.warning("No reference files found")
                return False
            
            reference_files.sort(reverse=True)
            file_path = reference_files[0]
        
        try:
            data = load_json(file_path)
            self.reference_distribution = data.get("distribution")
            self.reference_metadata = data.get("metadata", {})
            
            self.logger.info(f"Loaded reference distribution from {file_path}")
            return True
        except Exception as e:
            self.logger.error(f"Failed to load reference distribution: {e}")
            return False
    
    def update(self, record: Dict[str, Any], auto_check: bool = True) -> bool:
        self.current_window.append(record)
        
        if self.reference_distribution is None:
            return False
        
        if len(self.current_window) < self.window_size:
            return False
        
        if auto_check:
            current_time = time.time()
            check_interval = self.config.get("drift_detector", "check_interval", 60.0)
            
            if current_time - self.last_check_time > check_interval:
                drift_result = self.check_for_drift()
                self.last_check_time = current_time
                return bool(drift_result.get("drift_detected", False))
        
        return False
    
    def check_for_drift(self) -> Dict[str, Any]:
        if self.reference_distribution is None:
            self.logger.warning("Cannot check for drift: No reference distribution")
            return {"drift_detected": False, "error": "No reference distribution"}
        
        if len(self.current_window) < self.window_size / 2:
            self.logger.warning(f"Not enough data to check for drift: {len(self.current_window)}/{self.window_size}")
            return {
                "drift_detected": False, 
                "error": f"Not enough data: {len(self.current_window)}/{self.window_size}"
            }
        
        drift_result = self._calculate_drift()
        
        drift_detected = bool(drift_result.get("drift_detected", False))
        drift_result["drift_detected"] = drift_detected
        
        if drift_detected:
            self.logger.warning(f"Drift detected: {drift_result.get('drift_type', 'unknown')}")
            
            drift_result["snapshot"] = self._save_current_state()
            
            callback_data = drift_result.copy()
            
            for callback in self.drift_callbacks:
                try:
                    callback(callback_data)
                except Exception as e:
                    self.logger.error(f"Error in drift callback: {e}")
        
        return drift_result
    
    def _calculate_drift(self) -> Dict[str, Any]:
        return {
            "drift_detected": False,
            "timestamp": time.time(),
            "session_id": self.session_id,
            "window_size": len(self.current_window),
            "threshold": self.threshold
        }
    
    def reset(self) -> None:
        self.current_window.clear()
        self.last_check_time = 0
        self.logger.info("Reset drift detector state")
    
    def export_current_state(self) -> str:
        return self._save_current_state()
    
    def _save_current_state(self) -> str:
        timestamp = int(time.time())
        file_path = self.snapshots_dir / f"snapshot_{timestamp}_{self.session_id}.json"
        
        state = {
            "timestamp": timestamp,
            "session_id": self.session_id,
            "window_size": self.window_size,
            "threshold": self.threshold,
            "current_window": list(self.current_window)
        }
        
        save_json(state, file_path)
        self.logger.debug(f"Saved current state to {file_path}")
        
        return str(file_path)
    
    def close(self) -> None:
        self.logger.info(f"Closing drift detector session: {self.session_id}")