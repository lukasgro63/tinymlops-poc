from pathlib import Path
from typing import Any, Dict, List, Optional, Union
import time

from tinylcm.core.drift_detector.base import DriftDetector
from tinylcm.utils.config import Config

class CompositeDriftDetector(DriftDetector):
    def __init__(
        self,
        storage_dir: Optional[Union[str, Path]] = None,
        detectors: Optional[List[DriftDetector]] = None,
        window_size: int = 100,
        threshold: float = 0.1,
        config: Optional[Config] = None
    ):
        super().__init__(
            storage_dir=storage_dir, 
            window_size=window_size,
            threshold=threshold,
            config=config
        )
        self.detectors = detectors or []
        self.logger.info(f"Initialized composite drift detector with {len(self.detectors)} sub-detectors")
    
    def create_reference_distribution(self, data: Dict[str, Any]) -> None:
        super().create_reference_distribution(data)
        for detector in self.detectors:
            detector.create_reference_distribution(data)
    
    def update(self, record: Dict[str, Any], auto_check: bool = True) -> bool:
        self.current_window.append(record)
        for detector in self.detectors:
            detector.current_window.append(record)
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
        for detector in self.detectors:
            detector.current_window.clear()
            for record in self.current_window:
                detector.current_window.append(record)
        detector_results = []
        for detector in self.detectors:
            detector_result = detector.check_for_drift()
            detector_results.append(detector_result)
        any_drift = bool(any(bool(result.get("drift_detected", False)) for result in detector_results))
        result = {
            "drift_detected": any_drift,
            "drift_type": "composite",
            "timestamp": time.time(),
            "session_id": self.session_id,
            "detector_results": detector_results,
            "threshold": self.threshold,
            "window_size": len(self.current_window)
        }
        if any_drift:
            self.logger.warning("Composite drift detected")
            result["snapshot"] = self._save_current_state()
            callback_data = result.copy()
            for callback in self.drift_callbacks:
                try:
                    callback(callback_data)
                except Exception as e:
                    self.logger.error(f"Error in drift callback: {e}")
        return result
    
    def reset(self) -> None:
        super().reset()
        for detector in self.detectors:
            detector.reset()
    
    def close(self) -> None:
        super().close()
        for detector in self.detectors:
            detector.close()