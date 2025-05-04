from abc import ABC, abstractmethod
from typing import Any, Dict, Optional, List, Callable, Tuple, Union

class DriftDetector(ABC):
    """Base abstract class for drift detectors.
    
    This defines a common interface for both label-based detectors (like AccuracyCUSUM)
    and autonomous detectors that monitor features or model properties without requiring
    ground truth labels.
    """
    
    @abstractmethod
    def update(self, record: Any) -> Tuple[bool, Optional[Any]]:
        """Update the detector with new data.
        
        Args:
            record: New data point. For label-based detectors, this will be accuracy.
                   For autonomous detectors, this could be a feature vector, confidence score,
                   prediction distribution, or any other relevant data.
        
        Returns:
            Tuple of (drift_detected, drift_point) where drift_point contains
            additional information about the detected drift (type depends on the detector).
        """
        pass
    
    @abstractmethod
    def check_for_drift(self) -> Tuple[bool, Optional[Any]]:
        """Check if drift has been detected.
        
        This is mainly used for batch detectors that accumulate data for a while
        before performing drift detection. For streaming detectors, this might
        simply return the latest detection result from update().
        
        Returns:
            Tuple of (drift_detected, drift_point)
        """
        pass
    
    @abstractmethod
    def reset(self) -> None:
        """Reset the detector state after drift has been handled."""
        pass
    
    @abstractmethod
    def get_state(self) -> Dict[str, Any]:
        """Get the current state of the detector as a serializable dictionary.
        
        Returns:
            State dictionary
        """
        pass
    
    @abstractmethod
    def set_state(self, state: Dict[str, Any]) -> None:
        """Restore the detector state from a dictionary.
        
        Args:
            state: Previously saved state dictionary
        """
        pass


class AutonomousDriftDetector(DriftDetector):
    """Base class for autonomous drift detectors that don't require labels.
    
    This class extends the basic DriftDetector interface with specific methods
    for autonomous drift detection and callback registration.
    """
    
    def __init__(self):
        """Initialize the autonomous drift detector."""
        self.callbacks = []
    
    def register_callback(self, callback: Callable[[Dict[str, Any]], None]) -> None:
        """Register a callback function to be called when drift is detected.
        
        Args:
            callback: Function to call when drift is detected. It receives a dictionary
                     with information about the detected drift.
        """
        self.callbacks.append(callback)
    
    def _notify_callbacks(self, drift_info: Dict[str, Any]) -> None:
        """Notify all registered callbacks about detected drift.
        
        Args:
            drift_info: Dictionary with information about the detected drift
        """
        for callback in self.callbacks:
            try:
                callback(drift_info)
            except Exception as e:
                # Log the error but don't propagate it to avoid interrupting the detection process
                from tinylcm.utils.logging import setup_logger
                logger = setup_logger(__name__)
                logger.error(f"Error in drift callback: {str(e)}")