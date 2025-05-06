from dataclasses import dataclass, field
from typing import Dict, Any, List, Optional, Tuple, Union
import copy
import numpy as np

from tinylcm.utils.logging import setup_logger
from tinylcm.core.drift_detection.base import DriftDetector

logger = setup_logger(__name__)


@dataclass
class CUSUMState:
    """Represents the state of a CUSUM detector with sample tracking.
    
    This maintains the cumulative sum, minimum cumulative sum, number of samples processed,
    and the index at which drift was detected.
    """
    cumsum: float = 0.0
    min_cumsum: float = float('inf')
    min_cumsum_index: int = 0
    n_samples: int = 0
    drift_point_index: Optional[int] = None
    accuracy_history: List[float] = field(default_factory=list)
    cumsum_history: List[float] = field(default_factory=list)
    

class AccuracyCUSUM(DriftDetector):
    """CUSUM detector for monitoring classification accuracy.
    
    The Cumulative Sum (CUSUM) detector monitors the accuracy of a classifier
    over time and detects when the accuracy drops below a threshold, indicating
    potential concept drift.
    
    This implementation is optimized for resource-constrained devices and can
    operate with or without NumPy.
    
    Note: This is a label-based drift detector that requires ground truth labels
    to calculate accuracy. For autonomous drift detection without labels, see the
    modules confidence.py, distribution.py, and features.py.
    """
    
    def __init__(
        self,
        baseline_accuracy: float = 0.9,
        threshold: float = 5.0,
        drift_magnitude: float = 0.1,
        use_numpy: bool = True,
        min_samples_for_detection: int = 10  # Minimum samples before drift can be detected
    ):
        """Initialize the CUSUM detector.
        
        Args:
            baseline_accuracy: Expected accuracy of the classifier under normal conditions
            threshold: Detection threshold for the cumulative sum
            drift_magnitude: Expected magnitude of accuracy drop (delta)
            use_numpy: Whether to use NumPy for calculations (faster but uses more memory)
            min_samples_for_detection: Minimum number of samples before drift can be detected
        """
        self.baseline_accuracy = baseline_accuracy
        self.threshold = threshold
        self.drift_magnitude = drift_magnitude
        self.use_numpy = use_numpy
        self.min_samples_for_detection = min_samples_for_detection
        
        # Initialize internal state
        self.state = CUSUMState()
        self._last_detection_result = (False, None)
        
        # Register callbacks
        self.callbacks = []
        
        logger.debug(
            f"Initialized AccuracyCUSUM with baseline={baseline_accuracy}, "
            f"threshold={threshold}, drift_magnitude={drift_magnitude}"
        )
    
    def register_callback(self, callback):
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
                logger.error(f"Error in drift callback: {str(e)}")
    
    def update(self, record: Dict[str, Any]) -> Tuple[bool, Optional[Dict[str, Any]]]:
        """Update the CUSUM detector with a new accuracy observation.
        
        This implements the core CUSUM algorithm, which tracks the cumulative sum
        of deviations from the baseline accuracy. When this sum exceeds the threshold,
        drift is detected.
        
        Args:
            record: Dictionary containing 'accuracy' key with batch accuracy value
        
        Returns:
            Tuple of (drift_detected, drift_info)
        """
        # Extract accuracy from record
        if 'accuracy' not in record:
            logger.warning("Accuracy missing in record, skipping update")
            return False, None
        
        accuracy = record.get('accuracy')
        
        # Handle invalid accuracy values
        if accuracy is None or not (0 <= accuracy <= 1):
            logger.warning(f"Invalid accuracy value: {accuracy}, skipping update")
            return False, None
            
        # Update internal tracking
        self.state.accuracy_history.append(accuracy)
        self.state.n_samples += 1
        
        # Don't check for drift until we have enough samples
        if self.state.n_samples < self.min_samples_for_detection:
            return False, None
        
        # Calculate the deviation from baseline, adjusted by drift magnitude
        deviation = self.baseline_accuracy - accuracy - self.drift_magnitude
        
        # Update the cumulative sum
        self.state.cumsum = max(0, self.state.cumsum + deviation)
        self.state.cumsum_history.append(self.state.cumsum)
        
        # Track minimum cumsum and its position
        if self.state.cumsum < self.state.min_cumsum:
            self.state.min_cumsum = self.state.cumsum
            self.state.min_cumsum_index = self.state.n_samples - 1
        
        # Check for drift
        drift_detected = self.state.cumsum > self.threshold
        
        drift_info = None
        
        if drift_detected:
            # Calculate drift point - this is the point where the CUSUM first
            # started rising above the minimum, indicating the start of drift
            self.state.drift_point_index = self.state.min_cumsum_index
            
            drift_info = {
                'detector': 'AccuracyCUSUM',
                'detected_at_sample': self.state.drift_point_index,
                'current_accuracy': accuracy,
                'baseline_accuracy': self.baseline_accuracy,
                'cumsum': self.state.cumsum,
                'threshold': self.threshold,
                'drift_magnitude': self.drift_magnitude
            }
            
            logger.info(
                f"Drift detected at sample {self.state.drift_point_index} "
                f"(current_accuracy={accuracy:.4f}, cumsum={self.state.cumsum:.4f})"
            )
            
            # Notify callbacks
            self._notify_callbacks(drift_info)
            
            self._last_detection_result = (True, drift_info)
            return True, drift_info
        
        self._last_detection_result = (False, None)
        return False, None
    
    def check_for_drift(self) -> Tuple[bool, Optional[Dict[str, Any]]]:
        """Check if drift has been detected.
        
        For AccuracyCUSUM, this simply returns the most recent detection result.
        
        Returns:
            Tuple of (drift_detected, drift_info)
        """
        return self._last_detection_result
    
    def estimate_drift_timing(self, batch_size: int) -> int:
        """Estimate the exact sample index where drift likely began.
        
        This is useful for determining which samples to keep or discard when
        retraining a model after drift is detected.
        
        Args:
            batch_size: The size of the batches used when updating the detector
            
        Returns:
            Estimated index of the sample where drift began
        """
        if self.state.drift_point_index is None:
            # No drift detected yet
            return -1
        
        # Convert batch index to sample index
        # This is an estimate as we only have accuracy at batch level
        estimated_sample_index = self.state.drift_point_index * batch_size
        
        return estimated_sample_index
    
    def reset(self) -> None:
        """Reset the detector state after drift has been handled."""
        # Keep history for analysis but reset detection state
        self.state.cumsum = 0.0
        self.state.min_cumsum = float('inf')
        self.state.min_cumsum_index = 0
        self.state.drift_point_index = None
        self._last_detection_result = (False, None)
        
        logger.debug("CUSUM detector reset")
    
    def get_state(self) -> Dict[str, Any]:
        """Get the current state of the CUSUM detector.
        
        Returns:
            Dict containing the serializable state
        """
        state_dict = {
            "baseline_accuracy": self.baseline_accuracy,
            "threshold": self.threshold,
            "drift_magnitude": self.drift_magnitude,
            "use_numpy": self.use_numpy,
            "min_samples_for_detection": self.min_samples_for_detection,
            "cusum_state": {
                "cumsum": self.state.cumsum,
                "min_cumsum": self.state.min_cumsum,
                "min_cumsum_index": self.state.min_cumsum_index,
                "n_samples": self.state.n_samples,
                "drift_point_index": self.state.drift_point_index,
                
                # History arrays are converted to lists for serialization
                "accuracy_history": self.state.accuracy_history.copy(),
                "cumsum_history": self.state.cumsum_history.copy()
            }
        }
        
        return state_dict
    
    def set_state(self, state: Dict[str, Any]) -> None:
        """Restore the detector's state from a dictionary.
        
        Args:
            state: Previously saved state dictionary
        """
        self.baseline_accuracy = state.get("baseline_accuracy", self.baseline_accuracy)
        self.threshold = state.get("threshold", self.threshold)
        self.drift_magnitude = state.get("drift_magnitude", self.drift_magnitude)
        self.use_numpy = state.get("use_numpy", self.use_numpy)
        self.min_samples_for_detection = state.get("min_samples_for_detection", self.min_samples_for_detection)
        
        cusum_state = state.get("cusum_state", {})
        
        # Create a new state object
        self.state = CUSUMState(
            cumsum=cusum_state.get("cumsum", 0.0),
            min_cumsum=cusum_state.get("min_cumsum", float('inf')),
            min_cumsum_index=cusum_state.get("min_cumsum_index", 0),
            n_samples=cusum_state.get("n_samples", 0),
            drift_point_index=cusum_state.get("drift_point_index"),
            accuracy_history=cusum_state.get("accuracy_history", []).copy(),
            cumsum_history=cusum_state.get("cumsum_history", []).copy()
        )
        
        # Update detection result
        if self.state.drift_point_index is not None:
            drift_info = {
                'detector': 'AccuracyCUSUM',
                'detected_at_sample': self.state.drift_point_index,
                'cumsum': self.state.cumsum,
                'threshold': self.threshold,
                'drift_magnitude': self.drift_magnitude
            }
            self._last_detection_result = (True, drift_info)
        else:
            self._last_detection_result = (False, None)