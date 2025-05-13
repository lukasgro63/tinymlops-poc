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
    for autonomous drift detection, callback registration, and reference statistics
    management. It implements the warm-up, rolling update, and pausing logic for
    the reference statistics.
    """
    
    def __init__(
        self,
        warm_up_samples: int = 100,
        reference_update_interval: int = 50,
        reference_update_factor: float = 0.05,
        pause_reference_update_during_drift: bool = True,
        drift_cooldown_period: int = 20,  # Reduced to make the system more responsive to new drift events
    ):
        """Initialize the autonomous drift detector.
        
        Args:
            warm_up_samples: Number of samples to collect during the warm-up phase
            reference_update_interval: Number of samples between reference updates
            reference_update_factor: Factor for updating reference (β)
            pause_reference_update_during_drift: Whether to pause updating during detected drift
            drift_cooldown_period: Number of samples to wait before triggering another drift event
        """
        self.warm_up_samples = warm_up_samples
        self.reference_update_interval = reference_update_interval
        self.reference_update_factor = reference_update_factor
        self.pause_reference_update_during_drift = pause_reference_update_during_drift
        self.drift_cooldown_period = drift_cooldown_period
        
        # State flags
        self.in_warm_up_phase = True
        self.samples_processed = 0
        self.drift_detected = False
        self.samples_since_last_update = 0
        
        # Drift cooldown tracking
        self.samples_since_last_drift = 0
        self.in_cooldown_period = False
        self.last_drift_timestamp = 0
        
        # Callback registration
        self.callbacks = []
    
    def register_callback(self, callback: Callable[[Dict[str, Any]], None]) -> None:
        """Register a callback function to be called when drift is detected.
        
        Args:
            callback: Function to call when drift is detected. It receives a dictionary
                    with information about the detected drift.
        """
        self.callbacks.append(callback)
    
    def _notify_callbacks(self, drift_info: Dict[str, Any]) -> bool:
        """Notify all registered callbacks about detected drift.

        Args:
            drift_info: Dictionary with information about the detected drift

        Returns:
            bool: True if callbacks were notified, False if in cooldown period
        """
        # Import logger here to avoid circular imports
        from tinylcm.utils.logging import setup_logger
        logger = setup_logger(__name__)

        # Add timestamp to drift info if not present
        if "timestamp" not in drift_info:
            import time
            drift_info["timestamp"] = time.time()

        # Check if we're in the cooldown period
        if self.in_cooldown_period and self.samples_since_last_drift < self.drift_cooldown_period:
            logger.debug(f"In drift cooldown period ({self.samples_since_last_drift}/{self.drift_cooldown_period} samples since last drift), skipping callback notifications")

            # Add cooldown info to the drift_info for downstream consumers
            drift_info["in_cooldown_period"] = True
            drift_info["samples_since_last_drift"] = self.samples_since_last_drift
            drift_info["drift_cooldown_period"] = self.drift_cooldown_period
            return False

        # Reset cooldown tracking
        self.in_cooldown_period = True
        self.samples_since_last_drift = 0
        self.last_drift_timestamp = drift_info.get("timestamp", 0) or 0

        # Log drift detection
        detector_name = drift_info.get('detector_type', type(self).__name__)
        metric_info = ""
        if "metric" in drift_info:
            current = drift_info.get("current_value", "unknown")
            threshold = drift_info.get("threshold", "unknown")
            metric_info = f" - {drift_info['metric']} (current: {current}, threshold: {threshold})"

        logger.info(f"Drift detector {detector_name} detected drift{metric_info}, callbacks will be notified")

        # Execute callbacks
        callbacks_executed = 0
        for callback in self.callbacks:
            try:
                # Try different calling conventions to be flexible with callback signatures
                try:
                    # First try with just drift_info
                    callback(drift_info)
                except TypeError:
                    try:
                        # Then try with drift_info and an empty dict
                        callback(drift_info, {})
                    except TypeError:
                        # Finally try with just drift_info wrapped in *args
                        callback(*(drift_info,))
                
                callbacks_executed += 1
            except Exception as e:
                # Log the error but don't propagate it to avoid interrupting the detection process
                logger.error(f"Error in drift callback: {str(e)}")

        logger.info(f"Notified {callbacks_executed} callbacks about drift event")
        return True
    
    def should_update_reference(self) -> bool:
        """Determine if reference statistics should be updated.
        
        Returns:
            True if reference should be updated, False otherwise
        """
        # Don't update during warm-up phase
        if self.in_warm_up_phase:
            return False
        
        # If configured to pause during drift and drift is detected, don't update
        if self.pause_reference_update_during_drift and self.drift_detected:
            return False
        
        # Check if it's time for a periodic update
        return self.samples_since_last_update >= self.reference_update_interval
    
    def _update_reference_statistics(self, new_stats: Any, ref_stats: Any) -> Any:
        """Update reference statistics using rolling update formula.
        
        Uses the formula: μ_ref_t ← β·μ_ref_{t-1} + (1−β)·x̄_batch
        
        Args:
            new_stats: New statistics from recent batch
            ref_stats: Current reference statistics
            
        Returns:
            Updated reference statistics
        """
        # This method should be implemented by subclasses based on their specific statistics
        raise NotImplementedError("Subclasses must implement _update_reference_statistics")
    
    def _process_sample(self, record: Any) -> None:
        """Process a new sample and update internal state.
        
        This method should be called at the beginning of the update method in subclasses
        to handle the common logic for warm-up phase and reference updates.
        
        Args:
            record: The new data point
        """
        # Just increment counters - don't try to process the record here
        # That should be handled by the specific drift detector classes
        self.samples_processed += 1
        self.samples_since_last_update += 1
        
        # Increment cooldown counter if in cooldown period
        if self.in_cooldown_period:
            self.samples_since_last_drift += 1
            
            # If we've passed the cooldown period, exit cooldown mode
            if self.samples_since_last_drift >= self.drift_cooldown_period:
                self.in_cooldown_period = False
        
        # Handle warm-up phase
        if self.in_warm_up_phase and self.samples_processed >= self.warm_up_samples:
            self.in_warm_up_phase = False
            from tinylcm.utils.logging import setup_logger
            logger = setup_logger(__name__)
            logger.debug(f"Warm-up phase completed after {self.samples_processed} samples")
    
    def _get_base_state(self) -> Dict[str, Any]:
        """Get the base state for the autonomous drift detector.
        
        Returns:
            Dictionary with base state information
        """
        return {
            "warm_up_samples": self.warm_up_samples,
            "reference_update_interval": self.reference_update_interval,
            "reference_update_factor": self.reference_update_factor,
            "pause_reference_update_during_drift": self.pause_reference_update_during_drift,
            "drift_cooldown_period": self.drift_cooldown_period,
            "in_warm_up_phase": self.in_warm_up_phase,
            "samples_processed": self.samples_processed,
            "drift_detected": self.drift_detected,
            "samples_since_last_update": self.samples_since_last_update,
            "in_cooldown_period": self.in_cooldown_period,
            "samples_since_last_drift": self.samples_since_last_drift,
            "last_drift_timestamp": self.last_drift_timestamp
        }
    
    def _set_base_state(self, state: Dict[str, Any]) -> None:
        """Set the base state for the autonomous drift detector.
        
        Args:
            state: Dictionary with base state information
        """
        self.warm_up_samples = state.get("warm_up_samples", self.warm_up_samples)
        self.reference_update_interval = state.get("reference_update_interval", self.reference_update_interval)
        self.reference_update_factor = state.get("reference_update_factor", self.reference_update_factor)
        self.pause_reference_update_during_drift = state.get(
            "pause_reference_update_during_drift", self.pause_reference_update_during_drift
        )
        self.drift_cooldown_period = state.get("drift_cooldown_period", self.drift_cooldown_period)
        self.in_warm_up_phase = state.get("in_warm_up_phase", self.in_warm_up_phase)
        self.samples_processed = state.get("samples_processed", self.samples_processed)
        self.drift_detected = state.get("drift_detected", self.drift_detected)
        self.samples_since_last_update = state.get("samples_since_last_update", self.samples_since_last_update)
        self.in_cooldown_period = state.get("in_cooldown_period", self.in_cooldown_period)
        self.samples_since_last_drift = state.get("samples_since_last_drift", self.samples_since_last_drift)
        self.last_drift_timestamp = state.get("last_drift_timestamp", self.last_drift_timestamp)