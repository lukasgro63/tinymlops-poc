from dataclasses import dataclass, field
from typing import Dict, Any, List, Optional, Tuple, Union, Callable, Deque
import time
import math
from collections import deque
import numpy as np

from tinylcm.utils.logging import setup_logger
from tinylcm.core.drift_detection.base import AutonomousDriftDetector

logger = setup_logger(__name__)


@dataclass
class EWMAConfidenceState:
    """State class for Exponentially Weighted Moving Average confidence monitoring."""
    # Sample history
    confidence_history: List[float] = field(default_factory=list)
    
    # EWMA statistics
    ewma_values: List[float] = field(default_factory=list)
    current_ewma: float = 0.0
    
    # Control limits
    ucl_values: List[float] = field(default_factory=list)  # Upper control limit
    lcl_values: List[float] = field(default_factory=list)  # Lower control limit
    current_ucl: float = 0.0
    current_lcl: float = 0.0
    
    # Reference statistics
    reference_mean: float = 0.0
    reference_std: float = 0.0
    
    # Drift detection
    drift_detected_at: Optional[int] = None
    last_update_time: float = field(default_factory=time.time)


class EWMAConfidenceMonitor(AutonomousDriftDetector):
    """Autonomous drift detector based on prediction confidence using EWMA.
    
    This detector uses Exponentially Weighted Moving Average (EWMA) to track model
    confidence scores over time, detecting drift when confidence falls outside the
    control limits for a significant period.
    
    It implements the following calibration and adaptation mechanisms:
    1. Warm-Up Phase: Collects initial samples to establish baseline statistics
    2. Rolling Update: Gradually updates reference statistics using the formula
       μ_ref_t ← β·μ_ref_{t-1} + (1−β)·x̄_batch
    3. Pause during drift: Optionally pauses reference updates when drift is detected
    
    Unlike AccuracyCUSUM, this detector doesn't require ground truth labels, making
    it suitable for ongoing monitoring in production where labels aren't available.
    """
    
    def __init__(
        self,
        lambda_param: float = 0.1,
        threshold_factor: float = 3.0,
        drift_window: int = 5,
        min_confidence: float = 0.2,
        warm_up_samples: int = 30,
        reference_update_interval: int = 50,
        reference_update_factor: float = 0.05,
        pause_reference_update_during_drift: bool = True,
        use_numpy: bool = True
    ):
        """Initialize the EWMA confidence monitor.
        
        Args:
            lambda_param: Smoothing factor (0-1), controls how much weight is given to recent vs. old data.
                Lower values are more stable but less responsive to recent changes.
            threshold_factor: Factor multiplied by standard deviation to set control limits.
                Higher values make the detector less sensitive.
            drift_window: Number of consecutive samples outside control limits to trigger drift detection.
            min_confidence: Minimum confidence level to consider (values below this are clamped).
            warm_up_samples: Number of initial samples to use for establishing baseline.
            reference_update_interval: Number of samples between reference statistic updates.
            reference_update_factor: Factor (β) for the reference rolling update formula.
            pause_reference_update_during_drift: Whether to pause reference updates during detected drift.
            use_numpy: Whether to use NumPy for calculations (faster but uses more memory).
        """
        super().__init__(
            warm_up_samples=warm_up_samples,
            reference_update_interval=reference_update_interval,
            reference_update_factor=reference_update_factor,
            pause_reference_update_during_drift=pause_reference_update_during_drift
        )
        
        self.lambda_param = lambda_param
        self.threshold_factor = threshold_factor
        self.drift_window = drift_window
        self.min_confidence = min_confidence
        self.use_numpy = use_numpy
        
        # Initialize state
        self.state = EWMAConfidenceState()
        
        # Use deque with maxlen for efficient sliding window
        self.recent_violations = deque(maxlen=drift_window)
        
        logger.debug(
            f"Initialized EWMAConfidenceMonitor with lambda={lambda_param}, "
            f"threshold_factor={threshold_factor}, drift_window={drift_window}, "
            f"warm_up_samples={warm_up_samples}"
        )
    
    def update(self, record: Dict[str, Any]) -> Tuple[bool, Optional[Dict[str, Any]]]:
        """Update the detector with a new confidence score.
        
        Args:
            record: Dictionary containing the inference data. Should include a 'confidence'
                   key with the model's confidence score for the prediction.
                   
        Returns:
            Tuple of (drift_detected, drift_info)
        """
        # Process sample using common base class logic
        self._process_sample(record)
        
        # Extract confidence from record
        if 'confidence' not in record:
            # No confidence information - can't update
            logger.warning("Confidence value missing in record, skipping update")
            return False, None
        
        confidence = record['confidence']
        
        # Handle invalid confidence values
        if confidence is None or not isinstance(confidence, (int, float)):
            logger.warning(f"Invalid confidence value: {confidence}, skipping update")
            return False, None
            
        # Apply minimum confidence floor
        confidence = max(confidence, self.min_confidence)
        
        # Update internal tracking
        self.state.confidence_history.append(confidence)
        self.state.last_update_time = time.time()
        
        # If in warm-up phase, just collect samples without checking for drift
        if self.in_warm_up_phase:
            # Initialize EWMA statistics if warm-up phase is complete
            if not self.in_warm_up_phase:  # This could change after _process_sample
                self._initialize_ewma()
            return False, None
        
        # Update EWMA
        prev_ewma = self.state.current_ewma
        self.state.current_ewma = self._calculate_ewma(confidence, prev_ewma)
        self.state.ewma_values.append(self.state.current_ewma)
        
        # Update reference statistics if needed
        if self.should_update_reference():
            # Calculate batch statistics for update
            recent_window = self.state.confidence_history[-self.reference_update_interval:]
            if self.use_numpy and np:
                batch_mean = float(np.mean(recent_window))
                batch_std = float(np.std(recent_window))
            else:
                batch_mean = sum(recent_window) / len(recent_window)
                sum_squared_diff = sum((x - batch_mean) ** 2 for x in recent_window)
                batch_std = math.sqrt(sum_squared_diff / len(recent_window))
            
            # Update reference statistics
            self.state.reference_mean = self._update_reference_statistic(
                batch_mean, self.state.reference_mean
            )
            self.state.reference_std = self._update_reference_statistic(
                batch_std, self.state.reference_std
            )
            
            # Update control limits
            self._update_control_limits()
            self.samples_since_last_update = 0
            
            logger.debug(
                f"Updated reference statistics: mean={self.state.reference_mean:.4f}, "
                f"std={self.state.reference_std:.4f}"
            )
        
        # Check if current EWMA is outside control limits
        is_violation = (self.state.current_ewma > self.state.current_ucl or 
                       self.state.current_ewma < self.state.current_lcl)
        
        # Update recent_violations with 1 for violation, 0 for normal
        self.recent_violations.append(1 if is_violation else 0)
        
        # Check for drift condition (consecutive violations)
        consecutive_violations = sum(self.recent_violations)
        self.drift_detected = consecutive_violations >= self.drift_window
        
        # If this is the first time drift is detected
        if self.drift_detected and self.state.drift_detected_at is None:
            # Record when drift was first detected
            self.state.drift_detected_at = self.samples_processed
            
            # Create drift info
            drift_info = {
                "detector": "EWMAConfidenceMonitor",
                "detected_at_sample": self.state.drift_detected_at,
                "timestamp": time.time(),
                "metric": "confidence",
                "current_value": confidence,
                "ewma_value": self.state.current_ewma,
                "upper_limit": self.state.current_ucl,
                "lower_limit": self.state.current_lcl,
                "reference_mean": self.state.reference_mean,
                "reference_std": self.state.reference_std,
                "consecutive_violations": consecutive_violations
            }
            
            logger.info(
                f"Confidence drift detected at sample {self.state.drift_detected_at} "
                f"(current_confidence={confidence:.4f}, ewma={self.state.current_ewma:.4f})"
            )
            
            # Notify callbacks
            self._notify_callbacks(drift_info)
            
            return True, drift_info
        
        # Update control limit history for plotting/analysis
        self.state.ucl_values.append(self.state.current_ucl)
        self.state.lcl_values.append(self.state.current_lcl)
        
        return False, None
    
    def check_for_drift(self) -> Tuple[bool, Optional[Dict[str, Any]]]:
        """Check if drift has been detected.
        
        Returns:
            Tuple of (drift_detected, drift_info)
        """
        if self.drift_detected and self.state.drift_detected_at is not None:
            drift_info = {
                "detector": "EWMAConfidenceMonitor",
                "detected_at_sample": self.state.drift_detected_at,
                "timestamp": self.state.last_update_time,
                "metric": "confidence",
                "ewma_value": self.state.current_ewma,
                "upper_limit": self.state.current_ucl,
                "lower_limit": self.state.current_lcl,
                "reference_mean": self.state.reference_mean,
                "reference_std": self.state.reference_std,
                "consecutive_violations": sum(self.recent_violations) if self.recent_violations else 0
            }
            return True, drift_info
        
        return False, None
    
    def reset(self) -> None:
        """Reset the detector state after drift has been handled."""
        # Clear drift detection state but keep the statistics
        self.state.drift_detected_at = None
        self.drift_detected = False
        self.recent_violations.clear()

        # Reset drift cooldown tracking
        self.in_cooldown_period = False
        self.samples_since_last_drift = 0

        logger.debug("EWMAConfidenceMonitor reset")
    
    def _initialize_ewma(self) -> None:
        """Initialize EWMA statistics from collected training samples during warm-up."""
        if self.use_numpy and np:
            # Calculate initial EWMA using mean of training samples
            initial_samples = np.array(self.state.confidence_history)
            self.state.reference_mean = float(np.mean(initial_samples))
            self.state.current_ewma = self.state.reference_mean
            
            # Calculate standard deviation for control limits
            self.state.reference_std = float(np.std(initial_samples))
        else:
            # Pure Python implementation for devices without NumPy
            initial_samples = self.state.confidence_history
            mean = sum(initial_samples) / len(initial_samples)
            self.state.reference_mean = mean
            self.state.current_ewma = mean
            
            # Calculate standard deviation
            sum_squared_diff = sum((x - mean) ** 2 for x in initial_samples)
            self.state.reference_std = math.sqrt(sum_squared_diff / len(initial_samples))
        
        # Initialize control limits
        self._update_control_limits()
        
        # Add initial values to history
        self.state.ewma_values.append(self.state.current_ewma)
        self.state.ucl_values.append(self.state.current_ucl)
        self.state.lcl_values.append(self.state.current_lcl)
        
        logger.debug(
            f"Initialized EWMA with mean={self.state.reference_mean:.4f}, "
            f"std={self.state.reference_std:.4f}, UCL={self.state.current_ucl:.4f}, "
            f"LCL={self.state.current_lcl:.4f}"
        )
    
    def _calculate_ewma(self, new_value: float, prev_ewma: float) -> float:
        """Calculate new EWMA value based on new observation and previous EWMA."""
        return self.lambda_param * new_value + (1 - self.lambda_param) * prev_ewma
    
    def _update_control_limits(self) -> None:
        """Update EWMA control limits based on reference statistics."""
        # Control limit width
        limit_width = self.threshold_factor * self.state.reference_std * math.sqrt(
            self.lambda_param / (2 - self.lambda_param)
        )
        
        # Upper and lower control limits
        self.state.current_ucl = self.state.reference_mean + limit_width
        self.state.current_lcl = max(self.state.reference_mean - limit_width, 0)  # Confidence can't be negative
    
    def _update_reference_statistic(self, new_value: float, ref_value: float) -> float:
        """Update a reference statistic using the rolling update formula."""
        return (self.reference_update_factor * ref_value + 
                (1 - self.reference_update_factor) * new_value)
    
    def get_state(self) -> Dict[str, Any]:
        """Get the current state of the detector as a serializable dictionary.
        
        Returns:
            State dictionary
        """
        # Get base state from parent class
        state_dict = self._get_base_state()
        
        # Add detector-specific parameters
        state_dict.update({
            "lambda_param": self.lambda_param,
            "threshold_factor": self.threshold_factor,
            "drift_window": self.drift_window,
            "min_confidence": self.min_confidence,
            "use_numpy": self.use_numpy,
        })
        
        # Convert deque to list for serialization
        recent_violations_list = list(self.recent_violations)
        
        # Add detector-specific state
        state_dict["ewma_state"] = {
            "confidence_history": self.state.confidence_history.copy(),
            "ewma_values": self.state.ewma_values.copy(),
            "ucl_values": self.state.ucl_values.copy(),
            "lcl_values": self.state.lcl_values.copy(),
            "current_ewma": self.state.current_ewma,
            "current_ucl": self.state.current_ucl,
            "current_lcl": self.state.current_lcl,
            "reference_mean": self.state.reference_mean,
            "reference_std": self.state.reference_std,
            "drift_detected_at": self.state.drift_detected_at,
            "last_update_time": self.state.last_update_time
        }
        state_dict["recent_violations"] = recent_violations_list
        
        return state_dict
    
    def set_state(self, state: Dict[str, Any]) -> None:
        """Restore the detector state from a dictionary.
        
        Args:
            state: Previously saved state dictionary
        """
        # Restore base state using parent class
        self._set_base_state(state)
        
        # Restore detector-specific parameters
        self.lambda_param = state.get("lambda_param", self.lambda_param)
        self.threshold_factor = state.get("threshold_factor", self.threshold_factor)
        self.drift_window = state.get("drift_window", self.drift_window)
        self.min_confidence = state.get("min_confidence", self.min_confidence)
        self.use_numpy = state.get("use_numpy", self.use_numpy)
        
        # Restore detector-specific state
        ewma_state = state.get("ewma_state", {})
        
        # Create new state object
        self.state = EWMAConfidenceState(
            confidence_history=ewma_state.get("confidence_history", []).copy(),
            ewma_values=ewma_state.get("ewma_values", []).copy(),
            ucl_values=ewma_state.get("ucl_values", []).copy(),
            lcl_values=ewma_state.get("lcl_values", []).copy(),
            current_ewma=ewma_state.get("current_ewma", 0.0),
            current_ucl=ewma_state.get("current_ucl", 0.0),
            current_lcl=ewma_state.get("current_lcl", 0.0),
            reference_mean=ewma_state.get("reference_mean", 0.0),
            reference_std=ewma_state.get("reference_std", 0.0),
            drift_detected_at=ewma_state.get("drift_detected_at"),
            last_update_time=ewma_state.get("last_update_time", time.time())
        )
        
        # Restore recent violations as deque
        recent_violations = state.get("recent_violations", [])
        self.recent_violations = deque(recent_violations, maxlen=self.drift_window)


@dataclass
class PageHinkleyState:
    """State class for Page-Hinkley test monitoring."""
    # Sample history
    confidence_history: List[float] = field(default_factory=list)
    
    # Reference statistics
    reference_mean: float = 0.0
    
    # Page-Hinkley statistics
    sum_pos: float = 0.0  # Cumulative sum for detecting positive shifts
    sum_neg: float = 0.0  # Cumulative sum for detecting negative shifts
    sum_pos_history: List[float] = field(default_factory=list)
    sum_neg_history: List[float] = field(default_factory=list)
    reference_mean_history: List[float] = field(default_factory=list)
    
    # Drift detection
    drift_detected_at: Optional[int] = None
    drift_type: Optional[str] = None  # 'increase' or 'decrease'
    last_update_time: float = field(default_factory=time.time)


class PageHinkleyConfidenceMonitor(AutonomousDriftDetector):
    """Autonomous drift detector based on prediction confidence using Page-Hinkley Test.
    
    This detector applies the Page-Hinkley test to model confidence scores, detecting drift
    when there is a significant and sustained change in the confidence level. The test is
    particularly effective at detecting sudden shifts in the probability distribution.
    
    It implements the following calibration and adaptation mechanisms:
    1. Warm-Up Phase: Collects initial samples to establish baseline statistics
    2. Rolling Update: Gradually updates reference statistics using the formula
       μ_ref_t ← β·μ_ref_{t-1} + (1−β)·x̄_batch
    3. Pause during drift: Optionally pauses reference updates when drift is detected
    
    Like the EWMA detector, this doesn't require ground truth labels, making it suitable
    for production monitoring.
    """
    
    def __init__(
        self,
        delta: float = 0.05,
        lambda_param: float = 10.0,
        alpha: float = 0.9999,
        min_confidence: float = 0.2,
        warm_up_samples: int = 30,
        reference_update_interval: int = 50,
        reference_update_factor: float = 0.05,
        pause_reference_update_during_drift: bool = True,
        use_numpy: bool = True
    ):
        """Initialize the Page-Hinkley confidence monitor.
        
        Args:
            delta: Magnitude of allowed changes (smaller values are more sensitive)
            lambda_param: Detection threshold (higher values reduce false positives)
            alpha: Forgetting factor for internal EWMA calculation
            min_confidence: Minimum confidence value to consider
            warm_up_samples: Number of initial samples to use for establishing baseline
            reference_update_interval: Number of samples between reference statistic updates
            reference_update_factor: Factor (β) for the reference rolling update formula
            pause_reference_update_during_drift: Whether to pause reference updates during detected drift
            use_numpy: Whether to use NumPy for calculations
        """
        super().__init__(
            warm_up_samples=warm_up_samples,
            reference_update_interval=reference_update_interval,
            reference_update_factor=reference_update_factor,
            pause_reference_update_during_drift=pause_reference_update_during_drift
        )
        
        self.delta = delta
        self.lambda_param = lambda_param
        self.alpha = alpha
        self.min_confidence = min_confidence
        self.use_numpy = use_numpy
        
        # Initialize state
        self.state = PageHinkleyState()
        
        logger.debug(
            f"Initialized PageHinkleyConfidenceMonitor with delta={delta}, "
            f"lambda_param={lambda_param}, alpha={alpha}, warm_up_samples={warm_up_samples}"
        )
    
    def update(self, record: Dict[str, Any]) -> Tuple[bool, Optional[Dict[str, Any]]]:
        """Update the detector with a new confidence score.
        
        Args:
            record: Dictionary containing the inference data. Should include a 'confidence'
                   key with the model's confidence score for the prediction.
                   
        Returns:
            Tuple of (drift_detected, drift_info)
        """
        # Process sample using common base class logic
        self._process_sample(record)
        
        # Extract confidence from record
        if 'confidence' not in record:
            # No confidence information - can't update
            logger.warning("Confidence value missing in record, skipping update")
            return False, None
        
        confidence = record['confidence']
        
        # Handle invalid confidence values
        if confidence is None or not isinstance(confidence, (int, float)):
            logger.warning(f"Invalid confidence value: {confidence}, skipping update")
            return False, None
            
        # Apply minimum confidence floor
        confidence = max(confidence, self.min_confidence)
        
        # Update internal tracking
        self.state.confidence_history.append(confidence)
        self.state.last_update_time = time.time()
        
        # If in warm-up phase, just collect samples without checking for drift
        if self.in_warm_up_phase:
            # Initialize statistics if warm-up phase is complete
            if not self.in_warm_up_phase:  # This could change after _process_sample
                self._initialize_reference()
            return False, None
        
        # Update reference statistics if needed
        if self.should_update_reference():
            # Calculate batch mean for update
            recent_window = self.state.confidence_history[-self.reference_update_interval:]
            if self.use_numpy and np:
                batch_mean = float(np.mean(recent_window))
            else:
                batch_mean = sum(recent_window) / len(recent_window)
            
            # Update reference mean
            self.state.reference_mean = self._update_reference_statistic(
                batch_mean, self.state.reference_mean
            )
            self.samples_since_last_update = 0
            
            logger.debug(f"Updated reference mean: {self.state.reference_mean:.4f}")
        
        # Calculate deviation from reference mean
        deviation_pos = confidence - (self.state.reference_mean + self.delta)
        deviation_neg = (self.state.reference_mean - self.delta) - confidence
        
        # Update cumulative sums using Page-Hinkley formula
        self.state.sum_pos = max(0, self.state.sum_pos + deviation_pos)
        self.state.sum_neg = max(0, self.state.sum_neg + deviation_neg)
        
        # Record history
        self.state.sum_pos_history.append(self.state.sum_pos)
        self.state.sum_neg_history.append(self.state.sum_neg)
        self.state.reference_mean_history.append(self.state.reference_mean)
        
        # Check for drift
        if not self.drift_detected:
            if self.state.sum_pos > self.lambda_param:
                # Positive drift (confidence increasing)
                self.drift_detected = True
                self.state.drift_detected_at = self.samples_processed
                self.state.drift_type = 'increase'
                
                drift_info = {
                    'detector': 'PageHinkleyConfidenceMonitor',
                    'drift_type': 'increase',
                    'detected_at_sample': self.state.drift_detected_at,
                    'timestamp': self.state.last_update_time,
                    'metric': 'confidence',
                    'current_value': confidence,
                    'reference_mean': self.state.reference_mean,
                    'sum_statistic': self.state.sum_pos,
                    'threshold': self.lambda_param
                }
                
                logger.info(
                    f"Confidence increase drift detected at sample {self.state.drift_detected_at} "
                    f"(current_confidence={confidence:.4f}, mean={self.state.reference_mean:.4f})"
                )
                
                # Notify callbacks
                self._notify_callbacks(drift_info)
                
                return True, drift_info
                
            elif self.state.sum_neg > self.lambda_param:
                # Negative drift (confidence decreasing)
                self.drift_detected = True
                self.state.drift_detected_at = self.samples_processed
                self.state.drift_type = 'decrease'
                
                drift_info = {
                    'detector': 'PageHinkleyConfidenceMonitor',
                    'drift_type': 'decrease',
                    'detected_at_sample': self.state.drift_detected_at,
                    'timestamp': self.state.last_update_time,
                    'metric': 'confidence',
                    'current_value': confidence,
                    'reference_mean': self.state.reference_mean,
                    'sum_statistic': self.state.sum_neg,
                    'threshold': self.lambda_param
                }
                
                logger.info(
                    f"Confidence decrease drift detected at sample {self.state.drift_detected_at} "
                    f"(current_confidence={confidence:.4f}, mean={self.state.reference_mean:.4f})"
                )
                
                # Notify callbacks
                self._notify_callbacks(drift_info)
                
                return True, drift_info
        
        return False, None
    
    def check_for_drift(self) -> Tuple[bool, Optional[Dict[str, Any]]]:
        """Check if drift has been detected.
        
        Returns:
            Tuple of (drift_detected, drift_info)
        """
        if self.drift_detected and self.state.drift_detected_at is not None:
            drift_info = {
                'detector': 'PageHinkleyConfidenceMonitor',
                'drift_type': self.state.drift_type,
                'detected_at_sample': self.state.drift_detected_at,
                'timestamp': self.state.last_update_time,
                'metric': 'confidence',
                'reference_mean': self.state.reference_mean,
                'sum_pos': self.state.sum_pos,
                'sum_neg': self.state.sum_neg,
                'threshold': self.lambda_param
            }
            return True, drift_info
        
        return False, None
    
    def reset(self) -> None:
        """Reset the detector state after drift has been handled."""
        # Reset drift detection state but keep the reference statistics
        self.state.sum_pos = 0.0
        self.state.sum_neg = 0.0
        self.drift_detected = False
        self.state.drift_detected_at = None
        self.state.drift_type = None

        # Reset drift cooldown tracking
        self.in_cooldown_period = False
        self.samples_since_last_drift = 0

        logger.debug("PageHinkleyConfidenceMonitor reset")
    
    def _initialize_reference(self) -> None:
        """Initialize reference statistics from warm-up phase samples."""
        if self.use_numpy and np:
            # Calculate initial reference mean using training samples
            initial_samples = np.array(self.state.confidence_history)
            self.state.reference_mean = float(np.mean(initial_samples))
        else:
            # Pure Python implementation for devices without NumPy
            initial_samples = self.state.confidence_history
            self.state.reference_mean = sum(initial_samples) / len(initial_samples)
        
        # Initialize history
        self.state.reference_mean_history = [self.state.reference_mean] * len(self.state.confidence_history)
        self.state.sum_pos_history = [0.0] * len(self.state.confidence_history)
        self.state.sum_neg_history = [0.0] * len(self.state.confidence_history)
        
        logger.debug(f"Initialized Page-Hinkley with reference mean={self.state.reference_mean:.4f}")
    
    def _update_reference_statistic(self, new_value: float, ref_value: float) -> float:
        """Update a reference statistic using the rolling update formula."""
        return (self.reference_update_factor * ref_value + 
                (1 - self.reference_update_factor) * new_value)
    
    def get_state(self) -> Dict[str, Any]:
        """Get the current state of the detector as a serializable dictionary.
        
        Returns:
            State dictionary
        """
        # Get base state from parent class
        state_dict = self._get_base_state()
        
        # Add detector-specific parameters
        state_dict.update({
            'delta': self.delta,
            'lambda_param': self.lambda_param,
            'alpha': self.alpha,
            'min_confidence': self.min_confidence,
            'use_numpy': self.use_numpy,
        })
        
        # Add detector-specific state
        state_dict["page_hinkley_state"] = {
            'confidence_history': self.state.confidence_history.copy(),
            'reference_mean': self.state.reference_mean,
            'sum_pos': self.state.sum_pos,
            'sum_neg': self.state.sum_neg,
            'sum_pos_history': self.state.sum_pos_history.copy(),
            'sum_neg_history': self.state.sum_neg_history.copy(),
            'reference_mean_history': self.state.reference_mean_history.copy(),
            'drift_detected_at': self.state.drift_detected_at,
            'drift_type': self.state.drift_type,
            'last_update_time': self.state.last_update_time
        }
        
        return state_dict
    
    def set_state(self, state: Dict[str, Any]) -> None:
        """Restore the detector state from a dictionary.
        
        Args:
            state: Previously saved state dictionary
        """
        # Restore base state using parent class
        self._set_base_state(state)
        
        # Restore detector-specific parameters
        self.delta = state.get('delta', self.delta)
        self.lambda_param = state.get('lambda_param', self.lambda_param)
        self.alpha = state.get('alpha', self.alpha)
        self.min_confidence = state.get('min_confidence', self.min_confidence)
        self.use_numpy = state.get('use_numpy', self.use_numpy)
        
        # Restore detector-specific state
        page_hinkley_state = state.get('page_hinkley_state', {})
        
        # Create new state object
        self.state = PageHinkleyState(
            confidence_history=page_hinkley_state.get('confidence_history', []).copy(),
            reference_mean=page_hinkley_state.get('reference_mean', 0.0),
            sum_pos=page_hinkley_state.get('sum_pos', 0.0),
            sum_neg=page_hinkley_state.get('sum_neg', 0.0),
            sum_pos_history=page_hinkley_state.get('sum_pos_history', []).copy(),
            sum_neg_history=page_hinkley_state.get('sum_neg_history', []).copy(),
            reference_mean_history=page_hinkley_state.get('reference_mean_history', []).copy(),
            drift_detected_at=page_hinkley_state.get('drift_detected_at'),
            drift_type=page_hinkley_state.get('drift_type'),
            last_update_time=page_hinkley_state.get('last_update_time', time.time())
        )