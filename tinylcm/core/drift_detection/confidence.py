from dataclasses import dataclass, field
from typing import Dict, Any, List, Optional, Tuple, Union, Callable
import time
import math
from collections import deque

import numpy as np

from tinylcm.utils.logging import setup_logger
from tinylcm.core.drift_detection.base import AutonomousDriftDetector

logger = setup_logger(__name__)


@dataclass
class EWMAState:
    """State class for Exponentially Weighted Moving Average monitoring."""
    history: List[float] = field(default_factory=list)
    ewma_values: List[float] = field(default_factory=list)
    ucl_values: List[float] = field(default_factory=list)  # Upper control limit
    lcl_values: List[float] = field(default_factory=list)  # Lower control limit
    n_samples: int = 0
    current_ewma: float = 0.0
    current_ucl: float = 0.0  # Upper control limit
    current_lcl: float = 0.0  # Lower control limit
    drift_detected_at: Optional[int] = None
    last_update_time: float = field(default_factory=time.time)
    training_mode: bool = True
    min_samples: int = 30


class EWMAConfidenceMonitor(AutonomousDriftDetector):
    """Autonomous drift detector based on prediction confidence using EWMA.
    
    This detector uses Exponentially Weighted Moving Average (EWMA) to track model
    confidence scores over time, detecting drift when confidence falls outside the
    control limits for a significant period.
    
    Unlike AccuracyCUSUM, this detector doesn't require ground truth labels, making
    it suitable for ongoing monitoring in production where labels aren't available.
    
    The EWMA approach is well-suited for resource-constrained environments as it
    has minimal memory requirements and computational overhead.
    """
    
    def __init__(
        self,
        lambda_param: float = 0.1,
        threshold_factor: float = 3.0,
        drift_window: int = 5,
        min_confidence: float = 0.2,
        training_size: int = 30,
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
            training_size: Number of initial samples to use for establishing baseline.
            use_numpy: Whether to use NumPy for calculations (faster but uses more memory).
        """
        super().__init__()
        self.lambda_param = lambda_param
        self.threshold_factor = threshold_factor
        self.drift_window = drift_window
        self.min_confidence = min_confidence
        self.training_size = training_size
        self.use_numpy = use_numpy
        
        # Initialize state
        self.state = EWMAState(min_samples=training_size)
        self.out_of_bounds_count = 0
        
        # Use deque with maxlen for efficient sliding window
        self.recent_scores = deque(maxlen=drift_window)
        
        logger.debug(
            f"Initialized EWMAConfidenceMonitor with lambda={lambda_param}, "
            f"threshold_factor={threshold_factor}, drift_window={drift_window}, "
            f"training_size={training_size}"
        )
    
    def update(self, record: Dict[str, Any]) -> Tuple[bool, Optional[Dict[str, Any]]]:
        """Update the detector with a new confidence score.
        
        Args:
            record: Dictionary containing the inference data. Should include a 'confidence'
                   key with the model's confidence score for the prediction.
                   
        Returns:
            Tuple of (drift_detected, drift_info)
        """
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
        self.state.history.append(confidence)
        self.state.n_samples += 1
        self.state.last_update_time = time.time()
        
        # If in training mode, just collect samples without checking for drift
        if self.state.training_mode:
            if self.state.n_samples >= self.state.min_samples:
                # Exit training mode and initialize EWMA statistics
                self._initialize_ewma()
                self.state.training_mode = False
                logger.debug(f"Finished training phase with {self.state.n_samples} samples")
            return False, None
        
        # Update EWMA
        prev_ewma = self.state.current_ewma
        self.state.current_ewma = self._calculate_ewma(confidence, prev_ewma)
        self.state.ewma_values.append(self.state.current_ewma)
        
        # Update control limits if needed
        self._update_control_limits()
        
        # Check if current EWMA is outside control limits
        out_of_bounds = (self.state.current_ewma > self.state.current_ucl or 
                         self.state.current_ewma < self.state.current_lcl)
        
        # Update recent_scores with 1 for out of bounds, 0 for in bounds
        self.recent_scores.append(1 if out_of_bounds else 0)
        
        # Check for drift condition
        drift_detected = sum(self.recent_scores) >= self.drift_window
        
        if drift_detected and self.state.drift_detected_at is None:
            # Record when drift was first detected
            self.state.drift_detected_at = self.state.n_samples
            
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
                "consecutive_violations": sum(self.recent_scores)
            }
            
            logger.info(
                f"Confidence drift detected at sample {self.state.drift_detected_at} "
                f"(current_confidence={confidence:.4f}, ewma={self.state.current_ewma:.4f})"
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
        if self.state.drift_detected_at is not None:
            drift_info = {
                "detector": "EWMAConfidenceMonitor",
                "detected_at_sample": self.state.drift_detected_at,
                "timestamp": self.state.last_update_time,
                "metric": "confidence",
                "ewma_value": self.state.current_ewma,
                "upper_limit": self.state.current_ucl,
                "lower_limit": self.state.current_lcl,
                "consecutive_violations": sum(self.recent_scores) if self.recent_scores else 0
            }
            return True, drift_info
        
        return False, None
    
    def reset(self) -> None:
        """Reset the detector state after drift has been handled."""
        # Clear drift detection state but keep the statistics
        self.state.drift_detected_at = None
        self.recent_scores.clear()
        
        logger.debug("EWMAConfidenceMonitor reset")
    
    def _initialize_ewma(self) -> None:
        """Initialize EWMA statistics from collected training samples."""
        if self.use_numpy and np:
            # Calculate initial EWMA using mean of training samples
            initial_samples = np.array(self.state.history)
            self.state.current_ewma = float(np.mean(initial_samples))
            
            # Calculate standard deviation for control limits
            std_dev = float(np.std(initial_samples))
        else:
            # Pure Python implementation for devices without NumPy
            initial_samples = self.state.history
            mean = sum(initial_samples) / len(initial_samples)
            self.state.current_ewma = mean
            
            # Calculate standard deviation
            sum_squared_diff = sum((x - mean) ** 2 for x in initial_samples)
            std_dev = math.sqrt(sum_squared_diff / len(initial_samples))
        
        # Initialize control limits
        self._set_control_limits(std_dev)
        
        # Add initial values to history
        self.state.ewma_values.append(self.state.current_ewma)
        self.state.ucl_values.append(self.state.current_ucl)
        self.state.lcl_values.append(self.state.current_lcl)
        
        logger.debug(
            f"Initialized EWMA with mean={self.state.current_ewma:.4f}, "
            f"std_dev={std_dev:.4f}, UCL={self.state.current_ucl:.4f}, "
            f"LCL={self.state.current_lcl:.4f}"
        )
    
    def _calculate_ewma(self, new_value: float, prev_ewma: float) -> float:
        """Calculate new EWMA value based on new observation and previous EWMA."""
        return self.lambda_param * new_value + (1 - self.lambda_param) * prev_ewma
    
    def _set_control_limits(self, std_dev: float) -> None:
        """Set upper and lower control limits based on standard deviation."""
        # Control limit width
        limit_width = self.threshold_factor * std_dev * math.sqrt(
            self.lambda_param / (2 - self.lambda_param)
        )
        
        # Upper and lower control limits
        self.state.current_ucl = self.state.current_ewma + limit_width
        self.state.current_lcl = max(self.state.current_ewma - limit_width, 0)  # Confidence can't be negative
    
    def _update_control_limits(self) -> None:
        """Update control limits based on current EWMA value."""
        # In this implementation, we keep the control limits static after initialization
        # This is a common approach for EWMA control charts
        # Alternative: could recalculate limits periodically or adapt them based on recent values
        
        # Just add current values to history for plotting/analysis
        self.state.ucl_values.append(self.state.current_ucl)
        self.state.lcl_values.append(self.state.current_lcl)
    
    def get_state(self) -> Dict[str, Any]:
        """Get the current state of the detector as a serializable dictionary.
        
        Returns:
            State dictionary
        """
        # Convert deque to list for serialization
        recent_scores_list = list(self.recent_scores)
        
        state_dict = {
            "lambda_param": self.lambda_param,
            "threshold_factor": self.threshold_factor,
            "drift_window": self.drift_window,
            "min_confidence": self.min_confidence,
            "training_size": self.training_size,
            "use_numpy": self.use_numpy,
            "ewma_state": {
                "history": self.state.history.copy(),
                "ewma_values": self.state.ewma_values.copy(),
                "ucl_values": self.state.ucl_values.copy(),
                "lcl_values": self.state.lcl_values.copy(),
                "n_samples": self.state.n_samples,
                "current_ewma": self.state.current_ewma,
                "current_ucl": self.state.current_ucl,
                "current_lcl": self.state.current_lcl,
                "drift_detected_at": self.state.drift_detected_at,
                "last_update_time": self.state.last_update_time,
                "training_mode": self.state.training_mode,
                "min_samples": self.state.min_samples
            },
            "out_of_bounds_count": self.out_of_bounds_count,
            "recent_scores": recent_scores_list
        }
        
        return state_dict
    
    def set_state(self, state: Dict[str, Any]) -> None:
        """Restore the detector state from a dictionary.
        
        Args:
            state: Previously saved state dictionary
        """
        self.lambda_param = state.get("lambda_param", self.lambda_param)
        self.threshold_factor = state.get("threshold_factor", self.threshold_factor)
        self.drift_window = state.get("drift_window", self.drift_window)
        self.min_confidence = state.get("min_confidence", self.min_confidence)
        self.training_size = state.get("training_size", self.training_size)
        self.use_numpy = state.get("use_numpy", self.use_numpy)
        
        ewma_state = state.get("ewma_state", {})
        
        # Create new state object
        self.state = EWMAState(
            history=ewma_state.get("history", []).copy(),
            ewma_values=ewma_state.get("ewma_values", []).copy(),
            ucl_values=ewma_state.get("ucl_values", []).copy(),
            lcl_values=ewma_state.get("lcl_values", []).copy(),
            n_samples=ewma_state.get("n_samples", 0),
            current_ewma=ewma_state.get("current_ewma", 0.0),
            current_ucl=ewma_state.get("current_ucl", 0.0),
            current_lcl=ewma_state.get("current_lcl", 0.0),
            drift_detected_at=ewma_state.get("drift_detected_at"),
            last_update_time=ewma_state.get("last_update_time", time.time()),
            training_mode=ewma_state.get("training_mode", True),
            min_samples=ewma_state.get("min_samples", self.training_size)
        )
        
        self.out_of_bounds_count = state.get("out_of_bounds_count", 0)
        
        # Restore recent scores as deque
        recent_scores = state.get("recent_scores", [])
        self.recent_scores = deque(recent_scores, maxlen=self.drift_window)


class PageHinkleyConfidenceMonitor(AutonomousDriftDetector):
    """Autonomous drift detector based on prediction confidence using Page-Hinkley Test.
    
    This detector applies the Page-Hinkley test to model confidence scores, detecting drift
    when there is a significant and sustained change in the confidence level. The test is
    particularly effective at detecting sudden shifts in the probability distribution.
    
    Like the EWMA detector, this doesn't require ground truth labels, making it suitable
    for production monitoring.
    
    The Page-Hinkley test is computationally efficient and has minimal memory requirements,
    making it suitable for resource-constrained environments.
    """
    
    def __init__(
        self,
        delta: float = 0.05,
        lambda_param: float = 10.0,
        alpha: float = 0.9999,
        min_confidence: float = 0.2,
        training_size: int = 30,
        use_numpy: bool = True
    ):
        """Initialize the Page-Hinkley confidence monitor.
        
        Args:
            delta: Magnitude of allowed changes (smaller values are more sensitive)
            lambda_param: Detection threshold (higher values reduce false positives)
            alpha: Forgetting factor for updating the mean
            min_confidence: Minimum confidence value to consider
            training_size: Number of initial samples to use for establishing baseline
            use_numpy: Whether to use NumPy for calculations
        """
        super().__init__()
        self.delta = delta
        self.lambda_param = lambda_param
        self.alpha = alpha
        self.min_confidence = min_confidence
        self.training_size = training_size
        self.use_numpy = use_numpy
        
        # Initialize state
        self.mean = None
        self.sum_pos = 0.0
        self.sum_neg = 0.0
        self.n_samples = 0
        self.training_samples = []
        self.history = {
            'confidence': [],
            'mean': [],
            'sum_pos': [],
            'sum_neg': []
        }
        
        self.drift_detected = False
        self.drift_type = None  # 'increase' or 'decrease'
        self.drift_point_index = None
        self.last_update_time = time.time()
        
        logger.debug(
            f"Initialized PageHinkleyConfidenceMonitor with delta={delta}, "
            f"lambda_param={lambda_param}, alpha={alpha}, training_size={training_size}"
        )
    
    def update(self, record: Dict[str, Any]) -> Tuple[bool, Optional[Dict[str, Any]]]:
        """Update the detector with a new confidence score.
        
        Args:
            record: Dictionary containing the inference data. Should include a 'confidence'
                   key with the model's confidence score for the prediction.
                   
        Returns:
            Tuple of (drift_detected, drift_info)
        """
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
        
        # Update counter and last update time
        self.n_samples += 1
        self.last_update_time = time.time()
        
        # Track history
        self.history['confidence'].append(confidence)
        
        # Training phase - collect samples to establish baseline
        if self.mean is None:
            self.training_samples.append(confidence)
            
            if len(self.training_samples) >= self.training_size:
                # Initialize mean from training samples
                if self.use_numpy and np:
                    self.mean = float(np.mean(self.training_samples))
                else:
                    self.mean = sum(self.training_samples) / len(self.training_samples)
                
                logger.debug(f"Initialized Page-Hinkley with mean={self.mean:.4f}")
                
                # Initialize historical values for the mean
                self.history['mean'] = [self.mean] * len(self.history['confidence'])
                self.history['sum_pos'] = [0.0] * len(self.history['confidence'])
                self.history['sum_neg'] = [0.0] * len(self.history['confidence'])
            
            return False, None
        
        # Update mean using forgetting factor
        self.mean = self.alpha * self.mean + (1 - self.alpha) * confidence
        self.history['mean'].append(self.mean)
        
        # Calculate deviation
        deviation = confidence - (self.mean + self.delta)
        
        # Update cumulative sums
        self.sum_pos = max(0, self.sum_pos + deviation)
        self.sum_neg = max(0, self.sum_neg - deviation)
        
        # Record history
        self.history['sum_pos'].append(self.sum_pos)
        self.history['sum_neg'].append(self.sum_neg)
        
        # Check for drift
        if not self.drift_detected:
            if self.sum_pos > self.lambda_param:
                # Positive drift (confidence increasing)
                self.drift_detected = True
                self.drift_type = 'increase'
                self.drift_point_index = self.n_samples
                
                drift_info = {
                    'detector': 'PageHinkleyConfidenceMonitor',
                    'drift_type': 'increase',
                    'detected_at_sample': self.drift_point_index,
                    'timestamp': self.last_update_time,
                    'metric': 'confidence',
                    'current_value': confidence,
                    'mean': self.mean,
                    'sum_statistic': self.sum_pos,
                    'threshold': self.lambda_param
                }
                
                logger.info(
                    f"Confidence increase drift detected at sample {self.drift_point_index} "
                    f"(current_confidence={confidence:.4f}, mean={self.mean:.4f})"
                )
                
                # Notify callbacks
                self._notify_callbacks(drift_info)
                
                return True, drift_info
                
            elif self.sum_neg > self.lambda_param:
                # Negative drift (confidence decreasing)
                self.drift_detected = True
                self.drift_type = 'decrease'
                self.drift_point_index = self.n_samples
                
                drift_info = {
                    'detector': 'PageHinkleyConfidenceMonitor',
                    'drift_type': 'decrease',
                    'detected_at_sample': self.drift_point_index,
                    'timestamp': self.last_update_time,
                    'metric': 'confidence',
                    'current_value': confidence,
                    'mean': self.mean,
                    'sum_statistic': self.sum_neg,
                    'threshold': self.lambda_param
                }
                
                logger.info(
                    f"Confidence decrease drift detected at sample {self.drift_point_index} "
                    f"(current_confidence={confidence:.4f}, mean={self.mean:.4f})"
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
        if self.drift_detected:
            drift_info = {
                'detector': 'PageHinkleyConfidenceMonitor',
                'drift_type': self.drift_type,
                'detected_at_sample': self.drift_point_index,
                'timestamp': self.last_update_time,
                'metric': 'confidence',
                'mean': self.mean,
                'sum_pos': self.sum_pos,
                'sum_neg': self.sum_neg,
                'threshold': self.lambda_param
            }
            return True, drift_info
        
        return False, None
    
    def reset(self) -> None:
        """Reset the detector state after drift has been handled."""
        # Reset drift detection state but keep the mean 
        self.sum_pos = 0.0
        self.sum_neg = 0.0
        self.drift_detected = False
        self.drift_type = None
        self.drift_point_index = None
        
        logger.debug("PageHinkleyConfidenceMonitor reset")
    
    def get_state(self) -> Dict[str, Any]:
        """Get the current state of the detector as a serializable dictionary.
        
        Returns:
            State dictionary
        """
        state_dict = {
            'delta': self.delta,
            'lambda_param': self.lambda_param,
            'alpha': self.alpha,
            'min_confidence': self.min_confidence,
            'training_size': self.training_size,
            'use_numpy': self.use_numpy,
            'mean': self.mean,
            'sum_pos': self.sum_pos,
            'sum_neg': self.sum_neg,
            'n_samples': self.n_samples,
            'training_samples': self.training_samples.copy(),
            'history': {
                'confidence': self.history['confidence'].copy(),
                'mean': self.history['mean'].copy(),
                'sum_pos': self.history['sum_pos'].copy(),
                'sum_neg': self.history['sum_neg'].copy()
            },
            'drift_detected': self.drift_detected,
            'drift_type': self.drift_type,
            'drift_point_index': self.drift_point_index,
            'last_update_time': self.last_update_time
        }
        
        return state_dict
    
    def set_state(self, state: Dict[str, Any]) -> None:
        """Restore the detector state from a dictionary.
        
        Args:
            state: Previously saved state dictionary
        """
        self.delta = state.get('delta', self.delta)
        self.lambda_param = state.get('lambda_param', self.lambda_param)
        self.alpha = state.get('alpha', self.alpha)
        self.min_confidence = state.get('min_confidence', self.min_confidence)
        self.training_size = state.get('training_size', self.training_size)
        self.use_numpy = state.get('use_numpy', self.use_numpy)
        
        self.mean = state.get('mean')
        self.sum_pos = state.get('sum_pos', 0.0)
        self.sum_neg = state.get('sum_neg', 0.0)
        self.n_samples = state.get('n_samples', 0)
        
        training_samples = state.get('training_samples', [])
        self.training_samples = training_samples.copy()
        
        history = state.get('history', {})
        self.history = {
            'confidence': history.get('confidence', []).copy(),
            'mean': history.get('mean', []).copy(),
            'sum_pos': history.get('sum_pos', []).copy(),
            'sum_neg': history.get('sum_neg', []).copy()
        }
        
        self.drift_detected = state.get('drift_detected', False)
        self.drift_type = state.get('drift_type')
        self.drift_point_index = state.get('drift_point_index')
        self.last_update_time = state.get('last_update_time', time.time())