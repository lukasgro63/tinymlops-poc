from typing import Dict, Any, List, Optional, Tuple, Union, Callable
import time
import math
from collections import deque, Counter

import numpy as np

from tinylcm.utils.logging import setup_logger
from tinylcm.core.drift_detection.base import AutonomousDriftDetector

logger = setup_logger(__name__)


class PredictionDistributionMonitor(AutonomousDriftDetector):
    """Autonomous drift detector that monitors the distribution of model predictions.
    
    This detector tracks the frequency distribution of class predictions over time,
    detecting drift when the distribution significantly changes from the baseline.
    
    No ground truth labels are required, making this suitable for production monitoring.
    Two detection methods are supported:
    1. Block comparison - compares the distribution in consecutive blocks of samples
    2. EWMA control chart - tracks exponentially weighted probability of each class
    
    The detector uses Jensen-Shannon divergence to measure distribution difference,
    which is more suitable than KL divergence as it's symmetric and always finite.
    """
    
    def __init__(
        self,
        window_size: int = 50,
        threshold: float = 0.2,
        method: str = 'block',
        alpha: float = 0.05,  # Significance level for detection
        smoothing: float = 0.1,  # Smoothing factor for EWMA
        min_samples: int = 100,  # Minimum samples to establish baseline
        use_numpy: bool = True
    ):
        """Initialize the prediction distribution monitor.
        
        Args:
            window_size: Number of samples in a sliding window for distribution calculation
            threshold: Threshold for detecting distribution changes 
                      (higher values make detection less sensitive)
            method: Detection method ('block' or 'ewma')
            alpha: Significance level for statistical tests
            smoothing: Smoothing factor for EWMA method
            min_samples: Minimum samples needed to establish baseline
            use_numpy: Whether to use NumPy for calculations
        """
        super().__init__()
        self.window_size = window_size
        self.threshold = threshold
        self.method = method.lower()
        self.alpha = alpha
        self.smoothing = smoothing
        self.min_samples = min_samples
        self.use_numpy = use_numpy
        
        # Validate method
        if self.method not in ['block', 'ewma']:
            raise ValueError("Method must be 'block' or 'ewma'")
        
        # Initialize state
        self.prediction_history = []
        self.class_counts = Counter()
        self.n_samples = 0
        self.classes = set()
        self.baseline_distribution = {}
        self.current_window = deque(maxlen=window_size)
        self.divergence_scores = []
        
        # EWMA specific state
        self.ewma_distributions = {}  # Class -> EWMA probability
        self.ewma_ucl = {}  # Upper control limits for each class
        self.ewma_lcl = {}  # Lower control limits for each class
        
        # Drift detection state
        self.training_mode = True
        self.drift_detected = False
        self.drift_point_index = None
        self.last_update_time = time.time()
        
        logger.debug(
            f"Initialized PredictionDistributionMonitor with window_size={window_size}, "
            f"threshold={threshold}, method={method}, min_samples={min_samples}"
        )
    
    def update(self, record: Dict[str, Any]) -> Tuple[bool, Optional[Dict[str, Any]]]:
        """Update the detector with a new prediction.
        
        Args:
            record: Dictionary containing the inference data. Should include a 'prediction'
                   key with the model's class prediction.
                   
        Returns:
            Tuple of (drift_detected, drift_info)
        """
        # Extract prediction from record
        if 'prediction' not in record:
            logger.warning("Prediction missing in record, skipping update")
            return False, None
        
        prediction = record['prediction']
        
        # Handle invalid predictions
        if prediction is None:
            logger.warning("Invalid prediction value: None, skipping update")
            return False, None
        
        # Update internal state
        self.prediction_history.append(prediction)
        self.class_counts[prediction] += 1
        self.classes.add(prediction)
        self.n_samples += 1
        self.current_window.append(prediction)
        self.last_update_time = time.time()
        
        # Training phase - collect samples to establish baseline
        if self.training_mode:
            if self.n_samples >= self.min_samples:
                self._initialize_baseline()
                self.training_mode = False
                logger.debug(f"Finished training phase with {self.n_samples} samples")
            return False, None
        
        # Check for drift using the selected method
        if self.method == 'block' and len(self.current_window) == self.window_size:
            return self._check_block_drift()
        elif self.method == 'ewma':
            return self._check_ewma_drift(prediction)
        
        return False, None
    
    def check_for_drift(self) -> Tuple[bool, Optional[Dict[str, Any]]]:
        """Check if drift has been detected.
        
        Returns:
            Tuple of (drift_detected, drift_info)
        """
        if self.drift_detected:
            drift_info = {
                'detector': 'PredictionDistributionMonitor',
                'method': self.method,
                'detected_at_sample': self.drift_point_index,
                'timestamp': self.last_update_time,
                'baseline_distribution': self.baseline_distribution,
                'current_distribution': self._get_current_distribution(),
                'threshold': self.threshold
            }
            
            if self.divergence_scores:
                drift_info['recent_divergence'] = self.divergence_scores[-1]
            
            return True, drift_info
        
        return False, None
    
    def reset(self) -> None:
        """Reset the detector state after drift has been handled."""
        # Reset drift detection flags but keep distributions
        self.drift_detected = False
        self.drift_point_index = None
        
        # Optionally update baseline to the current distribution
        # Uncomment to enable automatic adaptation:
        # self.baseline_distribution = self._get_current_distribution()
        
        logger.debug("PredictionDistributionMonitor reset")
    
    def _initialize_baseline(self) -> None:
        """Initialize the baseline distribution from collected samples."""
        self.baseline_distribution = self._calculate_distribution(self.prediction_history)
        
        # For EWMA method, initialize control limits
        if self.method == 'ewma':
            for class_label in self.classes:
                p = self.baseline_distribution.get(class_label, 0.0)
                self.ewma_distributions[class_label] = p
                
                # Calculate control limits
                # Standard deviation of a proportion = sqrt(p*(1-p)/n)
                std_dev = math.sqrt(p * (1 - p) / self.min_samples)
                control_factor = 3 * math.sqrt(self.smoothing / (2 - self.smoothing))
                
                self.ewma_ucl[class_label] = min(p + control_factor * std_dev, 1.0)
                self.ewma_lcl[class_label] = max(p - control_factor * std_dev, 0.0)
        
        logger.debug(f"Initialized baseline distribution: {self.baseline_distribution}")
    
    def _calculate_distribution(self, predictions: List[Any]) -> Dict[Any, float]:
        """Calculate the probability distribution from a list of predictions."""
        if not predictions:
            return {}
            
        counter = Counter(predictions)
        total = len(predictions)
        
        return {class_label: count / total for class_label, count in counter.items()}
    
    def _get_current_distribution(self) -> Dict[Any, float]:
        """Get the current distribution based on the sliding window."""
        if not self.current_window:
            return {}
            
        return self._calculate_distribution(list(self.current_window))
    
    def _calculate_js_divergence(self, dist1: Dict[Any, float], dist2: Dict[Any, float]) -> float:
        """Calculate the Jensen-Shannon divergence between two distributions."""
        # Ensure all classes are in both distributions
        all_classes = set(dist1.keys()) | set(dist2.keys())
        
        # For Jensen-Shannon divergence we need distributions over the same support
        p = [dist1.get(c, 0.0) for c in all_classes]
        q = [dist2.get(c, 0.0) for c in all_classes]
        
        # If using NumPy, use its implementation
        if self.use_numpy and np:
            # Convert to arrays
            p_array = np.array(p)
            q_array = np.array(q)
            
            # Calculate midpoint distribution
            m_array = 0.5 * (p_array + q_array)
            
            # Calculate KL divergence but avoid 0 in denominator
            kl_p_m = np.sum(p_array * np.log2(p_array / m_array + 1e-10) * (p_array > 0))
            kl_q_m = np.sum(q_array * np.log2(q_array / m_array + 1e-10) * (q_array > 0))
            
            # Jensen-Shannon divergence
            js_divergence = 0.5 * (kl_p_m + kl_q_m)
            
        else:
            # Pure Python implementation
            # Calculate midpoint distribution
            m = [0.5 * (p[i] + q[i]) for i in range(len(p))]
            
            # Calculate KL divergence for p||m
            kl_p_m = 0.0
            for i in range(len(p)):
                if p[i] > 0:
                    kl_p_m += p[i] * math.log2(p[i] / (m[i] + 1e-10))
            
            # Calculate KL divergence for q||m
            kl_q_m = 0.0
            for i in range(len(q)):
                if q[i] > 0:
                    kl_q_m += q[i] * math.log2(q[i] / (m[i] + 1e-10))
            
            # Jensen-Shannon divergence
            js_divergence = 0.5 * (kl_p_m + kl_q_m)
        
        return float(js_divergence)
    
    def _check_block_drift(self) -> Tuple[bool, Optional[Dict[str, Any]]]:
        """Check for drift using the block comparison method."""
        # Calculate current distribution from the sliding window
        current_dist = self._get_current_distribution()
        
        # Calculate JS divergence between current and baseline
        divergence = self._calculate_js_divergence(self.baseline_distribution, current_dist)
        self.divergence_scores.append(divergence)
        
        # Check if divergence exceeds threshold
        if divergence > self.threshold and not self.drift_detected:
            self.drift_detected = True
            self.drift_point_index = self.n_samples
            
            drift_info = {
                'detector': 'PredictionDistributionMonitor',
                'method': 'block',
                'detected_at_sample': self.drift_point_index,
                'timestamp': self.last_update_time,
                'js_divergence': divergence,
                'threshold': self.threshold,
                'baseline_distribution': self.baseline_distribution,
                'current_distribution': current_dist
            }
            
            logger.info(
                f"Prediction distribution drift detected at sample {self.drift_point_index} "
                f"(JS divergence={divergence:.4f}, threshold={self.threshold:.4f})"
            )
            
            # Notify callbacks
            self._notify_callbacks(drift_info)
            
            return True, drift_info
        
        return False, None
    
    def _check_ewma_drift(self, prediction: Any) -> Tuple[bool, Optional[Dict[str, Any]]]:
        """Check for drift using the EWMA method."""
        # Update EWMA for each class
        for class_label in self.classes:
            current_p = 1.0 if class_label == prediction else 0.0
            prev_ewma = self.ewma_distributions.get(class_label, self.baseline_distribution.get(class_label, 0.0))
            
            # Update EWMA
            new_ewma = self.smoothing * current_p + (1 - self.smoothing) * prev_ewma
            self.ewma_distributions[class_label] = new_ewma
            
            # Check if EWMA is outside control limits
            if (new_ewma > self.ewma_ucl.get(class_label, 1.0) or 
                new_ewma < self.ewma_lcl.get(class_label, 0.0)):
                
                # Only signal drift if not already detected
                if not self.drift_detected:
                    self.drift_detected = True
                    self.drift_point_index = self.n_samples
                    
                    # Calculate current distribution for reporting
                    current_dist = self._get_current_distribution()
                    
                    drift_info = {
                        'detector': 'PredictionDistributionMonitor',
                        'method': 'ewma',
                        'detected_at_sample': self.drift_point_index,
                        'timestamp': self.last_update_time,
                        'drifted_class': class_label,
                        'ewma_value': new_ewma,
                        'upper_limit': self.ewma_ucl.get(class_label, 1.0),
                        'lower_limit': self.ewma_lcl.get(class_label, 0.0),
                        'baseline_probability': self.baseline_distribution.get(class_label, 0.0),
                        'baseline_distribution': self.baseline_distribution,
                        'current_distribution': current_dist
                    }
                    
                    logger.info(
                        f"Prediction distribution drift detected at sample {self.drift_point_index} "
                        f"for class {class_label} (EWMA={new_ewma:.4f}, UCL={self.ewma_ucl.get(class_label, 1.0):.4f}, "
                        f"LCL={self.ewma_lcl.get(class_label, 0.0):.4f})"
                    )
                    
                    # Notify callbacks
                    self._notify_callbacks(drift_info)
                    
                    return True, drift_info
        
        return False, None
    
    def get_state(self) -> Dict[str, Any]:
        """Get the current state of the detector as a serializable dictionary.
        
        Returns:
            State dictionary
        """
        # Convert deque to list for serialization
        current_window_list = list(self.current_window)
        
        state_dict = {
            'window_size': self.window_size,
            'threshold': self.threshold,
            'method': self.method,
            'alpha': self.alpha,
            'smoothing': self.smoothing,
            'min_samples': self.min_samples,
            'use_numpy': self.use_numpy,
            'prediction_history': self.prediction_history.copy(),
            'class_counts': dict(self.class_counts),
            'classes': list(self.classes),
            'n_samples': self.n_samples,
            'baseline_distribution': self.baseline_distribution.copy(),
            'current_window': current_window_list,
            'divergence_scores': self.divergence_scores.copy(),
            'training_mode': self.training_mode,
            'drift_detected': self.drift_detected,
            'drift_point_index': self.drift_point_index,
            'last_update_time': self.last_update_time
        }
        
        # Add EWMA specific state if applicable
        if self.method == 'ewma':
            state_dict['ewma_distributions'] = self.ewma_distributions.copy()
            state_dict['ewma_ucl'] = self.ewma_ucl.copy()
            state_dict['ewma_lcl'] = self.ewma_lcl.copy()
        
        return state_dict
    
    def set_state(self, state: Dict[str, Any]) -> None:
        """Restore the detector state from a dictionary.
        
        Args:
            state: Previously saved state dictionary
        """
        self.window_size = state.get('window_size', self.window_size)
        self.threshold = state.get('threshold', self.threshold)
        self.method = state.get('method', self.method)
        self.alpha = state.get('alpha', self.alpha)
        self.smoothing = state.get('smoothing', self.smoothing)
        self.min_samples = state.get('min_samples', self.min_samples)
        self.use_numpy = state.get('use_numpy', self.use_numpy)
        
        # Restore core state
        self.prediction_history = state.get('prediction_history', []).copy()
        self.class_counts = Counter(state.get('class_counts', {}))
        self.classes = set(state.get('classes', []))
        self.n_samples = state.get('n_samples', 0)
        self.baseline_distribution = state.get('baseline_distribution', {}).copy()
        
        # Restore sliding window
        current_window_list = state.get('current_window', [])
        self.current_window = deque(current_window_list, maxlen=self.window_size)
        
        self.divergence_scores = state.get('divergence_scores', []).copy()
        self.training_mode = state.get('training_mode', True)
        self.drift_detected = state.get('drift_detected', False)
        self.drift_point_index = state.get('drift_point_index')
        self.last_update_time = state.get('last_update_time', time.time())
        
        # Restore EWMA specific state if applicable
        if self.method == 'ewma':
            self.ewma_distributions = state.get('ewma_distributions', {}).copy()
            self.ewma_ucl = state.get('ewma_ucl', {}).copy()
            self.ewma_lcl = state.get('ewma_lcl', {}).copy()