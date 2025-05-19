import math
import random
import time
from collections import Counter, deque
from typing import Any, Callable, Dict, List, Optional, Tuple, Union

import numpy as np

from tinylcm.core.drift_detection.base import AutonomousDriftDetector
from tinylcm.utils.logging import setup_logger

logger = setup_logger(__name__)


class FeatureMonitor(AutonomousDriftDetector):
    """Autonomous drift detector that tracks statistical properties of feature vectors.
    
    This detector monitors feature vectors (embeddings or raw features) for significant
    changes in their statistical properties, detecting drift when the data distribution
    changes. It uses incremental tracking of mean and variance for efficiency.
    
    By monitoring the feature space, it can detect changes in the input distribution
    even before they cause changes in model outputs, allowing for early drift detection.
    """
    
    def __init__(
        self,
        window_size: int = 100,
        threshold: float = 3.0,
        reference_size: int = 300,
        max_features: Optional[int] = None,
        sampling_rate: float = 1.0,
        use_numpy: bool = True,
        distance_metric: str = 'euclidean',
        warm_up_samples: int = 300,
        reference_update_interval: int = 100,
        reference_update_factor: float = 0.05,
        pause_reference_update_during_drift: bool = True
    ):
        """Initialize the feature monitor.
        
        Args:
            window_size: Size of sliding window for recent feature statistics
            threshold: Threshold for detecting feature drift (standard deviations)
            reference_size: Number of samples to use for establishing baseline
            max_features: Maximum number of features to monitor (uses random selection if needed)
            sampling_rate: Fraction of samples to process (for resource constraints)
            use_numpy: Whether to use NumPy for calculations
            distance_metric: Distance metric for feature space ('euclidean', 'cosine', 'manhattan')
            warm_up_samples: Number of samples to collect during warm-up
            reference_update_interval: Number of samples between reference updates
            reference_update_factor: Factor for updating reference (β)
            pause_reference_update_during_drift: Whether to pause updating during detected drift
        """
        super().__init__(
            warm_up_samples=warm_up_samples,
            reference_update_interval=reference_update_interval,
            reference_update_factor=reference_update_factor,
            pause_reference_update_during_drift=pause_reference_update_during_drift
        )
        
        self.window_size = window_size
        self.threshold = threshold
        self.reference_size = reference_size
        self.max_features = max_features
        self.sampling_rate = sampling_rate
        self.use_numpy = use_numpy
        self.distance_metric = distance_metric
        
        # Validate distance metric
        if self.distance_metric not in ['euclidean', 'cosine', 'manhattan']:
            raise ValueError("Distance metric must be 'euclidean', 'cosine', or 'manhattan'")
        
        # Initial state
        self.n_samples = 0
        self.reference_samples = []
        self.selected_features = None
        self.reference_mean = None
        self.reference_std = None
        self.reference_min = None
        self.reference_max = None
        
        # Sliding window for recent samples
        self.current_window = deque(maxlen=window_size)
        self.current_distances = deque(maxlen=window_size)
        
        # Drift detection state
        self.training_mode = True
        self.drift_detected = False
        self.drift_point_index = None
        self.last_update_time = time.time()
        
        # Control chart variables
        self.ewma_value = None
        self.ewma_history = []
        self.distance_history = []
        self.ucl = None
        self.lcl = None
        
        logger.debug(
            f"Initialized FeatureMonitor with window_size={window_size}, "
            f"threshold={threshold}, reference_size={reference_size}, "
            f"metric={distance_metric}"
        )
    
    def update(self, record: Dict[str, Any]) -> Tuple[bool, Optional[Dict[str, Any]]]:
        """Update the detector with a new feature vector.
        
        Args:
            record: Dictionary containing the inference data. Should include a 'features'
                   key with the feature vector.
                   
        Returns:
            Tuple of (drift_detected, drift_info)
        """
        # Process base sample counting and warm-up phase
        self._process_sample(record)
        
        # Extract features from record
        if 'features' not in record:
            logger.warning("Features missing in record, skipping update")
            return False, None
        
        features = record['features']
        
        # Handle invalid features
        if features is None:
            logger.warning("Invalid features value: None, skipping update")
            return False, None
        
        # Apply sampling rate - probabilistically skip some samples
        if random.random() > self.sampling_rate:
            return False, None
            
        # Convert to numpy array if not already
        if self.use_numpy and np:
            if not isinstance(features, np.ndarray):
                features = np.array(features, dtype=np.float32)
                
        # Select features if needed
        if self.max_features is not None and len(features) > self.max_features:
            if self.selected_features is None:
                # First time - randomly select features to monitor
                if self.use_numpy and np:
                    self.selected_features = np.sort(
                        np.random.choice(len(features), self.max_features, replace=False)
                    )
                else:
                    indices = list(range(len(features)))
                    random.shuffle(indices)
                    self.selected_features = sorted(indices[:self.max_features])
                    
                logger.debug(f"Selected {len(self.selected_features)} features to monitor")
            
            # Extract only the selected features
            if self.use_numpy and np:
                features = features[self.selected_features]
            else:
                features = [features[i] for i in self.selected_features]
        
        # Update counter
        self.n_samples += 1
        self.last_update_time = time.time()
        
        # Handle training phase
        if self.in_warm_up_phase:
            self.reference_samples.append(features)
            return False, None
        
        # Initialize reference if needed
        if self.reference_mean is None and not self.in_warm_up_phase:
            self._initialize_reference()
            logger.debug(f"Finished warm-up phase with {len(self.reference_samples)} samples")
            return False, None
        
        # Calculate distance from reference
        distance = self._calculate_distance(features)
        self.current_distances.append(distance)
        self.distance_history.append(distance)
        
        # Add to sliding window
        self.current_window.append(features)
        
        # Update EWMA
        if self.ewma_value is None:
            self.ewma_value = distance
        else:
            # Use smoothing factor of 0.2 for responsiveness
            self.ewma_value = 0.2 * distance + 0.8 * self.ewma_value
        
        self.ewma_history.append(self.ewma_value)
        
        # Check for drift
        drift_info = None
        was_drift_detected = self.drift_detected
        
        if not self.drift_detected and self.ewma_value > self.ucl:
            self.drift_detected = True
            self.drift_point_index = self.n_samples
            
            drift_info = {
                'detector': 'FeatureMonitor',
                'detected_at_sample': self.drift_point_index,
                'timestamp': self.last_update_time,
                'metric': 'feature_distance',
                'current_value': distance,
                'ewma_value': self.ewma_value,
                'distance_metric': self.distance_metric,
                'upper_control_limit': self.ucl,
                'reference_mean': self._serialize_value(self.reference_mean),
                'reference_std': self._serialize_value(self.reference_std)
            }
            
            logger.info(
                f"Feature drift detected at sample {self.drift_point_index} "
                f"(distance={distance:.4f}, ewma={self.ewma_value:.4f}, ucl={self.ucl:.4f})"
            )
            
            # Notify callbacks
            self._notify_callbacks(drift_info)
        
        # Update reference statistics if needed
        if self.should_update_reference():
            self._update_reference_statistics()
            self.samples_since_last_update = 0
        
        return self.drift_detected, drift_info
    
    def _serialize_value(self, value):
        """Convert numpy arrays to lists for serialization."""
        if self.use_numpy and np and isinstance(value, np.ndarray):
            return value.tolist()
        return value
    
    def _update_reference_statistics(self):
        """Update reference statistics using current window."""
        if not self.current_window:
            return
            
        # Calculate statistics from current window
        window_statistics = self._calculate_window_statistics()
        window_mean = window_statistics.get('mean')
        
        if window_mean is None:
            return
            
        # Update reference mean using the rolling update formula
        if self.use_numpy and np:
            self.reference_mean = (
                self.reference_update_factor * np.array(window_mean) + 
                (1 - self.reference_update_factor) * self.reference_mean
            )
            
            # Also update reference standard deviation if possible
            window_std = window_statistics.get('std')
            if window_std is not None and self.reference_std is not None:
                # Update reference_std with a similar formula to smooth changes
                self.reference_std = (
                    self.reference_update_factor * np.array(window_std) + 
                    (1 - self.reference_update_factor) * self.reference_std
                )
        else:
            # Update each feature dimension
            if isinstance(window_mean, list) and isinstance(self.reference_mean, list):
                for i in range(min(len(window_mean), len(self.reference_mean))):
                    self.reference_mean[i] = (
                        self.reference_update_factor * window_mean[i] +
                        (1 - self.reference_update_factor) * self.reference_mean[i]
                    )
                    
                # Update reference_std if available
                window_std = window_statistics.get('std')
                if isinstance(window_std, list) and isinstance(self.reference_std, list):
                    for i in range(min(len(window_std), len(self.reference_std))):
                        self.reference_std[i] = (
                            self.reference_update_factor * window_std[i] +
                            (1 - self.reference_update_factor) * self.reference_std[i]
                        )
        
        # Update control limits based on new reference
        distances = []
        for sample in self.current_window:
            distance = self._calculate_distance(sample)
            distances.append(distance)
            
        if self.use_numpy and np:
            mean_distance = np.mean(distances)
            std_distance = np.std(distances)
        else:
            mean_distance = sum(distances) / len(distances)
            sum_squared_diff = sum((d - mean_distance) ** 2 for d in distances)
            std_distance = math.sqrt(sum_squared_diff / len(distances))
            
        self.ucl = mean_distance + self.threshold * std_distance
        self.lcl = max(0, mean_distance - self.threshold * std_distance)
        
        logger.debug(f"Updated reference statistics. New control limits: UCL={self.ucl:.4f}, LCL={self.lcl:.4f}")
    
    def check_for_drift(self) -> Tuple[bool, Optional[Dict[str, Any]]]:
        """Check if drift has been detected.
        
        Returns:
            Tuple of (drift_detected, drift_info)
        """
        if self.drift_detected:
            current_stats = self._calculate_window_statistics()
            
            drift_info = {
                'detector': 'FeatureMonitor',
                'detected_at_sample': self.drift_point_index,
                'timestamp': self.last_update_time,
                'metric': 'feature_distance',
                'ewma_value': self.ewma_value,
                'upper_control_limit': self.ucl,
                'distance_metric': self.distance_metric,
                'current_stats': current_stats,
                'reference_stats': {
                    'mean': self._serialize_value(self.reference_mean),
                    'std': self._serialize_value(self.reference_std),
                    'min': self._serialize_value(self.reference_min),
                    'max': self._serialize_value(self.reference_max)
                }
            }
            
            return True, drift_info
        
        return False, None
    
    def reset(self) -> None:
        """Reset the detector state after drift has been handled."""
        # Reset drift detection flags but keep reference statistics
        self.drift_detected = False
        self.drift_point_index = None

        # Reset drift cooldown tracking
        self.in_cooldown_period = False
        self.samples_since_last_drift = 0

        # Optionally update reference to current window
        # self._initialize_reference_from_window()

        logger.debug("FeatureMonitor reset")
    
    def _initialize_reference(self) -> None:
        """Initialize reference statistics from collected samples."""
        if self.use_numpy and np:
            # Convert reference samples to array
            reference_array = np.array(self.reference_samples)
            
            # Calculate reference statistics
            self.reference_mean = np.mean(reference_array, axis=0)
            self.reference_std = np.std(reference_array, axis=0)
            self.reference_min = np.min(reference_array, axis=0)
            self.reference_max = np.max(reference_array, axis=0)
            
            # Calculate distances for each reference sample
            distances = []
            for sample in self.reference_samples:
                distance = self._calculate_distance(sample)
                distances.append(distance)
            
            mean_distance = np.mean(distances)
            std_distance = np.std(distances)
            
            # Set control limits
            self.ucl = mean_distance + self.threshold * std_distance
            self.lcl = max(0, mean_distance - self.threshold * std_distance)
            
        else:
            # Pure Python implementation for devices without NumPy
            n_features = len(self.reference_samples[0])
            n_samples = len(self.reference_samples)
            
            # Calculate mean
            self.reference_mean = [0.0] * n_features
            for i in range(n_features):
                for sample in self.reference_samples:
                    self.reference_mean[i] += sample[i]
                self.reference_mean[i] /= n_samples
            
            # Calculate std_dev
            self.reference_std = [0.0] * n_features
            for i in range(n_features):
                for sample in self.reference_samples:
                    self.reference_std[i] += (sample[i] - self.reference_mean[i]) ** 2
                self.reference_std[i] = math.sqrt(self.reference_std[i] / n_samples)
            
            # Calculate min/max
            self.reference_min = [float('inf')] * n_features
            self.reference_max = [float('-inf')] * n_features
            for i in range(n_features):
                for sample in self.reference_samples:
                    self.reference_min[i] = min(self.reference_min[i], sample[i])
                    self.reference_max[i] = max(self.reference_max[i], sample[i])
            
            # Calculate distances for control limits
            distances = []
            for sample in self.reference_samples:
                distance = self._calculate_distance(sample)
                distances.append(distance)
            
            mean_distance = sum(distances) / len(distances)
            sum_squared_diff = sum((d - mean_distance) ** 2 for d in distances)
            std_distance = math.sqrt(sum_squared_diff / len(distances))
            
            # Set control limits
            self.ucl = mean_distance + self.threshold * std_distance
            self.lcl = max(0, mean_distance - self.threshold * std_distance)
        
        logger.debug(
            f"Initialized reference statistics with {len(self.reference_samples)} samples. "
            f"Control limits: UCL={self.ucl:.4f}, LCL={self.lcl:.4f}"
        )
    
    def _calculate_distance(self, features) -> float:
        """Calculate the distance between a feature vector and the reference."""
        if self.reference_mean is None:
            return 0.0
            
        if self.distance_metric == 'euclidean':
            return self._euclidean_distance(features, self.reference_mean)
        elif self.distance_metric == 'cosine':
            return self._cosine_distance(features, self.reference_mean)
        elif self.distance_metric == 'manhattan':
            return self._manhattan_distance(features, self.reference_mean)
        else:
            return self._euclidean_distance(features, self.reference_mean)
    
    def _euclidean_distance(self, x, y) -> float:
        """Calculate Euclidean distance between two vectors."""
        if self.use_numpy and np:
            return float(np.sqrt(np.sum((x - y) ** 2)))
        else:
            return math.sqrt(sum((x[i] - y[i]) ** 2 for i in range(len(x))))
    
    def _cosine_distance(self, x, y) -> float:
        """Calculate cosine distance between two vectors."""
        if self.use_numpy and np:
            dot_product = np.dot(x, y)
            norm_x = np.linalg.norm(x)
            norm_y = np.linalg.norm(y)
            similarity = dot_product / (norm_x * norm_y)
            return float(1.0 - similarity)
        else:
            dot_product = sum(x[i] * y[i] for i in range(len(x)))
            norm_x = math.sqrt(sum(x[i] ** 2 for i in range(len(x))))
            norm_y = math.sqrt(sum(y[i] ** 2 for i in range(len(y))))
            similarity = dot_product / (norm_x * norm_y) if norm_x * norm_y != 0 else 0
            return 1.0 - similarity
    
    def _manhattan_distance(self, x, y) -> float:
        """Calculate Manhattan distance between two vectors."""
        if self.use_numpy and np:
            return float(np.sum(np.abs(x - y)))
        else:
            return sum(abs(x[i] - y[i]) for i in range(len(x)))
    
    def _calculate_window_statistics(self) -> Dict[str, Any]:
        """Calculate statistics for the current window of samples."""
        if not self.current_window:
            return {}
            
        if self.use_numpy and np:
            window_array = np.array(list(self.current_window))
            stats = {
                'mean': self._serialize_value(np.mean(window_array, axis=0)),
                'std': self._serialize_value(np.std(window_array, axis=0)),
                'min': self._serialize_value(np.min(window_array, axis=0)),
                'max': self._serialize_value(np.max(window_array, axis=0)),
                'avg_distance': float(np.mean(self.current_distances)),
                'max_distance': float(np.max(self.current_distances))
            }
        else:
            # Simple statistics for current window
            window_list = list(self.current_window)
            distances_list = list(self.current_distances)
            stats = {
                'avg_distance': sum(distances_list) / len(distances_list),
                'max_distance': max(distances_list)
            }
            
        return stats
    
    def _initialize_reference_from_window(self) -> None:
        """Update reference statistics from the current window (for adaptation)."""
        if not self.current_window or len(self.current_window) < self.window_size // 2:
            logger.warning("Not enough samples in current window to update reference")
            return
        
        # Save old reference samples
        old_reference = self.reference_samples
        
        # Use current window as new reference
        self.reference_samples = list(self.current_window)
        
        # Re-initialize reference statistics
        self._initialize_reference()
        
        logger.info(
            f"Updated reference statistics with {len(self.reference_samples)} samples from current window"
        )
    
    def get_state(self) -> Dict[str, Any]:
        """Get the current state of the detector as a serializable dictionary.
        
        Returns:
            State dictionary
        """
        # Get base state from parent class
        state_dict = self._get_base_state()
        
        # Add feature monitor specific state
        state_dict.update({
            'window_size': self.window_size,
            'threshold': self.threshold,
            'reference_size': self.reference_size,
            'max_features': self.max_features,
            'sampling_rate': self.sampling_rate,
            'use_numpy': self.use_numpy,
            'distance_metric': self.distance_metric,
            'n_samples': self.n_samples,
            'selected_features': self._serialize_value(self.selected_features),
            'reference_mean': self._serialize_value(self.reference_mean),
            'reference_std': self._serialize_value(self.reference_std),
            'reference_min': self._serialize_value(self.reference_min),
            'reference_max': self._serialize_value(self.reference_max),
            'ewma_value': self.ewma_value,
            'ucl': self.ucl,
            'lcl': self.lcl,
            'training_mode': self.training_mode,
            'drift_detected': self.drift_detected,
            'drift_point_index': self.drift_point_index,
            'last_update_time': self.last_update_time
        })
        
        # Don't store full reference samples - too large
        # Just store the count
        state_dict['reference_sample_count'] = len(self.reference_samples)
        
        # Store recent history (not full history)
        recent_length = min(100, len(self.ewma_history))
        state_dict['recent_ewma_history'] = self.ewma_history[-recent_length:] if self.ewma_history else []
        state_dict['recent_distance_history'] = self.distance_history[-recent_length:] if self.distance_history else []
        
        return state_dict
    
    def set_state(self, state: Dict[str, Any]) -> None:
        """Restore the detector state from a dictionary.
        
        Args:
            state: Previously saved state dictionary.
            Note: Reference samples are not restored, only the statistics.
        """
        # Restore base state from parent class
        self._set_base_state(state)
        
        # Restore feature monitor specific state
        self.window_size = state.get('window_size', self.window_size)
        self.threshold = state.get('threshold', self.threshold)
        self.reference_size = state.get('reference_size', self.reference_size)
        self.max_features = state.get('max_features', self.max_features)
        self.sampling_rate = state.get('sampling_rate', self.sampling_rate)
        self.use_numpy = state.get('use_numpy', self.use_numpy)
        self.distance_metric = state.get('distance_metric', self.distance_metric)
        
        self.n_samples = state.get('n_samples', 0)
        self.selected_features = state.get('selected_features')
        self.reference_mean = state.get('reference_mean')
        self.reference_std = state.get('reference_std')
        self.reference_min = state.get('reference_min')
        self.reference_max = state.get('reference_max')
        
        self.ewma_value = state.get('ewma_value')
        self.ucl = state.get('ucl')
        self.lcl = state.get('lcl')
        
        # Restore histories if available
        self.ewma_history = state.get('recent_ewma_history', []).copy()
        self.distance_history = state.get('recent_distance_history', []).copy()
        
        # Clear current window and distances
        self.current_window = deque(maxlen=self.window_size)
        self.current_distances = deque(maxlen=self.window_size)
        
        # Restore state flags
        self.training_mode = state.get('training_mode', True)
        self.drift_detected = state.get('drift_detected', False)
        self.drift_point_index = state.get('drift_point_index')
        self.last_update_time = state.get('last_update_time', time.time())
        
        # Reference samples are not restored to save memory
        # This means the detector can perform detection but not retrain reference
        # from scratch without new samples
        self.reference_samples = []


class PageHinkleyFeatureMonitor(AutonomousDriftDetector):
    """Page-Hinkley test for monitoring feature statistics to detect drift.
    
    This detector implements the Page-Hinkley test on feature statistics.
    For a given feature statistic f(x_t), it tracks the cumulative sum of deviations
    m_t = sum_{i=1}^{t} (f(x_i) - μ_ref - δ) along with the running minimum M_t = min_{i≤t} m_i.
    Drift is signaled if m_t - M_t > λ.
    
    The feature_statistic_fn can be any function that reduces a feature vector to a scalar value,
    such as norm, variance, max, min, etc.
    """
    
    def __init__(
        self,
        feature_statistic_fn: Callable[[np.ndarray], float],
        lambda_threshold: float = 50.0,
        delta: float = 0.005,
        warm_up_samples: int = 100,
        reference_update_interval: int = 50,
        reference_update_factor: float = 0.05,
        pause_reference_update_during_drift: bool = True,
    ):
        """Initialize the Page-Hinkley feature monitor.
        
        Args:
            feature_statistic_fn: Function that computes a scalar statistic from a feature vector
            lambda_threshold: Threshold for drift detection (λ)
            delta: Magnitude parameter to match expected drift direction (δ)
            warm_up_samples: Number of samples to collect during warm-up
            reference_update_interval: Number of samples between reference updates
            reference_update_factor: Factor for updating reference (β)
            pause_reference_update_during_drift: Whether to pause updating during detected drift
        """
        super().__init__(
            warm_up_samples=warm_up_samples,
            reference_update_interval=reference_update_interval,
            reference_update_factor=reference_update_factor,
            pause_reference_update_during_drift=pause_reference_update_during_drift,
        )
        
        self.feature_statistic_fn = feature_statistic_fn
        self.lambda_threshold = lambda_threshold
        self.delta = delta
        
        # PH statistics
        self.reference_mean = None  # μ_ref
        self.cumulative_sum = 0.0   # m_t
        self.minimum_sum = 0.0      # M_t
        
        # For warm-up calculations
        self.warm_up_values = [] if self.in_warm_up_phase else None
    
    def update(self, record: Dict[str, Any]) -> Tuple[bool, Optional[Dict[str, Any]]]:
        """Update the detector with new data.
        
        Args:
            record: Dictionary containing 'features' key with feature vector
            
        Returns:
            Tuple of (drift_detected, drift_info)
        """
        # Process base sample counting and warm-up phase
        self._process_sample(record)
        
        # Extract features from record
        features = record.get('features')
        if features is None:
            logger.warning("Record does not contain 'features' key")
            return False, None
        
        # Calculate the feature statistic
        try:
            statistic_value = self.feature_statistic_fn(features)
        except Exception as e:
            logger.error(f"Error computing feature statistic: {str(e)}")
            return False, None
        
        drift_info = None
        
        # Handle warm-up phase
        if self.in_warm_up_phase:
            self.warm_up_values.append(statistic_value)
            return False, None
        elif self.reference_mean is None and not self.in_warm_up_phase:
            # Initialize reference after warm-up
            self.reference_mean = np.mean(self.warm_up_values)
            logger.debug(f"Initialized reference mean to {self.reference_mean}")
            self.warm_up_values = None  # Free memory
            self.cumulative_sum = 0.0
            self.minimum_sum = 0.0
            return False, None
        
        # Page-Hinkley test update
        deviation = statistic_value - self.reference_mean - self.delta
        self.cumulative_sum += deviation
        self.minimum_sum = min(self.minimum_sum, self.cumulative_sum)
        
        # Check for drift
        ph_value = self.cumulative_sum - self.minimum_sum
        was_drift_detected = self.drift_detected
        self.drift_detected = ph_value > self.lambda_threshold
        
        # If drift is newly detected, prepare drift info
        if self.drift_detected and not was_drift_detected:
            drift_info = {
                'detector_type': 'PageHinkleyFeatureMonitor',
                'statistic_value': statistic_value,
                'reference_mean': self.reference_mean,
                'ph_value': ph_value,
                'threshold': self.lambda_threshold
            }
            self._notify_callbacks(drift_info)
        
        # If drift ends, reset detection state but keep statistics
        elif was_drift_detected and not self.drift_detected:
            logger.debug("Drift condition no longer met")
            # Keep the reference statistics but reset the PH test
            self.cumulative_sum = 0.0
            self.minimum_sum = 0.0
        
        # Update reference statistics if needed
        if self.should_update_reference():
            self._update_statistics(statistic_value)
            self.samples_since_last_update = 0
        
        return self.drift_detected, drift_info
    
    def _update_statistics(self, new_value: float) -> None:
        """Update reference statistics with new value.
        
        Args:
            new_value: New statistic value
        """
        # Using the rolling update formula: μ_ref_t ← β·μ_ref_t-1 + (1-β)·new_value
        self.reference_mean = (self.reference_update_factor * self.reference_mean + 
                              (1 - self.reference_update_factor) * new_value)
        logger.debug(f"Updated reference mean to {self.reference_mean}")
    
    def _update_reference_statistics(self, new_stats: float, ref_stats: float) -> float:
        """Implement the abstract method from base class.
        
        Args:
            new_stats: New statistic value
            ref_stats: Current reference value
            
        Returns:
            Updated reference value
        """
        # Using the rolling update formula: μ_ref_t ← β·μ_ref_t-1 + (1-β)·new_value
        return self.reference_update_factor * ref_stats + (1 - self.reference_update_factor) * new_stats
    
    def check_for_drift(self) -> Tuple[bool, Optional[Dict[str, Any]]]:
        """Check if drift has been detected.
        
        For this detector, this simply returns the current drift state.
        
        Returns:
            Tuple of (drift_detected, drift_info)
        """
        drift_info = None
        if self.drift_detected:
            drift_info = {
                'detector_type': 'PageHinkleyFeatureMonitor',
                'reference_mean': self.reference_mean,
                'ph_value': self.cumulative_sum - self.minimum_sum,
                'threshold': self.lambda_threshold
            }
        return self.drift_detected, drift_info
    
    def reset(self) -> None:
        """Reset the detector state.

        This resets the Page-Hinkley statistics but keeps the reference statistics.
        """
        self.drift_detected = False
        self.cumulative_sum = 0.0
        self.minimum_sum = 0.0

        # Reset drift cooldown tracking
        self.in_cooldown_period = False
        self.samples_since_last_drift = 0

        logger.debug("Reset Page-Hinkley detector state")
    
    def get_state(self) -> Dict[str, Any]:
        """Get the current state of the detector.
        
        Returns:
            State dictionary
        """
        state = self._get_base_state()
        state.update({
            'reference_mean': self.reference_mean,
            'cumulative_sum': self.cumulative_sum,
            'minimum_sum': self.minimum_sum,
            'lambda_threshold': self.lambda_threshold,
            'delta': self.delta,
            'warm_up_values': self.warm_up_values if self.warm_up_values is not None else []
        })
        return state
    
    def set_state(self, state: Dict[str, Any]) -> None:
        """Restore the detector state.
        
        Args:
            state: Previously saved state dictionary
        """
        self._set_base_state(state)
        self.reference_mean = state.get('reference_mean', self.reference_mean)
        self.cumulative_sum = state.get('cumulative_sum', self.cumulative_sum)
        self.minimum_sum = state.get('minimum_sum', self.minimum_sum)
        self.lambda_threshold = state.get('lambda_threshold', self.lambda_threshold)
        self.delta = state.get('delta', self.delta)
        warm_up_values = state.get('warm_up_values', [])
        self.warm_up_values = warm_up_values if self.in_warm_up_phase else None


class EWMAFeatureMonitor(AutonomousDriftDetector):
    """EWMA-based monitor for feature statistics to detect drift.
    
    This detector implements an Exponentially Weighted Moving Average (EWMA) control chart
    for monitoring feature statistics. It tracks S_t = α·stat_t + (1-α)·S_{t-1} and signals
    drift if |S_t - μ_ref| > h·σ_ref, where h is a threshold multiplier.
    
    The feature_statistic_fn can be any function that reduces a feature vector to a scalar value,
    such as norm, variance, max, min, etc.
    """
    
    def __init__(
        self,
        feature_statistic_fn: Callable[[np.ndarray], float],
        alpha: float = 0.1,
        threshold_multiplier: float = 3.0,
        warm_up_samples: int = 100,
        reference_update_interval: int = 50,
        reference_update_factor: float = 0.05,
        pause_reference_update_during_drift: bool = True,
    ):
        """Initialize the EWMA feature monitor.
        
        Args:
            feature_statistic_fn: Function that computes a scalar statistic from a feature vector
            alpha: Smoothing factor for EWMA (α)
            threshold_multiplier: Multiplier for σ_ref to set threshold (h)
            warm_up_samples: Number of samples to collect during warm-up
            reference_update_interval: Number of samples between reference updates
            reference_update_factor: Factor for updating reference (β)
            pause_reference_update_during_drift: Whether to pause updating during detected drift
        """
        super().__init__(
            warm_up_samples=warm_up_samples,
            reference_update_interval=reference_update_interval,
            reference_update_factor=reference_update_factor,
            pause_reference_update_during_drift=pause_reference_update_during_drift,
        )
        
        self.feature_statistic_fn = feature_statistic_fn
        self.alpha = alpha
        self.threshold_multiplier = threshold_multiplier
        
        # EWMA statistics
        self.reference_mean = None    # μ_ref
        self.reference_std = None     # σ_ref
        self.ewma_statistic = None    # S_t
        
        # For warm-up calculations
        self.warm_up_values = [] if self.in_warm_up_phase else None
    
    def update(self, record: Dict[str, Any]) -> Tuple[bool, Optional[Dict[str, Any]]]:
        """Update the detector with new data.
        
        Args:
            record: Dictionary containing 'features' key with feature vector
            
        Returns:
            Tuple of (drift_detected, drift_info)
        """
        # Process base sample counting and warm-up phase
        self._process_sample(record)
        
        # Extract features from record
        features = record.get('features')
        if features is None:
            logger.warning("Record does not contain 'features' key")
            return False, None
        
        # Calculate the feature statistic
        try:
            statistic_value = self.feature_statistic_fn(features)
        except Exception as e:
            logger.error(f"Error computing feature statistic: {str(e)}")
            return False, None
        
        drift_info = None
        
        # Handle warm-up phase
        if self.in_warm_up_phase:
            self.warm_up_values.append(statistic_value)
            return False, None
        elif self.reference_mean is None and not self.in_warm_up_phase:
            # Initialize reference after warm-up
            self.reference_mean = np.mean(self.warm_up_values)
            self.reference_std = np.std(self.warm_up_values) if len(self.warm_up_values) > 1 else 1.0
            self.ewma_statistic = self.reference_mean
            logger.debug(f"Initialized reference mean={self.reference_mean}, std={self.reference_std}")
            self.warm_up_values = None  # Free memory
            return False, None
        
        # Update EWMA statistic
        self.ewma_statistic = self.alpha * statistic_value + (1 - self.alpha) * self.ewma_statistic
        
        # Check for drift
        threshold = self.threshold_multiplier * self.reference_std
        deviation = abs(self.ewma_statistic - self.reference_mean)
        was_drift_detected = self.drift_detected
        self.drift_detected = deviation > threshold
        
        # If drift is newly detected, prepare drift info
        if self.drift_detected and not was_drift_detected:
            drift_info = {
                'detector_type': 'EWMAFeatureMonitor',
                'statistic_value': statistic_value,
                'ewma_value': self.ewma_statistic,
                'reference_mean': self.reference_mean,
                'reference_std': self.reference_std,
                'deviation': deviation,
                'threshold': threshold
            }
            self._notify_callbacks(drift_info)
        
        # If drift ends, reset detection state but keep statistics
        elif was_drift_detected and not self.drift_detected:
            logger.debug("Drift condition no longer met")
        
        # Update reference statistics if needed
        if self.should_update_reference():
            self._update_statistics(statistic_value)
            self.samples_since_last_update = 0
        
        return self.drift_detected, drift_info
    
    def _update_statistics(self, new_value: float) -> None:
        """Update reference statistics with new value.
        
        Args:
            new_value: New statistic value
        """
        # Update reference mean using the rolling update formula
        old_mean = self.reference_mean
        self.reference_mean = self.reference_update_factor * old_mean + (
                              1 - self.reference_update_factor) * new_value
        
        # Update reference standard deviation
        # We use a simplified approach that maintains the coefficient of variation
        if old_mean != 0:
            cv = self.reference_std / old_mean
            self.reference_std = cv * self.reference_mean
        
        logger.debug(f"Updated reference mean={self.reference_mean}, std={self.reference_std}")
    
    def _update_reference_statistics(self, new_stats: float, ref_stats: Dict[str, float]) -> Dict[str, float]:
        """Implement the abstract method from base class.
        
        Args:
            new_stats: New statistic value
            ref_stats: Current reference statistics (dict with 'mean' and 'std' keys)
            
        Returns:
            Updated reference statistics
        """
        # This method isn't directly used in this implementation, as we have a specialized 
        # _update_statistics method that handles both mean and std updates
        return {
            'mean': self.reference_update_factor * ref_stats['mean'] + 
                    (1 - self.reference_update_factor) * new_stats,
            'std': ref_stats['std']  # Would need more complex logic to update properly
        }
    
    def check_for_drift(self) -> Tuple[bool, Optional[Dict[str, Any]]]:
        """Check if drift has been detected.
        
        For this detector, this simply returns the current drift state.
        
        Returns:
            Tuple of (drift_detected, drift_info)
        """
        drift_info = None
        if self.drift_detected:
            threshold = self.threshold_multiplier * self.reference_std
            deviation = abs(self.ewma_statistic - self.reference_mean)
            drift_info = {
                'detector_type': 'EWMAFeatureMonitor',
                'ewma_value': self.ewma_statistic,
                'reference_mean': self.reference_mean,
                'reference_std': self.reference_std,
                'deviation': deviation,
                'threshold': threshold
            }
        return self.drift_detected, drift_info
    
    def reset(self) -> None:
        """Reset the detector state.

        This resets the drift flag but keeps the reference statistics and current EWMA value.
        """
        self.drift_detected = False

        # Reset drift cooldown tracking
        self.in_cooldown_period = False
        self.samples_since_last_drift = 0

        logger.debug("Reset EWMA detector state")
    
    def get_state(self) -> Dict[str, Any]:
        """Get the current state of the detector.
        
        Returns:
            State dictionary
        """
        state = self._get_base_state()
        state.update({
            'reference_mean': self.reference_mean,
            'reference_std': self.reference_std,
            'ewma_statistic': self.ewma_statistic,
            'alpha': self.alpha,
            'threshold_multiplier': self.threshold_multiplier,
            'warm_up_values': self.warm_up_values if self.warm_up_values is not None else []
        })
        return state
    
    def set_state(self, state: Dict[str, Any]) -> None:
        """Restore the detector state.
        
        Args:
            state: Previously saved state dictionary
        """
        self._set_base_state(state)
        self.reference_mean = state.get('reference_mean', self.reference_mean)
        self.reference_std = state.get('reference_std', self.reference_std)
        self.ewma_statistic = state.get('ewma_statistic', self.ewma_statistic)
        self.alpha = state.get('alpha', self.alpha)
        self.threshold_multiplier = state.get('threshold_multiplier', self.threshold_multiplier)
        warm_up_values = state.get('warm_up_values', [])
        self.warm_up_values = warm_up_values if self.in_warm_up_phase else None

class KNNDistanceMonitor(AutonomousDriftDetector):
    """Specialized drift detector that directly monitors KNN neighbor distances.
    
    This detector is particularly effective for detecting unknown classes or outliers,
    as they typically have much larger distances to their nearest neighbors compared
    to known classes. It works by tracking statistics of the distance values rather
    than just prediction confidences, making it more sensitive to novel objects.
    
    It uses the Page-Hinkley test to detect significant increases in the average
    distance to nearest neighbors, which is a strong signal for unknown objects.
    """
    
    def __init__(
        self,
        delta: float = 0.1,
        lambda_threshold: float = 5.0,
        exit_threshold_factor: float = 0.7,
        high_confidence_threshold: float = 0.9,
        stable_known_classes: List[str] = None,
        warm_up_samples: int = 5,
        reference_update_interval: int = 50,
        reference_update_factor: float = 0.05,
        pause_reference_update_during_drift: bool = True,
        drift_cooldown_period: int = 10,
        initial_reference_mean: float = None,
        initial_reference_std: float = None,
        reference_stats_path: str = None,
        use_adaptive_thresholds: bool = False,
        allow_runtime_threshold_adaptation: bool = False,
        adaptive_delta_std_multiplier: float = 1.5,
        adaptive_lambda_std_multiplier: float = 5.0,
    ):
        """Initialize the KNN distance monitor.
        
        Args:
            delta: Magnitude parameter to allow for small fluctuations (δ)
            lambda_threshold: Threshold for drift detection (λ)
            exit_threshold_factor: Factor to multiply lambda_threshold when determining threshold for exiting drift state
            high_confidence_threshold: Confidence threshold for identifying stable predictions
            stable_known_classes: List of class names considered stable (to skip/reset drift detection when predicted with high confidence)
            warm_up_samples: Number of samples to collect during warm-up
            reference_update_interval: Number of samples between reference updates
            reference_update_factor: Factor for updating reference (β)
            pause_reference_update_during_drift: Whether to pause updating during detected drift
            drift_cooldown_period: Number of samples to wait before triggering another drift event
            initial_reference_mean: Pre-computed reference mean to use instead of warm-up calculation
            initial_reference_std: Pre-computed reference standard deviation for adaptive thresholds
            reference_stats_path: Path to JSON file containing pre-computed reference statistics
            use_adaptive_thresholds: Whether to use reference_std to set delta and lambda thresholds 
            allow_runtime_threshold_adaptation: Whether to update thresholds during runtime (vs. only during initialization)
            adaptive_delta_std_multiplier: Multiplier for reference_std to calculate delta threshold
            adaptive_lambda_std_multiplier: Multiplier for reference_std to calculate lambda threshold
        """
        super().__init__(
            warm_up_samples=warm_up_samples,
            reference_update_interval=reference_update_interval,
            reference_update_factor=reference_update_factor,
            pause_reference_update_during_drift=pause_reference_update_during_drift,
            drift_cooldown_period=drift_cooldown_period,
        )
        
        self.delta = delta
        self.lambda_threshold = lambda_threshold
        self.exit_threshold_factor = exit_threshold_factor
        self.high_confidence_threshold = high_confidence_threshold
        self.stable_known_classes = stable_known_classes or []
        self.use_adaptive_thresholds = use_adaptive_thresholds
        self.allow_runtime_threshold_adaptation = allow_runtime_threshold_adaptation
        self.adaptive_delta_std_multiplier = adaptive_delta_std_multiplier
        self.adaptive_lambda_std_multiplier = adaptive_lambda_std_multiplier
        
        # Store pre-computed reference statistics values
        self.initial_reference_mean = initial_reference_mean
        self.initial_reference_std = initial_reference_std
        self.reference_stats_path = reference_stats_path
        
        # PH statistics
        self.reference_mean = None  # μ_ref
        self.reference_std = None   # σ_ref (for adaptive thresholds)
        self.cumulative_sum = 0.0   # m_t
        self.minimum_sum = 0.0      # M_t
        
        # Distance tracking
        self.last_distances = []
        # Use deque with max length to prevent unbounded growth
        from collections import deque
        self.distance_history = deque(maxlen=max(500, warm_up_samples * 5))
        
        # Initialize reference_mean from pre-computed statistics if provided
        if self.initial_reference_mean is not None:
            # Skip warm-up and use the pre-computed reference mean
            self.reference_mean = self.initial_reference_mean
            self.in_warm_up_phase = False
            self.warm_up_values = None
            logger.info(f"Using pre-computed reference mean: {self.reference_mean}")
        elif self.reference_stats_path:
            # Try to load reference statistics from file
            try:
                self._load_reference_statistics()
            except Exception as e:
                logger.warning(f"Failed to load reference statistics from {self.reference_stats_path}: {e}")
                # Fall back to warm-up
                self.warm_up_values = [] if self.in_warm_up_phase else None
        else:
            # Use regular warm-up
            self.warm_up_values = [] if self.in_warm_up_phase else None
        
        logger.debug(
            f"Initialized KNNDistanceMonitor with lambda_threshold={lambda_threshold}, "
            f"delta={delta}, warm_up_samples={warm_up_samples}, "
            f"use_precomputed_stats={self.reference_mean is not None}"
        )
    
    def update(self, record: Dict[str, Any]) -> Tuple[bool, Optional[Dict[str, Any]]]:
        """Update the detector with new data.
        
        Args:
            record: Dictionary containing prediction data. Should have either:
                  - '_knn_distances' key with a list of distances to nearest neighbors
                  - 'classifier' key with an object having '_last_distances' attribute
                  - 'confidence' key which can be used as a fallback
            
        Returns:
            Tuple of (drift_detected, drift_info)
        """
        # Safety check - record should not be None
        if record is None:
            logger.warning("Record is None, skipping update")
            return False, None
        
        # Check if record is a dict
        if not isinstance(record, dict):
            logger.warning(f"Record is not a dictionary, received: {type(record).__name__}, skipping update")
            return False, None
        
        # Check for high-confidence predictions of known classes to skip or reset drift detection
        # This prevents the detector from being unnecessarily affected by normal variations in known objects
        prediction = record.get('prediction')
        confidence = record.get('confidence')
        
        if prediction and confidence is not None and self.stable_known_classes:
            if confidence >= self.high_confidence_threshold and prediction in self.stable_known_classes:
                # If we're in a drift state and a high-confidence known object appears, reset the PH stats
                if self.drift_detected:
                    logger.info(f"KNNDistanceMonitor: Reset due to high-confidence ({confidence:.2f}) known class '{prediction}'")
                    self.reset_ph_stats()  # Reset Page-Hinkley stats but keep reference statistics
                    return False, None
                else:
                    # If we're not in drift state, just skip this sample to avoid affecting PH statistics
                    # for normal variations in known classes
                    logger.debug(f"KNNDistanceMonitor: Skipping update for high-confidence ({confidence:.2f}) known class '{prediction}'")
                    return False, None
            
        try:    
            # Process base sample counting and warm-up phase
            self._process_sample(record)
            
            # Extract distances from record using multiple possible sources
            distances = self._extract_distances(record)
            if distances is None or len(distances) == 0:
                logger.warning(f"Could not extract valid distances from record with keys: {list(record.keys())}. Skipping update for this sample.")
                return False, None
        except Exception as e:
            logger.warning(f"Error processing record in KNNDistanceMonitor: {str(e)}")
            return False, None
        
        # Calculate average distance
        avg_distance = sum(distances) / len(distances)
        self.last_distances = distances
        self.distance_history.append(avg_distance)
        
        # Occasionally log the distances to help with debugging
        if len(self.distance_history) % 10 == 0:
            logger.debug(f"KNNDistanceMonitor: Avg distance = {avg_distance:.4f}")
        
        drift_info = None
        
        # Handle warm-up phase
        if self.in_warm_up_phase:
            self.warm_up_values.append(avg_distance)
            return False, None
        elif self.reference_mean is None and not self.in_warm_up_phase:
            # Initialize reference after warm-up
            self.reference_mean = np.mean(self.warm_up_values) if len(self.warm_up_values) > 0 else avg_distance
            
            # Also calculate standard deviation if we have enough warm-up values
            if len(self.warm_up_values) > 1:
                self.reference_std = np.std(self.warm_up_values)
                logger.info(f"KNNDistanceMonitor: Initialized reference mean={self.reference_mean:.4f}, std={self.reference_std:.4f}")
            else:
                # If we don't have enough samples, use a default
                self.reference_std = self.reference_mean * 0.1  # Fallback: assume 10% relative variation
                logger.info(f"KNNDistanceMonitor: Initialized reference mean={self.reference_mean:.4f}, using default std={self.reference_std:.4f}")
            
            # If adaptive thresholds are enabled, update the thresholds based on learned statistics
            if self.use_adaptive_thresholds and self.reference_std > 0:
                original_delta = self.delta
                original_lambda = self.lambda_threshold
                
                # Set thresholds based on reference_std
                self.delta = self.reference_std * self.adaptive_delta_std_multiplier
                self.lambda_threshold = self.reference_std * self.adaptive_lambda_std_multiplier
                
                logger.info(f"KNNDistanceMonitor: Using adaptive thresholds after warm-up: "
                          f"delta={original_delta:.4f}->{self.delta:.4f}, "
                          f"lambda={original_lambda:.4f}->{self.lambda_threshold:.4f}")
            
            self.warm_up_values = None  # Free memory
            self.cumulative_sum = 0.0
            self.minimum_sum = 0.0
            return False, None
        
        # Page-Hinkley test update - we're looking for increases in distance
        # as unknown objects typically have larger distances
        try:
            # Calculate the deviation from the reference (plus delta)
            # When using fixed delta from config:
            deviation = avg_distance - (self.reference_mean + self.delta)
            
            # Track cumulative sum of deviations 
            self.cumulative_sum += deviation
            
            # Track minimum cumulative sum seen so far
            self.minimum_sum = min(self.minimum_sum, self.cumulative_sum)
            
            # Calculate the Page-Hinkley test statistic
            ph_value = self.cumulative_sum - self.minimum_sum
            
            # Remember the previous drift detection state
            was_drift_detected = self.drift_detected
            
            # Define exit threshold using factor (hysteresis approach)
            exit_drift_threshold = self.lambda_threshold * self.exit_threshold_factor
            
            # Track if drift was newly detected in this update for callback notification
            was_drift_detected = self.drift_detected
            newly_detected_drift_by_ph = False
            
            # Compare to appropriate threshold based on current state
            # For unknown objects, distances will be much larger, causing positive deviations
            # which will accumulate and cause ph_value to exceed the threshold
            if not self.drift_detected:
                # Not in drift state, use regular threshold to enter drift state
                if ph_value > self.lambda_threshold:
                    self.drift_detected = True
                    newly_detected_drift_by_ph = True
                    logger.info(f"KNNDistanceMonitor: Entering drift state. PH value {ph_value:.2f} > threshold {self.lambda_threshold:.2f}")
                    # Log additional debug info about the current state
                    logger.info(f"KNNDistanceMonitor: avg_distance={avg_distance:.2f}, reference_mean={self.reference_mean:.2f}, " +
                              f"deviation={deviation:.2f}, cumulative_sum={self.cumulative_sum:.2f}")
                    
                    if 'prediction' in record and 'confidence' in record:
                        logger.info(f"KNNDistanceMonitor: Current prediction='{record['prediction']}', confidence={record['confidence']:.2f}")
            else:
                # Already in drift state, use lower threshold to exit drift state
                if ph_value <= exit_drift_threshold:
                    self.drift_detected = False
                    logger.info(f"KNNDistanceMonitor: Exited drift state. PH value {ph_value:.2f} <= exit_threshold {exit_drift_threshold:.2f}")
                    # Reset Page-Hinkley test statistics when exiting drift state
                    # so the detector starts fresh for the next potential drift
                    self.reset_ph_stats()
            
            # Log when PH test triggers new drift detection for better debugging
            if self.drift_detected and not was_drift_detected:
                # Include reference std in logging if available
                std_info = ""
                if self.reference_std:
                    # Calculate how many standard deviations we are from the mean
                    if self.reference_mean > 0 and self.reference_std > 0:
                        std_distance = (avg_distance - self.reference_mean) / self.reference_std
                        std_info = f", deviation: {std_distance:.2f} standard deviations"
                
                logger.warning(f"KNNDistanceMonitor: DRIFT DETECTED by Page-Hinkley test! " +
                             f"PH value: {ph_value:.2f} > threshold: {self.lambda_threshold:.2f}, " +
                             f"avg_distance: {avg_distance:.2f}, reference: {self.reference_mean:.2f}, " +
                             f"ratio: {(avg_distance/max(0.01, self.reference_mean)):.2f}x" + std_info)
            
            # Log the PH value occasionally for regular debugging
            if self.samples_processed % 10 == 0:
                logger.debug(f"KNNDistanceMonitor PH value: {ph_value:.2f}, threshold: {self.lambda_threshold:.2f}, " +
                            f"avg_distance: {avg_distance:.2f}, reference: {self.reference_mean:.2f}")
        except Exception as e:
            logger.warning(f"Error in Page-Hinkley calculation: {str(e)}")
            was_drift_detected = self.drift_detected
            self.drift_detected = False
        
        # The custom unknown object detection logic has been removed
        # Now only the Page-Hinkley test is used for drift detection
        is_unknown_object = False  # Always false now that we removed the custom detection
        
        # Check if we are in the cooldown period - respect this to avoid too many drift events
        if self.in_cooldown_period and self.samples_since_last_drift < self.drift_cooldown_period:
            # We're in cooldown - suppress new drift detections
            self.drift_detected = False
            is_unknown_object = False
            logger.debug(f"In cooldown period ({self.samples_since_last_drift}/{self.drift_cooldown_period}) - suppressing drift detection")
        
        # If drift is newly detected, prepare drift info
        if newly_detected_drift_by_ph:
            # We use Page-Hinkley test for drift detection, drift flag is already set above
            
            drift_info = {
                'detector_type': 'KNNDistanceMonitor',
                'metric': 'neighbor_distance',
                'current_value': avg_distance,
                'reference_mean': self.reference_mean,
                'reference_std': self.reference_std,  # Include reference_std in drift info
                'ph_value': ph_value if 'ph_value' in locals() else 0.0,
                'threshold': self.lambda_threshold,
                'distances': distances,
                'timestamp': time.time()
            }
            
            # Add standardized distance info if available
            if self.reference_std and self.reference_std > 0:
                std_distance = (avg_distance - self.reference_mean) / self.reference_std
                drift_info['std_distance'] = std_distance  # How many standard deviations away
            
            # Include prediction in log message if available
            prediction_info = ""
            if 'prediction' in record:
                prediction_info = f", Prediction={record['prediction']}"
                
            logger.info(
                f"KNNDistanceMonitor: Drift detected! Avg distance={avg_distance:.4f}, "
                f"Reference={self.reference_mean:.4f}{prediction_info}, " +
                (f"PH value={ph_value:.4f}" if 'ph_value' in locals() else "Direct distance-based detection")
            )
            
            # Notify callbacks
            self._notify_callbacks(drift_info)
        
        # Update reference statistics if needed
        if self.should_update_reference():
            # Using the rolling update formula for mean
            self.reference_mean = (self.reference_update_factor * self.reference_mean + 
                                  (1 - self.reference_update_factor) * avg_distance)
            
            # Also update reference standard deviation if it exists
            if self.reference_std is not None:
                # Calculate the current standard deviation over recent distances
                recent_distances = list(self.distance_history)[-30:]  # Use last 30 samples
                if recent_distances and len(recent_distances) > 5:  # Ensure enough samples
                    current_std = np.std(recent_distances)
                    # Update reference_std using the same rolling formula
                    self.reference_std = (self.reference_update_factor * self.reference_std + 
                                         (1 - self.reference_update_factor) * current_std)
                    
                    # If adaptive thresholds AND runtime adaptation are enabled, update thresholds 
                    if self.use_adaptive_thresholds and self.allow_runtime_threshold_adaptation:
                        # Store original values for logging
                        original_delta = self.delta
                        original_lambda = self.lambda_threshold
                        
                        # Adjust delta and lambda threshold based on new reference_std
                        self.delta = self.reference_std * self.adaptive_delta_std_multiplier
                        self.lambda_threshold = self.reference_std * self.adaptive_lambda_std_multiplier
                        
                        # Only log when there's a substantial change
                        if (abs(original_delta - self.delta) > 0.1 * original_delta or 
                            abs(original_lambda - self.lambda_threshold) > 0.1 * original_lambda):
                            logger.info(f"KNNDistanceMonitor: Updated adaptive thresholds during runtime: "
                                      f"delta={original_delta:.4f}->{self.delta:.4f}, "
                                      f"lambda={original_lambda:.4f}->{self.lambda_threshold:.4f}")
            
            self.samples_since_last_update = 0
            std_value = f"{self.reference_std:.4f}" if self.reference_std is not None else "None"
            logger.debug(f"KNNDistanceMonitor: Updated reference mean to {self.reference_mean:.4f}, std to {std_value}")
        
        return self.drift_detected, drift_info
    
    def _extract_distances(self, record: Dict[str, Any]) -> Optional[List[float]]:
        """Extract distance information from the record.
        
        Tries multiple sources in order of preference:
        1. '_knn_distances' key in the record
        2. 'classifier' object with '_last_distances' attribute
        3. '_last_distances' key in the record
        
        Args:
            record: Dictionary with prediction data
        
        Returns:
            List of distances or None if not found
        """
        try:
            # Case 1: Direct distances in record
            if '_knn_distances' in record and isinstance(record['_knn_distances'], (list, tuple, np.ndarray)):
                return list(record['_knn_distances'])
            
            # Case 2: Classifier object with distances attribute
            if 'classifier' in record and record['classifier'] is not None:
                if hasattr(record['classifier'], '_last_distances'):
                    distances = record['classifier']._last_distances
                    if distances is not None:
                        return list(distances)
                # Some classifiers store distances in a different attribute
                if hasattr(record['classifier'], 'last_distances'):
                    distances = record['classifier'].last_distances
                    if distances is not None:
                        return list(distances)
            
            # Case 3: Last distances in record
            if '_last_distances' in record and isinstance(record['_last_distances'], (list, tuple, np.ndarray)):
                return list(record['_last_distances'])
            
            # Case 4: Extract from raw result if available
            if 'raw_result' in record and isinstance(record['raw_result'], dict):
                raw = record['raw_result']
                if 'distances' in raw and isinstance(raw['distances'], (list, tuple, np.ndarray)):
                    return list(raw['distances'])
            
            # Case 5: Try to extract from metadata if present
            if 'metadata' in record and isinstance(record['metadata'], dict):
                meta = record['metadata']
                if 'distances' in meta and isinstance(meta['distances'], (list, tuple, np.ndarray)):
                    return list(meta['distances'])
                if 'neighbor_distances' in meta and isinstance(meta['neighbor_distances'], (list, tuple, np.ndarray)):
                    return list(meta['neighbor_distances'])
            
            # These fallbacks have been removed since we shouldn't hardcode values
            # and instead rely on the actual distances computed by the classifier
            
            # Log record keys to help debug
            logger.debug(f"KNNDistanceMonitor could not extract distances. Record keys: {list(record.keys())}")
            
            # Return None to indicate that no valid distances were found
            return None
            
        except Exception as e:
            logger.warning(f"Error extracting distances from record: {str(e)}")
            return None  # Return None to indicate extraction error
    
    def check_for_drift(self) -> Tuple[bool, Optional[Dict[str, Any]]]:
        """Check if drift has been detected.
        
        Returns:
            Tuple of (drift_detected, drift_info)
        """
        if not self.drift_detected:
            return False, None
        
        # Calculate average distance if available
        avg_distance = None
        if self.last_distances:
            avg_distance = sum(self.last_distances) / len(self.last_distances)
        
        drift_info = {
            'detector_type': 'KNNDistanceMonitor',
            'metric': 'neighbor_distance',
            'current_value': avg_distance,
            'reference_mean': self.reference_mean,
            'reference_std': self.reference_std,  # Include reference_std in check_for_drift too
            'ph_value': self.cumulative_sum - self.minimum_sum,
            'threshold': self.lambda_threshold,
            'std_distance': (avg_distance - self.reference_mean) / self.reference_std if self.reference_std and self.reference_std > 0 else None
        }
        
        return True, drift_info
    
    def reset_ph_stats(self) -> None:
        """Reset only the Page-Hinkley test statistics.
        
        This resets the drift flag and Page-Hinkley statistics but keeps
        the reference statistics for continuous monitoring.
        """
        self.drift_detected = False
        self.cumulative_sum = 0.0
        self.minimum_sum = 0.0
        
        logger.debug("KNNDistanceMonitor: Reset Page-Hinkley statistics")
    
    def reset(self) -> None:
        """Reset the detector state completely.
        
        This resets the drift flag, Page-Hinkley statistics, and the base
        class cooldown state, but keeps the reference statistics.
        """
        # Reset base class state (cooldown tracking)
        super().reset()
        
        # Reset Page-Hinkley stats
        self.reset_ph_stats()
        
        logger.debug("KNNDistanceMonitor: Reset detector state completely")
    
    def _load_reference_statistics(self) -> None:
        """Load pre-computed reference statistics from a JSON file.
        
        This method loads reference statistics (mean, std, etc.) from a JSON file
        specified by self.reference_stats_path.
        """
        if not self.reference_stats_path:
            logger.warning("No reference statistics path provided")
            return
            
        import json
        import os
        
        if not os.path.exists(self.reference_stats_path):
            logger.warning(f"Reference statistics file not found: {self.reference_stats_path}")
            return
            
        try:
            with open(self.reference_stats_path, 'r') as f:
                stats = json.load(f)
                
            # Extract relevant statistics
            if 'reference_mean' in stats:
                self.reference_mean = float(stats['reference_mean'])
                logger.info(f"Loaded reference mean: {self.reference_mean}")
                
                # Load standard deviation for adaptive thresholds
                if 'reference_std' in stats:
                    self.reference_std = float(stats['reference_std'])
                    self.initial_reference_std = self.reference_std
                    logger.info(f"Loaded reference std: {self.reference_std}")
                    
                    # Adjust thresholds based on loaded statistics if they were provided in config
                    original_delta = self.delta
                    original_lambda = self.lambda_threshold
                    
                    # Only adjust if explicitly using loaded statistics (keeping configured values as is)
                    # This allows for explicit configuration but also provides sensible defaults
                    adjusted_message = ""
                    if self.reference_std > 0:
                        # Use reference_std to validate if the current thresholds make sense
                        suggested_delta = self.reference_std * self.adaptive_delta_std_multiplier
                        suggested_lambda = self.reference_std * self.adaptive_lambda_std_multiplier
                        
                        # Log comparison between configured and suggested values
                        logger.info(f"Current delta: {self.delta}, Suggested delta: {suggested_delta}")
                        logger.info(f"Current lambda: {self.lambda_threshold}, Suggested lambda: {suggested_lambda}")
                        
                        # If adaptive thresholds are enabled, update parameters
                        if self.use_adaptive_thresholds:
                            self.delta = suggested_delta
                            self.lambda_threshold = suggested_lambda
                            adjusted_message = f" (using adaptive thresholds: delta={suggested_delta:.2f}, lambda={suggested_lambda:.2f})"
                            logger.info(f"Using adaptive thresholds based on reference statistics")
                        else:
                            adjusted_message = " (kept explicit configuration)"
                    
                # Skip warm-up phase if reference mean is available
                self.in_warm_up_phase = False
                self.warm_up_values = None
                
                logger.info(f"Successfully loaded reference statistics from file{adjusted_message}")
            else:
                logger.warning(f"No reference mean found in statistics file: {self.reference_stats_path}")
        except Exception as e:
            logger.error(f"Error loading reference statistics: {e}")
            import traceback
            traceback.print_exc()
    
    def get_state(self) -> Dict[str, Any]:
        """Get the current state of the detector.
        
        Returns:
            State dictionary
        """
        state = self._get_base_state()
        state.update({
            'reference_mean': self.reference_mean,
            'reference_std': self.reference_std,  # Added reference_std to state
            'cumulative_sum': self.cumulative_sum,
            'minimum_sum': self.minimum_sum,
            'lambda_threshold': self.lambda_threshold,
            'exit_threshold_factor': self.exit_threshold_factor,
            'high_confidence_threshold': self.high_confidence_threshold,
            'stable_known_classes': self.stable_known_classes,
            'delta': self.delta,
            'initial_reference_mean': self.initial_reference_mean,
            'initial_reference_std': self.initial_reference_std,
            'reference_stats_path': self.reference_stats_path,
            'use_adaptive_thresholds': self.use_adaptive_thresholds,
            'allow_runtime_threshold_adaptation': self.allow_runtime_threshold_adaptation,
            'adaptive_delta_std_multiplier': self.adaptive_delta_std_multiplier,
            'adaptive_lambda_std_multiplier': self.adaptive_lambda_std_multiplier,
            'warm_up_values': self.warm_up_values if self.warm_up_values is not None else [],
            'last_distances': self.last_distances,
            'recent_distance_history': list(self.distance_history)[-100:] if self.distance_history else []
        })
        return state
    
    def set_state(self, state: Dict[str, Any]) -> None:
        """Restore the detector state.
        
        Args:
            state: Previously saved state dictionary
        """
        self._set_base_state(state)
        self.reference_mean = state.get('reference_mean', self.reference_mean)
        self.reference_std = state.get('reference_std', self.reference_std)  # Restore reference_std
        self.cumulative_sum = state.get('cumulative_sum', self.cumulative_sum)
        self.minimum_sum = state.get('minimum_sum', self.minimum_sum)
        self.lambda_threshold = state.get('lambda_threshold', self.lambda_threshold)
        self.exit_threshold_factor = state.get('exit_threshold_factor', self.exit_threshold_factor)
        self.high_confidence_threshold = state.get('high_confidence_threshold', self.high_confidence_threshold)
        self.stable_known_classes = state.get('stable_known_classes', self.stable_known_classes)
        self.delta = state.get('delta', self.delta)
        self.initial_reference_mean = state.get('initial_reference_mean', self.initial_reference_mean)
        self.initial_reference_std = state.get('initial_reference_std', self.initial_reference_std)
        self.reference_stats_path = state.get('reference_stats_path', self.reference_stats_path)
        self.use_adaptive_thresholds = state.get('use_adaptive_thresholds', self.use_adaptive_thresholds)
        self.allow_runtime_threshold_adaptation = state.get('allow_runtime_threshold_adaptation', self.allow_runtime_threshold_adaptation)
        self.adaptive_delta_std_multiplier = state.get('adaptive_delta_std_multiplier', self.adaptive_delta_std_multiplier)
        self.adaptive_lambda_std_multiplier = state.get('adaptive_lambda_std_multiplier', self.adaptive_lambda_std_multiplier)
        
        warm_up_values = state.get('warm_up_values', [])
        self.warm_up_values = warm_up_values if self.in_warm_up_phase else None
        
        self.last_distances = state.get('last_distances', [])
        
        # Restore distance history if available
        if 'recent_distance_history' in state:
            self.distance_history = state['recent_distance_history']


class NeighborDiversityDriftDetector(AutonomousDriftDetector):
    """Drift detector that monitors the diversity of class labels among nearest neighbors.
    
    When a known object is presented to the KNN classifier, most or all of the nearest
    neighbors typically have the same class label (high homogeneity). However, when an
    unknown object is encountered, the neighbors may come from different classes (low
    homogeneity), indicating that the object doesn't clearly match any known class.
    
    This detector tracks the number of neighbors that don't belong to the dominant class
    and signals drift when this count exceeds a specified threshold.
    """
    
    def __init__(
        self,
        k_neighbors: int = 5,  # The number of neighbors used by KNN 
        max_mismatched_neighbors_threshold: int = 1,  # Threshold for drift (2+ neighbors not from dominant class)
        warm_up_samples: int = 10,  # Short warm-up period since we decide per sample
        reference_update_interval: int = 30,
        reference_update_factor: float = 0.05,
        pause_reference_update_during_drift: bool = True,
        drift_cooldown_period: int = 20,
    ):
        """Initialize the neighbor diversity drift detector.
        
        Args:
            k_neighbors: Number of neighbors used by the KNN classifier
            max_mismatched_neighbors_threshold: Maximum acceptable neighbors not from dominant class
            warm_up_samples: Number of samples to collect during warm-up
            reference_update_interval: Number of samples between reference updates
            reference_update_factor: Factor for updating reference (β)
            pause_reference_update_during_drift: Whether to pause updating during drift 
            drift_cooldown_period: Number of samples to wait before triggering new drift
        """
        super().__init__(
            warm_up_samples=warm_up_samples,
            reference_update_interval=reference_update_interval,
            reference_update_factor=reference_update_factor,
            pause_reference_update_during_drift=pause_reference_update_during_drift,
            drift_cooldown_period=drift_cooldown_period,
        )
        
        self.k_neighbors = k_neighbors
        self.max_mismatched_neighbors_threshold = max_mismatched_neighbors_threshold
        
        # Validate parameters
        if max_mismatched_neighbors_threshold >= k_neighbors:
            logger.warning(f"max_mismatched_neighbors_threshold ({max_mismatched_neighbors_threshold}) should be less than k_neighbors ({k_neighbors})")
            self.max_mismatched_neighbors_threshold = k_neighbors - 1  # Set to reasonable default
            
        # Keep track of recent label diversity metrics
        self.last_mismatched_count = 0
        self.last_dominant_class = None
        self.last_neighbor_labels = []
        
        logger.debug(
            f"Initialized NeighborDiversityDriftDetector with k={k_neighbors}, "
            f"threshold={max_mismatched_neighbors_threshold}, cooldown={drift_cooldown_period}"
        )
    
    def update(self, record: Dict[str, Any]) -> Tuple[bool, Optional[Dict[str, Any]]]:
        """Update the detector with new data.
        
        Args:
            record: Dictionary containing prediction data with neighbor labels
                   either via classifier.get_last_neighbor_labels() or _knn_neighbor_labels
            
        Returns:
            Tuple of (drift_detected, drift_info)
        """
        # Process base sample counting and warm-up phase
        self._process_sample(record)
        
        # This detector doesn't need extensive warm-up for reference statistics,
        # but we still respect the base class warm-up and cooldown
        if self.in_warm_up_phase:
            return False, None
            
        # Extract the labels of k nearest neighbors
        neighbor_labels = self._extract_neighbor_labels(record)
        if not neighbor_labels or len(neighbor_labels) != self.k_neighbors:
            logger.warning(f"Could not extract {self.k_neighbors} neighbor labels from record")
            return False, None
            
        # Keep track of the neighbor labels for logging/debugging
        self.last_neighbor_labels = neighbor_labels
        
        # Count the occurrences of each class among the neighbors
        label_counts = Counter(neighbor_labels)
        
        # If no labels were found (should not happen if neighbor_labels is not empty)
        if not label_counts:
            return False, None
            
        # Find the dominant class and its count
        dominant_class, dominant_count = label_counts.most_common(1)[0]
        
        # Number of neighbors that are NOT from the dominant class
        mismatched_neighbors_count = self.k_neighbors - dominant_count
        
        # Save for later reference
        self.last_mismatched_count = mismatched_neighbors_count
        self.last_dominant_class = dominant_class
        
        # Determine drift based on mismatched neighbors
        current_drift_detected = mismatched_neighbors_count > self.max_mismatched_neighbors_threshold
        
        drift_info = None
        previous_drift_state = self.drift_detected
        self.drift_detected = current_drift_detected
        
        # If drift is newly detected, prepare drift info
        if self.drift_detected and not previous_drift_state:
            # Construct detailed drift information
            drift_info = {
                'detector_type': 'NeighborDiversityDriftDetector',
                'metric': 'mismatched_knn_neighbors',
                'current_value': mismatched_neighbors_count,
                'threshold': self.max_mismatched_neighbors_threshold,
                'k_neighbors': self.k_neighbors,
                'dominant_class_in_neighbors': dominant_class,
                'dominant_class_count': dominant_count,
                'neighbor_labels': neighbor_labels,
                'timestamp': time.time()
            }
            
            # Include prediction in log message if available
            prediction_info = ""
            if 'prediction' in record:
                prediction_info = f", Prediction={record['prediction']}"
                drift_info['prediction'] = record['prediction']
                
            logger.warning(
                f"NeighborDiversityDriftDetector: DRIFT DETECTED! "
                f"Mismatched neighbors: {mismatched_neighbors_count} > threshold: {self.max_mismatched_neighbors_threshold}"
                f"{prediction_info}, Dominant class={dominant_class} ({dominant_count}/{self.k_neighbors})"
            )
            
            # Log the neighbor labels for debugging
            logger.info(f"Neighbor labels: {neighbor_labels}")
            
            # Notify callbacks
            self._notify_callbacks(drift_info)
            
        # Periodically log diversity stats for debugging
        if self.samples_processed % 10 == 0:
            logger.debug(
                f"Neighbor diversity: {mismatched_neighbors_count}/{self.k_neighbors} mismatched, "
                f"dominant class={dominant_class} ({dominant_count}/{self.k_neighbors})"
            )
            
        return self.drift_detected, drift_info
    
    def _extract_neighbor_labels(self, record: Dict[str, Any]) -> Optional[List[Any]]:
        """Extract neighbor labels from the record.
        
        Tries multiple sources in order of preference:
        1. 'classifier' object with 'get_last_neighbor_labels' method
        2. '_knn_neighbor_labels' key in the record
        
        Args:
            record: Dictionary with prediction data
        
        Returns:
            List of neighbor labels or None if not found
        """
        try:
            # Case 1: Classifier object with get_last_neighbor_labels method
            if 'classifier' in record and record['classifier'] is not None:
                if hasattr(record['classifier'], 'get_last_neighbor_labels'):
                    labels = record['classifier'].get_last_neighbor_labels()
                    if labels is not None and len(labels) > 0:
                        return labels
                # Direct access to _last_neighbor_labels as fallback
                if hasattr(record['classifier'], '_last_neighbor_labels'):
                    labels = record['classifier']._last_neighbor_labels
                    if labels is not None and len(labels) > 0:
                        return labels
            
            # Case 2: Neighbor labels directly in record
            if '_knn_neighbor_labels' in record:
                return record['_knn_neighbor_labels']
            
            # Log debug info about what's available in the record
            logger.debug(f"Could not find neighbor labels in record with keys: {list(record.keys())}")
            
            # If we have prediction and confidence, and they were directly extracted from neighbors
            # we can infer that all neighbors have the same label
            if 'prediction' in record and 'confidence' in record:
                # High confidence means neighbors are likely homogeneous
                if record['confidence'] > 0.9:
                    return [record['prediction']] * self.k_neighbors
            
            return None
            
        except Exception as e:
            logger.warning(f"Error extracting neighbor labels: {str(e)}")
            return None
    
    def check_for_drift(self) -> Tuple[bool, Optional[Dict[str, Any]]]:
        """Check if drift has been detected.
        
        Returns:
            Tuple of (drift_detected, drift_info)
        """
        if not self.drift_detected:
            return False, None
        
        drift_info = {
            'detector_type': 'NeighborDiversityDriftDetector',
            'metric': 'mismatched_knn_neighbors',
            'current_value': self.last_mismatched_count,
            'threshold': self.max_mismatched_neighbors_threshold,
            'k_neighbors': self.k_neighbors,
            'dominant_class': self.last_dominant_class,
            'neighbor_labels': self.last_neighbor_labels
        }
        
        return True, drift_info
    
    def reset(self) -> None:
        """Reset the detector state.
        
        This resets the drift flag but keeps any statistics.
        """
        self.drift_detected = False
        
        # Reset drift cooldown tracking
        self.in_cooldown_period = False
        self.samples_since_last_drift = 0
        
        logger.debug("NeighborDiversityDriftDetector reset")
    
    def get_state(self) -> Dict[str, Any]:
        """Get the current state of the detector.
        
        Returns:
            State dictionary
        """
        state = self._get_base_state()
        state.update({
            'k_neighbors': self.k_neighbors,
            'max_mismatched_neighbors_threshold': self.max_mismatched_neighbors_threshold,
            'last_mismatched_count': self.last_mismatched_count,
            'last_dominant_class': self.last_dominant_class,
            'last_neighbor_labels': self.last_neighbor_labels
        })
        return state
    
    def set_state(self, state: Dict[str, Any]) -> None:
        """Restore the detector state.
        
        Args:
            state: Previously saved state dictionary
        """
        self._set_base_state(state)
        self.k_neighbors = state.get('k_neighbors', self.k_neighbors)
        self.max_mismatched_neighbors_threshold = state.get('max_mismatched_neighbors_threshold', self.max_mismatched_neighbors_threshold)
        self.last_mismatched_count = state.get('last_mismatched_count', 0)
        self.last_dominant_class = state.get('last_dominant_class')
        self.last_neighbor_labels = state.get('last_neighbor_labels', [])