from typing import Any, Dict, List, Optional, Tuple, Union
import numpy as np
import random
from collections import Counter, defaultdict
import math

from tinylcm.utils.logging import setup_logger

logger = setup_logger(__name__)


class CondensingAlgorithm:
    """Algorithms for condensing training samples while maintaining diversity.
    
    This class provides methods to reduce the size of a training set while
    preserving its distribution and decision boundaries, making it useful
    for memory-constrained devices.
    """
    
    @staticmethod
    def condense_samples(
        features: np.ndarray,
        labels: List[Any],
        max_size: Optional[int] = None,
        method: str = "class_balanced",
        use_numpy: bool = True,
        timestamps: Optional[List[float]] = None
    ) -> Tuple[np.ndarray, List[Any], List[float]]:
        """Condense training samples while maintaining diversity.
        
        Args:
            features: Matrix of feature vectors, shape (n_samples, n_features)
            labels: List of corresponding labels
            max_size: Maximum number of samples after condensing (if None, 
                     determined by method)
            method: Condensing method to use:
                   - "class_balanced": Equal number of samples per class
                   - "distance_based": Keep samples with maximum distance
                   - "random": Random selection
            use_numpy: Whether to use NumPy for calculations
            timestamps: Optional list of timestamps for each sample
            
        Returns:
            Tuple of (condensed_features, condensed_labels, condensed_timestamps)
        """
        if len(features) != len(labels):
            raise ValueError(f"Number of features ({len(features)}) and labels ({len(labels)}) must match")
        
        if timestamps is None:
            timestamps = [0.0] * len(labels)
        elif len(timestamps) != len(labels):
            raise ValueError(f"Number of timestamps ({len(timestamps)}) must match number of samples ({len(labels)})")
        
        # If dataset is already smaller than max_size, return as is
        if max_size is not None and len(features) <= max_size:
            return features, labels, timestamps
        
        logger.debug(f"Condensing {len(features)} samples using {method} method")
        
        if method == "class_balanced":
            return CondensingAlgorithm._class_balanced_condensing(
                features, labels, timestamps, max_size, use_numpy
            )
        elif method == "distance_based":
            return CondensingAlgorithm._distance_based_condensing(
                features, labels, timestamps, max_size, use_numpy
            )
        elif method == "random":
            return CondensingAlgorithm._random_condensing(
                features, labels, timestamps, max_size
            )
        else:
            raise ValueError(f"Unsupported condensing method: {method}")
    
    @staticmethod
    def _class_balanced_condensing(
        features: np.ndarray,
        labels: List[Any],
        timestamps: List[float],
        max_size: Optional[int] = None,
        use_numpy: bool = True
    ) -> Tuple[np.ndarray, List[Any], List[float]]:
        """Condense samples by keeping an equal number of samples per class.
        
        Args:
            features: Matrix of feature vectors
            labels: List of corresponding labels
            timestamps: List of timestamps for each sample
            max_size: Maximum number of samples after condensing
            use_numpy: Whether to use NumPy for calculations
            
        Returns:
            Tuple of (condensed_features, condensed_labels, condensed_timestamps)
        """
        # Group samples by class
        class_samples = defaultdict(list)
        
        for i, (feature, label, timestamp) in enumerate(zip(features, labels, timestamps)):
            class_samples[label].append((i, timestamp))
        
        # If max_size is not specified, keep all samples
        if max_size is None:
            max_size = len(features)
        
        # Number of classes
        n_classes = len(class_samples)
        
        # Samples per class
        samples_per_class = max(1, max_size // n_classes)
        
        # Selected indices
        selected_indices = []
        
        for label, samples in class_samples.items():
            # Sort samples by timestamp (descending) to prioritize newer samples
            samples.sort(key=lambda x: -x[1])
            
            # Select up to samples_per_class samples
            selected = samples[:samples_per_class]
            selected_indices.extend([idx for idx, _ in selected])
        
        # If we still have room, add more samples evenly
        remaining = max_size - len(selected_indices)
        if remaining > 0:
            # Collect samples not yet selected
            remaining_samples = []
            for label, samples in class_samples.items():
                # Skip samples already selected
                samples = [(idx, ts) for idx, ts in samples if idx not in selected_indices]
                remaining_samples.extend([(idx, ts, label) for idx, ts in samples])
            
            # Sort by timestamp (descending)
            remaining_samples.sort(key=lambda x: -x[1])
            
            # Add remaining samples
            for idx, _, _ in remaining_samples[:remaining]:
                selected_indices.append(idx)
        
        # Extract selected samples
        condensed_features = [features[i] for i in selected_indices]
        condensed_labels = [labels[i] for i in selected_indices]
        condensed_timestamps = [timestamps[i] for i in selected_indices]
        
        # Convert to numpy array if needed
        if use_numpy:
            condensed_features = np.array(condensed_features)
        
        logger.debug(f"Class-balanced condensing: {len(features)} -> {len(condensed_features)} samples")
        
        return condensed_features, condensed_labels, condensed_timestamps
    
    @staticmethod
    def _distance_based_condensing(
        features: np.ndarray,
        labels: List[Any],
        timestamps: List[float],
        max_size: Optional[int] = None,
        use_numpy: bool = True
    ) -> Tuple[np.ndarray, List[Any], List[float]]:
        """Condense samples by keeping those with maximum distance between them.
        
        This uses a greedy approach to select samples that maximize the diversity
        of the condensed dataset.
        
        Args:
            features: Matrix of feature vectors
            labels: List of corresponding labels
            timestamps: List of timestamps for each sample
            max_size: Maximum number of samples after condensing
            use_numpy: Whether to use NumPy for calculations
            
        Returns:
            Tuple of (condensed_features, condensed_labels, condensed_timestamps)
        """
        # If max_size is not specified, keep all samples
        if max_size is None:
            max_size = len(features)
        
        # If there are fewer samples than max_size, return all
        if len(features) <= max_size:
            return features, labels, timestamps
        
        # Group samples by class
        class_samples = defaultdict(list)
        
        for i, (feature, label, timestamp) in enumerate(zip(features, labels, timestamps)):
            class_samples[label].append((i, timestamp))
        
        # Selected indices
        selected_indices = []
        
        # Select one sample from each class (newest)
        for label, samples in class_samples.items():
            # Sort samples by timestamp (descending) to prioritize newer samples
            samples.sort(key=lambda x: -x[1])
            
            # Select the newest sample
            selected_indices.append(samples[0][0])
        
        # Compute distances function
        def compute_distance(idx1, idx2):
            """Compute Euclidean distance between two feature vectors."""
            if use_numpy:
                return np.sqrt(np.sum((features[idx1] - features[idx2]) ** 2))
            else:
                return math.sqrt(sum((features[idx1][j] - features[idx2][j]) ** 2 
                               for j in range(len(features[idx1]))))
        
        # While we haven't selected enough samples
        while len(selected_indices) < max_size:
            # Find the sample with maximum minimum distance to selected samples
            max_min_distance = -1
            max_idx = -1
            
            for i in range(len(features)):
                # Skip if already selected
                if i in selected_indices:
                    continue
                
                # Compute minimum distance to selected samples
                min_distance = float('inf')
                for sel_idx in selected_indices:
                    distance = compute_distance(i, sel_idx)
                    min_distance = min(min_distance, distance)
                
                # Update if this sample has a larger minimum distance
                if min_distance > max_min_distance:
                    max_min_distance = min_distance
                    max_idx = i
            
            # Add the sample with maximum minimum distance
            if max_idx != -1:
                selected_indices.append(max_idx)
            else:
                # This shouldn't happen, but just in case
                break
        
        # Extract selected samples
        condensed_features = [features[i] for i in selected_indices]
        condensed_labels = [labels[i] for i in selected_indices]
        condensed_timestamps = [timestamps[i] for i in selected_indices]
        
        # Convert to numpy array if needed
        if use_numpy:
            condensed_features = np.array(condensed_features)
        
        logger.debug(f"Distance-based condensing: {len(features)} -> {len(condensed_features)} samples")
        
        return condensed_features, condensed_labels, condensed_timestamps
    
    @staticmethod
    def _random_condensing(
        features: np.ndarray,
        labels: List[Any],
        timestamps: List[float],
        max_size: Optional[int] = None
    ) -> Tuple[np.ndarray, List[Any], List[float]]:
        """Condense samples by random selection.
        
        Args:
            features: Matrix of feature vectors
            labels: List of corresponding labels
            timestamps: List of timestamps for each sample
            max_size: Maximum number of samples after condensing
            
        Returns:
            Tuple of (condensed_features, condensed_labels, condensed_timestamps)
        """
        # If max_size is not specified, keep half of the samples
        if max_size is None:
            max_size = len(features) // 2
        
        # If there are fewer samples than max_size, return all
        if len(features) <= max_size:
            return features, labels, timestamps
        
        # Indices of all samples
        indices = list(range(len(features)))
        
        # Randomly select max_size indices
        selected_indices = random.sample(indices, max_size)
        
        # Extract selected samples
        condensed_features = [features[i] for i in selected_indices]
        condensed_labels = [labels[i] for i in selected_indices]
        condensed_timestamps = [timestamps[i] for i in selected_indices]
        
        # If features is a numpy array, convert condensed_features to numpy array
        if isinstance(features, np.ndarray):
            condensed_features = np.array(condensed_features)
        
        logger.debug(f"Random condensing: {len(features)} -> {len(condensed_features)} samples")
        
        return condensed_features, condensed_labels, condensed_timestamps