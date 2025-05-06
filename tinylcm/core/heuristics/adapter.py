"""
HeuristicAdapter module for TinyLCM.

This module analyzes samples in the QuarantineBuffer to identify emerging patterns
and generate potential labels for cautious on-device adaptation without requiring
external validation.
"""

import numpy as np
import time
from typing import Any, Dict, List, Tuple, Callable, Optional, Set
from collections import defaultdict

from tinylcm.utils.logging import setup_logger
from tinylcm.core.quarantine.buffer import QuarantineBuffer
from tinylcm.core.data_structures import FeatureSample

logger = setup_logger(__name__)


class HeuristicAdapter:
    """Component for heuristic adaptation based on quarantined samples.
    
    The HeuristicAdapter analyzes samples in the QuarantineBuffer to identify
    patterns and generate potential labels. It follows a simple algorithm:
    
    1. Group samples by their original prediction
    2. For each group, calculate intra-group feature dispersion
    3. If a group is sufficiently coherent (low dispersion), assign potential labels
    4. Select representative samples for adaptation
    
    This enables cautious on-device adaptation when ground truth labels
    are not available.
    """
    
    def __init__(
        self,
        quarantine_buffer: QuarantineBuffer,
        min_cluster_size: int = 5,
        variance_threshold: float = 0.1,
        k_representatives: int = 3,
        use_numpy: bool = True
    ):
        """Initialize the heuristic adapter.
        
        Args:
            quarantine_buffer: Buffer containing quarantined samples
            min_cluster_size: Minimum size of a cluster to consider it valid
            variance_threshold: Maximum allowed variance within a cluster
            k_representatives: Number of representative samples to select per cluster
            use_numpy: Whether to use NumPy for calculations
        """
        self.quarantine_buffer = quarantine_buffer
        self.min_cluster_size = min_cluster_size
        self.variance_threshold = variance_threshold
        self.k_representatives = k_representatives
        self.use_numpy = use_numpy
        
        # Tracking state
        self.created_labels = set()
        self.new_label_counter = 0
        
        logger.debug(
            f"Initialized HeuristicAdapter with min_cluster_size={min_cluster_size}, "
            f"variance_threshold={variance_threshold}"
        )
    
    def process_buffer(self) -> List[Dict[str, Any]]:
        """Process samples in the quarantine buffer.
        
        This implements the main algorithm for identifying patterns and
        generating potential labels.
        
        Returns:
            List of adaptation actions, each containing:
            - sample_id: ID of the sample
            - features: Feature vector
            - potential_label: Suggested potential label
            - reason: Reason for the adaptation
        """
        # Get all samples from quarantine buffer
        buffer_data = self.quarantine_buffer.get_data()
        
        # Skip if buffer is empty
        if not buffer_data:
            return []
        
        # Group samples by original prediction
        groups = self._group_by_prediction(buffer_data)
        
        # Process each group
        adaptation_actions = []
        processed_sample_ids = []
        
        for original_label, samples in groups.items():
            # Skip small groups
            if len(samples) < self.min_cluster_size:
                continue
            
            # Calculate intra-group dispersion
            dispersion = self._calculate_dispersion(samples)
            
            # Check if group is coherent enough
            if dispersion <= self.variance_threshold:
                # Generate potential label
                potential_label = self._generate_potential_label(original_label)
                
                # Select representative samples
                representatives = self._select_representatives(samples)
                
                for sample in representatives:
                    # Create adaptation action
                    action = {
                        'sample_id': sample.sample_id,
                        'features': sample.features,
                        'potential_label': potential_label,
                        'original_label': original_label,
                        'reason': f"Low dispersion cluster (dispersion={dispersion:.4f})"
                    }
                    
                    adaptation_actions.append(action)
                    processed_sample_ids.append(sample.sample_id)
                
                logger.info(
                    f"Identified potential new label '{potential_label}' "
                    f"for {len(representatives)} samples (dispersion={dispersion:.4f})"
                )
        
        # Mark processed samples
        if processed_sample_ids:
            self.quarantine_buffer.mark_as_processed(processed_sample_ids)
        
        return adaptation_actions
    
    def _group_by_prediction(self, buffer_data: List[Dict[str, Any]]) -> Dict[Any, List[FeatureSample]]:
        """Group samples by their original prediction.
        
        Args:
            buffer_data: List of buffer entries
            
        Returns:
            Dictionary mapping predictions to lists of samples
        """
        groups = defaultdict(list)
        
        for entry in buffer_data:
            # Skip processed samples
            if entry['processed']:
                continue
                
            sample = entry['sample']
            prediction = sample.prediction
            
            groups[prediction].append(sample)
        
        return groups
    
    def _calculate_dispersion(self, samples: List[FeatureSample]) -> float:
        """Calculate feature dispersion within a group of samples.
        
        Uses average pairwise distance or variance around centroid.
        
        Args:
            samples: List of samples in the group
            
        Returns:
            Dispersion score (lower is more coherent)
        """
        # Extract feature vectors
        if self.use_numpy and np:
            features = np.array([sample.features for sample in samples])
            
            # Calculate centroid
            centroid = np.mean(features, axis=0)
            
            # Calculate variance around centroid (average squared distance)
            squared_distances = np.sum((features - centroid) ** 2, axis=1)
            dispersion = np.mean(squared_distances)
            
            return float(dispersion)
        else:
            # Pure Python implementation for devices without NumPy
            feature_list = [sample.features for sample in samples]
            
            # Calculate centroid
            n_features = len(feature_list[0])
            centroid = [0.0] * n_features
            
            for feature_vector in feature_list:
                for i in range(n_features):
                    centroid[i] += feature_vector[i]
            
            for i in range(n_features):
                centroid[i] /= len(feature_list)
            
            # Calculate variance around centroid
            squared_distances = []
            
            for feature_vector in feature_list:
                squared_distance = sum((feature_vector[i] - centroid[i]) ** 2 for i in range(n_features))
                squared_distances.append(squared_distance)
            
            dispersion = sum(squared_distances) / len(squared_distances)
            
            return dispersion
    
    def _generate_potential_label(self, original_label: Any) -> str:
        """Generate a potential label for a new cluster.
        
        Args:
            original_label: Original prediction
            
        Returns:
            Potential label string
        """
        # Increment counter
        self.new_label_counter += 1
        
        # Create new label
        potential_label = f"{original_label}_drift_variant_{self.new_label_counter}"
        
        # Track created label
        self.created_labels.add(potential_label)
        
        return potential_label
    
    def _select_representatives(self, samples: List[FeatureSample]) -> List[FeatureSample]:
        """Select representative samples from a cluster.
        
        Chooses samples closest to the centroid.
        
        Args:
            samples: List of samples in the cluster
            
        Returns:
            List of representative samples
        """
        # Limit to k_representatives or the number of samples, whichever is smaller
        k = min(self.k_representatives, len(samples))
        
        if self.use_numpy and np:
            # Extract feature vectors
            features = np.array([sample.features for sample in samples])
            
            # Calculate centroid
            centroid = np.mean(features, axis=0)
            
            # Calculate distances to centroid
            distances = np.sqrt(np.sum((features - centroid) ** 2, axis=1))
            
            # Get indices of k samples closest to centroid
            closest_indices = np.argsort(distances)[:k]
            
            # Return corresponding samples
            return [samples[i] for i in closest_indices]
        else:
            # Pure Python implementation
            feature_list = [sample.features for sample in samples]
            
            # Calculate centroid
            n_features = len(feature_list[0])
            centroid = [0.0] * n_features
            
            for feature_vector in feature_list:
                for i in range(n_features):
                    centroid[i] += feature_vector[i]
            
            for i in range(n_features):
                centroid[i] /= len(feature_list)
            
            # Calculate distances to centroid
            distances = []
            
            for i, feature_vector in enumerate(feature_list):
                distance = sum((feature_vector[j] - centroid[j]) ** 2 for j in range(n_features)) ** 0.5
                distances.append((i, distance))
            
            # Sort by distance
            distances.sort(key=lambda x: x[1])
            
            # Take k closest
            closest_indices = [idx for idx, _ in distances[:k]]
            
            # Return corresponding samples
            return [samples[i] for i in closest_indices]
    
    def provide_feedback_callback(
        self, on_adaptation_fn: Callable[[str, Any, Any], None]
    ) -> None:
        """Set callback function to provide feedback to pipeline.
        
        Args:
            on_adaptation_fn: Function to call with (sample_id, features, potential_label)
        """
        # Process buffer
        adaptation_actions = self.process_buffer()
        
        # Skip if no actions
        if not adaptation_actions:
            return
        
        # Call callback for each action
        for action in adaptation_actions:
            sample_id = action['sample_id']
            features = action['features']
            potential_label = action['potential_label']
            
            # Provide feedback to pipeline
            on_adaptation_fn(sample_id, features, potential_label)
        
        # Remove processed samples from buffer
        self.quarantine_buffer.remove_processed()
    
    def get_statistics(self) -> Dict[str, Any]:
        """Get statistics about the heuristic adapter.
        
        Returns:
            Dictionary with adapter statistics
        """
        return {
            "min_cluster_size": self.min_cluster_size,
            "variance_threshold": self.variance_threshold,
            "k_representatives": self.k_representatives,
            "created_labels": list(self.created_labels),
            "new_label_counter": self.new_label_counter
        }