"""
HeuristicAdapter module for TinyLCM.

This module analyzes samples in the QuarantineBuffer to identify emerging patterns
and generate potential labels for cautious on-device adaptation without requiring
external validation.
"""

import numpy as np
import time
import math
from typing import Any, Dict, List, Tuple, Callable, Optional, Set
from collections import defaultdict

from tinylcm.utils.logging import setup_logger
from tinylcm.core.quarantine.buffer import QuarantineBuffer
from tinylcm.core.data_structures import FeatureSample, AdaptationAction, AdaptationEvent

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
        
    def store_pre_adaptation_metrics(self) -> None:
        """Store metrics before adaptation for later comparison.
        
        This method captures the current performance metrics before applying
        heuristic adaptations, for later comparison to determine if the
        adaptation was beneficial.
        """
        self.pre_adaptation_metrics = {
            "timestamp": time.time(),
            "samples_in_quarantine": len(self.quarantine_buffer.get_data()),
            "buffer_stats": self.quarantine_buffer.get_statistics()
        }
        logger.debug("Stored pre-adaptation metrics")
        
    def apply_pseudo_labels(self, min_confidence: float = 0.7) -> List[FeatureSample]:
        """Apply pseudo-labels to quarantined samples and prepare them for adaptation.
        
        Args:
            min_confidence: Minimum confidence threshold for pseudo-labeling
            
        Returns:
            List of feature samples with pseudo-labels applied
        """
        # Process the quarantine buffer to identify candidate samples
        adaptation_actions = self.process_buffer()
        
        if not adaptation_actions:
            logger.debug("No adaptation actions identified")
            return []
            
        # Convert adaptation actions to feature samples
        adapted_samples = []
        for action in adaptation_actions:
            if "confidence" in action and action["confidence"] < min_confidence:
                logger.debug(f"Skipping low confidence action: {action['confidence']:.2f} < {min_confidence}")
                continue
                
            # Create a feature sample with the pseudo-label
            sample = FeatureSample(
                features=action["features"],
                label=action["potential_label"],
                prediction=action["original_label"],
                sample_id=action["sample_id"],
                metadata={
                    "adaptation_type": "heuristic",
                    "adaptation_reason": action["reason"],
                    "adaptation_time": time.time()
                }
            )
            adapted_samples.append(sample)
            
        logger.info(f"Applied pseudo-labels to {len(adapted_samples)} samples")
        return adapted_samples
        
    def create_adaptation_event(self, adapted_samples: List[FeatureSample]) -> AdaptationEvent:
        """Create an adaptation event from the adapted samples.
        
        Args:
            adapted_samples: List of samples that were adapted
            
        Returns:
            AdaptationEvent object with details about the adaptation
        """
        # Create an adaptation event
        event = AdaptationEvent(
            event_type="heuristic",
            samples_added=len(adapted_samples),
            samples_removed=0,
            drift_detected=True,
            metadata={
                "quarantine_stats": self.quarantine_buffer.get_statistics(),
                "adaptation_time": time.time(),
                "heuristic_params": {
                    "min_cluster_size": self.min_cluster_size,
                    "variance_threshold": self.variance_threshold,
                    "k_representatives": self.k_representatives
                }
            }
        )
        
        # Add performance metrics if available
        if hasattr(self, "pre_adaptation_metrics"):
            event.metadata["pre_adaptation_metrics"] = self.pre_adaptation_metrics
            
        return event
        
    def get_samples_for_adaptation(self, min_confidence: float = 0.7) -> List[FeatureSample]:
        """Get samples that are ready for adaptation.
        
        This method performs additional filtering beyond apply_pseudo_labels
        to ensure only high-quality samples with sufficient confidence are
        used for adaptation. It applies stability and coherence checks.
        
        Args:
            min_confidence: Minimum confidence threshold for adaptation
            
        Returns:
            List of feature samples ready for adaptation
        """
        # First get candidate samples with pseudo-labels
        candidate_samples = self.apply_pseudo_labels(min_confidence=min_confidence)
        
        if not candidate_samples:
            return []
            
        # Apply additional quality filters
        filtered_samples = []
        
        # Group samples by label to check coherence
        samples_by_label = {}
        for sample in candidate_samples:
            if sample.label not in samples_by_label:
                samples_by_label[sample.label] = []
            samples_by_label[sample.label].append(sample)
        
        # Check coherence within each label group
        for label, samples in samples_by_label.items():
            if len(samples) < 2:
                # Need at least two samples to evaluate coherence
                continue
                
            # Extract features as numpy arrays for calculation
            features_list = [s.features for s in samples]
            if all(isinstance(f, np.ndarray) for f in features_list):
                # Calculate centroid of this group
                centroid = np.mean(features_list, axis=0)
                
                # Calculate distances from centroid
                if self.use_numpy:
                    distances = [np.linalg.norm(f - centroid) for f in features_list]
                else:
                    # Pure Python distance calculation
                    distances = []
                    for f in features_list:
                        squared_dist = sum((a - b) ** 2 for a, b in zip(f, centroid))
                        distances.append(math.sqrt(squared_dist))
                
                # Calculate average distance and standard deviation
                avg_distance = sum(distances) / len(distances)
                if len(distances) > 1:
                    variance = sum((d - avg_distance) ** 2 for d in distances) / len(distances)
                    std_dev = math.sqrt(variance)
                else:
                    std_dev = 0
                
                # Filter out samples that are too far from centroid (outliers)
                # Keep samples within 2 standard deviations from centroid
                for i, sample in enumerate(samples):
                    if distances[i] <= avg_distance + 2 * std_dev:
                        filtered_samples.append(sample)
            else:
                # If we can't do the calculation, include all samples
                filtered_samples.extend(samples)
        
        # Log filtering results
        filtered_out = len(candidate_samples) - len(filtered_samples)
        if filtered_out > 0:
            logger.info(f"Filtered out {filtered_out} samples, keeping {len(filtered_samples)} for adaptation")
        
        return filtered_samples
        
    def update_metrics(self, current_metrics: Dict[str, Any]) -> None:
        """Update metrics after adaptation for comparison.
        
        Args:
            current_metrics: Current metrics from the handler
        """
        self.post_adaptation_metrics = {
            "timestamp": time.time(),
            "metrics": current_metrics,
            "samples_in_quarantine": len(self.quarantine_buffer.get_data()),
            "buffer_stats": self.quarantine_buffer.get_statistics()
        }
        logger.debug("Updated post-adaptation metrics")
        
    def rollback_needed(self) -> bool:
        """Check if rollback is needed based on metrics comparison.
        
        Analyzes pre- and post-adaptation metrics to determine if the
        adaptation was beneficial or detrimental to model performance.
        
        Returns:
            True if rollback is recommended, False otherwise
        """
        # Can't determine rollback need if we don't have both metrics
        if not hasattr(self, "pre_adaptation_metrics") or not hasattr(self, "post_adaptation_metrics"):
            logger.debug("Cannot determine if rollback is needed: missing metrics")
            return False
            
        # Get pre and post metrics
        pre = self.pre_adaptation_metrics
        post = self.post_adaptation_metrics
        
        # Check confidence metrics if available in post metrics
        if "metrics" in post and "confidence" in post["metrics"]:
            # Check confidence degradation
            pre_confidence = pre.get("metrics", {}).get("confidence", {}).get("mean", 0.0)
            post_confidence = post["metrics"]["confidence"].get("mean", 0.0)
            
            # Significant confidence drop indicates potential rollback need
            if pre_confidence > 0 and post_confidence < pre_confidence * 0.9:
                logger.warning(f"Confidence dropped significantly: {pre_confidence:.2f} -> {post_confidence:.2f}")
                return True
                
        # Check prediction stability
        if "metrics" in post and "prediction_distribution" in post["metrics"]:
            pre_dist = pre.get("metrics", {}).get("prediction_distribution", {})
            post_dist = post["metrics"].get("prediction_distribution", {})
            
            # Significant change in prediction distribution indicates potential issues
            if pre_dist and post_dist:
                # Calculate Jensen-Shannon distance between distributions
                # (simplified here to just check for major class shifts)
                pre_classes = set(pre_dist.keys())
                post_classes = set(post_dist.keys())
                
                # Check if previously dominant classes disappeared
                for cls in pre_classes:
                    if cls in pre_dist and pre_dist[cls] > max(pre_dist.values()) * 0.8:
                        if cls not in post_dist or post_dist[cls] < pre_dist[cls] * 0.5:
                            logger.warning(f"Major class shift detected for {cls}")
                            return True
        
        # Check if quarantine size grew significantly after adaptation
        # (which could indicate the adaptation is causing more confusion)
        pre_quarantine = pre.get("samples_in_quarantine", 0)
        post_quarantine = post.get("samples_in_quarantine", 0)
        
        if pre_quarantine > 0 and post_quarantine > pre_quarantine * 1.5:
            logger.warning(f"Quarantine size grew significantly: {pre_quarantine} -> {post_quarantine}")
            return True
            
        logger.debug("Adaptation metrics indicate positive or neutral effect, no rollback needed")
        return False
    
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