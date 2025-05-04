from enum import Enum, auto
from typing import Dict, List, Any, Optional, Callable, Tuple, Set, Union
import numpy as np
import time
import threading
import json
import uuid
import copy
from collections import Counter, defaultdict
from pathlib import Path

from tinylcm.utils.logging import setup_logger
from tinylcm.utils.file_utils import ensure_directory_exists
from tinylcm.core.quarantine.buffer import (
    QuarantineBuffer,
    QuarantinedSample,
    QuarantineStatus
)
from tinylcm.core.data_structures import FeatureSample, AdaptationEvent

logger = setup_logger(__name__)


class HeuristicStrategy(Enum):
    """Strategies for heuristic adaptation."""
    FREQUENCY_BASED = auto()    # Assign labels based on frequency of similar features
    CLUSTER_BASED = auto()      # Use clustering to identify new classes
    CONFIDENCE_BASED = auto()   # Use samples with high confidence scores
    NEAR_BOUNDARY = auto()      # Add samples near decision boundaries
    HYBRID = auto()             # Combine multiple strategies
    CUSTOM = auto()             # Custom strategy defined by user-provided function


class ClusteringMethod(Enum):
    """Clustering methods for the CLUSTER_BASED strategy."""
    KMEANS = auto()             # K-means clustering
    DBSCAN = auto()             # DBSCAN clustering
    MEAN_SHIFT = auto()         # Mean shift clustering
    SIMPLE_DISTANCE = auto()    # Simple distance-based clustering (for resource constraints)


class AdaptationAction(Enum):
    """Actions to take during adaptation."""
    ADD_SAMPLES = auto()        # Add new samples to the classifier
    CREATE_NEW_CLASS = auto()   # Create a new class
    ADJUST_BOUNDARIES = auto()  # Adjust decision boundaries
    REJECT = auto()             # Reject adaptation (samples not useful)


class HeuristicAdapter:
    """Component for heuristic adaptation based on quarantined samples.
    
    This class analyzes quarantined samples that have not been validated
    externally and applies heuristic strategies to:
    
    1. Assign pseudo-labels to samples
    2. Identify potential new classes
    3. Select samples for model adaptation
    4. Track adaptation decisions and their effectiveness
    
    The adapter uses various strategies like clustering, frequency analysis,
    and confidence thresholds to make adaptation decisions when ground truth
    labels are not available.
    """
    
    def __init__(
        self,
        quarantine_buffer: QuarantineBuffer,
        strategy: HeuristicStrategy = HeuristicStrategy.HYBRID,
        min_cluster_size: int = 5,
        min_samples_for_adaptation: int = 10,
        confidence_threshold: float = 0.7,
        distance_threshold: float = 0.8,
        max_new_classes: int = 3,
        clustering_method: ClusteringMethod = ClusteringMethod.SIMPLE_DISTANCE,
        adaptation_cooldown: int = 100,  # Min samples between adaptation events
        log_dir: Optional[str] = None,
        custom_strategy: Optional[Callable[[List[QuarantinedSample]], List[Tuple[str, Any, float, AdaptationAction]]]] = None,
        use_numpy: bool = True
    ):
        """Initialize the heuristic adapter.
        
        Args:
            quarantine_buffer: Buffer containing quarantined samples
            strategy: Strategy for heuristic adaptation
            min_cluster_size: Minimum size of a cluster to consider it a new class
            min_samples_for_adaptation: Minimum samples needed before attempting adaptation
            confidence_threshold: Minimum confidence for confidence-based adaptation
            distance_threshold: Maximum distance for samples to be considered similar
            max_new_classes: Maximum number of new classes to create
            clustering_method: Method to use for clustering
            adaptation_cooldown: Minimum samples between adaptation attempts
            log_dir: Directory for logging adaptation decisions
            custom_strategy: Custom function for CUSTOM strategy
            use_numpy: Whether to use NumPy for calculations
        """
        self.quarantine_buffer = quarantine_buffer
        self.strategy = strategy
        self.min_cluster_size = min_cluster_size
        self.min_samples_for_adaptation = min_samples_for_adaptation
        self.confidence_threshold = confidence_threshold
        self.distance_threshold = distance_threshold
        self.max_new_classes = max_new_classes
        self.clustering_method = clustering_method
        self.adaptation_cooldown = adaptation_cooldown
        self.custom_strategy = custom_strategy
        self.use_numpy = use_numpy
        
        # Set up logging
        if log_dir:
            self.log_dir = Path(log_dir)
            ensure_directory_exists(self.log_dir)
        else:
            self.log_dir = None
        
        # Tracking state
        self.created_classes = set()
        self.adaptation_history = []
        self.samples_since_last_adaptation = 0
        self.total_adaptations = 0
        self.total_samples_processed = 0
        self.new_class_counter = 0
        
        # Class tracking
        self.known_classes = set()
        
        # State to track adaptation success
        self.last_adaptation_metrics = {}
        self.current_metrics = {}
        
        logger.debug(f"HeuristicAdapter initialized with strategy: {strategy.name}")
    
    def update_known_classes(self, classes: Set[Any]) -> None:
        """Update the set of known classes.
        
        Args:
            classes: Set of class labels known to the classifier
        """
        self.known_classes = set(classes)
        logger.debug(f"Updated known classes: {self.known_classes}")
    
    def should_attempt_adaptation(self) -> bool:
        """Check if we should attempt adaptation based on current state.
        
        Returns:
            True if adaptation should be attempted, False otherwise
        """
        # Check cooldown period
        if self.samples_since_last_adaptation < self.adaptation_cooldown:
            return False
        
        # Check if we have enough pending samples
        pending_samples = self.quarantine_buffer.get_pending_samples()
        if len(pending_samples) < self.min_samples_for_adaptation:
            return False
        
        return True
    
    def process_pending_samples(self) -> Optional[List[Tuple[QuarantinedSample, AdaptationAction]]]:
        """Process pending samples and determine adaptation actions.
        
        This is the main method that analyzes quarantined samples and decides
        how to adapt the model based on the selected strategy.
        
        Returns:
            List of tuples (sample, action) or None if no adaptation needed
        """
        if not self.should_attempt_adaptation():
            return None
        
        # Get pending samples
        pending_samples = self.quarantine_buffer.get_pending_samples()
        
        # Apply the selected strategy
        if self.strategy == HeuristicStrategy.FREQUENCY_BASED:
            return self._apply_frequency_strategy(pending_samples)
        
        elif self.strategy == HeuristicStrategy.CLUSTER_BASED:
            return self._apply_clustering_strategy(pending_samples)
        
        elif self.strategy == HeuristicStrategy.CONFIDENCE_BASED:
            return self._apply_confidence_strategy(pending_samples)
        
        elif self.strategy == HeuristicStrategy.NEAR_BOUNDARY:
            return self._apply_boundary_strategy(pending_samples)
        
        elif self.strategy == HeuristicStrategy.HYBRID:
            return self._apply_hybrid_strategy(pending_samples)
        
        elif self.strategy == HeuristicStrategy.CUSTOM and self.custom_strategy:
            return self._apply_custom_strategy(pending_samples)
        
        # Default: no adaptation
        return None
    
    def _apply_frequency_strategy(
        self, samples: List[QuarantinedSample]
    ) -> List[Tuple[QuarantinedSample, AdaptationAction]]:
        """Apply frequency-based adaptation strategy.
        
        This strategy looks for frequently occurring similar samples and
        assigns the most common prediction as the pseudo-label.
        
        Args:
            samples: List of quarantined samples
            
        Returns:
            List of tuples (sample, action)
        """
        # Group samples by prediction
        samples_by_prediction = defaultdict(list)
        for sample in samples:
            samples_by_prediction[sample.prediction].append(sample)
        
        # Process each prediction group
        results = []
        
        for prediction, pred_samples in samples_by_prediction.items():
            # Skip small groups
            if len(pred_samples) < self.min_cluster_size:
                continue
            
            # Check consistency in features for this prediction
            if self._feature_consistency(pred_samples) > 0.7:
                # Consistent group - use as-is
                for sample in pred_samples:
                    # Update sample status
                    self.quarantine_buffer.update_sample_status(
                        sample_id=sample.sample_id,
                        status=QuarantineStatus.PSEUDO_LABELED,
                        pseudo_label=prediction,
                        pseudo_label_confidence=0.8
                    )
                    
                    # Add to results
                    results.append((sample, AdaptationAction.ADD_SAMPLES))
        
        return results
    
    def _feature_consistency(self, samples: List[QuarantinedSample]) -> float:
        """Calculate feature consistency for a group of samples.
        
        Args:
            samples: List of samples to calculate consistency for
            
        Returns:
            Consistency score (0-1)
        """
        if len(samples) < 2:
            return 1.0
        
        # Extract features
        features_list = [sample.features for sample in samples]
        
        # Calculate pairwise distances
        total_distance = 0.0
        count = 0
        
        for i in range(len(features_list)):
            for j in range(i+1, len(features_list)):
                dist = self._calculate_distance(features_list[i], features_list[j])
                total_distance += dist
                count += 1
        
        # Calculate average distance
        avg_distance = total_distance / count if count > 0 else 0.0
        
        # Convert to consistency score (1 - normalized distance)
        return max(0.0, 1.0 - (avg_distance / self.distance_threshold))
    
    def _calculate_distance(self, a: np.ndarray, b: np.ndarray) -> float:
        """Calculate distance between two feature vectors.
        
        Args:
            a: First feature vector
            b: Second feature vector
            
        Returns:
            Distance between vectors
        """
        if self.use_numpy and isinstance(a, np.ndarray) and isinstance(b, np.ndarray):
            # Euclidean distance
            return float(np.sqrt(np.sum((a - b) ** 2)))
        else:
            # Fallback for non-NumPy
            return float(sum((x - y) ** 2 for x, y in zip(a, b)) ** 0.5)
    
    def _apply_clustering_strategy(
        self, samples: List[QuarantinedSample]
    ) -> List[Tuple[QuarantinedSample, AdaptationAction]]:
        """Apply clustering-based adaptation strategy.
        
        This strategy uses clustering to identify potential new classes
        in the data.
        
        Args:
            samples: List of quarantined samples
            
        Returns:
            List of tuples (sample, action)
        """
        # Extract features
        feature_array = np.array([sample.features for sample in samples])
        
        # Check if we should create a new class
        if self._is_new_class_candidate(feature_array, samples):
            return self._create_new_class(samples)
        else:
            # Try to assign existing classes using simple distance clustering
            return self._assign_to_existing_classes(samples)
    
    def _is_new_class_candidate(
        self, features: np.ndarray, samples: List[QuarantinedSample]
    ) -> bool:
        """Determine if a set of samples represents a candidate for a new class.
        
        Args:
            features: Array of feature vectors
            samples: Corresponding samples
            
        Returns:
            True if samples likely represent a new class
        """
        # Check if we've reached the maximum allowed new classes
        if len(self.created_classes) >= self.max_new_classes:
            return False
        
        # Check if samples are tightly clustered
        if not self._is_tight_cluster(features):
            return False
        
        # Check if predictions are diverse (suggesting confusion)
        predictions = [sample.prediction for sample in samples]
        unique_predictions = set(predictions)
        if len(unique_predictions) < 2:
            return False
        
        return True
    
    def _is_tight_cluster(self, features: np.ndarray) -> bool:
        """Check if features form a tight cluster.
        
        Args:
            features: Array of feature vectors
            
        Returns:
            True if features form a tight cluster
        """
        if len(features) < self.min_cluster_size:
            return False
        
        # Calculate centroid
        centroid = np.mean(features, axis=0)
        
        # Calculate distances to centroid
        distances = np.sqrt(np.sum((features - centroid) ** 2, axis=1))
        
        # Check if most points are close to centroid
        close_points = np.sum(distances < self.distance_threshold)
        ratio = close_points / len(features)
        
        return ratio > 0.8
    
    def _create_new_class(
        self, samples: List[QuarantinedSample]
    ) -> List[Tuple[QuarantinedSample, AdaptationAction]]:
        """Create a new class from samples.
        
        Args:
            samples: List of samples to assign to new class
            
        Returns:
            List of tuples (sample, action)
        """
        # Generate new class label
        self.new_class_counter += 1
        new_class = f"potential_class_{self.new_class_counter}"
        
        # Track created class
        self.created_classes.add(new_class)
        
        # Select representative samples (most central in feature space)
        if len(samples) > self.min_cluster_size:
            # Use only the most central samples
            central_samples = self._find_central_samples(
                samples, n=self.min_cluster_size
            )
        else:
            central_samples = samples
        
        # Update sample status
        results = []
        for sample in central_samples:
            self.quarantine_buffer.update_sample_status(
                sample_id=sample.sample_id,
                status=QuarantineStatus.PSEUDO_LABELED,
                pseudo_label=new_class,
                pseudo_label_confidence=0.7
            )
            
            # Add to results
            results.append((sample, AdaptationAction.CREATE_NEW_CLASS))
        
        logger.info(f"Created new class {new_class} with {len(central_samples)} samples")
        
        return results
    
    def _find_central_samples(
        self, samples: List[QuarantinedSample], n: int
    ) -> List[QuarantinedSample]:
        """Find the n most central samples in feature space.
        
        Args:
            samples: List of samples
            n: Number of central samples to find
            
        Returns:
            List of n most central samples
        """
        # Extract features
        features = np.array([sample.features for sample in samples])
        
        # Calculate centroid
        centroid = np.mean(features, axis=0)
        
        # Calculate distances to centroid
        distances = np.sqrt(np.sum((features - centroid) ** 2, axis=1))
        
        # Sort samples by distance to centroid
        sorted_indices = np.argsort(distances)
        
        # Take n closest samples
        return [samples[i] for i in sorted_indices[:n]]
    
    def _assign_to_existing_classes(
        self, samples: List[QuarantinedSample]
    ) -> List[Tuple[QuarantinedSample, AdaptationAction]]:
        """Assign samples to existing classes using distance-based clustering.
        
        Args:
            samples: List of samples to assign
            
        Returns:
            List of tuples (sample, action)
        """
        # Group samples by current prediction
        samples_by_prediction = defaultdict(list)
        for sample in samples:
            samples_by_prediction[sample.prediction].append(sample)
        
        # Process each prediction group
        results = []
        
        for prediction, pred_samples in samples_by_prediction.items():
            # Skip small groups
            if len(pred_samples) < 3:
                continue
            
            # Samples with high confidence
            high_conf_samples = [s for s in pred_samples if s.confidence and s.confidence > 0.7]
            
            if high_conf_samples:
                # Use samples with consistent, high confidence predictions
                for sample in high_conf_samples:
                    self.quarantine_buffer.update_sample_status(
                        sample_id=sample.sample_id,
                        status=QuarantineStatus.PSEUDO_LABELED,
                        pseudo_label=prediction,
                        pseudo_label_confidence=min(1.0, float(sample.confidence) * 1.2 if sample.confidence else 0.8)
                    )
                    
                    # Add to results
                    results.append((sample, AdaptationAction.ADJUST_BOUNDARIES))
        
        return results
    
    def _apply_confidence_strategy(
        self, samples: List[QuarantinedSample]
    ) -> List[Tuple[QuarantinedSample, AdaptationAction]]:
        """Apply confidence-based adaptation strategy.
        
        This strategy selects samples with high confidence predictions.
        
        Args:
            samples: List of quarantined samples
            
        Returns:
            List of tuples (sample, action)
        """
        # Filter samples with high confidence
        high_conf_samples = [
            s for s in samples
            if s.confidence and s.confidence > self.confidence_threshold
        ]
        
        results = []
        
        for sample in high_conf_samples:
            self.quarantine_buffer.update_sample_status(
                sample_id=sample.sample_id,
                status=QuarantineStatus.PSEUDO_LABELED,
                pseudo_label=sample.prediction,
                pseudo_label_confidence=sample.confidence
            )
            
            # Add to results
            results.append((sample, AdaptationAction.ADD_SAMPLES))
        
        return results
    
    def _apply_boundary_strategy(
        self, samples: List[QuarantinedSample]
    ) -> List[Tuple[QuarantinedSample, AdaptationAction]]:
        """Apply boundary-focused adaptation strategy.
        
        This strategy focuses on samples near decision boundaries.
        
        Args:
            samples: List of quarantined samples
            
        Returns:
            List of tuples (sample, action)
        """
        # Find samples with moderate confidence (near boundaries)
        boundary_samples = [
            s for s in samples
            if s.confidence and 0.4 < s.confidence < 0.7
        ]
        
        # Group by prediction
        by_prediction = defaultdict(list)
        for sample in boundary_samples:
            by_prediction[sample.prediction].append(sample)
        
        results = []
        
        # For each prediction class, select a few representatives
        for prediction, pred_samples in by_prediction.items():
            if len(pred_samples) < 3:
                continue
            
            # Use a subset of samples to adjust boundaries
            for sample in pred_samples[:5]:
                self.quarantine_buffer.update_sample_status(
                    sample_id=sample.sample_id,
                    status=QuarantineStatus.PSEUDO_LABELED,
                    pseudo_label=prediction,
                    pseudo_label_confidence=0.5
                )
                
                # Add to results
                results.append((sample, AdaptationAction.ADJUST_BOUNDARIES))
        
        return results
    
    def _apply_hybrid_strategy(
        self, samples: List[QuarantinedSample]
    ) -> List[Tuple[QuarantinedSample, AdaptationAction]]:
        """Apply hybrid adaptation strategy.
        
        This strategy combines multiple approaches for more robust adaptation.
        
        Args:
            samples: List of quarantined samples
            
        Returns:
            List of tuples (sample, action)
        """
        results = []
        
        # Step 1: Try to identify new classes
        if len(self.created_classes) < self.max_new_classes:
            cluster_results = self._apply_clustering_strategy(samples)
            if cluster_results:
                # Successfully created a new class
                # Mark used samples as processed
                used_sample_ids = set(sample.sample_id for sample, _ in cluster_results)
                samples = [s for s in samples if s.sample_id not in used_sample_ids]
                
                results.extend(cluster_results)
        
        # Step 2: Use high confidence samples for existing classes
        if samples:
            conf_results = self._apply_confidence_strategy(samples)
            if conf_results:
                # Mark used samples as processed
                used_sample_ids = set(sample.sample_id for sample, _ in conf_results)
                samples = [s for s in samples if s.sample_id not in used_sample_ids]
                
                results.extend(conf_results)
        
        # Step 3: For remaining samples, try frequency-based approach
        if samples:
            freq_results = self._apply_frequency_strategy(samples)
            if freq_results:
                results.extend(freq_results)
        
        return results
    
    def _apply_custom_strategy(
        self, samples: List[QuarantinedSample]
    ) -> List[Tuple[QuarantinedSample, AdaptationAction]]:
        """Apply custom adaptation strategy.
        
        Args:
            samples: List of quarantined samples
            
        Returns:
            List of tuples (sample, action)
        """
        if not self.custom_strategy:
            return []
        
        # Call custom strategy
        custom_results = self.custom_strategy(samples)
        
        results = []
        for sample_id, pseudo_label, confidence, action in custom_results:
            # Get the sample
            sample = self.quarantine_buffer.get_sample(sample_id)
            if not sample:
                continue
            
            # Update sample status
            self.quarantine_buffer.update_sample_status(
                sample_id=sample_id,
                status=QuarantineStatus.PSEUDO_LABELED,
                pseudo_label=pseudo_label,
                pseudo_label_confidence=confidence
            )
            
            # Add to results
            results.append((sample, action))
        
        return results
    
    def apply_pseudo_labels(
        self, min_confidence: float = 0.5
    ) -> List[Tuple[QuarantinedSample, AdaptationAction]]:
        """Apply pseudo-labels to quarantined samples.
        
        This is the main method to call when you want to apply heuristic
        adaptation. It:
        1. Processes pending samples
        2. Filters samples that meet confidence threshold
        3. Prepares them for adaptation
        
        Args:
            min_confidence: Minimum confidence threshold for pseudo-labels
            
        Returns:
            List of samples with adaptation actions
        """
        # Reset samples since last adaptation
        self.samples_since_last_adaptation = 0
        
        # Process pending samples
        adaptation_samples = self.process_pending_samples()
        
        if not adaptation_samples:
            return []
        
        # Log adaptation event
        self.adaptation_history.append({
            "timestamp": time.time(),
            "strategy": self.strategy.name,
            "num_samples": len(adaptation_samples),
            "new_classes": list(self.created_classes),
            "actions": {action.name: 0 for action in AdaptationAction}
        })
        
        # Update action counts
        for _, action in adaptation_samples:
            self.adaptation_history[-1]["actions"][action.name] += 1
        
        # Update stats
        self.total_adaptations += 1
        
        return adaptation_samples
    
    def get_samples_for_adaptation(
        self, min_confidence: float = 0.7
    ) -> List[FeatureSample]:
        """Get samples ready for model adaptation.
        
        Args:
            min_confidence: Minimum confidence threshold for pseudo-labels
            
        Returns:
            List of FeatureSample objects ready for adaptation
        """
        # Get pseudo-labeled samples with sufficient confidence
        samples = self.quarantine_buffer.get_pseudo_labeled_samples(min_confidence)
        
        # Convert to FeatureSample for AdaptivePipeline
        return [sample.create_feature_sample() for sample in samples]
    
    def track_sample(self) -> None:
        """Track a new sample (for cooldown purposes)."""
        self.samples_since_last_adaptation += 1
        self.total_samples_processed += 1
    
    def update_metrics(self, metrics: Dict[str, Any]) -> None:
        """Update current performance metrics.
        
        This is used to track if adaptations have positive or negative effects.
        
        Args:
            metrics: Dictionary of current performance metrics
        """
        self.current_metrics = copy.deepcopy(metrics)
    
    def evaluate_adaptation_impact(self) -> Dict[str, Any]:
        """Evaluate the impact of the last adaptation.
        
        This compares current metrics with metrics before the last adaptation.
        
        Returns:
            Dictionary with impact evaluation
        """
        if not self.last_adaptation_metrics or not self.current_metrics:
            return {"impact": "unknown", "details": {}}
        
        # Compare metrics
        impact = {}
        
        # Compare confidence
        if "confidence" in self.current_metrics and "confidence" in self.last_adaptation_metrics:
            curr_conf = self.current_metrics["confidence"].get("mean", 0)
            prev_conf = self.last_adaptation_metrics["confidence"].get("mean", 0)
            impact["confidence_change"] = curr_conf - prev_conf
        
        # Compare latency
        if "latency" in self.current_metrics and "latency" in self.last_adaptation_metrics:
            curr_lat = self.current_metrics["latency"].get("mean", 0)
            prev_lat = self.last_adaptation_metrics["latency"].get("mean", 0)
            impact["latency_change"] = curr_lat - prev_lat
        
        # Overall assessment
        if "confidence_change" in impact:
            if impact["confidence_change"] > 0.05:
                impact["impact"] = "positive"
            elif impact["confidence_change"] < -0.1:
                impact["impact"] = "negative"
            else:
                impact["impact"] = "neutral"
        else:
            impact["impact"] = "unknown"
        
        return impact
    
    def store_pre_adaptation_metrics(self) -> None:
        """Store current metrics before adaptation for later comparison."""
        self.last_adaptation_metrics = copy.deepcopy(self.current_metrics)
    
    def rollback_needed(self) -> bool:
        """Determine if adaptation rollback is needed based on metrics.
        
        Returns:
            True if adaptation had negative impact and rollback is recommended
        """
        impact = self.evaluate_adaptation_impact()
        
        # Recommend rollback if adaptation had clear negative impact
        return impact.get("impact") == "negative" and impact.get("confidence_change", 0) < -0.15
    
    def log_adaptation_event(self, event: Dict[str, Any]) -> None:
        """Log an adaptation event to disk.
        
        Args:
            event: Dictionary with adaptation event details
        """
        if not self.log_dir:
            return
        
        try:
            # Create timestamp
            timestamp = int(time.time())
            
            # Create log file
            log_file = self.log_dir / f"heuristic_adaptation_{timestamp}.json"
            
            # Save event
            with open(log_file, 'w') as f:
                json.dump(event, f, indent=2)
                
            logger.debug(f"Logged adaptation event to {log_file}")
        except Exception as e:
            logger.error(f"Error logging adaptation event: {str(e)}")
    
    def get_statistics(self) -> Dict[str, Any]:
        """Get statistics about the heuristic adapter.
        
        Returns:
            Dictionary of statistics
        """
        # Basic stats
        stats = {
            "total_adaptations": self.total_adaptations,
            "total_samples_processed": self.total_samples_processed,
            "created_classes": list(self.created_classes),
            "known_classes": list(self.known_classes),
            "samples_since_last_adaptation": self.samples_since_last_adaptation,
            "strategy": self.strategy.name,
            "clustering_method": self.clustering_method.name,
            "adaptation_cooldown": self.adaptation_cooldown,
            "min_cluster_size": self.min_cluster_size,
            "min_samples_for_adaptation": self.min_samples_for_adaptation,
            "confidence_threshold": self.confidence_threshold,
            "distance_threshold": self.distance_threshold,
            "max_new_classes": self.max_new_classes
        }
        
        # Add recent adaptations
        if self.adaptation_history:
            stats["recent_adaptations"] = self.adaptation_history[-min(5, len(self.adaptation_history)):]
        
        # Add adaptation impact if available
        impact = self.evaluate_adaptation_impact()
        if impact:
            stats["last_adaptation_impact"] = impact
        
        return stats
    
    def create_adaptation_event(
        self,
        adapted_samples: List[Tuple[QuarantinedSample, AdaptationAction]]
    ) -> AdaptationEvent:
        """Create an adaptation event from adapted samples.
        
        Args:
            adapted_samples: List of samples and actions
            
        Returns:
            AdaptationEvent object
        """
        # Count samples by action type
        action_counts = Counter([action.name for _, action in adapted_samples])
        
        # Determine main event type
        if AdaptationAction.CREATE_NEW_CLASS.name in action_counts:
            event_type = "heuristic_new_class"
        elif AdaptationAction.ADD_SAMPLES.name in action_counts:
            event_type = "heuristic_add_samples"
        elif AdaptationAction.ADJUST_BOUNDARIES.name in action_counts:
            event_type = "heuristic_adjust_boundaries"
        else:
            event_type = "heuristic_adaptation"
        
        # Create event
        event = AdaptationEvent(
            event_type=event_type,
            timestamp=time.time(),
            samples_added=len(adapted_samples),
            samples_removed=0,
            drift_detected=True,  # Heuristic adaptation is always triggered by drift
            metadata={
                "strategy": self.strategy.name,
                "action_counts": action_counts,
                "new_classes": list(self.created_classes),
                "heuristic_adaptation": True,
                "confidence_threshold": self.confidence_threshold
            }
        )
        
        return event