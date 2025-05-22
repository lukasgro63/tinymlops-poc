import math
import time
from collections import Counter, deque
from enum import Enum
from typing import Any, Callable, Dict, List, Optional, Tuple, Union

import numpy as np

from tinylcm.core.base import AdaptiveComponent
from tinylcm.core.classifiers.base import BaseAdaptiveClassifier
from tinylcm.utils.logging import setup_logger

logger = setup_logger(__name__)


class DistanceMetric(str, Enum):
    """Enumeration of supported distance metrics."""
    EUCLIDEAN = "euclidean"  # d(a,b) = sqrt(sum((a_i - b_i)^2))
    MANHATTAN = "manhattan"  # d(a,b) = sum(|a_i - b_i|)
    COSINE = "cosine"        # d(a,b) = 1 - (a·b)/(||a||·||b||)


class LightweightKNN(BaseAdaptiveClassifier, AdaptiveComponent):
    """A lightweight k-Nearest Neighbors classifier optimized for resource-constrained devices.
    
    This implementation is specifically designed for edge devices like Raspberry Pi Zero,
    with options to operate without NumPy for maximum compatibility with restricted
    environments.
    
    Features:
    - Multiple distance metrics: euclidean, manhattan, cosine
    - Optional NumPy acceleration (`use_numpy=False` for pure Python)
    - Sample timestamps for tie-breaking (prioritize newer samples)
    - Maximum sample limit to prevent memory growth
    - Incremental learning with efficient single-sample addition
    - Sample-level management for adaptive memory usage
    """
    
    def __init__(
        self,
        k: int = 5,
        distance_metric: Union[str, DistanceMetric] = DistanceMetric.EUCLIDEAN,
        max_samples: int = 100,
        use_numpy: bool = True,
        weight_by_distance: bool = False,
        tie_break_by_time: bool = True,
        simple_confidence: bool = False
    ):
        """Initialize the k-NN classifier.
        
        Args:
            k: Number of neighbors to consider
            distance_metric: Distance metric to use (euclidean, manhattan, cosine)
            max_samples: Maximum number of training samples to store
            use_numpy: Whether to use NumPy for calculations (faster but uses more memory)
            weight_by_distance: Whether to weight votes by inverse distance
            tie_break_by_time: Whether to break ties by preferring more recent samples
            simple_confidence: Whether to use simple vote-based confidence (recommended for KNN Distance Monitor)
        """
        self.k = k
        
        # Convert string to enum if needed
        if isinstance(distance_metric, str):
            try:
                distance_metric = DistanceMetric(distance_metric.lower())
            except ValueError:
                logger.warning(f"Unknown distance metric: {distance_metric}, using euclidean instead")
                distance_metric = DistanceMetric.EUCLIDEAN
                
        self.distance_metric = distance_metric
        self.max_samples = max_samples
        self.use_numpy = use_numpy
        self.weight_by_distance = weight_by_distance
        self.tie_break_by_time = tie_break_by_time
        self.simple_confidence = simple_confidence
        
        # Training data
        self.X_train = []  # Feature vectors
        self.y_train = []  # Labels
        self.timestamps = []  # Timestamps for each sample (for tie-breaking)
        self._classes = set()  # Set of unique classes
        
        # Stored information for drift detection
        self._last_distances = []  # Most recent neighbor distances (for drift detection)
        self._last_neighbor_labels = []  # Most recent neighbor labels (for drift detection)
        
        # Performance metrics
        self._total_prediction_time = 0.0
        self._total_predictions = 0
        
        logger.debug(
            f"Initialized LightweightKNN with k={k}, "
            f"metric={distance_metric}, max_samples={max_samples}, use_numpy={use_numpy}"
        )
    
    def fit(self, features: np.ndarray, labels: List[Any], timestamps: Optional[List[float]] = None) -> None:
        """Train the classifier on the provided data.
        
        This overwrites any existing training data.
        
        Args:
            features: Matrix of feature vectors, shape (n_samples, n_features)
            labels: List of corresponding labels
            timestamps: Optional list of timestamps for each sample
        """
        if len(features) != len(labels):
            raise ValueError(f"Number of features ({len(features)}) and labels ({len(labels)}) must match")
            
        if timestamps is not None and len(timestamps) != len(features):
            raise ValueError(f"Number of timestamps ({len(timestamps)}) must match number of samples")
            
        # Clear existing data
        self.X_train = []
        self.y_train = []
        self.timestamps = []
        self._classes = set()
        
        # Generate timestamps for new data if not provided
        if timestamps is None:
            current_time = time.time()
            timestamps = [current_time] * len(labels)
        
        # Add new data
        self._add_samples(features, labels, timestamps)
        
        logger.debug(f"Fitted {len(self.X_train)} samples with {len(self._classes)} classes")
    
    def add_sample(self, feature: np.ndarray, label: Any, timestamp: Optional[float] = None) -> None:
        """Add a single sample to the classifier.
        
        This is more efficient than calling incremental_fit for a single sample.
        
        Args:
            feature: Feature vector, shape (n_features,)
            label: Label for the sample
            timestamp: Optional timestamp for the sample
        """
        # Update set of unique classes
        self._classes.add(label)
        
        # Use current time if timestamp not provided
        if timestamp is None:
            timestamp = time.time()
        
        # If we're using NumPy, convert feature to numpy array if not already
        if self.use_numpy and not isinstance(feature, np.ndarray):
            feature = np.array(feature)
        
        # Check if we're at max capacity
        if len(self.X_train) >= self.max_samples:
            # Find oldest sample
            oldest_idx = self.timestamps.index(min(self.timestamps))
            
            # Replace oldest with new sample
            self.X_train[oldest_idx] = feature
            self.y_train[oldest_idx] = label
            self.timestamps[oldest_idx] = timestamp
        else:
            # Add new sample
            self.X_train.append(feature)
            self.y_train.append(label)
            self.timestamps.append(timestamp)
    
    def incremental_fit(self, features: np.ndarray, labels: List[Any], timestamps: Optional[List[float]] = None) -> None:
        """Incrementally train the classifier on new data.
        
        This preserves existing training data and adds new samples.
        If the total number of samples exceeds max_samples, the oldest
        samples will be removed.
        
        Args:
            features: Matrix of feature vectors, shape (n_samples, n_features)
            labels: List of corresponding labels
            timestamps: Optional list of timestamps for each sample
        """
        if len(features) != len(labels):
            raise ValueError(f"Number of features ({len(features)}) and labels ({len(labels)}) must match")
            
        if timestamps is not None and len(timestamps) != len(features):
            raise ValueError(f"Number of timestamps ({len(timestamps)}) must match number of samples")
            
        # Generate timestamps for new data if not provided
        if timestamps is None:
            current_time = time.time()
            timestamps = [current_time] * len(labels)
        
        # Add new data
        self._add_samples(features, labels, timestamps)
        
        logger.debug(f"Incrementally fitted {len(features)} new samples, total {len(self.X_train)}")
    
    def _add_samples(self, features: np.ndarray, labels: List[Any], timestamps: List[float]) -> None:
        """Add samples to the training set, respecting max_samples limit.
        
        Args:
            features: Feature vectors to add
            labels: Labels to add
            timestamps: Timestamps for each sample
        """
        # Update set of unique classes
        self._classes.update(labels)
        
        # If we're using NumPy, convert features to numpy array if not already
        if self.use_numpy and not isinstance(features, np.ndarray):
            features = np.array(features)
        
        # Add each sample
        for i, (feature, label, timestamp) in enumerate(zip(features, labels, timestamps)):
            # Check if we're at max capacity
            if len(self.X_train) >= self.max_samples:
                # Find oldest sample
                oldest_idx = self.timestamps.index(min(self.timestamps))
                
                # Replace oldest with new sample
                self.X_train[oldest_idx] = feature
                self.y_train[oldest_idx] = label
                self.timestamps[oldest_idx] = timestamp
            else:
                # Add new sample
                self.X_train.append(feature)
                self.y_train.append(label)
                self.timestamps.append(timestamp)
    
    def predict(self, features: np.ndarray) -> List[Any]:
        """Predict labels for the provided feature vectors.
        
        IMPORTANT: This is a synchronous, blocking operation. The execution time
        depends on the number of training samples, the dimensionality of the feature
        vectors, and whether NumPy acceleration is enabled (use_numpy).
        
        For resource-constrained devices like the Raspberry Pi Zero, consider:
        1. Using a smaller value for max_samples to limit the number of distance calculations
        2. Using simpler distance metrics like 'manhattan' instead of 'euclidean'
        3. Setting use_numpy=True on devices that support it for faster computation
        4. Batch processing inputs when possible
        
        Args:
            features: Matrix of feature vectors, shape (n_samples, n_features)
            
        Returns:
            List of predicted labels
        """
        start_time = time.time()
        
        if not self.X_train:
            logger.warning("Classifier has not been trained yet - returning default predictions")
            # Return a default prediction to avoid raising an exception
            if self.use_numpy and isinstance(features, np.ndarray):
                if len(features.shape) == 1:
                    # Single sample
                    return ["unknown"]
                else:
                    # Multiple samples
                    return ["unknown"] * features.shape[0]
            else:
                # If not using numpy, assume a single sample if features is not a list
                # or return a list of "unknown" for each sample if features is a list
                if isinstance(features, list):
                    return ["unknown"] * len(features)
                else:
                    return ["unknown"]
        
        predictions = []
        
        # Ensure features is a numpy array if we're using numpy
        if self.use_numpy and not isinstance(features, np.ndarray):
            features = np.array(features)
        
        # Handle case where features is a single sample
        if self.use_numpy and len(features.shape) == 1:
            features = np.expand_dims(features, axis=0)
        elif not self.use_numpy and not isinstance(features[0], (list, np.ndarray)):
            features = [features]
        
        # Make predictions for each sample
        for feature in features:
            # Find k nearest neighbors
            neighbors = self._find_neighbors(feature)
            
            # Get the most common class among neighbors
            if self.weight_by_distance:
                # Weight votes by inverse distance
                class_votes = {}
                for neighbor, distance in neighbors:
                    weight = 1.0 / (distance + 1e-6)  # Avoid division by zero
                    class_votes[self.y_train[neighbor]] = class_votes.get(self.y_train[neighbor], 0) + weight
                
                # Find class with maximum weight
                predicted_class = max(class_votes.items(), key=lambda x: x[1])[0]
            else:
                # Simple majority voting
                votes = [self.y_train[neighbor[0]] for neighbor in neighbors]
                predicted_class = Counter(votes).most_common(1)[0][0]
            
            predictions.append(predicted_class)
        
        # Update performance metrics
        self._total_prediction_time += time.time() - start_time
        self._total_predictions += len(features)
        
        return predictions
    
    def predict_proba(self, features: np.ndarray) -> np.ndarray:
        """Predict probability distributions over classes for the provided features.

        IMPORTANT: This is a synchronous, blocking operation. Similar to predict(),
        the execution time depends on the number of training samples, the dimensionality
        of the feature vectors, and whether NumPy acceleration is enabled (use_numpy).

        This method supports two confidence calculation modes:
        1. Simple confidence (simple_confidence=True): Traditional vote counting - recommended for KNN Distance Monitor
        2. Enhanced confidence (simple_confidence=False): Distance-weighted confidence - for advanced use cases

        Args:
            features: Matrix of feature vectors, shape (n_samples, n_features)

        Returns:
            Matrix of class probabilities, shape (n_samples, n_classes)
        """
        if self.simple_confidence:
            return self._predict_proba_simple(features)
        else:
            return self._predict_proba_enhanced(features)
        start_time = time.time()

        if not self.X_train:
            logger.warning("Classifier has not been trained yet - returning uniform probabilities")
            # Handle the case where classifier has not been trained yet
            # Return a probability distribution with a single "unknown" class with prob 1.0

            # Determine the number of samples in the input
            if self.use_numpy and isinstance(features, np.ndarray):
                if len(features.shape) == 1:
                    n_samples = 1
                else:
                    n_samples = features.shape[0]
            else:
                if isinstance(features, list):
                    n_samples = len(features)
                else:
                    n_samples = 1

            # Create an array with a single class probability
            if self.use_numpy:
                return np.ones((n_samples, 1))
            else:
                return [[1.0] for _ in range(n_samples)]

        # Ensure classes are ordered consistently
        classes = sorted(list(self._classes))

        # Ensure features is a numpy array if we're using numpy
        if self.use_numpy and not isinstance(features, np.ndarray):
            features = np.array(features)

        # Handle case where features is a single sample
        if self.use_numpy and len(features.shape) == 1:
            features = np.expand_dims(features, axis=0)
        elif not self.use_numpy and not isinstance(features[0], (list, np.ndarray)):
            features = [features]

        # Initialize probabilities array
        n_samples = len(features)
        n_classes = len(classes)

        if self.use_numpy:
            probas = np.zeros((n_samples, n_classes))
        else:
            probas = [[0.0 for _ in range(n_classes)] for _ in range(n_samples)]

        # Distance-based confidence scaling factor
        # - Higher values make confidence drop more quickly with distance
        # - Lower values make confidence more resilient to distance changes
        confidence_scaling = 10  # Dramatically increased to make confidence more sensitive to distance with small feature dimensions

        # Track stats for logging
        prediction_stats = {
            "avg_distances": [],
            "max_probas": []
        }

        # Compute probabilities for each sample
        for i, feature in enumerate(features):
            # Find k nearest neighbors
            neighbors = self._find_neighbors(feature)

            if not neighbors:
                # No neighbors found (shouldn't happen unless k > num samples)
                # Return uniform distribution
                for j in range(n_classes):
                    probas[i][j] = 1.0 / n_classes
                continue

            # Calculate average distance to neighbors for diagnostic purposes
            avg_distance = sum(dist for _, dist in neighbors) / len(neighbors)
            prediction_stats["avg_distances"].append(avg_distance)

            # Enhanced confidence calculation:
            # 1. Get vote counts for each class
            # 2. Calculate average distance for samples of each class
            # 3. Scale confidence based on both vote count and distance

            # Count votes and distances by class
            class_stats = {}
            for neighbor_idx, distance in neighbors:
                label = self.y_train[neighbor_idx]
                if label not in class_stats:
                    class_stats[label] = {"count": 0, "total_distance": 0.0}
                class_stats[label]["count"] += 1
                class_stats[label]["total_distance"] += distance

            # Calculate adjusted probabilities
            total_adjusted_vote = 0.0
            adjusted_votes = {}

            for label, vote_info in class_stats.items():
                # Base vote probability
                vote_prob = vote_info["count"] / self.k

                # Average distance for this class
                class_avg_distance = vote_info["total_distance"] / vote_info["count"]

                # Distance factor: higher distances = lower confidence
                # Use a smooth decay function (sigmoid-like)
                distance_factor = 1.0 / (1.0 + confidence_scaling * class_avg_distance)

                # Final adjusted vote includes both vote count and distance
                adjusted_vote = vote_prob * distance_factor
                adjusted_votes[label] = adjusted_vote
                total_adjusted_vote += adjusted_vote

            # Normalize to get probabilities
            if total_adjusted_vote > 0:
                for j, c in enumerate(classes):
                    if c in adjusted_votes:
                        probas[i][j] = adjusted_votes[c] / total_adjusted_vote
                    else:
                        probas[i][j] = 0.0
            else:
                # Fallback to simple vote counting if adjustment fails
                votes = Counter([self.y_train[neighbor[0]] for neighbor in neighbors])
                for j, c in enumerate(classes):
                    probas[i][j] = votes.get(c, 0) / self.k

            # Track max probability for logging
            if self.use_numpy:
                max_proba = np.max(probas[i])
            else:
                max_proba = max(probas[i])
            prediction_stats["max_probas"].append(max_proba)

        # Periodically log stats
        if hasattr(self, '_prediction_count'):
            self._prediction_count += n_samples
        else:
            self._prediction_count = n_samples

        # Always log for debugging confidence values
        avg_distance = sum(prediction_stats["avg_distances"]) / len(prediction_stats["avg_distances"])
        avg_max_proba = sum(prediction_stats["max_probas"]) / len(prediction_stats["max_probas"])

        # Log at INFO level to ensure it appears in logs
        logger.info(f"KNN ENHANCED CONFIDENCE STATS (v2) - Avg distance: {avg_distance:.6f}, Avg max probability: {avg_max_proba:.6f}")
        if n_samples == 1:  # Only show detailed probas for single predictions to avoid log spam
            logger.info(f"DETAILED PROBAS: {probas[0] if self.use_numpy else probas[0]}")

        # Update performance metrics
        self._total_prediction_time += time.time() - start_time
        self._total_predictions += len(features)

        return np.array(probas) if self.use_numpy else probas
    
    def _predict_proba_simple(self, features: np.ndarray) -> np.ndarray:
        """Simple vote-based confidence calculation (recommended for KNN Distance Monitor).
        
        This method uses traditional KNN vote counting without distance weighting,
        resulting in interpretable confidence values that represent the fraction
        of neighbors agreeing on the prediction.
        
        Args:
            features: Matrix of feature vectors, shape (n_samples, n_features)
            
        Returns:
            Matrix of class probabilities, shape (n_samples, n_classes)
        """
        start_time = time.time()
        
        if not self.X_train:
            logger.warning("Classifier has not been trained yet - returning uniform probabilities")
            # Determine the number of samples in the input
            if self.use_numpy and isinstance(features, np.ndarray):
                if len(features.shape) == 1:
                    n_samples = 1
                else:
                    n_samples = features.shape[0]
            else:
                if isinstance(features, list):
                    n_samples = len(features)
                else:
                    n_samples = 1
            
            # Create an array with a single class probability
            if self.use_numpy:
                return np.ones((n_samples, 1))
            else:
                return [[1.0] for _ in range(n_samples)]
        
        # Ensure classes are ordered consistently
        classes = sorted(list(self._classes))
        
        # Ensure features is a numpy array if we're using numpy
        if self.use_numpy and not isinstance(features, np.ndarray):
            features = np.array(features)
        
        # Handle case where features is a single sample
        if self.use_numpy and len(features.shape) == 1:
            features = np.expand_dims(features, axis=0)
        elif not self.use_numpy and not isinstance(features[0], (list, np.ndarray)):
            features = [features]
        
        # Initialize probabilities array
        n_samples = len(features)
        n_classes = len(classes)
        
        if self.use_numpy:
            probas = np.zeros((n_samples, n_classes))
        else:
            probas = [[0.0 for _ in range(n_classes)] for _ in range(n_samples)]
        
        # Track stats for logging
        prediction_stats = {
            "avg_distances": [],
            "max_probas": []
        }
        
        # Compute probabilities for each sample
        for i, feature in enumerate(features):
            # Find k nearest neighbors
            neighbors = self._find_neighbors(feature)
            
            if not neighbors:
                # No neighbors found (shouldn't happen unless k > num samples)
                # Return uniform distribution
                for j in range(n_classes):
                    probas[i][j] = 1.0 / n_classes
                continue
            
            # Calculate average distance for diagnostic purposes
            avg_distance = sum(dist for _, dist in neighbors) / len(neighbors)
            prediction_stats["avg_distances"].append(avg_distance)
            
            # Simple vote counting (traditional KNN)
            votes = {}
            for neighbor_idx, _ in neighbors:
                label = self.y_train[neighbor_idx]
                votes[label] = votes.get(label, 0) + 1
            
            # Convert votes to probabilities
            for j, class_label in enumerate(classes):
                probas[i][j] = votes.get(class_label, 0) / self.k
            
            # Track max probability for logging
            if self.use_numpy:
                max_proba = np.max(probas[i])
            else:
                max_proba = max(probas[i])
            prediction_stats["max_probas"].append(max_proba)
        
        # Log statistics
        if prediction_stats["avg_distances"]:
            avg_distance = sum(prediction_stats["avg_distances"]) / len(prediction_stats["avg_distances"])
            avg_max_proba = sum(prediction_stats["max_probas"]) / len(prediction_stats["max_probas"])
            
            logger.info(f"KNN SIMPLE CONFIDENCE STATS - Avg distance: {avg_distance:.6f}, Avg max probability: {avg_max_proba:.6f}")
            if n_samples == 1:  # Only show detailed probas for single predictions
                logger.info(f"DETAILED PROBAS: {probas[0] if self.use_numpy else probas[0]}")
        
        # Update performance metrics
        self._total_prediction_time += time.time() - start_time
        self._total_predictions += len(features)
        
        return np.array(probas) if self.use_numpy else probas
    
    def _predict_proba_enhanced(self, features: np.ndarray) -> np.ndarray:
        """Enhanced distance-weighted confidence calculation (original method).
        
        This method scales votes by distance, so samples that are further away
        result in lower confidence scores, even if the predicted class is the same.
        This is computationally more intensive but may be useful for advanced use cases.
        
        Args:
            features: Matrix of feature vectors, shape (n_samples, n_features)
            
        Returns:
            Matrix of class probabilities, shape (n_samples, n_classes)
        """
        start_time = time.time()

        if not self.X_train:
            logger.warning("Classifier has not been trained yet - returning uniform probabilities")
            # Handle the case where classifier has not been trained yet
            # Return a probability distribution with a single "unknown" class with prob 1.0

            # Determine the number of samples in the input
            if self.use_numpy and isinstance(features, np.ndarray):
                if len(features.shape) == 1:
                    n_samples = 1
                else:
                    n_samples = features.shape[0]
            else:
                if isinstance(features, list):
                    n_samples = len(features)
                else:
                    n_samples = 1

            # Create an array with a single class probability
            if self.use_numpy:
                return np.ones((n_samples, 1))
            else:
                return [[1.0] for _ in range(n_samples)]

        # Ensure classes are ordered consistently
        classes = sorted(list(self._classes))

        # Ensure features is a numpy array if we're using numpy
        if self.use_numpy and not isinstance(features, np.ndarray):
            features = np.array(features)

        # Handle case where features is a single sample
        if self.use_numpy and len(features.shape) == 1:
            features = np.expand_dims(features, axis=0)
        elif not self.use_numpy and not isinstance(features[0], (list, np.ndarray)):
            features = [features]

        # Initialize probabilities array
        n_samples = len(features)
        n_classes = len(classes)

        if self.use_numpy:
            probas = np.zeros((n_samples, n_classes))
        else:
            probas = [[0.0 for _ in range(n_classes)] for _ in range(n_samples)]

        # Distance-based confidence scaling factor
        # - Higher values make confidence drop more quickly with distance
        # - Lower values make confidence more resilient to distance changes
        confidence_scaling = 10  # Dramatically increased to make confidence more sensitive to distance with small feature dimensions

        # Track stats for logging
        prediction_stats = {
            "avg_distances": [],
            "max_probas": []
        }

        # Compute probabilities for each sample
        for i, feature in enumerate(features):
            # Find k nearest neighbors
            neighbors = self._find_neighbors(feature)

            if not neighbors:
                # No neighbors found (shouldn't happen unless k > num samples)
                # Return uniform distribution
                for j in range(n_classes):
                    probas[i][j] = 1.0 / n_classes
                continue

            # Calculate average distance to neighbors for diagnostic purposes
            avg_distance = sum(dist for _, dist in neighbors) / len(neighbors)
            prediction_stats["avg_distances"].append(avg_distance)

            # Enhanced confidence calculation:
            # 1. Get vote counts for each class
            # 2. Calculate average distance for samples of each class
            # 3. Scale confidence based on both vote count and distance

            # Count votes and distances by class
            class_stats = {}
            for neighbor_idx, distance in neighbors:
                label = self.y_train[neighbor_idx]
                if label not in class_stats:
                    class_stats[label] = {"count": 0, "total_distance": 0.0}
                class_stats[label]["count"] += 1
                class_stats[label]["total_distance"] += distance

            # Calculate adjusted probabilities
            total_adjusted_vote = 0.0
            adjusted_votes = {}

            for label, vote_info in class_stats.items():
                # Base vote probability
                vote_prob = vote_info["count"] / self.k

                # Average distance for this class
                class_avg_distance = vote_info["total_distance"] / vote_info["count"]

                # Distance factor: higher distances = lower confidence
                # Use a smooth decay function (sigmoid-like)
                distance_factor = 1.0 / (1.0 + confidence_scaling * class_avg_distance)

                # Final adjusted vote includes both vote count and distance
                adjusted_vote = vote_prob * distance_factor
                adjusted_votes[label] = adjusted_vote
                total_adjusted_vote += adjusted_vote

            # Normalize to get probabilities
            if total_adjusted_vote > 0:
                for j, c in enumerate(classes):
                    if c in adjusted_votes:
                        probas[i][j] = adjusted_votes[c] / total_adjusted_vote
                    else:
                        probas[i][j] = 0.0
            else:
                # Fallback to simple vote counting if adjustment fails
                votes = Counter([self.y_train[neighbor[0]] for neighbor in neighbors])
                for j, c in enumerate(classes):
                    probas[i][j] = votes.get(c, 0) / self.k

            # Track max probability for logging
            if self.use_numpy:
                max_proba = np.max(probas[i])
            else:
                max_proba = max(probas[i])
            prediction_stats["max_probas"].append(max_proba)

        # Periodically log stats
        if hasattr(self, '_prediction_count'):
            self._prediction_count += n_samples
        else:
            self._prediction_count = n_samples

        # Always log for debugging confidence values
        avg_distance = sum(prediction_stats["avg_distances"]) / len(prediction_stats["avg_distances"])
        avg_max_proba = sum(prediction_stats["max_probas"]) / len(prediction_stats["max_probas"])

        # Log at INFO level to ensure it appears in logs
        logger.info(f"KNN ENHANCED CONFIDENCE STATS (v2) - Avg distance: {avg_distance:.6f}, Avg max probability: {avg_max_proba:.6f}")
        if n_samples == 1:  # Only show detailed probas for single predictions to avoid log spam
            logger.info(f"DETAILED PROBAS: {probas[0] if self.use_numpy else probas[0]}")

        # Update performance metrics
        self._total_prediction_time += time.time() - start_time
        self._total_predictions += len(features)

        return np.array(probas) if self.use_numpy else probas
    
    def _find_neighbors(self, feature: np.ndarray) -> List[Tuple[int, float]]:
        """Find the k nearest neighbors for a feature vector.
        
        Args:
            feature: Feature vector to find neighbors for
            
        Returns:
            List of tuples (neighbor_index, distance)
        """
        # Check if there are any training samples
        if not self.X_train:
            logger.warning("Cannot find neighbors - no training data available")
            return []
            
        # Compute distances to all training samples
        distances = []
        
        for i, train_feature in enumerate(self.X_train):
            distance = self._calculate_distance(feature, train_feature)
            distances.append((i, distance, self.timestamps[i]))
        
        # Sort by distance (ascending)
        if self.tie_break_by_time:
            # Sort by distance, then by -timestamp (more recent first)
            distances.sort(key=lambda x: (x[1], -x[2]))
        else:
            # Sort by distance only
            distances.sort(key=lambda x: x[1])
        
        # Get the k nearest neighbors
        nearest_neighbors = [(idx, dist) for idx, dist, _ in distances[:self.k]]

        # Log detailed information about the nearest neighbors for debugging
        debug_str = "NEAREST NEIGHBORS DEBUG:\n"
        neighbors_debug = []
        for i, (idx, dist) in enumerate(nearest_neighbors):
            label = self.y_train[idx]
            neighbor_info = {
                "index": i+1,
                "label": label,
                "distance": dist
            }
            neighbors_debug.append(neighbor_info)
            debug_str += f"  Neighbor {i+1}: Label={label}, Distance={dist:.6f}\n"
        
        # Store in a thread local for operational logging to access
        if not hasattr(self, '_thread_local'):
            import threading
            self._thread_local = threading.local()
        
        # Only store the structured data for operational logging
        self._thread_local.neighbors_debug = neighbors_debug
        
        # Log debug info to console but don't store the string in thread_local
        logger.info(debug_str)
        
        # Store the distances and labels for drift detection to access directly
        self._last_distances = [dist for _, dist in nearest_neighbors]
        self._last_neighbor_labels = [self.y_train[idx] for idx, _ in nearest_neighbors]

        return nearest_neighbors
    
    def _calculate_distance(self, a: np.ndarray, b: np.ndarray) -> float:
        """Calculate the distance between two feature vectors.
        
        This method dispatches to the appropriate distance function based on
        the selected distance metric.
        
        Args:
            a: First feature vector
            b: Second feature vector
            
        Returns:
            Distance between the vectors
        """
        if self.distance_metric == DistanceMetric.EUCLIDEAN:
            return self._euclidean_distance(a, b)
        elif self.distance_metric == DistanceMetric.MANHATTAN:
            return self._manhattan_distance(a, b)
        elif self.distance_metric == DistanceMetric.COSINE:
            return self._cosine_distance(a, b)
        else:
            raise ValueError(f"Unsupported distance metric: {self.distance_metric}")
    
    def _euclidean_distance(self, a: np.ndarray, b: np.ndarray) -> float:
        """Calculate the Euclidean distance between two vectors.
        
        Euclidean distance is defined as:
        d(a,b) = sqrt(sum((a_i - b_i)^2))
        
        Args:
            a: First vector
            b: Second vector
            
        Returns:
            Euclidean distance
        """
        # For small feature dimensions, add a small base distance
        # to ensure distances are never exactly zero, which can lead to artificially high confidence
        base_distance = 0.01
        
        if self.use_numpy:
            raw_distance = np.sqrt(np.sum((a - b) ** 2))
            # Normalize distance by vector dimension for more consistent scaling
            # This helps when dealing with very small feature dimensions
            dimension_factor = max(1.0, np.sqrt(a.shape[0]) / 10.0)
            return base_distance + (raw_distance * dimension_factor)
        else:
            # Pure Python implementation (optimized for edge devices without NumPy)
            raw_distance = math.sqrt(sum((a_i - b_i) ** 2 for a_i, b_i in zip(a, b)))
            dimension_factor = max(1.0, math.sqrt(len(a)) / 10.0)
            return base_distance + (raw_distance * dimension_factor)
    
    def _manhattan_distance(self, a: np.ndarray, b: np.ndarray) -> float:
        """Calculate the Manhattan distance between two vectors.
        
        Manhattan distance is defined as:
        d(a,b) = sum(|a_i - b_i|)
        
        Args:
            a: First vector
            b: Second vector
            
        Returns:
            Manhattan distance
        """
        # Similar to euclidean distance, add a base distance and dimension scaling factor
        base_distance = 0.01
        
        if self.use_numpy:
            raw_distance = np.sum(np.abs(a - b))
            # Normalize by dimension for consistent scaling
            dimension_factor = max(1.0, a.shape[0] / 10.0)
            return base_distance + (raw_distance * dimension_factor)
        else:
            # Pure Python implementation (optimized for edge devices without NumPy)
            raw_distance = sum(abs(a_i - b_i) for a_i, b_i in zip(a, b))
            dimension_factor = max(1.0, len(a) / 10.0)
            return base_distance + (raw_distance * dimension_factor)
    
    def _cosine_distance(self, a: np.ndarray, b: np.ndarray) -> float:
        """Calculate the cosine distance between two vectors.
        
        Cosine distance is defined as:
        d(a,b) = 1 - (a·b)/(||a||·||b||)
        
        Args:
            a: First vector
            b: Second vector
            
        Returns:
            Cosine distance (0 = identical, 2 = opposite)
        """
        if self.use_numpy:
            # Calculate dot product
            dot_product = np.dot(a, b)
            
            # Calculate magnitudes
            magnitude_a = np.sqrt(np.sum(a ** 2))
            magnitude_b = np.sqrt(np.sum(b ** 2))
            
            # Calculate cosine similarity
            if magnitude_a > 0 and magnitude_b > 0:
                cosine_similarity = dot_product / (magnitude_a * magnitude_b)
                # Ensure it's within valid range due to floating point errors
                cosine_similarity = max(-1.0, min(1.0, cosine_similarity))
            else:
                cosine_similarity = 0.0
        else:
            # Pure Python implementation (optimized for edge devices without NumPy)
            
            # Calculate dot product
            dot_product = sum(a_i * b_i for a_i, b_i in zip(a, b))
            
            # Calculate magnitudes
            magnitude_a = math.sqrt(sum(a_i ** 2 for a_i in a))
            magnitude_b = math.sqrt(sum(b_i ** 2 for b_i in b))
            
            # Calculate cosine similarity
            if magnitude_a > 0 and magnitude_b > 0:
                cosine_similarity = dot_product / (magnitude_a * magnitude_b)
                # Ensure it's within valid range due to floating point errors
                cosine_similarity = max(-1.0, min(1.0, cosine_similarity))
            else:
                cosine_similarity = 0.0
        
        # Convert to cosine distance
        base_distance = 0.01
        raw_distance = 1.0 - cosine_similarity
        
        # For small feature dimensions, scale the distance
        # Small number of dimensions can lead to artificially high cosine similarity
        dimension_factor = max(1.0, min(5.0, 10.0 / max(1, len(a) if not self.use_numpy else a.shape[0])))
        
        return base_distance + (raw_distance * dimension_factor)
    
    def get_performance_metrics(self) -> Dict[str, float]:
        """Get performance metrics for this classifier.
        
        Returns:
            Dictionary containing performance metrics:
            - total_prediction_time: Total time spent in prediction
            - total_predictions: Number of predictions performed
            - avg_prediction_time: Average time per prediction
        """
        avg_time = 0.0
        if self._total_predictions > 0:
            avg_time = self._total_prediction_time / self._total_predictions
            
        return {
            "total_prediction_time": self._total_prediction_time,
            "total_predictions": self._total_predictions,
            "avg_prediction_time": avg_time,
            "sample_count": len(self.X_train),
            "class_count": len(self._classes)
        }
    
    def get_state(self) -> Dict[str, Any]:
        """Get the current state of the classifier.
        
        Returns:
            Dict containing the serializable state
        """
        # Convert features and classes to lists for serialization
        X_train_serialized = [x.tolist() if isinstance(x, np.ndarray) else x for x in self.X_train]
        
        return {
            "type": "LightweightKNN",
            "k": self.k,
            "distance_metric": self.distance_metric.value,
            "max_samples": self.max_samples,
            "use_numpy": self.use_numpy,
            "weight_by_distance": self.weight_by_distance,
            "tie_break_by_time": self.tie_break_by_time,
            "X_train": X_train_serialized,
            "y_train": self.y_train,
            "timestamps": self.timestamps,
            "classes": list(self._classes),
            "metrics": {
                "total_prediction_time": self._total_prediction_time,
                "total_predictions": self._total_predictions
            }
        }
    
    def set_state(self, state: Dict[str, Any]) -> None:
        """Restore the classifier's state.
        
        Args:
            state: Previously saved state dictionary
        """
        self.k = state.get("k", self.k)
        
        # Handle distance metric (which might be a string or enum)
        metric = state.get("distance_metric", self.distance_metric)
        if isinstance(metric, str):
            try:
                self.distance_metric = DistanceMetric(metric.lower())
            except ValueError:
                logger.warning(f"Unknown distance metric in state: {metric}, using current setting")
        else:
            self.distance_metric = metric
            
        self.max_samples = state.get("max_samples", self.max_samples)
        self.use_numpy = state.get("use_numpy", self.use_numpy)
        self.weight_by_distance = state.get("weight_by_distance", self.weight_by_distance)
        self.tie_break_by_time = state.get("tie_break_by_time", self.tie_break_by_time)
        
        # Restore training data
        X_train = state.get("X_train", [])
        self.y_train = state.get("y_train", [])
        self.timestamps = state.get("timestamps", [])
        
        # Convert features to numpy arrays if using numpy
        if self.use_numpy:
            self.X_train = [np.array(x) for x in X_train]
        else:
            self.X_train = X_train
        
        # Restore classes
        self._classes = set(state.get("classes", []))
        
        # Restore metrics if available
        metrics = state.get("metrics", {})
        self._total_prediction_time = metrics.get("total_prediction_time", 0.0)
        self._total_predictions = metrics.get("total_predictions", 0)
    
    def get_samples(self) -> Tuple[List[Any], List[Any], List[float]]:
        """Get the current training samples.
        
        Returns:
            Tuple of (features, labels, timestamps)
        """
        return self.X_train, self.y_train, self.timestamps
    
    def get_classes(self) -> List[Any]:
        """Get the current list of classes.
        
        Returns:
            List of unique classes
        """
        return sorted(list(self._classes))
        
    def get_last_neighbor_labels(self) -> List[Any]:
        """Get the labels of the k nearest neighbors from the most recent prediction.
        
        Returns:
            List of labels of the nearest neighbors
        """
        return self._last_neighbor_labels
    
    def clear(self) -> None:
        """Clear all training samples."""
        self.X_train = []
        self.y_train = []
        self.timestamps = []
        self._classes = set()
        logger.debug("Cleared all training samples from classifier")