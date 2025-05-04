from typing import Any, Dict, List, Optional, Tuple, Union, Callable
import numpy as np
from collections import Counter, deque
import time
import math
from enum import Enum

from tinylcm.core.classifiers.base import BaseAdaptiveClassifier
from tinylcm.utils.logging import setup_logger

logger = setup_logger(__name__)


class DistanceMetric(str, Enum):
    """Enumeration of supported distance metrics."""
    EUCLIDEAN = "euclidean"
    MANHATTAN = "manhattan"
    COSINE = "cosine"


class LightweightKNN(BaseAdaptiveClassifier):
    """A lightweight k-Nearest Neighbors classifier optimized for resource-constrained devices.
    
    This implementation can operate with or without NumPy for maximum compatibility
    and performance on edge devices like Raspberry Pi Zero.
    
    Features:
    - Multiple distance metrics: euclidean, manhattan, cosine
    - Optional NumPy acceleration
    - Sample timestamps for tie-breaking (prioritize newer samples)
    - Maximum sample limit to prevent memory growth
    - Incremental learning
    """
    
    def __init__(
        self,
        k: int = 5,
        distance_metric: Union[str, DistanceMetric] = DistanceMetric.EUCLIDEAN,
        max_samples: int = 100,
        use_numpy: bool = True,
        weight_by_distance: bool = False
    ):
        """Initialize the k-NN classifier.
        
        Args:
            k: Number of neighbors to consider
            distance_metric: Distance metric to use (euclidean, manhattan, cosine)
            max_samples: Maximum number of training samples to store
            use_numpy: Whether to use NumPy for calculations (faster but uses more memory)
            weight_by_distance: Whether to weight votes by inverse distance
        """
        self.k = k
        self.distance_metric = distance_metric
        self.max_samples = max_samples
        self.use_numpy = use_numpy
        self.weight_by_distance = weight_by_distance
        
        # Training data
        self.X_train = []  # Feature vectors
        self.y_train = []  # Labels
        self.timestamps = []  # Timestamps for each sample (for tie-breaking)
        self._classes = set()  # Set of unique classes
        
        logger.debug(
            f"Initialized LightweightKNN with k={k}, "
            f"metric={distance_metric}, max_samples={max_samples}"
        )
    
    def fit(self, features: np.ndarray, labels: List[Any]) -> None:
        """Train the classifier on the provided data.
        
        This overwrites any existing training data.
        
        Args:
            features: Matrix of feature vectors, shape (n_samples, n_features)
            labels: List of corresponding labels
        """
        if len(features) != len(labels):
            raise ValueError(f"Number of features ({len(features)}) and labels ({len(labels)}) must match")
            
        # Clear existing data
        self.X_train = []
        self.y_train = []
        self.timestamps = []
        self._classes = set()
        
        # Generate timestamps for new data
        current_time = time.time()
        timestamps = [current_time] * len(labels)
        
        # Add new data
        self._add_samples(features, labels, timestamps)
        
        logger.debug(f"Fitted {len(self.X_train)} samples with {len(self._classes)} classes")
    
    def incremental_fit(self, features: np.ndarray, labels: List[Any]) -> None:
        """Incrementally train the classifier on new data.
        
        This preserves existing training data and adds new samples.
        If the total number of samples exceeds max_samples, the oldest
        samples will be removed.
        
        Args:
            features: Matrix of feature vectors, shape (n_samples, n_features)
            labels: List of corresponding labels
        """
        if len(features) != len(labels):
            raise ValueError(f"Number of features ({len(features)}) and labels ({len(labels)}) must match")
            
        # Generate timestamps for new data
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
        if not self.X_train:
            raise ValueError("Classifier has not been trained yet")
        
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
        
        return predictions
    
    def predict_proba(self, features: np.ndarray) -> np.ndarray:
        """Predict probability distributions over classes for the provided features.
        
        IMPORTANT: This is a synchronous, blocking operation. Similar to predict(),
        the execution time depends on the number of training samples, the dimensionality
        of the feature vectors, and whether NumPy acceleration is enabled (use_numpy).
        
        This method is more computationally intensive than predict() as it needs
        to calculate probabilities for each class.
        
        Args:
            features: Matrix of feature vectors, shape (n_samples, n_features)
            
        Returns:
            Matrix of class probabilities, shape (n_samples, n_classes)
        """
        if not self.X_train:
            raise ValueError("Classifier has not been trained yet")
        
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
        
        # Compute probabilities for each sample
        for i, feature in enumerate(features):
            # Find k nearest neighbors
            neighbors = self._find_neighbors(feature)
            
            # Count class occurrences among neighbors
            if self.weight_by_distance:
                # Weight by inverse distance
                class_weights = {c: 0.0 for c in classes}
                total_weight = 0.0
                
                for neighbor, distance in neighbors:
                    weight = 1.0 / (distance + 1e-6)  # Avoid division by zero
                    class_weights[self.y_train[neighbor]] += weight
                    total_weight += weight
                
                # Convert to probabilities
                if total_weight > 0:
                    for j, c in enumerate(classes):
                        probas[i][j] = class_weights[c] / total_weight
            else:
                # Simple vote counting
                votes = Counter([self.y_train[neighbor[0]] for neighbor in neighbors])
                
                # Convert to probabilities
                for j, c in enumerate(classes):
                    probas[i][j] = votes.get(c, 0) / self.k
        
        return np.array(probas) if self.use_numpy else probas
    
    def _find_neighbors(self, feature: np.ndarray) -> List[Tuple[int, float]]:
        """Find the k nearest neighbors for a feature vector.
        
        Args:
            feature: Feature vector to find neighbors for
            
        Returns:
            List of tuples (neighbor_index, distance)
        """
        # Compute distances to all training samples
        distances = []
        
        for i, train_feature in enumerate(self.X_train):
            distance = self._calculate_distance(feature, train_feature)
            distances.append((i, distance, self.timestamps[i]))
        
        # Sort by distance (ascending)
        distances.sort(key=lambda x: (x[1], -x[2]))  # Sort by distance, then by -timestamp
        
        # Return the k nearest neighbors
        return [(idx, dist) for idx, dist, _ in distances[:self.k]]
    
    def _calculate_distance(self, a: np.ndarray, b: np.ndarray) -> float:
        """Calculate the distance between two feature vectors.
        
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
        
        Args:
            a: First vector
            b: Second vector
            
        Returns:
            Euclidean distance
        """
        if self.use_numpy:
            return np.sqrt(np.sum((a - b) ** 2))
        else:
            return math.sqrt(sum((a_i - b_i) ** 2 for a_i, b_i in zip(a, b)))
    
    def _manhattan_distance(self, a: np.ndarray, b: np.ndarray) -> float:
        """Calculate the Manhattan distance between two vectors.
        
        Args:
            a: First vector
            b: Second vector
            
        Returns:
            Manhattan distance
        """
        if self.use_numpy:
            return np.sum(np.abs(a - b))
        else:
            return sum(abs(a_i - b_i) for a_i, b_i in zip(a, b))
    
    def _cosine_distance(self, a: np.ndarray, b: np.ndarray) -> float:
        """Calculate the cosine distance between two vectors.
        
        Cosine distance = 1 - cosine similarity
        
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
        return 1.0 - cosine_similarity
    
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
            "distance_metric": self.distance_metric,
            "max_samples": self.max_samples,
            "use_numpy": self.use_numpy,
            "weight_by_distance": self.weight_by_distance,
            "X_train": X_train_serialized,
            "y_train": self.y_train,
            "timestamps": self.timestamps,
            "classes": list(self._classes)
        }
    
    def set_state(self, state: Dict[str, Any]) -> None:
        """Restore the classifier's state.
        
        Args:
            state: Previously saved state dictionary
        """
        self.k = state.get("k", self.k)
        self.distance_metric = state.get("distance_metric", self.distance_metric)
        self.max_samples = state.get("max_samples", self.max_samples)
        self.use_numpy = state.get("use_numpy", self.use_numpy)
        self.weight_by_distance = state.get("weight_by_distance", self.weight_by_distance)
        
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