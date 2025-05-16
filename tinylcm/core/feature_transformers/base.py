"""
Base module for feature transformation components.
"""

from abc import ABC, abstractmethod
import numpy as np
from typing import Dict, Any, Optional


class FeatureTransformer(ABC):
    """Base class for all feature transformation components."""

    @abstractmethod
    def transform(self, features: np.ndarray) -> np.ndarray:
        """
        Transform input features to a new representation.

        Args:
            features: Input feature array to transform (n_samples, n_features)

        Returns:
            Transformed feature array
        """
        pass

    @abstractmethod
    def get_state(self) -> Dict[str, Any]:
        """
        Get the current state of the transformer.

        Returns:
            Dictionary containing the transformer state
        """
        pass

    @abstractmethod
    def set_state(self, state: Dict[str, Any]) -> None:
        """
        Set the transformer state from a dictionary.

        Args:
            state: Dictionary containing transformer state
        """
        pass