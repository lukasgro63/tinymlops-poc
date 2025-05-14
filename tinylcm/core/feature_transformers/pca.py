"""
PCA-based feature transformation for dimensionality reduction.
"""

import os
import pickle
import numpy as np
from typing import Dict, Any, Optional
import logging

from .base import FeatureTransformer

logger = logging.getLogger(__name__)


class PCATransformer(FeatureTransformer):
    """
    PCA feature transformation for dimensionality reduction.
    Implements a lightweight wrapper around a pre-trained scikit-learn PCA model.
    """

    def __init__(self, model_path: str, n_components: int = 50):
        """
        Initialize PCATransformer.

        Args:
            model_path: Path to the pickle file containing the pre-trained PCA model
            n_components: Number of principal components (only used for logging/information)
        """
        self.model_path = model_path
        self.n_components = n_components
        self.pca_model = None
        self._load_model()

    def _load_model(self) -> None:
        """Load the pre-trained PCA model from disk."""
        try:
            if not os.path.exists(self.model_path):
                logger.error(f"PCA model file {self.model_path} not found!")
                raise FileNotFoundError(f"PCA model file {self.model_path} not found!")
            
            with open(self.model_path, 'rb') as f:
                self.pca_model = pickle.load(f)
            
            # Log details about the loaded model
            if hasattr(self.pca_model, 'n_components_'):
                actual_components = self.pca_model.n_components_
                logger.info(f"Loaded PCA model with {actual_components} components")
                if actual_components != self.n_components:
                    logger.warning(f"Loaded PCA model has {actual_components} components, "
                                  f"but {self.n_components} were specified in config.")
            else:
                logger.warning("Loaded PCA model doesn't have n_components_ attribute. "
                              "This might not be a scikit-learn PCA model.")
                
        except Exception as e:
            logger.error(f"Error loading PCA model: {e}")
            raise

    def transform(self, features: np.ndarray) -> np.ndarray:
        """
        Apply PCA transformation to features.

        Args:
            features: Input feature array (n_samples, n_features)
                     If a single sample is provided as a 1D array, it will be reshaped.

        Returns:
            Transformed features with reduced dimensionality
        """
        if self.pca_model is None:
            logger.error("PCA model not loaded. Cannot transform features.")
            return features
        
        try:
            # Ensure the input is properly shaped
            if features.ndim == 1:
                # Reshape a single sample into a 2D array with one row
                features = features.reshape(1, -1)
            
            # Apply the transformation
            transformed_features = self.pca_model.transform(features)
            
            # If it was a single sample, return it as a 1D array to match input format
            if transformed_features.shape[0] == 1:
                return transformed_features.flatten()
            
            return transformed_features
            
        except Exception as e:
            logger.error(f"Error applying PCA transformation: {e}")
            # In case of error, return original features
            return features

    def get_state(self) -> Dict[str, Any]:
        """
        Get the current state of the transformer.

        Returns:
            Dictionary containing the transformer state
        """
        return {
            "model_path": self.model_path,
            "n_components": self.n_components
        }

    def set_state(self, state: Dict[str, Any]) -> None:
        """
        Set the transformer state from a dictionary.

        Args:
            state: Dictionary containing transformer state
        """
        if "model_path" in state:
            self.model_path = state["model_path"]
        
        if "n_components" in state:
            self.n_components = state["n_components"]
            
        # Reload the model if the path changed
        self._load_model()