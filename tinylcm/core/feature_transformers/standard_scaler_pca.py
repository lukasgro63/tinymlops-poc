"""
StandardScaler + PCA feature transformation for dimensionality reduction.

This module implements a combined StandardScaler + PCA feature transformer that:
1. First standardizes features to have zero mean and unit variance
2. Then applies PCA dimensionality reduction

This two-step approach is particularly effective for drift detection and KNN classification
on resource-constrained devices like Raspberry Pi Zero.
"""

import os
import pickle
import numpy as np
from typing import Dict, Any, Optional, Union, Tuple
import logging

from .base import FeatureTransformer

logger = logging.getLogger(__name__)


class StandardScalerPCATransformer(FeatureTransformer):
    """
    Combined StandardScaler + PCA feature transformation for dimensionality reduction.
    
    This transformer implements a two-step feature processing pipeline that first
    standardizes features (zero mean, unit variance) and then applies PCA dimensionality
    reduction. It is designed for use on resource-constrained devices, improving both
    the quality of drift detection and the speed of KNN computations.
    """

    def __init__(self, processor_path: str):
        """
        Initialize StandardScalerPCATransformer.

        Args:
            processor_path: Path to the pickle file containing the pre-trained processor 
                           (which should include both StandardScaler and PCA models)
        """
        self.processor_path = processor_path
        self.processor = None
        self.scaler = None
        self.pca = None
        self.input_dim = None
        self.output_dim = None
        self.explained_variance = None
        self.pca_enabled = False
        self._load_processor()

    def _load_processor(self) -> None:
        """Load the pre-trained processor model from disk."""
        try:
            if not os.path.exists(self.processor_path):
                logger.error(f"Processor file {self.processor_path} not found!")
                raise FileNotFoundError(f"Processor file {self.processor_path} not found!")
            
            with open(self.processor_path, 'rb') as f:
                self.processor = pickle.load(f)
            
            # Handle compatibility format (created by modified train_tflite_object_model.py)
            if self.processor.get('compatible_format', False):
                logger.info(f"Detected compatible format processor (numpy {self.processor.get('numpy_version', 'unknown')})")
                
                # Reconstruct StandardScaler
                from sklearn.preprocessing import StandardScaler
                scaler = StandardScaler()
                scaler_data = self.processor['scaler_data']
                scaler.mean_ = np.array(scaler_data['mean'])
                scaler.var_ = np.array(scaler_data['var'])
                scaler.scale_ = np.array(scaler_data['scale'])
                scaler.n_features_in_ = scaler_data['n_features_in']
                scaler.n_samples_seen_ = scaler_data['n_samples_seen']
                
                # Reconstruct PCA if available
                pca = None
                if 'pca_data' in self.processor:
                    from sklearn.decomposition import PCA
                    pca_data = self.processor['pca_data']
                    pca = PCA()
                    pca.n_components_ = pca_data['n_components']
                    pca.components_ = np.array(pca_data['components'])
                    pca.explained_variance_ = np.array(pca_data['explained_variance'])
                    pca.explained_variance_ratio_ = np.array(pca_data['explained_variance_ratio'])
                    
                    if pca_data['mean'] is not None:
                        pca.mean_ = np.array(pca_data['mean'])
                    
                    if pca_data['singular_values'] is not None:
                        pca.singular_values_ = np.array(pca_data['singular_values'])
                    
                    if pca_data['noise_variance'] is not None:
                        pca.noise_variance_ = pca_data['noise_variance']
                
                # Set attributes
                self.scaler = scaler
                self.pca = pca
                self.input_dim = self.processor.get('input_dim')
                self.output_dim = self.processor.get('output_dim')
                self.explained_variance = self.processor.get('explained_variance', 0)
                
            else:
                # Original format - direct extraction
                self.scaler = self.processor.get('scaler')
                self.pca = self.processor.get('pca')
                self.input_dim = self.processor.get('input_dim')
                self.output_dim = self.processor.get('output_dim')
                self.explained_variance = self.processor.get('explained_variance', 0)
            
            # Validate necessary components
            if self.scaler is None:
                logger.error("StandardScaler not found in processor file!")
                raise ValueError("StandardScaler not found in processor file!")
            
            # PCA is optional - determine if it should be used
            self.pca_enabled = self.pca is not None
            
            # Log details about the loaded processor
            if self.pca_enabled:
                logger.info(f"Loaded feature processor: {self.input_dim}D → {self.output_dim}D")
                logger.info(f"PCA explains {self.explained_variance:.2f}% of variance")
            else:
                logger.info("Loaded feature processor with standardization only (no PCA)")
                
        except Exception as e:
            logger.error(f"Error loading feature processor: {e}")
            raise

    def transform(self, features: np.ndarray) -> np.ndarray:
        """
        Apply standardization and optional PCA transformation to features.

        Args:
            features: Input feature array (n_samples, n_features) or a single 1D feature vector.
                     If a single sample is provided as a 1D array, it will be reshaped.

        Returns:
            Transformed features (standardized and optionally dimensionality-reduced)
        """
        if self.scaler is None:
            logger.error("Feature processor not loaded. Cannot transform features.")
            return features
        
        try:
            # Get original shape for dimension checks and reshape a single feature vector if needed
            original_shape = features.shape
            is_single_sample = features.ndim == 1
            
            # Ensure input is flattened for single samples
            if is_single_sample:
                flat_features = features
            else:
                # If multiple samples, return error - we expect only one sample at a time
                if original_shape[0] > 1:
                    logger.warning(f"Multiple samples provided ({original_shape[0]}). " 
                                   "This transformer expects a single sample.")
                    flat_features = features[0]  # Process only the first sample
                else:
                    flat_features = features[0]
            
            # Dimension check
            if len(flat_features) != self.input_dim:
                logger.warning(f"Feature dimension mismatch: got {len(flat_features)}, "
                               f"expected {self.input_dim}")
                
                # Handle dimension mismatch by truncating or padding
                if len(flat_features) > self.input_dim:
                    logger.debug(f"Truncating features to match expected dimensions")
                    flat_features = flat_features[:self.input_dim]
                else:
                    logger.debug(f"Padding features with zeros to match expected dimensions")
                    padded = np.zeros(self.input_dim)
                    padded[:len(flat_features)] = flat_features
                    flat_features = padded
            
            # Reshape for transformation (scaler expects 2D array)
            features_reshaped = flat_features.reshape(1, -1)
            
            # Apply standardization
            scaled_features = self.scaler.transform(features_reshaped)
            
            # Apply PCA if available
            if self.pca_enabled:
                reduced_features = self.pca.transform(scaled_features)
                # Return as 1D array to match expected output format for a single sample
                return reduced_features[0]
            else:
                # Return standardized features
                return scaled_features[0]
            
        except Exception as e:
            logger.error(f"Error applying feature transformation: {e}")
            # In case of error, return original features
            return features

    def get_state(self) -> Dict[str, Any]:
        """
        Get the current state of the transformer.

        Returns:
            Dictionary containing the transformer state
        """
        state = {
            "processor_path": self.processor_path,
            "input_dim": self.input_dim,
            "output_dim": self.output_dim,
            "pca_enabled": self.pca_enabled
        }
        
        if self.explained_variance is not None:
            state["explained_variance"] = self.explained_variance
            
        return state

    def set_state(self, state: Dict[str, Any]) -> None:
        """
        Set the transformer state from a dictionary.

        Args:
            state: Dictionary containing transformer state
        """
        if "processor_path" in state:
            self.processor_path = state["processor_path"]
            # Reload the processor if the path changed
            self._load_processor()
        
        # Other properties will be loaded from the processor file