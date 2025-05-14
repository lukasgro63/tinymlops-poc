#!/usr/bin/env python3
"""
Feature Processor Compatibility Converter

This script converts existing feature_processor.pkl files from the original format
to a compatible format that works with numpy 1.24.2 on Raspberry Pi.

The original format directly pickles scikit-learn objects, which can create
compatibility issues between different numpy versions. The new format stores
the essential data needed to reconstruct these objects in a simpler format.

Usage:
    python convert_feature_processor.py <input_pkl_path> <output_pkl_path>

Example:
    python convert_feature_processor.py feature_processor.pkl feature_processor_compatible.pkl
"""

import os
import sys
import pickle
import json
import numpy as np
import argparse
import logging

logging.basicConfig(
    level=logging.INFO,
    format='%(asctime)s - %(levelname)s - %(message)s'
)
logger = logging.getLogger(__name__)

def load_original_processor(file_path):
    """Load the original processor pickle file."""
    logger.info(f"Loading original processor from: {file_path}")
    try:
        with open(file_path, 'rb') as f:
            processor = pickle.load(f)
        return processor
    except Exception as e:
        logger.error(f"Error loading processor: {e}")
        raise

def convert_to_compatible_format(processor):
    """Convert the processor to a compatible format."""
    logger.info("Converting processor to compatible format...")
    
    # Check if already in compatible format
    if processor.get('compatible_format', False):
        logger.info("Processor is already in compatible format")
        return processor
        
    # Extract components
    scaler = processor.get('scaler')
    pca = processor.get('pca')
    input_dim = processor.get('input_dim')
    output_dim = processor.get('output_dim')
    explained_variance = processor.get('explained_variance', 0)
    creation_timestamp = processor.get('creation_timestamp', '')
    
    if scaler is None:
        logger.error("StandardScaler not found in processor!")
        raise ValueError("StandardScaler not found in processor!")
    
    # Create scaler_data dictionary
    scaler_data = {
        'mean': scaler.mean_.tolist(),
        'var': scaler.var_.tolist(),
        'scale': scaler.scale_.tolist(),
        'n_features_in': scaler.n_features_in_,
        'n_samples_seen': int(scaler.n_samples_seen_)
    }
    
    # Create pca_data dictionary if PCA exists
    pca_data = None
    if pca is not None:
        pca_data = {
            'n_components': pca.n_components_,
            'mean': pca.mean_.tolist() if hasattr(pca, 'mean_') else None,
            'components': pca.components_.tolist(),
            'explained_variance': pca.explained_variance_.tolist(),
            'explained_variance_ratio': pca.explained_variance_ratio_.tolist(),
            'singular_values': pca.singular_values_.tolist() if hasattr(pca, 'singular_values_') else None,
            'noise_variance': pca.noise_variance_ if hasattr(pca, 'noise_variance_') else None
        }
    
    # Create compatible processor dictionary
    compatible_processor = {
        'scaler_data': scaler_data,
        'pca_data': pca_data,
        'input_dim': int(input_dim) if input_dim is not None else None,
        'output_dim': int(output_dim) if output_dim is not None else None,
        'explained_variance': float(explained_variance) if explained_variance is not None else 0.0,
        'creation_timestamp': creation_timestamp,
        'numpy_version': np.__version__,
        'compatible_format': True
    }
    
    return compatible_processor

def save_compatible_processor(processor, output_path, save_json=True):
    """Save the processor in compatible format."""
    logger.info(f"Saving compatible processor to: {output_path}")
    try:
        # Save pickle file
        with open(output_path, 'wb') as f:
            pickle.dump(processor, f, protocol=3)  # Protocol 3 for better compatibility
        
        # Optionally save as JSON for inspection
        if save_json:
            json_path = output_path.replace('.pkl', '.json')
            with open(json_path, 'w') as f:
                import json
                json.dump(processor, f, indent=2, default=str)
            logger.info(f"JSON version saved to: {json_path}")
            
        return True
    except Exception as e:
        logger.error(f"Error saving processor: {e}")
        return False

def main():
    # Parse command line arguments
    parser = argparse.ArgumentParser(description='Convert feature processor to compatible format')
    parser.add_argument('input_path', help='Path to the input feature_processor.pkl file')
    parser.add_argument('output_path', nargs='?', help='Path to save the compatible feature_processor.pkl (optional)')
    parser.add_argument('--force', action='store_true', help='Force overwrite if output file exists')
    args = parser.parse_args()
    
    # Set default output path if not provided
    if args.output_path is None:
        args.output_path = args.input_path.replace('.pkl', '_compatible.pkl')
    
    # Check if output file already exists
    if os.path.exists(args.output_path) and not args.force:
        logger.error(f"Output file already exists: {args.output_path}")
        logger.error("Use --force to overwrite")
        return 1
    
    try:
        # Load original processor
        processor = load_original_processor(args.input_path)
        
        # Convert to compatible format
        compatible_processor = convert_to_compatible_format(processor)
        
        # Save compatible processor
        success = save_compatible_processor(compatible_processor, args.output_path)
        
        if success:
            logger.info("✅ Conversion successful!")
            logger.info(f"Original processor: {args.input_path}")
            logger.info(f"Compatible processor: {args.output_path}")
            
            # Create a compatibility loader module
            loader_path = os.path.dirname(args.output_path)
            compatibility_path = os.path.join(loader_path, "feature_processor_compatibility.py")
            
            with open(compatibility_path, 'w') as f:
                f.write("""
# Compatibility loader for feature_processor.pkl
# This module can reconstruct StandardScaler and PCA from the simplified data format saved

import pickle
import numpy as np
from sklearn.preprocessing import StandardScaler
from sklearn.decomposition import PCA

def load_compatible_processor(file_path):
    """
    Load a feature processor saved in compatibility mode and reconstruct scikit-learn objects.
    
    Args:
        file_path: Path to the feature processor pickle file
        
    Returns:
        Dictionary with reconstructed scaler, pca, and metadata
    """
    with open(file_path, 'rb') as f:
        saved_data = pickle.load(f)
    
    # Check if this is a compatible format processor
    if not saved_data.get('compatible_format', False):
        # This might be the original format, just return it
        return saved_data
    
    # Reconstruct StandardScaler
    scaler = StandardScaler()
    scaler_data = saved_data['scaler_data']
    scaler.mean_ = np.array(scaler_data['mean'])
    scaler.var_ = np.array(scaler_data['var'])
    scaler.scale_ = np.array(scaler_data['scale'])
    scaler.n_features_in_ = scaler_data['n_features_in']
    scaler.n_samples_seen_ = scaler_data['n_samples_seen']
    
    # Reconstruct PCA
    pca = None
    if saved_data.get('pca_data') is not None:
        pca = PCA()
        pca_data = saved_data['pca_data']
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
    
    # Return reconstructed objects and metadata
    return {
        'scaler': scaler,
        'pca': pca,
        'input_dim': saved_data['input_dim'],
        'output_dim': saved_data['output_dim'],
        'explained_variance': saved_data['explained_variance'],
        'creation_timestamp': saved_data['creation_timestamp']
    }

def process_features(features, processor):
    """
    Apply feature processing (standardization and PCA) to a feature vector.
    
    Args:
        features: Input feature vector
        processor: Loaded and reconstructed processor 
                  (from load_compatible_processor)
        
    Returns:
        Processed (standardized and dimensionality-reduced) feature vector
    """
    # Get components from processor
    scaler = processor['scaler']
    pca = processor['pca']
    input_dim = processor['input_dim']
    
    # Flatten if needed
    if len(features.shape) > 1:
        features = features.flatten()
    
    # Check dimensions
    if len(features) != input_dim:
        # Handle dimension mismatch
        if len(features) > input_dim:
            # Truncate
            features = features[:input_dim]
        else:
            # Pad with zeros
            padded = np.zeros(input_dim)
            padded[:len(features)] = features
            features = padded
            
    # Standardize and reduce dimensions
    scaled = scaler.transform(features.reshape(1, -1))
    if pca is not None:
        reduced = pca.transform(scaled)[0]
        return reduced
    else:
        return scaled[0]
""")
            logger.info(f"✅ Compatibility loader saved to: {compatibility_path}")
            
            return 0
        else:
            logger.error("❌ Conversion failed!")
            return 1
            
    except Exception as e:
        logger.error(f"Error during conversion: {e}")
        import traceback
        traceback.print_exc()
        return 1

if __name__ == "__main__":
    sys.exit(main())