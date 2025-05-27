
# Example code for using the feature processor in TinyLCM:

import pickle
import numpy as np

# Load feature processor
with open("feature_processor.pkl", 'rb') as f:
    processor = pickle.load(f)

scaler = processor['scaler']
pca = processor['pca']
input_dim = processor['input_dim']
output_dim = processor['output_dim']

# In the pipeline after feature extraction:
def process_features(features):
    # Flatten if needed
    if len(features.shape) > 1:
        features = features.flatten()
        
    # Dimension check
    if len(features) != input_dim:
        # Handle dimension mismatch if needed
        pass
        
    # Apply standardization and PCA
    scaled = scaler.transform(features.reshape(1, -1))
    reduced = pca.transform(scaled)[0]
    
    # Return reduced features (dimension: 256)
    return reduced
