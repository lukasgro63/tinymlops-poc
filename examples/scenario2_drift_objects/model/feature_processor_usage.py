
# Beispielcode zur Verwendung des Feature-Prozessors in TinyLCM:

import pickle
import numpy as np

# Feature-Prozessor laden
with open("feature_processor.pkl", 'rb') as f:
    processor = pickle.load(f)

scaler = processor['scaler']
pca = processor['pca']
input_dim = processor['input_dim']
output_dim = processor['output_dim']

# In der Pipeline nach Feature-Extraktion:
def process_features(features):
    # Flatten, falls nötig
    if len(features.shape) > 1:
        features = features.flatten()
        
    # Dimensionscheck
    if len(features) != input_dim:
        # Hier ggf. Fehlerbehandlung oder Anpassung
        pass
        
    # Standardisierung und PCA anwenden
    scaled = scaler.transform(features.reshape(1, -1))
    reduced = pca.transform(scaled)[0]
    
    # Reduzierte Features (Dimension: 115) zurückgeben
    return reduced
