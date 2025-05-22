# Initial State Creation - Detailed Technical Analysis

This document provides an in-depth analysis of the initial state creation process in TinyLCM, which generates the reference statistics and KNN state required for autonomous drift detection on edge devices.

## Overview

The initial state creation process is critical for enabling autonomous operation on edge devices. It pre-computes reference statistics from training data that serve as the baseline for drift detection algorithms, particularly the Page-Hinkley test used in KNNDistanceMonitor.

## Process Architecture

### 1. Data Pipeline Flow

```
Training Images → TFLite Feature Extraction → StandardScaler+PCA → KNN State Creation
                                                                 → Reference Statistics
```

**Key Components:**
- **TFLite Model**: `model_object.tflite` (quantized MobileNetV2)
- **Feature Processor**: `feature_processor.pkl` (StandardScaler + PCA pipeline)
- **Output**: `knn_initial_state_objects.json` + `knn_reference_stats.json`

### 2. Feature Extraction Pipeline

#### Image Preprocessing
```python
def preprocess_image_for_feature_extraction(image_path: Path, target_size: tuple) -> np.ndarray:
    # Load image using OpenCV
    img = cv2.imread(str(image_path))
    
    # Resize maintaining aspect ratio
    img_resized = resize_image(img, target_size)  # Uses cv2.INTER_AREA
    
    # Convert BGR → RGB for TFLite compatibility
    img_rgb = cv2.cvtColor(img_resized, cv2.COLOR_BGR2RGB)
    
    return img_rgb.astype(np.uint8)  # Keep as uint8 for quantized model
```

#### TFLite Feature Extraction
```python
def extract_features_manually(image, interpreter, input_details, output_details, feature_processor_path=None):
    # Prepare quantized input tensor (uint8)
    processed_image = prepare_input_tensor_quantized(image, input_details)
    
    # Run TFLite inference
    interpreter.set_tensor(input_details['index'], processed_image)
    interpreter.invoke()
    
    # Extract 1280D features from GlobalAveragePooling2D
    feature_tensor = interpreter.get_tensor(output_details['index'])
    
    # Remove batch dimension: (1, 1280) → (1280,)
    if len(feature_tensor.shape) > 1 and feature_tensor.shape[0] == 1:
        feature_tensor = feature_tensor[0]
    
    # Apply StandardScaler + PCA transformation
    if feature_processor_path:
        return apply_feature_transformation(feature_tensor, feature_processor_path)
    
    return feature_tensor
```

#### Feature Transformation Application
```python
def apply_feature_transformation(raw_features, processor_path):
    with open(processor_path, 'rb') as f:
        processor = pickle.load(f)
    
    # Reconstruct StandardScaler
    scaler = StandardScaler()
    scaler_data = processor['scaler_data']
    scaler.mean_ = np.array(scaler_data['mean'])
    scaler.var_ = np.array(scaler_data['var'])
    scaler.scale_ = np.array(scaler_data['scale'])
    scaler.n_features_in_ = scaler_data['n_features_in']
    
    # Apply standardization: (x - mean) / scale
    scaled_features = scaler.transform(raw_features.reshape(1, -1))
    
    # Reconstruct and apply PCA
    if 'pca_data' in processor:
        pca = PCA()
        pca_data = processor['pca_data']
        pca.components_ = np.array(pca_data['components'])
        pca.mean_ = np.array(pca_data['mean'])
        pca.explained_variance_ = np.array(pca_data['explained_variance'])
        
        # Transform: 1280D → 256D
        transformed_features = pca.transform(scaled_features)[0]
        return transformed_features
    
    return scaled_features[0]
```

### 3. Training Data Processing

#### Dataset Configuration
```python
CLASSES = {
    "negative": "examples/assets/training_images/negative",   # ~161 samples
    "stone": "examples/assets/training_images/stone",        # ~161 samples  
    "lego": "examples/assets/training_images/lego",          # ~161 samples
    "leaf": "examples/assets/training_images/leaf"           # ~161 samples
}

# Total: ~644 balanced samples across 4 classes
```

#### Systematic Feature Collection
```python
def collect_all_features():
    all_features = []
    all_labels = []
    all_timestamps = []
    
    print("Collecting features from all training images...")
    
    for label_name, image_folder_path in CLASSES.items():
        print(f"Processing class: {label_name}")
        image_files = list(Path(image_folder_path).glob("*.jpg"))
        
        for i, image_file in enumerate(image_files):
            try:
                # Preprocess image
                preprocessed_img = preprocess_image_for_feature_extraction(
                    image_file, TARGET_IMG_SIZE
                )
                
                # Extract features using TFLite + feature processor
                features = extract_features_manually(
                    preprocessed_img, interpreter, input_details, output_details,
                    feature_processor_path=FEATURE_PROCESSOR_PATH
                )
                
                # Store processed data
                all_features.append(features.flatten())
                all_labels.append(label_name)
                all_timestamps.append(datetime.now().isoformat())
                
                if i % 20 == 0:
                    print(f"  Processed {i+1}/{len(image_files)} images")
                    
            except Exception as e:
                print(f"Error processing {image_file}: {e}")
                continue
    
    return all_features, all_labels, all_timestamps
```

### 4. KNN State Creation

#### Balanced Sampling Strategy
```python
def create_knn_with_balanced_samples():
    knn = LightweightKNN(
        k=KNN_K,                           # 5 nearest neighbors
        max_samples=KNN_MAX_SAMPLES,       # 200 total samples
        distance_metric=KNN_DISTANCE_METRIC,  # "euclidean"
        use_numpy=KNN_USE_NUMPY            # True for offline creation
    )
    
    # Calculate samples per class for balance
    samples_per_class = KNN_MAX_SAMPLES // len(CLASSES)  # 200 // 4 = 50
    
    # Group features by class
    features_by_class = {}
    for feature, label, timestamp in zip(all_features, all_labels, all_timestamps):
        if label not in features_by_class:
            features_by_class[label] = {'features': [], 'timestamps': []}
        features_by_class[label]['features'].append(feature)
        features_by_class[label]['timestamps'].append(timestamp)
    
    # Add balanced samples to KNN
    for label, data in features_by_class.items():
        available_samples = len(data['features'])
        samples_to_add = min(available_samples, samples_per_class)
        
        print(f"Adding {samples_to_add} samples for class '{label}'")
        
        # Random sampling without replacement
        selected_indices = np.random.choice(
            available_samples, 
            size=samples_to_add, 
            replace=False
        )
        
        for idx in selected_indices:
            knn.add_sample(
                feature=data['features'][idx],
                label=label,
                timestamp=data['timestamps'][idx]
            )
    
    return knn
```

#### KNN State Serialization
```python
def save_knn_state(knn_instance, output_path):
    # Convert KNN state to JSON-serializable format
    state_data = {
        'X_train': [x.tolist() for x in knn_instance.X_train],
        'y_train': knn_instance.y_train,
        'timestamps': knn_instance.timestamps,
        'k': knn_instance.k,
        'distance_metric': knn_instance.distance_metric,
        'max_samples': knn_instance.max_samples,
        'feature_dim': len(knn_instance.X_train[0]) if knn_instance.X_train else 0,
        'creation_timestamp': datetime.now().isoformat(),
        'sample_count': len(knn_instance.X_train),
        'class_distribution': {
            label: knn_instance.y_train.count(label) 
            for label in set(knn_instance.y_train)
        }
    }
    
    with open(output_path, 'w') as f:
        json.dump(state_data, f, indent=2)
```

### 5. Reference Statistics Calculation

#### KNN Distance Analysis
```python
def calculate_knn_distance_statistics(knn_instance, all_features, all_labels):
    print("Calculating KNN distance statistics for drift detection reference...")
    
    all_avg_distances = []
    avg_distances_by_class = {}
    
    # For each feature, calculate distances to k nearest neighbors
    for i, (feature, label) in enumerate(zip(all_features, all_labels)):
        # Find k nearest neighbors using KNN internal method
        nearest_neighbors = knn_instance._find_neighbors(feature)
        
        # Extract distances from (index, distance) tuples
        distances = [dist for _, dist in nearest_neighbors]
        
        if len(distances) > 0:
            avg_distance = sum(distances) / len(distances)
            all_avg_distances.append(avg_distance)
            
            # Group by class for detailed analysis
            if label not in avg_distances_by_class:
                avg_distances_by_class[label] = []
            avg_distances_by_class[label].append(avg_distance)
        
        if i % 50 == 0:
            print(f"Progress: {i}/{len(all_features)} samples processed")
    
    return compute_reference_statistics(all_avg_distances, avg_distances_by_class)
```

#### Statistical Analysis
```python
def compute_reference_statistics(all_distances, class_distances):
    # Global statistics for Page-Hinkley test
    global_stats = {
        'reference_mean': float(np.mean(all_distances)),      # μ_ref for PH test
        'reference_std': float(np.std(all_distances)),        # σ_ref for adaptive thresholds
        'min_distance': float(np.min(all_distances)),
        'max_distance': float(np.max(all_distances)),
        'sample_count': len(all_distances)
    }
    
    # Per-class statistics for detailed analysis
    class_stats = {}
    for label, distances in class_distances.items():
        class_stats[label] = {
            'mean': float(np.mean(distances)),
            'std': float(np.std(distances)),
            'min': float(np.min(distances)),
            'max': float(np.max(distances)),
            'count': len(distances)
        }
    
    # Combined reference statistics
    reference_stats = {
        **global_stats,
        'class_statistics': class_stats,
        'generation_timestamp': datetime.now().isoformat(),
        'description': f"Reference statistics for KNNDistanceMonitor based on {len(all_distances)} balanced samples"
    }
    
    return reference_stats
```

### 6. Generated Reference Statistics

#### Actual Production Statistics
Based on the analysis of 644 balanced samples (161 per class):

```json
{
    "reference_mean": 45.161649691184856,      // Global mean distance
    "reference_std": 10.17918530068497,        // Global standard deviation
    "min_distance": 15.394752197265626,        // Minimum observed distance
    "max_distance": 73.78734497070315,         // Maximum observed distance
    "sample_count": 644,                       // Total samples analyzed
    
    "class_statistics": {
        "negative": {
            "mean": 43.56, "std": 13.47, "count": 161
        },
        "stone": {
            "mean": 47.56, "std": 8.88, "count": 161  
        },
        "lego": {
            "mean": 44.50, "std": 7.99, "count": 161
        },
        "leaf": {
            "mean": 45.02, "std": 9.03, "count": 161
        }
    },
    
    "knn_k": 5,                               // K value used
    "feature_dimension": 256,                 // Post-PCA feature dimension
    "model_path": ".../model_object.tflite"  // Source model
}
```

#### Statistical Insights
- **Class Separation**: `stone` class has highest mean distance (47.56), indicating it's most distinct
- **Class Cohesion**: `lego` class has lowest standard deviation (7.99), indicating most cohesive
- **Global Distribution**: Mean ≈ 45.16, std ≈ 10.18 provides baseline for drift detection
- **Distance Range**: 15.39 to 73.78 defines the operational range

### 7. Deployment Integration

#### Configuration Linkage
```json
{
  "drift_detectors": [{
    "type": "KNNDistanceMonitor",
    "delta": 0.1,                           // Magnitude parameter (δ)
    "lambda_threshold": 5.0,                // Detection threshold (λ) 
    "reference_stats_path": "./initial_state/knn_reference_stats.json",
    "use_adaptive_thresholds": true,        // Use reference_std for adaptation
    "adaptive_delta_std_multiplier": 1.5,   // δ = reference_std * 1.5
    "adaptive_lambda_std_multiplier": 5.0   // λ = reference_std * 5.0
  }]
}
```

#### Runtime Usage
```python
# During KNNDistanceMonitor initialization
with open(reference_stats_path, 'r') as f:
    stats = json.load(f)

self.reference_mean = stats['reference_mean']        # 45.16
self.reference_std = stats['reference_std']          # 10.18

# Adaptive threshold calculation
if use_adaptive_thresholds:
    self.delta = self.reference_std * 1.5            # ≈ 15.27
    self.lambda_threshold = self.reference_std * 5.0  # ≈ 50.90

# Page-Hinkley test during runtime
deviation = avg_distance - (self.reference_mean + self.delta)
# If avg_distance > 45.16 + 15.27 = 60.43, accumulate positive deviation
```

### 8. Quality Assurance

#### Validation Checks
```python
def validate_initial_state():
    # Load and verify KNN state
    with open(knn_state_path, 'r') as f:
        knn_data = json.load(f)
    
    assert knn_data['sample_count'] == KNN_MAX_SAMPLES
    assert knn_data['feature_dim'] == 256  # Post-PCA dimension
    assert len(set(knn_data['y_train'])) == len(CLASSES)
    
    # Load and verify reference statistics
    with open(reference_stats_path, 'r') as f:
        stats = json.load(f)
    
    assert stats['sample_count'] > 0
    assert 'reference_mean' in stats
    assert 'reference_std' in stats
    assert len(stats['class_statistics']) == len(CLASSES)
    
    print("✅ Initial state validation passed")
```

#### Cross-Validation with Training
```python
def cross_validate_with_training():
    # Ensure feature dimensions match
    training_features = extract_training_features()  # From model training
    initial_state_features = extract_initial_features()  # From state creation
    
    assert training_features.shape[1] == initial_state_features.shape[1]
    
    # Verify feature quality correlation
    sample_correlation = np.corrcoef(
        training_features[0], 
        initial_state_features[0]
    )[0,1]
    
    assert sample_correlation > 0.8, "Feature extraction inconsistency detected"
```

## Key Innovation Points

### 1. End-to-End Consistency
- Same TFLite model and feature processor used in training and deployment
- Ensures feature extraction consistency across development and production
- Eliminates model/preprocessing version mismatches

### 2. Balanced Reference Creation
- Systematic sampling ensures equal representation of all classes
- Prevents bias in reference statistics toward over-represented classes
- Maintains statistical validity of drift detection baselines

### 3. Adaptive Threshold Generation
- Reference statistics enable automatic threshold tuning
- Eliminates manual hyperparameter tuning for different datasets
- Provides dataset-specific optimization for drift sensitivity

### 4. Comprehensive Statistical Foundation
- Both global and per-class statistics for detailed analysis
- Enables multiple drift detection strategies (global vs. class-specific)
- Provides rich metadata for post-deployment analysis

This initial state creation process establishes the foundation for truly autonomous drift detection on edge devices, enabling zero-configuration deployment while maintaining high detection quality.