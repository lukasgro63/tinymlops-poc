# TinyLCM Scenario 1: Autonomous Drift Detection on Edge Devices

This document provides a comprehensive explanation of the TinyLCM Scenario 1 (Inference Pipeline) example, focusing on the feature extraction process, KNN classifier, and drift detection mechanisms.

## Table of Contents

1. [Overview](#overview)
2. [Architecture](#architecture)
3. [Feature Extraction](#feature-extraction)
4. [KNN Classifier](#knn-classifier)
5. [Drift Detection Mechanisms](#drift-detection-mechanisms)
   - [KNN Distance Monitor](#knn-distance-monitor)
   - [Page-Hinkley Feature Monitor](#page-hinkley-feature-monitor)
6. [Pipeline Workflow](#pipeline-workflow)
7. [Advanced Concepts](#advanced-concepts)

## Overview

Scenario 1 demonstrates an autonomous drift detection system designed to run on resource-constrained edge devices like the Raspberry Pi Zero 2W. The system operates with limited computing resources and without relying on ground truth labels, making it suitable for real-world deployments where immediate feedback is unavailable.

The primary components include:

1. **TFLite Feature Extractor**: Extracts feature vectors from input images using a pre-trained TensorFlow Lite model
2. **Lightweight KNN Classifier**: Classifies feature vectors using an optimized k-Nearest Neighbors algorithm
3. **Autonomous Drift Detectors**: Monitor for distribution shifts without requiring ground truth labels
4. **InferencePipeline**: Orchestrates the workflow, routing data through the system components

The system is designed to:
- Perform inference on image data (classifying LEGO brick colors)
- Monitor for distribution shifts in real-time
- Log drift events for further analysis
- Optionally synchronize with a server for external validation

## Architecture

The overall architecture follows this flow:

```
┌───────────────┐      ┌───────────────┐      ┌────────────────┐      ┌───────────────┐
│ Camera Input  │ ───> │ TFLite        │ ───> │ Pre-trained    │ ───> │ Drift         │
│               │      │ Feature       │      │ LightweightKNN │      │ Detector      │
└───────────────┘      │ Extractor     │      │               │       └───────┬───────┘
                       └───────────────┘      └───────────────┘               │
                                                                              ▼
                                                                     ┌───────────────┐
                                                                     │ SyncClient    │
                                                                     │ (Optional)    │
                                                                     └───────────────┘
```

The system is initialized from a pre-trained state:
- The TFLite model is pre-trained on a dataset of images
- The KNN classifier is initialized with a set of labeled examples (stored in a JSON state file)
- The drift detectors are configured with appropriate parameters based on the use case

## Feature Extraction

The feature extraction process uses the TFLiteFeatureExtractor class from tinylcm to convert input images into feature vectors that can be used for classification and drift detection.

### TFLite Feature Extractor

The TFLiteFeatureExtractor loads a TensorFlow Lite model and extracts embeddings from a specified layer, typically the penultimate layer before the classification head.

**Key Properties:**
- **Model Path**: Path to the TensorFlow Lite model file
- **Feature Layer Index**: Which layer to extract features from (default: -1, the last layer)
- **Normalize Features**: Whether to normalize feature vectors to unit length (aids in similarity calculations)
- **Lazy Loading**: Whether to load the model only when needed (saves memory)

**Pseudocode for Feature Extraction:**

```python
def extract_features(input_image):
    # Preprocess the image
    processed_image = preprocess(input_image)
    
    # Ensure image has correct shape
    if processed_image.shape != input_shape:
        processed_image = reshape(processed_image)
    
    # Run inference
    interpreter.set_tensor(input_index, processed_image)
    interpreter.invoke()
    
    # Get output from feature layer
    features = interpreter.get_tensor(feature_layer_index)
    
    # Remove batch dimension if present
    if features.shape[0] == 1:
        features = features[0]
    
    # Normalize if requested
    if normalize_features:
        norm = sqrt(sum(features * features))
        features = features / max(norm, 1e-12)
    
    return features
```

In the scenario1 example, the feature extractor is configured with:
- A pre-trained TFLite model
- Feature normalization enabled
- BGR to RGB conversion preprocessing to handle OpenCV images

The extracted features serve as the input for both the KNN classifier and the drift detectors.

## KNN Classifier

The LightweightKNN classifier is a resource-efficient implementation of the k-Nearest Neighbors algorithm, optimized for edge devices with limited memory and computational power.

### Key Features

- **Multiple Distance Metrics**: Supports euclidean, manhattan, and cosine distance metrics
- **Memory Management**: Limits the number of stored training samples to prevent memory growth
- **Optimized Calculations**: Offers pure Python and NumPy calculation paths for different devices
- **Enhanced Confidence**: Provides distance-weighted confidence scores that are sensitive to distribution shifts
- **Tie Breaking**: Uses sample timestamps to prioritize more recent training data

### Classification Process

The LightweightKNN classifier works by storing feature vectors and their corresponding labels. During inference, it:

1. Calculates distances between the input feature vector and all stored training samples
2. Identifies the k nearest neighbors
3. Determines the final prediction through majority voting (optionally weighted by distance)
4. Calculates a confidence score based on distances and vote distribution

**Pseudocode for KNN Prediction:**

```python
def predict(features):
    if not training_samples:
        return "unknown"
    
    # Find k nearest neighbors
    distances = []
    for i, training_feature in enumerate(X_train):
        distance = calculate_distance(features, training_feature)
        distances.append((i, distance, timestamps[i]))
    
    # Sort by distance (and optionally by timestamp for tie-breaking)
    distances.sort(key=lambda x: (x[1], -x[2]))
    
    # Get the k nearest neighbors
    nearest_neighbors = distances[:k]
    
    # Count votes for each class
    votes = {}
    for idx, dist, _ in nearest_neighbors:
        label = y_train[idx]
        if weight_by_distance:
            weight = 1.0 / (dist + 1e-6)  # Avoid division by zero
            votes[label] = votes.get(label, 0) + weight
        else:
            votes[label] = votes.get(label, 0) + 1
    
    # Return the class with the most votes
    return max(votes.items(), key=lambda x: x[1])[0]
```

**Confidence Calculation:**

The confidence calculation is a critical aspect of the system, as it provides valuable information for drift detection. The method used in LightweightKNN goes beyond simple vote counting:

```python
def calculate_confidence(neighbors):
    # Group neighbors by class
    class_stats = {}
    for neighbor_idx, distance in neighbors:
        label = y_train[neighbor_idx]
        if label not in class_stats:
            class_stats[label] = {"count": 0, "total_distance": 0.0}
        class_stats[label]["count"] += 1
        class_stats[label]["total_distance"] += distance
    
    # Calculate adjusted probability for each class
    adjusted_votes = {}
    total_adjusted_vote = 0.0
    
    for label, stats in class_stats.items():
        # Base vote probability
        vote_prob = stats["count"] / k
        
        # Average distance for this class
        avg_distance = stats["total_distance"] / stats["count"]
        
        # Distance factor: higher distances = lower confidence
        distance_factor = 1.0 / (1.0 + scaling_factor * avg_distance)
        
        # Final adjusted vote includes both vote count and distance
        adjusted_vote = vote_prob * distance_factor
        adjusted_votes[label] = adjusted_vote
        total_adjusted_vote += adjusted_vote
    
    # Normalize to get probabilities
    probabilities = {
        label: vote / total_adjusted_vote 
        for label, vote in adjusted_votes.items()
    }
    
    return probabilities
```

This approach ensures that the confidence drops when samples are far from training data, even if the predicted class is still the same - a critical property for effective drift detection.

### Initial State Loading

In the scenario1 example, the KNN classifier is initialized from a pre-trained state stored in a JSON file. This state contains:
- Feature vectors for known examples
- Corresponding labels
- Configuration parameters (k, distance metric, etc.)

This approach allows the system to start with a good baseline classifier without having to train from scratch on the device.

## Drift Detection Mechanisms

The scenario1 example uses two primary drift detection mechanisms:

1. **KNNDistanceMonitor**: Tracks distances to nearest neighbors to detect unusual samples
2. **PageHinkleyFeatureMonitor**: Monitors statistical properties of feature vectors using the Page-Hinkley test

### KNN Distance Monitor

The KNNDistanceMonitor is particularly effective at detecting unknown classes or outliers by directly monitoring the distances to nearest neighbors.

**Working Principle**

When a new sample is very different from all training data, the distances to its nearest neighbors will be much larger than normal. The KNNDistanceMonitor uses this property to detect potential drift.

**Pseudocode for KNNDistanceMonitor:**

```python
class KNNDistanceMonitor:
    def __init__(self, delta, lambda_threshold, warm_up_samples, 
                 reference_update_interval, reference_update_factor):
        self.delta = delta  # Magnitude parameter for small fluctuations
        self.lambda_threshold = lambda_threshold  # Threshold for drift detection
        self.warm_up_samples = warm_up_samples  # Samples to collect during warm-up
        self.reference_update_interval = reference_update_interval  # How often to update reference
        self.reference_update_factor = reference_update_factor  # Factor for updating reference
        
        # State variables
        self.reference_mean = None  # Average distance to nearest neighbors
        self.cumulative_sum = 0.0  # Cumulative sum of deviations
        self.minimum_sum = 0.0  # Running minimum of cumulative sum
        self.in_warm_up_phase = True  # Whether we're still in warm-up
        self.samples_processed = 0  # Number of samples processed
    
    def update(self, record):
        # Process sample and update counters
        self.samples_processed += 1
        
        # Extract distances from record
        distances = extract_distances(record)
        avg_distance = sum(distances) / len(distances)
        
        # Handle warm-up phase
        if self.in_warm_up_phase:
            self.warm_up_values.append(avg_distance)
            if self.samples_processed >= self.warm_up_samples:
                self.in_warm_up_phase = False
                self.reference_mean = mean(self.warm_up_values)
            return False, None
        
        # Page-Hinkley test update
        deviation = avg_distance - (self.reference_mean + self.delta)
        self.cumulative_sum += deviation
        self.minimum_sum = min(self.minimum_sum, self.cumulative_sum)
        
        # Calculate Page-Hinkley test statistic
        ph_value = self.cumulative_sum - self.minimum_sum
        
        # Check for drift
        drift_detected = ph_value > self.lambda_threshold
        
        # If drift detected, prepare drift info
        if drift_detected:
            drift_info = {
                "detector_type": "KNNDistanceMonitor",
                "metric": "neighbor_distance",
                "current_value": avg_distance,
                "reference_mean": self.reference_mean,
                "ph_value": ph_value,
                "threshold": self.lambda_threshold
            }
        else:
            drift_info = None
        
        # Update reference statistics periodically
        if self.should_update_reference():
            self.reference_mean = (self.reference_update_factor * self.reference_mean + 
                                  (1 - self.reference_update_factor) * avg_distance)
        
        return drift_detected, drift_info
```

**Configuration in Scenario1**

In the scenario1 example, the KNNDistanceMonitor is configured with:
- `delta = 30.0`: Tolerance for small fluctuations in distances
- `lambda_threshold = 100.0`: Threshold for the Page-Hinkley test statistic
- `warm_up_samples = 3`: Number of samples to use for establishing the baseline
- `reference_update_interval = 5`: Update the reference every 5 samples
- `reference_update_factor = 0.1`: Weight for updating the reference (0.1 means 10% new, 90% old)

This configuration allows the monitor to quickly detect significant changes in distances while being somewhat tolerant of minor fluctuations.

### Page-Hinkley Feature Monitor

The PageHinkleyFeatureMonitor applies the Page-Hinkley test to a scalar statistic derived from feature vectors.

**Working Principle**

The Page-Hinkley test is a sequential analysis technique designed to detect changes in the mean of a Gaussian process. For drift detection, it:

1. Extracts a scalar statistic from feature vectors (e.g., a specific feature dimension)
2. Tracks the cumulative sum of deviations from a reference mean
3. Signals drift when the difference between the cumulative sum and its running minimum exceeds a threshold

**Pseudocode for PageHinkleyFeatureMonitor:**

```python
class PageHinkleyFeatureMonitor:
    def __init__(self, feature_statistic_fn, lambda_threshold, delta, 
                 warm_up_samples, reference_update_interval, reference_update_factor):
        self.feature_statistic_fn = feature_statistic_fn  # Function to extract scalar from features
        self.lambda_threshold = lambda_threshold  # Threshold for drift detection
        self.delta = delta  # Magnitude parameter for expected direction
        self.warm_up_samples = warm_up_samples  # Samples to collect during warm-up
        self.reference_update_interval = reference_update_interval  # How often to update reference
        self.reference_update_factor = reference_update_factor  # Factor for updating reference
        
        # State variables
        self.reference_mean = None  # Reference mean of the statistic
        self.cumulative_sum = 0.0  # Cumulative sum of deviations (m_t)
        self.minimum_sum = 0.0  # Running minimum of cumulative sum (M_t)
        self.in_warm_up_phase = True  # Whether we're still in warm-up
        self.samples_processed = 0  # Number of samples processed
    
    def update(self, record):
        # Process sample and update counters
        self.samples_processed += 1
        
        # Extract features and calculate statistic
        features = record.get('features')
        statistic_value = self.feature_statistic_fn(features)
        
        # Handle warm-up phase
        if self.in_warm_up_phase:
            self.warm_up_values.append(statistic_value)
            if self.samples_processed >= self.warm_up_samples:
                self.in_warm_up_phase = False
                self.reference_mean = mean(self.warm_up_values)
            return False, None
        
        # Page-Hinkley test update
        deviation = statistic_value - self.reference_mean - self.delta
        self.cumulative_sum += deviation
        self.minimum_sum = min(self.minimum_sum, self.cumulative_sum)
        
        # Calculate Page-Hinkley test statistic
        ph_value = self.cumulative_sum - self.minimum_sum
        
        # Check for drift
        drift_detected = ph_value > self.lambda_threshold
        
        # If drift detected, prepare drift info
        if drift_detected:
            drift_info = {
                "detector_type": "PageHinkleyFeatureMonitor",
                "statistic_value": statistic_value,
                "reference_mean": self.reference_mean,
                "ph_value": ph_value,
                "threshold": self.lambda_threshold
            }
        else:
            drift_info = None
        
        # Update reference statistics periodically
        if self.should_update_reference():
            self.reference_mean = (self.reference_update_factor * self.reference_mean + 
                                  (1 - self.reference_update_factor) * statistic_value)
        
        return drift_detected, drift_info
```

**Configuration in Scenario1**

In the scenario1 example, the feature statistic function extracts a specific feature dimension from the feature vector and normalizes it to reduce extreme values. The detector is configured with:

- `delta = 0.005`: Small positive value to detect increases in the statistic
- `lambda_threshold = 10.0`: Threshold for the Page-Hinkley test statistic
- `warm_up_samples = 30`: Number of samples to use for establishing the baseline
- `reference_update_interval = 50`: Update the reference every 50 samples
- `feature_index = 0`: The specific feature dimension to monitor

## Pipeline Workflow

The InferencePipeline class in tinylcm orchestrates the entire process, managing the flow of data through the system components.

### Main Processing Workflow

**Pseudocode for the Main Processing Flow:**

```python
def process(input_data, label=None, timestamp=None, sample_id=None, metadata=None):
    # Start timing for latency tracking
    start_time = time.time()
    
    # Extract features
    features = feature_extractor.extract_features(input_data)
    
    # Make prediction
    prediction = classifier.predict(features)
    
    # Get prediction probabilities and confidence
    probas = classifier.predict_proba(features)
    confidence = max(probas[0]) if probas is not None else None
    
    # Calculate inference latency
    latency_ms = (time.time() - start_time) * 1000
    
    # Create feature sample for tracking
    sample = FeatureSample(
        features=features,
        label=label,
        prediction=prediction,
        timestamp=timestamp or time.time(),
        sample_id=sample_id or generate_uuid(),
        metadata=metadata or {}
    )
    
    # Track operation in the operational monitor
    operational_monitor.track_inference(
        input_id=sample.sample_id,
        prediction=prediction,
        confidence=confidence,
        latency_ms=latency_ms,
        metadata=metadata or {},
        timestamp=sample.timestamp,
        features=features
    )
    
    # Log sample if data logger is available
    if data_logger:
        data_logger.log_sample(
            input_data, prediction, 
            label=label, 
            metadata={
                "sample_id": sample.sample_id,
                "timestamp": sample.timestamp,
                "confidence": confidence,
                **(metadata or {})
            }
        )
    
    # Create record for autonomous monitoring
    autonomous_record = {
        "features": features,
        "prediction": prediction,
        "confidence": confidence,
        "probabilities": probas[0].tolist() if probas is not None else None,
        "sample_id": sample.sample_id,
        "timestamp": sample.timestamp,
        "label": label,
        "metadata": metadata or {},
        "classifier": classifier  # Include reference for KNNDistanceMonitor
    }
    
    # Update autonomous monitors
    autonomous_drift_detected = False
    drift_info = None
    
    if enable_autonomous_detection and autonomous_monitors:
        for monitor in autonomous_monitors:
            # Update the monitor with the record
            drift_detected, monitor_drift_info = monitor.update(autonomous_record)
            
            if drift_detected:
                autonomous_drift_detected = True
                drift_info = {
                    "detector_type": type(monitor).__name__,
                    "sample_id": sample.sample_id,
                    "timestamp": sample.timestamp,
                    "details": monitor_drift_info
                }
    
    # Prepare result dictionary
    result = {
        "prediction": prediction,
        "confidence": confidence,
        "probabilities": probas[0] if probas is not None else None,
        "sample_id": sample.sample_id,
        "timestamp": sample.timestamp,
        "latency_ms": latency_ms,
        "autonomous_drift_detected": autonomous_drift_detected
    }
    
    if autonomous_drift_detected:
        result["drift_info"] = drift_info
    
    return result
```

### Drift Detection Workflow

The InferencePipeline also provides a method to explicitly check for drift across all autonomous monitors:

```python
def check_autonomous_drifts():
    drift_results = []
    
    for monitor in autonomous_monitors:
        # Check monitor for drift
        drift_detected, drift_info = monitor.check_for_drift()
        
        # Create result record
        result = {
            "detector_type": type(monitor).__name__,
            "drift_detected": drift_detected,
            "timestamp": time.time()
        }
        
        # Add drift info if available
        if drift_info:
            result.update(drift_info)
        
        drift_results.append(result)
    
    return drift_results
```

### Drift Callback Mechanism

When drift is detected, the system triggers registered callbacks to handle the event:

```python
def on_drift_detected(drift_info):
    # Log the drift event
    logger.warning(f"DRIFT DETECTED: {drift_info['detector_type']}")
    
    # Save the current frame if drift image saving is enabled
    if save_drift_images and current_frame is not None:
        # Create directory structure for organization
        image_path = save_drift_image(current_frame, drift_info)
    
    # Send drift event to server if sync client is available
    if sync_client:
        sync_client.create_and_send_drift_event_package(
            detector_name=drift_info['detector_type'],
            reason=drift_info.get('reason', 'Drift detected'),
            metrics=drift_info,
            image_path=image_path if image_path else None
        )
```

## Advanced Concepts

### Cooldown Mechanism

The drift detection system implements a cooldown mechanism to prevent excessive drift events from the same detector:

```python
def _notify_callbacks(self, drift_info):
    # Check if we're in the cooldown period
    if self.in_cooldown_period and self.samples_since_last_drift < self.drift_cooldown_period:
        logger.debug(f"In cooldown period, skipping callback notifications")
        drift_info["in_cooldown_period"] = True
        return False
    
    # Reset cooldown tracking
    self.in_cooldown_period = True
    self.samples_since_last_drift = 0
    
    # Execute callbacks
    for callback in self.callbacks:
        try:
            callback(drift_info)
        except Exception as e:
            logger.error(f"Error in drift callback: {e}")
    
    return True
```

### Rolling Reference Updates

The drift detectors implement a rolling update mechanism for their reference statistics to adapt to gradual, non-drift changes in the data distribution:

```python
def update_reference(current_value):
    # Using the rolling update formula
    self.reference_mean = (self.reference_update_factor * current_value + 
                           (1 - self.reference_update_factor) * self.reference_mean)
```

The update factor (`reference_update_factor`) determines how quickly the reference adapts to changes:
- Higher values (e.g., 0.5) cause faster adaptation
- Lower values (e.g., 0.01) cause slower adaptation

Updates are only performed when:
1. The warm-up phase is complete
2. The detector is not currently in a drift state
3. The interval since the last update exceeds `reference_update_interval`

### System Metrics Tracking

In addition to drift detection, the scenario1 example tracks system metrics to monitor the device's operational health:

```python
def track_system_metrics():
    # Collect basic system metrics
    metrics = {
        "timestamp": time.time(),
        "cpu_percent": psutil.cpu_percent(),
        "memory_percent": psutil.virtual_memory().percent
    }
    
    # Add to operational monitor
    operational_monitor.track_system_metrics(metrics)
```

These metrics can help identify performance bottlenecks or resource constraints that may affect the system's behavior.

### Synchronization with Server

The scenario1 example includes optional synchronization with a TinySphere server, enabling:
- Transfer of drift events and images for analysis
- Retrieval of validation results
- Management of device configuration

```python
def sync_with_server():
    # Send drift events to the server
    if autonomous_drift_detected:
        sync_client.create_and_send_drift_event_package(
            detector_name=drift_info['detector_type'],
            reason="Drift detected",
            metrics=drift_info,
            image_path=image_path
        )
    
    # Send operational metrics
    metrics = operational_monitor.get_current_metrics()
    sync_client.create_and_send_metrics_package(metrics)
    
    # Send operational logs
    sync_client.create_and_send_operational_logs_package(
        operational_logs_dir=operational_monitor.storage_dir,
        session_id=operational_monitor.session_id
    )
    
    # Sync all pending packages
    sync_client.sync_all_pending_packages()
```

## Conclusion

The TinyLCM Scenario 1 demonstrates a robust approach to autonomous drift detection on edge devices, combining feature extraction, efficient KNN classification, and multiple drift detection strategies. The system is designed to operate efficiently on resource-constrained hardware while providing valuable insights into model behavior and data distribution shifts.

Key strengths of this approach include:

1. **Label-Free Operation**: The system can detect drift without requiring ground truth labels
2. **Resource Efficiency**: Optimized implementations for edge device deployment
3. **Multi-Modal Detection**: Combines different detection strategies for robust monitoring
4. **Configurable Sensitivity**: Parameters can be tuned for different use cases and constraints
5. **Operational Monitoring**: Tracks system health alongside model behavior

This implementation provides a solid foundation for building adaptive edge AI systems that can maintain performance and reliability in challenging real-world environments.