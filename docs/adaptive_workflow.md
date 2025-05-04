# TinyLCM Adaptive Learning Workflow

This document explains the workflow of the refactored TinyLCM library with its adaptive learning capabilities for edge computing devices.

## Overview

TinyLCM is a lightweight MLOps framework for edge computing, now built around an adaptive learning architecture. The core components work together to enable:

1. Feature extraction from input data using TensorFlow Lite models
2. Classification using a lightweight k-NN classifier
3. Continuous adaptation through feedback
4. Drift detection and handling with CUSUM
5. Model state management and persistence
6. Performance monitoring

The entire system is designed for resource-constrained devices and provides non-blocking I/O operations to ensure minimal impact on inference performance.

## Component Architecture

![Architecture Diagram](img/tinylcm_architecture.png)

### Core Components

1. **AdaptivePipeline**
   - Central coordinator that connects all components
   - Manages the workflow from input to prediction
   - Handles feedback and adaptation

2. **Feature Extractors**
   - `TFLiteFeatureExtractor`: Extracts feature vectors using TensorFlow Lite models
   - Designed to work with minimal memory and compute requirements

3. **Classifiers**
   - `LightweightKNN`: Memory-efficient k-NN classifier optimized for edge devices
   - Uses condensing algorithms to maintain a representative, compact dataset

4. **Adaptation Handlers**
   - `PassiveHandler`: Updates the model only with explicit feedback
   - `ActiveHandler`: Proactively detects drift and triggers adaptation
   - `HybridHandler`: Combines passive and active approaches

5. **Drift Detection**
   - `AccuracyCUSUM`: Implements Cumulative Sum algorithm for detecting systematic drift
   - Efficiently tracks accuracy changes with minimal memory requirements

6. **State Management**
   - `StateManager`: Persists and loads model states
   - Manages versioning and rollback capabilities

7. **Performance Monitoring**
   - `InferenceMonitor`: Tracks inference metrics (latency, confidence, throughput)
   - Uses non-blocking I/O to minimize performance impact

8. **Adaptation Tracking**
   - `AdaptationTracker`: Logs adaptation events and metrics
   - Essential for understanding model evolution and performance over time

## Workflow Steps

### 1. Initialization

```python
# Create feature extractor
feature_extractor = TFLiteFeatureExtractor(
    model_path="model.tflite",
    feature_layer_index=-2  # Extract features from penultimate layer
)

# Create classifier
classifier = LightweightKNN(
    k=5,  # Number of neighbors
    distance_threshold=0.8,  # Limit influence of distant neighbors
    condensing_method="class_balanced"  # Keep dataset compact
)

# Create adaptation handler (choose one based on needs)
handler = HybridHandler(
    cusum_threshold=5.0,  # Sensitivity to drift
    cusum_delta=0.25,  # Minimum change considered significant
    batch_size=30  # Number of samples before checking for drift
)

# Create state manager
state_manager = StateManager(
    storage_dir="./model_states",
    max_states=10  # Number of states to keep
)

# Create adaptation tracker
adaptation_tracker = AdaptationTracker(
    log_dir="./adaptation_logs",
    max_events_in_memory=100
)

# Create metrics monitor
inference_monitor = InferenceMonitor(
    storage_dir="./inference_logs"
)

# Finally, create adaptive pipeline to coordinate all components
pipeline = AdaptivePipeline(
    feature_extractor=feature_extractor,
    classifier=classifier,
    handler=handler,
    state_manager=state_manager,
    adaptation_tracker=adaptation_tracker,
    auto_save=True  # Automatically save state on drift detection
)
```

### 2. Processing New Inputs

When a new input (e.g., image, text, sensor data) arrives, it passes through the pipeline:

```python
# Process a new input (e.g., an image)
result = pipeline.process(input_data)

# The result contains:
prediction = result.prediction  # Predicted class
confidence = result.confidence  # Confidence score (if available)
features = result.features      # Extracted feature vector
metadata = result.metadata      # Additional information

# Track inference performance
inference_monitor.track_inference(
    input_id=str(uuid.uuid4()),
    prediction=prediction,
    confidence=confidence,
    latency_ms=result.metadata.get("latency_ms"),
    metadata={"input_shape": input_data.shape}
)
```

#### What happens inside:

1. **Feature extraction**
   - Input is passed to the feature extractor
   - The TFLite model processes the input
   - Feature vector is extracted from a specified layer

2. **Classification**
   - Feature vector is passed to the classifier
   - The k-NN algorithm finds the closest neighbors
   - Prediction and confidence score are determined

3. **Inference monitoring**
   - Performance metrics are collected:
     - Latency (time taken for the process call)
     - Confidence score
     - Prediction distribution

### 3. Providing Feedback

When ground truth becomes available, it's provided to the pipeline:

```python
# Provide feedback for the previously processed input
pipeline.provide_feedback(
    features=result.features,  # Feature vector from earlier process call
    true_label="cat",          # The correct class label
    metadata={"source": "user_correction"}
)
```

#### What happens inside:

1. **Adaptation handler processing**
   - The handler receives the feedback with true label
   - Accuracy is calculated and tracked
   - In passive mode:
     - Model updates only if prediction was incorrect
   - In active/hybrid modes:
     - CUSUM algorithm tracks accuracy changes
     - Drift is detected when accuracy changes systematically
   - Adaptation events are generated

2. **Adaptation tracking**
   - Events are logged for later analysis
   - Metrics are updated (accuracy, drift events, etc.)

### 4. Drift Detection & Handling

When drift is detected (in active or hybrid mode):

```python
# This happens automatically inside the pipeline when drift is detected

# You can register a callback to be notified:
def on_drift_detected(event):
    print(f"Drift detected! Reason: {event.reason}")
    print(f"Adaptation: {event.adaptation_type}")
    
pipeline.register_drift_callback(on_drift_detected)
```

#### What happens inside:

1. **Drift detection (in the handler)**
   - CUSUM algorithm identifies statistically significant drops in accuracy
   - A drift event is generated

2. **Drift handling**
   - The classifier's training data is updated based on the handler strategy:
     - Recalculate decision boundaries
     - Adjust for new data distribution
     - Possibly remove outdated exemplars

3. **Model state management**
   - Current state is saved (with auto_save=True)
   - State includes:
     - Feature vectors and labels
     - Classifier parameters
     - Adaptation metadata

4. **Adaptation tracking**
   - Drift event is logged with metadata
   - Performance metrics before/after adaptation
   - Drift statistics are updated

### 5. Saving & Loading State

Model state can be saved and loaded:

```python
# Save current state manually (also happens automatically on drift)
state_id = pipeline.save_state()

# Later, load a previous state
pipeline.load_state(state_id)

# List available states
states = state_manager.list_states()
```

## Performance Considerations

The library is designed for resource-constrained devices:

1. **Memory efficiency**
   - Lightweight k-NN with condensing algorithms
   - Feature vector curation to maintain a representative set
   - Configurable memory limits for windowed metrics

2. **Compute efficiency**
   - Vectorized operations (via NumPy where available)
   - Options for Manhattan distance in k-NN (faster than Euclidean)
   - Batch processing of updates

3. **Non-blocking I/O**
   - Background threads for logging and persistence
   - Queue-based architecture to avoid blocking main thread
   - Configurable queue sizes and worker counts

4. **Storage efficiency**
   - Configurable log rotation
   - JSON-based storage for compatibility and readability
   - State versioning with cleanup policies

## Integration Examples

### Basic Camera Inference Loop

```python
# Initialize pipeline and components as above

import cv2

# Open camera
cap = cv2.VideoCapture(0)

while True:
    # Capture frame
    ret, frame = cap.read()
    
    # Preprocess (resize to model input size)
    input_tensor = preprocess_image(frame)
    
    # Process with adaptive pipeline
    start_time = time.time()
    result = pipeline.process(input_tensor)
    latency_ms = (time.time() - start_time) * 1000
    
    # Monitor performance
    inference_monitor.track_inference(
        input_id=str(uuid.uuid4()),
        prediction=result.prediction,
        confidence=result.confidence,
        latency_ms=latency_ms
    )
    
    # Display result
    cv2.putText(frame, f"{result.prediction}: {result.confidence:.2f}", 
                (10, 30), cv2.FONT_HERSHEY_SIMPLEX, 1, (0, 255, 0), 2)
    cv2.imshow('Frame', frame)
    
    # Exit on 'q' key press
    if cv2.waitKey(1) & 0xFF == ord('q'):
        break

# Release resources
cap.release()
cv2.destroyAllWindows()
```

### Asynchronous Processing with Feedback

For systems that need to handle inference and feedback asynchronously:

```python
import threading
import queue

# Create queues
inference_queue = queue.Queue()
feedback_queue = queue.Queue()

# Initialize pipeline and components as above

def inference_worker():
    while True:
        # Get item from queue
        input_data, input_id = inference_queue.get()
        if input_data is None:  # Shutdown signal
            break
            
        # Process with adaptive pipeline
        start_time = time.time()
        result = pipeline.process(input_data)
        latency_ms = (time.time() - start_time) * 1000
        
        # Monitor performance
        inference_monitor.track_inference(
            input_id=input_id,
            prediction=result.prediction,
            confidence=result.confidence,
            latency_ms=latency_ms
        )
        
        # Store result for potential feedback
        results[input_id] = result
        
        # Mark task as done
        inference_queue.task_done()

def feedback_worker():
    while True:
        # Get feedback from queue
        input_id, true_label = feedback_queue.get()
        if input_id is None:  # Shutdown signal
            break
            
        # Look up the corresponding result
        if input_id in results:
            result = results[input_id]
            
            # Provide feedback
            pipeline.provide_feedback(
                features=result.features,
                true_label=true_label
            )
            
            # Optional: clean up stored results
            del results[input_id]
        
        # Mark task as done
        feedback_queue.task_done()

# Start worker threads
results = {}
inference_thread = threading.Thread(target=inference_worker, daemon=True)
feedback_thread = threading.Thread(target=feedback_worker, daemon=True)
inference_thread.start()
feedback_thread.start()

# Later, submit work to the queues
inference_queue.put((image_data, "img_123"))

# Later, provide feedback
feedback_queue.put(("img_123", "cat"))

# On shutdown
inference_queue.put((None, None))  # Signal inference worker to stop
feedback_queue.put((None, None))   # Signal feedback worker to stop
inference_thread.join()
feedback_thread.join()
```

## Best Practices

1. **Handler Selection**
   - Use `PassiveHandler` when feedback is reliably available
   - Use `ActiveHandler` when accuracy needs to be maintained automatically
   - Use `HybridHandler` for most real-world applications (balanced approach)

2. **Feature Selection**
   - Choose feature layers that are:
     - Lower-dimensional (for efficiency)
     - Class-separable (for accuracy)
     - General rather than specific (for adaptability)

3. **Parameter Tuning**
   - Adjust `k` based on complexity:
     - Small `k` (3-5) for well-separated classes
     - Larger `k` (7-11) for overlapping classes
   - Tune CUSUM parameters:
     - Lower threshold for more sensitivity to drift
     - Higher delta for resilience to noise

4. **State Management**
   - Save states on significant events
   - Maintain multiple versions for rollback
   - Consider periodic state cleanup on constrained devices

5. **Performance Optimization**
   - Use `manhattan` distance for faster computation
   - Enable condensing to keep memory usage low
   - Configure worker threads based on device capabilities

