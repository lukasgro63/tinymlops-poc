# Dual Drift Detection and Adaptive Learning in TinyLCM

TinyLCM provides two complementary systems for detecting and responding to distribution shifts and concept drift in machine learning models deployed on edge devices:

1. **Label-based Adaptive Learning System** - A supervised approach that requires ground truth labels to adapt the model
2. **Autonomous Drift Detection System** - An unsupervised approach that monitors model behavior without requiring labels

These systems can operate independently or together, providing a flexible framework for maintaining model performance in resource-constrained environments.

## 1. Label-based Adaptive Learning System

The label-based system requires access to ground truth labels for at least some of the incoming data. It's designed for situations where feedback mechanisms are available, such as:

- User feedback loops
- Delayed labels (e.g., from a manual review process)
- Semi-supervised environments with occasional labeled data

### Components:

- **AdaptivePipeline** - The main entry point that integrates feature extraction, classification, and adaptation
- **AdaptiveHandler** (Passive/Active/Hybrid) - Manages when and how the model adapts to new data
- **AccuracyCUSUM** - Detects concept drift based on changes in accuracy
- **LightweightKNN** - A resource-efficient classifier that adapts to new data patterns
- **AdaptationTracker** - Records adaptation events and metrics
- **StateManager** - Persists and loads model states

### Workflow:

1. Extract features from input data
2. Make predictions using the current model
3. When ground truth labels become available, provide feedback to the system
4. The handler updates the model based on the feedback (adding samples, pruning, etc.)
5. AccuracyCUSUM monitors the accuracy to detect significant drift

```python
# Example of label-based adaptive learning
from tinylcm.core.pipeline import AdaptivePipeline

# Initialize pipeline
pipeline = AdaptivePipeline()

# Process input (prediction only)
result = pipeline.process(input_data)
prediction = result["prediction"]

# Later, when label becomes available:
feedback_result = pipeline.process(input_data, label=true_label)
```

## 2. Autonomous Drift Detection System

The autonomous system doesn't require ground truth labels, making it suitable for continuous monitoring in production environments where labels are unavailable. It detects potential data distribution shifts by monitoring proxy metrics:

### Components:

- **AutonomousDriftDetector** - Base class for all label-free detectors
- **EWMAConfidenceMonitor/PageHinkleyConfidenceMonitor** - Detects drops in model confidence
- **PredictionDistributionMonitor** - Detects shifts in the distribution of predicted classes
- **FeatureMonitor** - Detects changes in the statistical properties of input features

### Metrics Monitored:

| Detector | Metric | Approach | Use Case |
|----------|--------|----------|----------|
| EWMAConfidenceMonitor | Prediction confidence | Control chart with exponentially weighted moving average | Detect when model becomes less certain |
| PageHinkleyConfidenceMonitor | Prediction confidence | Sequential change detection test | Detect sudden changes in confidence |
| PredictionDistributionMonitor | Class distribution | Jensen-Shannon divergence between distributions | Detect when prediction patterns change |
| FeatureMonitor | Feature statistics | Distance in feature space from reference distribution | Detect when input patterns change |

### Workflow:

1. Initialize one or more autonomous detectors
2. After each inference, update each detector with relevant data
3. Check for drift detection and take appropriate actions
4. Reset detectors as needed after addressing drift

```python
# Example of autonomous drift detection
from tinylcm.core.drift_detection.confidence import EWMAConfidenceMonitor
from tinylcm.core.drift_detection.distribution import PredictionDistributionMonitor

# Initialize detectors
confidence_monitor = EWMAConfidenceMonitor()
distribution_monitor = PredictionDistributionMonitor()

# During inference
def on_inference(input_data, prediction, confidence):
    # Update detectors
    confidence_drift, info = confidence_monitor.update({
        "confidence": confidence
    })
    
    distribution_drift, info = distribution_monitor.update({
        "prediction": prediction
    })
    
    # Check for drift
    if confidence_drift or distribution_drift:
        # Take action (alert, logging, model switching, etc.)
        log_drift_event(info)
```

## 3. Combining Both Systems

The real power comes from using both systems together:

### Combined Workflow:

1. Process inputs through AdaptivePipeline for predictions
2. Update autonomous detectors with each inference result
3. When ground truth labels are available, provide feedback to AdaptivePipeline
4. Use autonomous detectors for continuous monitoring
5. Use label-based system for adaptation when labels are available

```python
# Example of combined approach
from tinylcm.core.pipeline import AdaptivePipeline
from tinylcm.core.drift_detection.confidence import EWMAConfidenceMonitor

# Initialize components
pipeline = AdaptivePipeline()
confidence_monitor = EWMAConfidenceMonitor()

# Process input
result = pipeline.process(input_data)
prediction = result["prediction"]
confidence = result["confidence"]

# Update autonomous detector
confidence_drift, info = confidence_monitor.update({
    "confidence": confidence,
    "prediction": prediction
})

# Take action if drift detected
if confidence_drift:
    log_drift_event(info)
    
# Later, when label becomes available:
if true_label is not None:
    feedback_result = pipeline.process(input_data, label=true_label)
    # Check if adaptation occurred
    if feedback_result["adaptation_occurred"]:
        # Model has been updated
        pass
```

## 4. Design Considerations

### Resource Constraints

Both systems are designed for resource-constrained environments:
- Minimal memory footprint with configurable history sizes
- Efficient algorithms with O(1) update complexity
- Optional NumPy acceleration where available
- Non-blocking I/O for state persistence

### Separation of Concerns

The two systems are intentionally decoupled:
- Autonomous detectors don't modify the model, they only signal drift
- Label-based system handles model adaptation
- Clear interfaces allow for custom integration strategies

### Multiple Detection Strategies

Different drift types require different detection approaches:
- Sudden concept drift (e.g., sensor calibration changes)
- Gradual drift (e.g., seasonal changes)
- Recurring contexts (e.g., different user behaviors)

## 5. Implementation Example

See `examples/adaptive_monitoring_picamera2.py` for a complete implementation demonstrating both systems working together on a Raspberry Pi with camera input.

## 6. Comparison

| Feature | Label-based System | Autonomous System |
|---------|-------------------|-------------------|
| Requires ground truth | Yes | No |
| Can adapt the model | Yes | No (signals only) |
| Detection sensitivity | High (with accurate labels) | Medium (depends on proxy metrics) |
| False positive rate | Low (with accurate labels) | Medium to high |
| Resource usage | Higher (stores samples) | Lower (stores statistics) |
| Best used for | Active learning, feedback loops | Continuous monitoring, alerting |