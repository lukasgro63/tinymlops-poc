# Scenario 1: TinyLCM Autonomous Monitoring Only

This scenario demonstrates how to use TinyLCM for autonomous drift monitoring on a Raspberry Pi Zero 2W without adaptation. The system monitors model performance using proxy metrics and reports drift events to a TinySphere server.

## Overview

The workflow in this scenario consists of:

1. A TFLite model used for object classification and feature extraction
2. A static LightweightKNN classifier (monitoring only, no adaptation)
3. Multiple autonomous drift detectors:
   - EWMAConfidenceMonitor: Tracks confidence scores for drift detection
   - PageHinkleyFeatureMonitor: Monitors statistical changes in feature values
   - PredictionDistributionMonitor: Detects shifts in prediction distribution
4. Synchronization with TinySphere server for drift event reporting and metrics collection

## Key Differences from InferencePipeline Scenario

While both scenarios use the same base TinyLCM framework, this "monitoring only" variant:

1. Uses an `InferencePipeline` instead of `AdaptivePipeline` (no adaptation)
2. Has `adapting_enabled` set to `false` in the classifier configuration
3. Has `enable_quarantine` and `enable_heuristic_adaptation` disabled
4. Employs multiple drift detectors for more robust monitoring
5. Focuses on reporting drift events rather than adapting to them

## Setup and Deployment

### Prerequisites

- Raspberry Pi Zero 2W (or other compatible device)
- Python 3.10+ with tflite_runtime or TensorFlow Lite
- Camera module connected to the Raspberry Pi
- Connection to a TinySphere server (optional but recommended)

### Installation

1. Copy the entire project to the Raspberry Pi, including:
   - The TinyLCM library
   - The example scripts
   - The TFLite model and labels

2. Install the required dependencies on the Pi:
   ```bash
   pip install -r examples/requirements.txt
   ```

3. Navigate to the scenario directory:
   ```bash
   cd examples/scenario1_monitoring_only
   ```

## Configuration

This scenario uses `config_scenario1.json` for all configuration. Key sections include:

- **adaptive_classifier**: Configures the LightweightKNN classifier (static in this scenario)
- **drift_detectors**: Configures multiple drift detectors with different sensitivity levels
- **operational_monitor**: Tracks system metrics and inference performance
- **sync_client**: Configures connection to TinySphere server for reporting

Example configuration highlights:

```json
"adaptive_classifier": {
  "type": "LightweightKNN",
  "k": 5,
  "max_samples": 100,
  "distance_metric": "cosine",
  "use_numpy": true,
  "adapting_enabled": false
},

"drift_detectors": [
  {
    "type": "EWMAConfidenceMonitor",
    "window_size": 50,
    "alpha": 0.1,
    "threshold_factor": 2.5,
    "min_samples": 100,
    "warmup_samples": 200
  },
  {
    "type": "PageHinkleyFeatureMonitor",
    "feature_index": 0,
    "delta": 0.005,
    "lambda_param": 50,
    "alpha": 0.1,
    "min_samples": 100,
    "warmup_samples": 200
  },
  {
    "type": "PredictionDistributionMonitor",
    "window_size": 50,
    "threshold": 0.2,
    "method": "block",
    "min_samples": 100
  }
]
```

## Running on the Raspberry Pi

1. Run the script:
   ```bash
   python main_scenario1.py
   ```

2. The script will:
   - Initialize the camera
   - Load the TFLite model
   - Register with the TinySphere server (if configured)
   - Start processing frames and detecting drift
   - Send drift events and metrics to the TinySphere server

## Flow Diagram

```
┌───────────────┐      ┌───────────────┐      ┌───────────────┐
│ Camera Input  │ ───> │ TFLite        │ ───> │ Static        │
│               │      │ Feature       │      │ LightweightKNN│
└───────────────┘      │ Extractor     │      │ (No Adaptation)
                       └───────────────┘      └───────┬───────┘
                                                      │
                                                      ▼
┌───────────────┐      ┌───────────────┐      ┌───────────────┐
│ TinySphere    │ <─── │ SyncClient    │ <─── │ Multiple Drift│
│ Server        │      │ (Reports)     │      │ Detectors     │
└───────────────┘      └───────────────┘      └───────────────┘
                                                      ▲
                                                      │
                                              ┌───────────────┐
                                              │ Operational   │
                                              │ Monitor       │
                                              └───────────────┘
```

## Important Code Sections

### LightweightKNN Configuration

In `main_scenario1.py`, the classifier is configured with these parameters:

```python
# Initialize LightweightKNN classifier
knn_config = tinylcm_config["adaptive_classifier"]
classifier = LightweightKNN(
    k=knn_config["k"],               # Number of nearest neighbors (5 in this example)
    distance_metric=knn_config["distance_metric"],  # Metric for distance calculation (cosine)
    max_samples=knn_config["max_samples"],       # Maximum number of samples to store (100)
    use_numpy=knn_config["use_numpy"]           # Use NumPy for faster calculations
)
```

### Drift Detection Initialization

In `main_scenario1.py`, around line 414:

```python
# Initialize drift detectors
drift_detectors = []
for detector_config in tinylcm_config["drift_detectors"]:
    if detector_config["type"] == "EWMAConfidenceMonitor":
        drift_detector = EWMAConfidenceMonitor(
            lambda_param=detector_config["alpha"],
            threshold_factor=detector_config["threshold_factor"],
            drift_window=detector_config["window_size"],
            warm_up_samples=detector_config["min_samples"],
            reference_update_interval=detector_config["warmup_samples"]
        )
        drift_detectors.append(drift_detector)
        
    elif detector_config["type"] == "PageHinkleyFeatureMonitor":
        # Create a function that extracts a specific feature
        feature_index = detector_config["feature_index"]
        feature_extractor_fn = lambda features: features[feature_index]
        
        drift_detector = PageHinkleyFeatureMonitor(
            feature_statistic_fn=feature_extractor_fn,
            delta=detector_config["delta"],
            lambda_threshold=detector_config["lambda_param"],
            warm_up_samples=detector_config["min_samples"],
            reference_update_interval=detector_config["warmup_samples"]
        )
        drift_detectors.append(drift_detector)
        
    elif detector_config["type"] == "PredictionDistributionMonitor":
        drift_detector = PredictionDistributionMonitor(
            window_size=detector_config.get("window_size", 100),
            threshold=detector_config.get("threshold", 0.15),
            method=detector_config.get("method", "block"),
            min_samples=detector_config.get("min_samples", 100)
        )
        drift_detectors.append(drift_detector)
```

### Drift Detection and Handling

The `on_drift_detected()` function in `main_scenario1.py` shows how drift events are handled:

```python
def on_drift_detected(drift_info: Dict[str, Any]) -> None:
    """Callback function for drift detection events."""
    timestamp = datetime.now().strftime("%Y-%m-%d %H:%M:%S")

    # Extract information from drift info
    detector_name = drift_info.get("detector", "Unknown")

    # Determine reason based on the detector type
    if "metric" in drift_info:
        metric = drift_info["metric"]
        current_value = drift_info.get("current_value", "unknown")
        reason = f"{metric} drift detected (current: {current_value})"
    else:
        reason = "Drift detected"

    # Log the drift detection
    logger.warning(f"DRIFT DETECTED by {detector_name}: {reason}")

    # Send the drift event to TinySphere if sync client is available
    if sync_client:
        # Save the current frame if drift image saving is enabled
        image_path = None
        if config["tinylcm"]["features"]["save_drift_images"] and current_frame is not None:
            drift_dir = Path("./drift_images")
            drift_dir.mkdir(exist_ok=True)

            image_path = drift_dir / f"drift_{timestamp.replace(' ', '_').replace(':', '-')}_{detector_name}.jpg"
            cv2.imwrite(str(image_path), current_frame)
            logger.info(f"Saved drift image to {image_path}")

        # Send the drift event to TinySphere
        success = sync_client.create_and_send_drift_event_package(
            detector_name=detector_name,
            reason=reason,
            metrics=metrics,
            sample=current_sample,
            image_path=str(image_path) if image_path else None
        )
```

## TinySphere Integration

The `ExtendedSyncClient` class in `examples/utils/sync_client.py` handles communication with the TinySphere server:

1. **Registration**: Registers the device with TinySphere on startup
2. **Drift Events**: Sends drift event packages when drift is detected
3. **Metrics**: Periodically sends system and operational metrics
4. **Model Information**: Sends the TFLite model and labels to TinySphere for reference

## Troubleshooting

1. **TFLite Installation**: On Raspberry Pi, it's recommended to use tflite_runtime instead of full TensorFlow:
   ```bash
   pip install https://github.com/google-coral/pycoral/releases/download/v2.0.0/tflite_runtime-2.5.0.post1-cp39-cp39-linux_armv7l.whl
   ```

2. **Camera Issues**: Ensure the camera is properly connected and permissions are set correctly:
   ```bash
   # Check if camera is detected
   vcgencmd get_camera
   
   # Add user to video group if needed
   sudo usermod -a -G video $USER
   ```

3. **TinySphere Connection**: Check server URL and API key in the config file if you're having trouble connecting

4. **Performance**: If inference is slow, consider:
   - Reducing the camera resolution in config_scenario1.json
   - Increasing the inference_interval_ms setting
   - Setting display_frames to false for headless operation

## Customization

To use your own custom model:

1. Train your own TFLite model or use a pre-trained one
2. Update the model path in config_scenario1.json
3. Adjust the feature_layer_index to select which layer to extract features from
4. Update the labels_path to point to your class labels file
5. Tune the drift detector parameters based on your model's characteristics

## Comparing Results in TinySphere

When using both this monitoring-only scenario and the adaptive pipeline scenario, you can compare drift detection results in TinySphere:

1. Deploy both scenarios on different devices or at different times
2. Monitor drift events in the TinySphere dashboard
3. Compare how quickly drift is detected in both scenarios
4. Analyze the effectiveness of multiple drift detectors versus the single detector approach