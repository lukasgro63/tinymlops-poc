# Scenario 1: TinyLCM Inference Pipeline with Pre-trained KNN

This scenario demonstrates how to use TinyLCM for autonomous drift monitoring on a Raspberry Pi Zero 2W using a pre-trained Lightweight KNN classifier combined with a TFLite feature extractor.

## Overview

The workflow in this scenario consists of:

1. A pre-trained TFLite model used as a feature extractor
2. A pre-trained KNN classifier for LEGO brick color classification (red vs. green)
3. Autonomous drift detection using PageHinkleyFeatureMonitor
4. Optional synchronization with TinySphere server

## How the Pre-trained KNN State Works

### Initial State Creation

We use a script at `examples/assets/initial_states/create_initial_knn_state.py` to create a pre-trained KNN state file:

1. The script loads the same TFLite model that will be used on the Raspberry Pi
2. It processes a set of red and green LEGO brick images from `examples/assets/initial_states/images/`
3. Feature vectors are extracted using the TFLite model
4. A LightweightKNN classifier is trained on these feature vectors
5. The trained KNN state is saved to `examples/assets/initial_states/knn_initial_state_RG.json`

### Loading the Pre-trained State on the Raspberry Pi

When `main_scenario1.py` runs:

1. The script checks for a pre-trained KNN state file at the path specified in the config 
   (`config_scenario1.json` → `tinylcm.adaptive_classifier.initial_state_path`)
2. If found, it loads the classifier state using `classifier.set_state()`
3. If not found or if loading fails, it falls back to random initialization (not recommended for production)

## Setup and Deployment

### Prerequisites

- Raspberry Pi Zero 2W (or other compatible device)
- Python 3.10+ with tflite_runtime or TensorFlow Lite
- Camera module connected to the Raspberry Pi
- (Optional) Connection to a TinySphere server

### Configuration

This scenario uses `config_scenario1.json` for all configuration:

```json
{
  "device": {
    "log_level": "INFO"
  },
  "camera": {
    "resolution": [640, 480],
    "framerate": 30,
    "rotation": 0,
    "inference_resolution": [224, 224]
  },
  "model": {
    "path": "../assets/model/model.tflite",
    "labels_path": "../assets/model/labels.txt"
  },
  "application": {
    "inference_interval_ms": 100,
    "save_debug_frames": false,
    "debug_output_dir": "./debug_frames"
  },
  "tinylcm": {
    "feature_extractor": {
      "model_path": "../assets/model/model.tflite",
      "feature_layer_index": -1,
      "normalize_features": true,
      "lazy_loading": false
    },
    "adaptive_classifier": {
      "k": 3,
      "max_samples": 20,
      "distance_metric": "cosine",
      "use_numpy": true,
      "initial_state_path": "../assets/initial_states/knn_initial_state_RG.json"
    },
    "drift_detectors": [
      {
        "type": "PageHinkleyFeatureMonitor",
        "feature_index": 0,
        "delta": 0.005,
        "lambda_param": 10,
        "min_samples": 30,
        "warmup_samples": 50
      }
    ],
    "features": {
      "save_drift_images": true
    },
    "data_logger": {
      "enabled": true,
      "log_dir": "./logs"
    },
    "operational_monitor": {
      "track_system_metrics": true,
      "report_interval_seconds": 30
    },
    "state_manager": {
      "state_dir": "./states"
    },
    "sync_client": {
      "server_url": "http://tinysphere:8000",
      "api_key": "test_key",
      "sync_dir": "./sync_data",
      "sync_interval_seconds": 30,
      "max_retries": 3,
      "auto_register": true
    }
  }
}
```

Key elements to note:
- **initial_state_path**: Points to the path of the pre-trained KNN state
- **drift_detectors**: Configures the PageHinkleyFeatureMonitor for drift detection

## Running on the Raspberry Pi

1. Copy the entire project to the Raspberry Pi, including:
   - The TinyLCM library
   - The example scripts
   - The pre-trained KNN state file
   - The TFLite model and labels

2. Install the required dependencies on the Pi:
   ```bash
   pip install -r requirements.txt
   ```

3. Navigate to the scenario directory:
   ```bash
   cd examples/scenario1_tinylcm_inferencepipeline
   ```

4. Run the script:
   ```bash
   python main_scenario1.py
   ```

5. The script will:
   - Load the pre-trained KNN classifier state
   - Initialize the camera
   - Start processing frames and detecting drift
   - Optionally sync with TinySphere server

## Flow Diagram

```
┌───────────────┐      ┌───────────────┐      ┌───────────────┐
│ Camera Input  │ ───> │ TFLite        │ ───> │ Pre-trained   │
│               │      │ Feature       │      │ LightweightKNN│
└───────────────┘      │ Extractor     │      │               │
                       └───────────────┘      └───────┬───────┘
                                                      │
                                                      ▼
┌───────────────┐      ┌───────────────┐      ┌───────────────┐
│ TinySphere    │ <─── │ SyncClient    │ <─── │ Drift         │
│ Server        │      │               │      │ Detector      │
└───────────────┘      └───────────────┘      └───────────────┘
```

## Important Code Sections

### Loading the Pre-trained KNN State

In `main_scenario1.py`, around line 449:

```python
# Lade den initialen KNN-Zustand, falls vorhanden
initial_state_path_str = knn_config.get("initial_state_path")
loaded_initial_state = False

if initial_state_path_str:
    initial_state_path = Path(initial_state_path_str)
    if initial_state_path.exists():
        logger.info(f"Versuche, initialen k-NN Zustand zu laden von: {initial_state_path}")
        try:
            with open(initial_state_path, 'r') as f:
                loaded_state_data = json.load(f)
            
            if "classifier" in loaded_state_data and isinstance(loaded_state_data["classifier"], dict):
                classifier.set_state(loaded_state_data["classifier"])
                logger.info(f"LightweightKNN initialisiert mit {len(classifier.X_train)} Samples aus Zustand: {initial_state_path}")
                loaded_initial_state = True
            elif isinstance(loaded_state_data, dict) and "X_train" in loaded_state_data:
                # Fallback, falls nur KNN state direkt gespeichert wurde
                classifier.set_state(loaded_state_data)
                logger.info(f"LightweightKNN initialisiert mit {len(classifier.X_train)} Samples aus direktem Zustand: {initial_state_path}")
                loaded_initial_state = True
            else:
                logger.error(f"Schlüssel 'classifier' nicht in Zustandsdatei gefunden oder ungültiges Format: {initial_state_path}")
        except Exception as e:
            logger.error(f"Fehler beim Laden des initialen k-NN Zustands von {initial_state_path}: {e}")
```

### Drift Detection and Handling

In `main_scenario1.py`, the `on_drift_detected()` function shows how drift events are handled:

```python
def on_drift_detected(drift_info: Dict[str, Any]) -> None:
    """Callback function for drift detection events."""
    # Log the drift event
    # Save the current frame (optional)
    # Notify TinySphere (if configured)
```

## Troubleshooting

1. **KNN State Not Loading**: Check that the path in config_scenario1.json correctly points to knn_initial_state_RG.json

2. **Camera Issues**: Ensure the camera is properly connected and permissions are set correctly

3. **TFLite Installation**: On Raspberry Pi, it's recommended to use tflite_runtime instead of full TensorFlow

4. **Performance**: If inference is slow, consider reducing the camera resolution or increasing the inference interval

## Customization

To use your own custom model and KNN state:

1. Train your own TFLite model or use a pre-trained one
2. Update the model path in config_scenario1.json
3. Run the create_initial_knn_state.py script with your own images to create a custom KNN state
4. Update the initial_state_path in config_scenario1.json