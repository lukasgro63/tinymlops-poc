# TinyLCM Scenario 2: Drift Detection with Objects

This example demonstrates autonomous drift monitoring for a multi-class object recognition model using the TinyLCM framework. It is designed for deployment on a Raspberry Pi Zero 2W.

## Features

- **Autonomous drift detection** using KNN distance, feature, and confidence monitors
- **Feature transformation** using StandardScalerPCA to improve drift detection quality and performance
- **Efficient feature extraction** from TFLite image classification model
- **Opportunistic synchronization** with TinySphere server for drift validation

## Configuration

The example uses a JSON configuration file (`config_scenario2.json`) with the following key sections:

- **Device information**: Name, description, and logging settings
- **Camera settings**: Resolution, framerate, and rotation
- **Model configuration**: Paths to TFLite model and label files
- **TinyLCM components**:
  - Feature extractor: TFLiteFeatureExtractor configuration
  - Feature transformer: StandardScalerPCA configuration
  - Adaptive classifier: LightweightKNN settings
  - Drift detectors: Configuration for various drift detection algorithms
  - Operational monitor: System and performance metrics collection
  - Sync client: TinySphere server connection settings
  - Features: Additional feature flags
  - State management: Persistent state storage
  - Data logger: Inference and drift event logging

## Feature Transformation

This scenario uses a `StandardScalerPCATransformer` for feature preprocessing, which:

1. Standardizes features to have zero mean and unit variance
2. Applies PCA dimensionality reduction

This two-step approach provides several benefits:
- Standardization ensures all features contribute equally to drift detection
- PCA reduces dimensionality for faster KNN computation
- The combined approach improves both drift detection quality and model responsiveness

## Usage

```bash
python main_scenario2.py --config config_scenario2.json
```

## Requirements

- Python 3.7+
- TensorFlow Lite or TensorFlow
- OpenCV
- TinyLCM library
- Optional: Connection to TinySphere server for drift validation

## Integration with TinySphere

When drift is detected, the scenario:
1. Logs the drift event locally
2. Captures the current frame as an image
3. Creates a drift event package
4. Synchronizes with the TinySphere server when connectivity is available