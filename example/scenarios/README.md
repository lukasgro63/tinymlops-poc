# TinyLCM Scenarios

This directory contains different scenarios for the TinyLCM example application. Each scenario demonstrates different aspects of the TinyLCM library's capabilities.

## Available Scenarios

### 1. Autonomous Stone Detection

**Files:**
- `main_autonomous.py`: Main entry point for the autonomous stone detection scenario
- `config_autonomous.json`: Configuration file for the autonomous scenario

**Features:**
- Autonomous drift detection using multiple detectors
- Quarantine buffer for storing samples with potential drift
- Heuristic-based adaptation
- Synchronization with TinySphere server
- Model performance monitoring

**Usage:**
```bash
python main_autonomous.py --config config_autonomous.json
```

### Running on Embedded Devices (Raspberry Pi)

When running on embedded devices like Raspberry Pi, the full TensorFlow library can be too resource-intensive. For these situations, we've provided alternatives:

1. **Use Detection Scores as Features**

   The configuration option `use_detection_scores_as_features` allows using the confidence scores from the object detector as feature vectors instead of extracting features from internal model layers. This is more efficient for resource-constrained devices.

   ```json
   "tinylcm": {
     "use_detection_scores_as_features": true
   }
   ```

2. **TensorFlow Lite Runtime**

   The example uses TensorFlow Lite Runtime (`tflite_runtime`) instead of the full TensorFlow library. This is already configured in the `requirements.txt` file.

   To install on Raspberry Pi:
   ```bash
   pip install tflite-runtime
   ```

3. **Input Data Type Handling**

   TFLite models often expect input data in a specific format (typically FLOAT32). When passing images from OpenCV (which are usually UINT8), we need to convert them. This is handled by preprocessor functions in the `preprocessors.py` module:

   ```python
   # Example usage in main_autonomous.py
   self.feature_extractor = TFLiteFeatureExtractor(
       model_path=self.config["model"]["path"],
       preprocessors=[
           # Resize and normalize the image
           lambda img: resize_and_normalize(img, target_size=(224, 224))
       ]
   )
   ```

   When you see an error like "Got value of type UINT8 but expected type FLOAT32 for input", make sure you're using the proper preprocessing.

## Creating New Scenarios

When creating new scenarios, follow this pattern:

1. Create a new Python file: `main_your_scenario.py`
2. Create a companion config file: `config_your_scenario.json`
3. Customize the components as needed for your scenario

## Common Configuration Options

| Option | Description | Default |
|--------|-------------|---------|
| `camera.resolution` | Camera resolution [width, height] | [640, 480] |
| `camera.framerate` | Frames per second | 10 |
| `model.path` | Path to TFLite model | "models/model.tflite" |
| `model.labels` | Path to labels file | "models/labels.txt" |
| `model.threshold` | Detection confidence threshold | 0.5 |
| `tinylcm.enable_autonomous_detection` | Enable autonomous drift detection | true |
| `tinylcm.enable_quarantine` | Enable quarantine buffer | true |
| `tinylcm.enable_heuristic_adaptation` | Enable heuristic adaptation | true |
| `tinylcm.use_detection_scores_as_features` | Use confidence scores as features instead of deep features | false |
| `application.detection_interval` | Interval between detections in seconds | 1 |
| `application.save_detected_stones` | Save detected stone images | true |
| `application.headless` | Run without GUI | true |