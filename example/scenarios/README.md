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

   TFLite models often expect input data in a specific format (typically FLOAT32). When passing images from OpenCV (which are usually UINT8), we need to convert them. This is handled by preprocessor functions in the `preprocessors.py` module.
   
   In our implementation, we handle this preprocessing directly in the `process_frame_async` method rather than in the feature extractor initialization:

   ```python
   # First preprocess the frame - resize and convert to float32
   preprocessed_frame = resize_and_normalize(frame, target_size=(224, 224))
   
   # Then extract features using the preprocessed frame
   features = self.feature_extractor.extract_features(preprocessed_frame)
   
   # Make sure features is 1D
   if features.ndim > 1 and features.size > 1:
       # Flatten to 1D if needed
       features = features.flatten()
   ```

   This approach provides more control over the preprocessing steps and better error handling.

   **Common Errors:**
   - "Got value of type UINT8 but expected type FLOAT32 for input" - This indicates you need to convert your image to float32.
   - "The truth value of an array with more than one element is ambiguous" - This happens when a multi-dimensional array is used in a boolean context. Make sure to flatten array features or explicitly check array properties with .any() or .all().
   - "Error processing detections: list index out of range" - This typically happens when the model outputs don't match expected formats. Use the model_inspector.py tool (see below) to diagnose.

## Troubleshooting Tools

For troubleshooting model-related issues, we provide a model inspector tool:

```bash
python model_inspector.py --model models/model.tflite --image path/to/test_image.jpg
```

This tool will:
1. Load the model and print input/output tensor details
2. Preprocess the image according to the model's input requirements
3. Run inference and analyze the outputs
4. For object detection models, it will parse common output formats and show detected objects

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