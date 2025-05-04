# TinyMLOps Stone Detector Examples

This directory contains example applications that demonstrate using TinyLCM for ML lifecycle management on resource-constrained devices like Raspberry Pi. The examples showcase different approaches to drift detection and adaptation.

## Example Structure

The examples are organized into different scenarios, each showcasing different features and capabilities of TinyLCM:

- **Common components** (`camera_handler.py`, `sync_client.py`, etc.) are shared across all scenarios
- **Scenario-specific implementations** (`scenarios/main_*.py`) demonstrate different adaptation and drift detection strategies
- **Configuration files** (`scenarios/config_*.json`) provide specific settings for each scenario

## Available Scenarios

### Original Stone Detector

**File:** `main.py`  
**Config:** `config.json`

The original stone detector example with basic TinyLCM integration.

### Autonomous Proxy-Based Drift Detection

**File:** `scenarios/main_autonomous.py`  
**Config:** `scenarios/config_autonomous.json`

This scenario demonstrates a fully autonomous adaptation system that:
- Uses proxy metrics (confidence, distribution, features) to detect drift without ground truth labels
- Quarantines samples that exhibit drift characteristics
- Applies heuristic adaptation techniques to update the model without requiring external validation
- Syncs with a central server for monitoring and tracking

Key features:
- Label-free drift detection
- Edge autonomy (no central server requirements for adaptation)
- Multiple proxy metrics for robust drift detection
- Adaptive heuristic strategies based on clustering and confidence patterns

## Running the Examples

Use the `run_example.py` script to run different scenarios:

```bash
# List all available scenarios
python run_example.py list

# Run the autonomous scenario (default)
python run_example.py run autonomous

# Run any scenario with a custom config
python run_example.py run autonomous --config path/to/custom_config.json

# Run in headless mode (no GUI)
python run_example.py run autonomous --headless
```

## Quick Setup (One-Line Installation)

```bash
curl -sSL https://raw.githubusercontent.com/lukasgro63/tinymlops-poc/main/example/one_line_install.sh | bash
```

This command will download and run the setup script that will:
1. Create the proper directory structure
2. Clone the repository
3. Set up TinyLCM
4. Install dependencies
5. Configure the application
6. Set up a launch script

## Features

- Real-time stone detection using TensorFlow Lite
- Integration with Raspberry Pi Camera using picamera2 (Bookworm-compatible)
- Seamless synchronization with TinySphere for remote monitoring
- Adaptive inference pipeline with streamlined metrics collection
- Non-blocking, asynchronous processing for responsive performance
- Headless operation support for embedded deployments
- Multiple drift detection and adaptation strategies

## Requirements

### Hardware
- Raspberry Pi Zero 2W (or any Raspberry Pi)
- Raspberry Pi Camera Module
- MicroSD card (8GB+)

### Software
- Raspberry Pi OS Bookworm (for picamera2 support)
- Python 3.9+

## Included Models

This example includes a demo TFLite model and labels file for detecting stones. These files are located in the `models/` directory and will be automatically installed during setup:

- `models/model.tflite` - A TensorFlow Lite model pre-trained to detect stones
- `models/labels.txt` - The corresponding labels file

For best results, you should train and use your own models, but these demo files allow you to test the system immediately after installation.

## Manual Setup Instructions

If you prefer a step-by-step setup, follow these instructions:

1. Create directory structure:
   ```bash
   mkdir -p ~/tinymlops
   mkdir -p ~/tinymlops/tinylcm
   mkdir -p ~/tinymlops/src
   ```

2. Clone the repository:
   ```bash
   git clone --filter=blob:none --sparse https://github.com/lukasgro63/tinymlops-poc.git ~/temp_repo
   cd ~/temp_repo
   git sparse-checkout set tinylcm example
   cp -r tinylcm/* ~/tinymlops/tinylcm/
   cp -r example/* ~/tinymlops/src/
   cd ~
   rm -rf ~/temp_repo
   ```

3. Install TinyLCM:
   ```bash
   cd ~/tinymlops/tinylcm
   pip3 install -e . --break-system-packages
   ```

4. Install dependencies:
   ```bash
   pip3 install numpy tflite-runtime requests psutil pyyaml --break-system-packages
   sudo apt install -y python3-picamera2 python3-libcamera python3-opencv
   ```

5. Place your TensorFlow Lite model and labels file in the models/ directory:
   ```
   ~/tinymlops/src/models/
        model.tflite
        labels.txt
   ```

6. Set up launch script:
   ```bash
   chmod +x ~/tinymlops/src/launch.sh
   ln -sf ~/tinymlops/src/launch.sh ~/tinymlops/launch.sh
   chmod +x ~/tinymlops/launch.sh
   ```

7. Make the run_example.py script executable:
   ```bash
   chmod +x ~/tinymlops/src/run_example.py
   ```

## Running the Original Application

### Headless Mode (Recommended for deployment)

```bash
./launch.sh --headless
```

### GUI Mode (For development and testing)

```bash
./launch.sh --gui
```

## Configuration

The `config.json` file contains all configuration options:

- **camera**: Camera settings (resolution, framerate, rotation)
- **model**: Model paths and threshold settings
- **tinylcm**: TinyLCM component settings
- **tinysphere**: TinySphere connection settings
- **application**: General application settings

Example:
```json
{
  "camera": {
    "resolution": [640, 480],
    "framerate": 10,
    "rotation": 0
  },
  "model": {
    "path": "models/model.tflite",
    "labels": "models/labels.txt",
    "threshold": 0.5
  },
  "tinylcm": {
    "model_dir": "tinylcm_data/models",
    "data_dir": "tinylcm_data/data_logs",
    "inference_dir": "tinylcm_data/inference_logs",
    "sync_interval_seconds": 300
  },
  "tinysphere": {
    "server_url": "http://192.168.0.66:8000",
    "api_key": "test_key",
    "device_id": "pi-stone-detector"
  },
  "application": {
    "detection_interval": 1,
    "save_detected_stones": true,
    "data_dir": "data",
    "log_level": "INFO",
    "headless": true
  }
}
```

## picamera2 Support

This example uses the newer picamera2 library, compatible with Raspberry Pi OS Bookworm. The camera handler includes a fallback to OpenCV for development environments without a Pi Camera.

## TinySphere Integration

The application regularly synchronizes data with TinySphere, including:

- Inference metrics (latency, confidence, throughput)
- Detected stone images and metadata
- System metrics (CPU, memory usage)
- Model status and performance data
- Quarantined samples (if using autonomous drift detection)

## Updated Directory Structure

```
example/
     main.py                # Original stone detector application
     run_example.py         # Script to run different example scenarios
     camera_handler.py      # Camera handling with picamera2
     stone_detector.py      # TFLite inference
     sync_client.py         # TinySphere communication
     device_id_manager.py   # Device identification
     system_metrics.py      # System metrics collection
     config.json            # Application configuration
     launch.sh              # Launcher script
     setup.sh               # Setup script
     setup_simple.sh        # Simple one-command setup
     test_tinysphere_connection.py # TinySphere connectivity test
     models/                # TFLite models
       └── model.tflite     # Pre-included demo model
       └── labels.txt       # Pre-included labels file
     scenarios/             # Different example scenarios
       └── main_autonomous.py  # Autonomous proxy-based drift detection
       └── config_autonomous.json # Configuration for autonomous mode
     tinylcm_data/          # TinyLCM data storage
     data/                  # Detection results
```

## Development Notes

- The application implements non-blocking I/O to prevent performance bottlenecks
- Inference runs in a dedicated thread pool
- Different adaptation strategies (label-based, autonomous proxy-based) for different use cases
- Data synchronization happens on a configurable interval
- Background threads for quarantine processing and heuristic adaptation to prevent blocking main execution

## Troubleshooting

- **Camera not working**: Ensure picamera2 is installed (`apt-get install python3-picamera2`)
- **TinySphere connection fails**: Check network settings and server URL in config.json
- **Model not found**: Ensure the models directory contains model.tflite and labels.txt
- **Drift detection not working**: Check that the necessary TinyLCM components are properly initialized

## License

This example is released under the same license as the TinyMLOps project.