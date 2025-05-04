# TinyMLOps on Raspberry Pi Zero 2W

This guide helps you set up TinyMLOps on a Raspberry Pi Zero 2W with Picamera v2.

## Prerequisites

- Raspberry Pi Zero 2W
- Picamera v2
- MicroSD card (8GB or larger)
- Raspberry Pi OS Lite (32-bit) Bookworm

## System Setup

### 1. Install Raspberry Pi OS

1. Download [Raspberry Pi Imager](https://www.raspberrypi.com/software/)
2. Select "Raspberry Pi OS (other)" → "Raspberry Pi OS Lite (32-bit)"
3. Ensure it's the Bookworm version
4. Configure SSH and WiFi in the imager settings
5. Flash to SD card and boot your Pi

### 2. Connect via SSH

```bash
# Find your Pi's IP address (on the Pi itself or router)
# Connect via SSH
ssh pi@<YOUR_PI_IP_ADDRESS>

# Default password is 'raspberry'
# Example:
ssh pi@192.168.1.100
```

### 3. Enable Camera

```bash
# Configure camera interface
sudo raspi-config
# Navigate to: Interface Options -> Camera -> Enable
# Then: Interface Options -> Legacy Camera -> Disable
sudo reboot
```

### 4. Create Project Structure

```bash
# Create main directory
mkdir -p ~/tinymlops
cd ~/tinymlops

# Create subdirectories
mkdir -p tinylcm
mkdir -p src
```

### 5. Install Dependencies

```bash
# Update system
sudo apt update && sudo apt upgrade -y

# Install system packages
sudo apt install -y python3-picamera2 python3-libcamera
sudo apt install -y python3-opencv
sudo apt install -y git

# Install Python packages
pip3 install numpy==1.23.5 tflite-runtime requests psutil --break-system-packages
```

### 6. Clone Repository (Sparse Checkout)

```bash
cd ~/tinymlops

# Clone only needed directories
git clone --filter=blob:none --sparse https://github.com/lukasgro63/tinymlops-poc.git temp_repo
cd temp_repo
git sparse-checkout set tinylcm examples

# Move directories to correct locations
cp -r tinylcm/* ../tinylcm/
cp -r examples/* ../src/
cd ..
rm -rf temp_repo
```

### 7. Install TinyLCM as Library

```bash
cd ~/tinymlops/tinylcm
pip3 install -e . --break-system-packages
```

### 8. Verify Installation

```python
cd ~/tinymlops/src
python3 -c "import tinylcm; import cv2; import picamera2; import tflite_runtime.interpreter as tflite; print('All packages imported successfully!')"
```

## Project Structure

```
~/tinymlops/
├── tinylcm/          # TinyLCM library
│   └── setup.py      # Library setup file
├── src/              # Application source code
│   └── [examples]    # Examples from repository
└── README.md         # This file
```

## Camera Test Script

Create a simple test script to verify camera functionality:

```python
# test_camera.py
from picamera2 import Picamera2
import cv2
import numpy as np

# Initialize camera
picam2 = Picamera2()
config = picam2.create_preview_configuration(main={"size": (640, 480)})
picam2.configure(config)
picam2.start()

# Capture frame
frame = picam2.capture_array()

# Convert to grayscale (example processing)
gray = cv2.cvtColor(frame, cv2.COLOR_BGR2GRAY)

# Save image
cv2.imwrite('test_capture.jpg', frame)
print("Image captured and saved as test_capture.jpg")

# Cleanup
picam2.stop()
```

## Usage Notes

### Headless Operation
This setup is designed for headless operation. Access your Pi via SSH and run scripts from the command line.

### Virtual Environment (Optional)
While not required for dedicated Pi projects, you can create a virtual environment:

```bash
python3 -m venv ~/tinymlops/venv
source ~/tinymlops/venv/bin/activate
```

### Package Versions
- Python: 3.11 (included in Bookworm)
- NumPy: 1.23.5 (specified version)
- OpenCV: Headless version
- Picamera2: Latest version for Bookworm
- TFLite Runtime: Latest version

## Troubleshooting

### Camera Not Detected
1. Check cable connection
2. Ensure camera is enabled in raspi-config
3. Verify with `libcamera-hello` command

### Import Errors
1. Verify all packages are installed
2. Check Python version: `python3 --version`
3. Ensure you're in the correct directory

### Memory Issues
The Pi Zero 2W has limited RAM (512MB). Monitor memory usage with:
```bash
free -h
# or
python3 -c "import psutil; print(f'Memory usage: {psutil.virtual_memory().percent}%')"
```


## Resources

- [Picamera2 Documentation](https://datasheets.raspberrypi.com/camera/picamera2-manual.pdf)
- [TensorFlow Lite Guide](https://www.tensorflow.org/lite/guide)
- [Raspberry Pi Documentation](https://www.raspberrypi.com/documentation/)