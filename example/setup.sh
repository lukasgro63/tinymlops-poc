#!/bin/bash
# Setup script for Stone Detector with picamera2 support

echo "Setting up Stone Detector project..."

# Create necessary directories
echo "Creating directories..."
mkdir -p models
mkdir -p data
mkdir -p tinylcm_data/{models,data_logs,inference_logs,sync}

# Make scripts executable
chmod +x launch.sh
chmod +x setup.sh

# Check if running on Raspberry Pi
IS_RASPBERRY_PI=false
if [ -f /proc/device-tree/model ]; then
    if grep -q "Raspberry Pi" /proc/device-tree/model; then
        IS_RASPBERRY_PI=true
    fi
fi

if [ "$IS_RASPBERRY_PI" = true ]; then
    echo "Detected Raspberry Pi environment"
    
    # Check OS version (we need Bookworm for picamera2)
    if grep -q "bookworm" /etc/os-release; then
        echo "Detected Bookworm OS, installing picamera2..."
        sudo apt-get update
        sudo apt-get install -y python3-picamera2
    else
        echo "WARNING: This example requires Raspberry Pi OS Bookworm for picamera2 support."
        echo "Please upgrade your OS or the camera functionality will not work."
    fi
    
    # Install additional system dependencies
    echo "Installing system dependencies..."
    sudo apt-get install -y python3-opencv
else
    echo "Not running on a Raspberry Pi, skipping picamera2 installation"
    echo "Will use OpenCV fallback camera for development purposes"
fi

# Install Python dependencies
echo "Installing Python dependencies..."
pip3 install -r requirements.txt

echo "Setup complete!"
echo ""
echo "Next steps:"
echo "1. Place your TensorFlow Lite model in the models/ directory"
echo "2. Update config.json with your TinySphere credentials"
echo "3. Run the application with: ./launch.sh"
echo ""
echo "Note: For headless operation, run: ./launch.sh --headless"