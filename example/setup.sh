#!/bin/bash
# Setup script for Stone Detector

echo "Setting up Stone Detector project..."

# Create necessary directories
echo "Creating directories..."
mkdir -p models
mkdir -p data
mkdir -p tinylcm_data/{models,data_logs,inference_logs,sync}

# Make scripts executable
chmod +x launch.sh
chmod +x setup.sh

# Install Python dependencies
echo "Installing Python dependencies..."
pip3 install -r requirements.txt

echo "Setup complete!"
echo ""
echo "Next steps:"
echo "1. Place your TensorFlow Lite model in the models/ directory"
echo "2. Update config.json with your TinySphere credentials"
echo "3. Run the application with: ./launch.sh"