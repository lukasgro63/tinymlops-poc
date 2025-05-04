#!/bin/bash
# Launch script for Stone Detector with picamera2 support

# Default to headless mode for Raspberry Pi environment
HEADLESS=""
CONFIG="config.json"

# Check if running on Raspberry Pi and default to headless mode
if [ -f /proc/device-tree/model ] && grep -q "Raspberry Pi" /proc/device-tree/model; then
    HEADLESS="--headless"
    echo "Detected Raspberry Pi environment, defaulting to headless mode"
fi

# Parse arguments
while [[ $# -gt 0 ]]; do
    case $1 in
        --gui)
            HEADLESS=""
            shift
            ;;
        --headless)
            HEADLESS="--headless"
            shift
            ;;
        --config)
            CONFIG="$2"
            shift 2
            ;;
        *)
            echo "Unknown option: $1"
            echo "Usage: $0 [--gui|--headless] [--config CONFIG_PATH]"
            exit 1
            ;;
    esac
done

# Activate virtual environment if it exists
if [ -d "venv" ]; then
    echo "Activating virtual environment..."
    source venv/bin/activate
fi

# Check if TinyLCM is available
if [ ! -d "../tinylcm" ]; then
    echo "Warning: TinyLCM directory not found at ../tinylcm"
    echo "Make sure TinyLCM is installed in the parent directory"
fi

# Add parent directory to PYTHONPATH so we can import tinylcm
export PYTHONPATH="${PYTHONPATH}:$(dirname $(pwd))"

# Check for config file existence
if [ ! -f "$CONFIG" ]; then
    echo "Error: Config file $CONFIG not found"
    exit 1
fi

# Check TinySphere connection if available
if [ -f "test_tinysphere_connection.py" ]; then
    echo "Testing TinySphere connection..."
    if python3 test_tinysphere_connection.py | grep -q "✅ Server is online"; then
        echo "TinySphere connection successful!"
    else
        echo "Warning: TinySphere connection might not be working correctly"
        echo "Continue anyway? (y/n)"
        read -n 1 -r
        echo
        if [[ ! $REPLY =~ ^[Yy]$ ]]; then
            exit 1
        fi
    fi
fi

# Launch the application
echo "Starting Stone Detector application $HEADLESS with config $CONFIG"
python3 main.py --config "$CONFIG" $HEADLESS