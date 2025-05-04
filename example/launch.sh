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

# Determine script's location and base directory
SCRIPT_DIR="$(cd "$(dirname "${BASH_SOURCE[0]}")" && pwd)"
BASE_DIR="$(dirname "$SCRIPT_DIR")"

# If running from a symlink (like in our new structure)
if [[ -L "${BASH_SOURCE[0]}" ]]; then
    SCRIPT_DIR="$(cd "$(dirname "$(readlink "${BASH_SOURCE[0]}")")" && pwd)"
    BASE_DIR="$(dirname "$SCRIPT_DIR")"
fi

# Activate virtual environment if it exists
if [ -d "$BASE_DIR/venv" ]; then
    echo "Activating virtual environment..."
    source "$BASE_DIR/venv/bin/activate"
fi

# Check if TinyLCM is available
if [ ! -d "$BASE_DIR/tinylcm" ]; then
    echo "Warning: TinyLCM directory not found at $BASE_DIR/tinylcm"
    echo "Make sure TinyLCM is installed in the correct directory"
fi

# Add TinyLCM directory to PYTHONPATH
export PYTHONPATH="${PYTHONPATH}:$BASE_DIR"

# Ensure CONFIG is an absolute path or relative to SCRIPT_DIR
if [[ "$CONFIG" != /* ]]; then  # If not an absolute path
    if [ -f "$SCRIPT_DIR/$CONFIG" ]; then
        # Config exists in script directory
        CONFIG="$SCRIPT_DIR/$CONFIG"
    elif [ -f "$BASE_DIR/$CONFIG" ]; then
        # Config exists in base directory
        CONFIG="$BASE_DIR/$CONFIG"
    else
        echo "Error: Config file $CONFIG not found in $SCRIPT_DIR or $BASE_DIR"
        exit 1
    fi
fi

# Double check that config file exists
if [ ! -f "$CONFIG" ]; then
    echo "Error: Config file $CONFIG not found"
    exit 1
fi

# Change to script directory for proper paths
cd "$SCRIPT_DIR"

# Check TinySphere connection if available
if [ -f "$SCRIPT_DIR/test_tinysphere_connection.py" ]; then
    echo "Testing TinySphere connection..."
    CONNECTION_RESULT=$(python3 test_tinysphere_connection.py)
    echo "$CONNECTION_RESULT"
    
    if echo "$CONNECTION_RESULT" | grep -q "✅"; then
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