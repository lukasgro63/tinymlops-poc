#!/bin/bash
# Launch script for Stone Detector

# Default to async version in headless mode
MODE="async"
HEADLESS=""

# Parse arguments
while [[ $# -gt 0 ]]; do
    case $1 in
        --sync)
            MODE="sync"
            shift
            ;;
        --gui)
            HEADLESS=""
            shift
            ;;
        --headless)
            HEADLESS="--headless"
            shift
            ;;
        *)
            echo "Unknown option: $1"
            echo "Usage: $0 [--sync|--gui|--headless]"
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

# Test TinySphere connection first
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

# Launch the appropriate version
if [ "$MODE" = "async" ]; then
    echo "Starting Stone Detector (Async version) $HEADLESS"
    python3 main_async.py $HEADLESS
else
    echo "Starting Stone Detector (Sync version) $HEADLESS"
    python3 main.py $HEADLESS
fi#!/bin/bash
# Launch script for Stone Detector

# Default to async version in headless mode
MODE="async"
HEADLESS=""

# Parse arguments
while [[ $# -gt 0 ]]; do
    case $1 in
        --sync)
            MODE="sync"
            shift
            ;;
        --gui)
            HEADLESS=""
            shift
            ;;
        --headless)
            HEADLESS="--headless"
            shift
            ;;
        *)
            echo "Unknown option: $1"
            echo "Usage: $0 [--sync|--gui|--headless]"
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

# Test TinySphere connection first
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

# Launch the appropriate version
if [ "$MODE" = "async" ]; then
    echo "Starting Stone Detector (Async version) $HEADLESS"
    python3 main_async.py $HEADLESS
else
    echo "Starting Stone Detector (Sync version) $HEADLESS"
    python3 main.py $HEADLESS
fi