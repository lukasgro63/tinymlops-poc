#!/bin/bash
# Setup TinySphere connection

echo "Setting up TinySphere connection..."

# Check if config.json exists
if [ ! -f "config.json" ]; then
    echo "Error: config.json not found!"
    exit 1
fi

# Test connection to TinySphere
echo "Testing connection to TinySphere server..."
echo "Server: http://192.168.0.66:8000"

# Try to ping the server
if curl -s -o /dev/null -w "%{http_code}" http://192.168.0.66:8000/api/status | grep -q "200"; then
    echo "✅ TinySphere server is reachable!"
else
    echo "❌ Cannot reach TinySphere server!"
    echo "Please check:"
    echo "1. Is the TinySphere Docker container running?"
    echo "2. Is the IP address correct?"
    echo "3. Is the Pi on the same network?"
    exit 1
fi

# Check if device_id.txt exists, create if not
if [ ! -f "device_id.txt" ]; then
    echo "Creating device ID file..."
    # Generate a unique device ID
    DEVICE_ID="pi-zero-stone-detector-$(hostname)"
    echo "$DEVICE_ID" > device_id.txt
    echo "Device ID created: $DEVICE_ID"
else
    DEVICE_ID=$(cat device_id.txt)
    echo "Using existing device ID: $DEVICE_ID"
fi

echo ""
echo "✅ TinySphere setup complete!"
echo "Device ID: $DEVICE_ID"
echo "Server: http://192.168.0.66:8000"
echo ""
echo "You can now run the stone detector with: ./launch.sh"