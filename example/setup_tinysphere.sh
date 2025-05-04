#!/bin/bash
# Setup TinySphere connection with improved testing

echo "Setting up TinySphere connection..."

# Make script executable
chmod +x test_tinysphere_connection.py

# Check if config.json exists
if [ ! -f "config.json" ]; then
    echo "Error: config.json not found!"
    exit 1
fi

# Extract server URL from config.json
SERVER_URL=$(grep -o '"server_url": *"[^"]*"' config.json | awk -F'"' '{print $4}')
if [ -z "$SERVER_URL" ]; then
    echo "Error: server_url not found in config.json!"
    exit 1
fi

# Test connection to TinySphere using Python script
echo "Testing connection to TinySphere server..."
echo "Server: $SERVER_URL"

if python3 test_tinysphere_connection.py; then
    echo "✅ TinySphere server is reachable!"
else
    echo "❌ Cannot reach TinySphere server!"
    echo "Please check:"
    echo "1. Is the TinySphere Docker container running?"
    echo "2. Is the server URL in config.json correct? ($SERVER_URL)"
    echo "3. Is the Pi on the same network as the TinySphere server?"
    
    # Ask if user wants to update the server URL
    echo ""
    echo "Would you like to update the server URL? (y/n)"
    read -n 1 -r
    echo ""
    if [[ $REPLY =~ ^[Yy]$ ]]; then
        echo "Enter the new server URL (including http:// or https://):"
        read NEW_SERVER_URL
        
        # Update config.json with the new server URL
        sed -i "s|\"server_url\": *\"[^\"]*\"|\"server_url\": \"$NEW_SERVER_URL\"|" config.json
        echo "Updated server URL to: $NEW_SERVER_URL"
        
        # Test connection again
        echo "Testing connection to updated server..."
        if python3 test_tinysphere_connection.py; then
            echo "✅ TinySphere server is now reachable!"
        else
            echo "❌ Still cannot reach TinySphere server. Continuing anyway."
        fi
    else
        echo "Continuing with existing server URL."
    fi
fi

# Use device_id_manager.py to handle device ID
echo "Setting up device ID..."
python3 -c "from device_id_manager import DeviceIDManager; print(f'Device ID: {DeviceIDManager().get_device_id()}')"

echo ""
echo "✅ TinySphere setup complete!"
echo "You can now run the stone detector with: ./launch.sh"