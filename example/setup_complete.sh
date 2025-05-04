#!/bin/bash
# Complete setup script for TinyMLOps on Raspberry Pi
# This script automates the entire setup process including:
# - Creating directory structure
# - Cloning repository
# - Setting up TinyLCM
# - Installing dependencies
# - Configuring the application
# - Setting up launch script

set -e  # Exit on any error

# Color codes for output
RED='\033[0;31m'
GREEN='\033[0;32m'
YELLOW='\033[0;33m'
BLUE='\033[0;34m'
NC='\033[0m' # No Color

echo -e "${BLUE}====================================================================${NC}"
echo -e "${BLUE}       TinyMLOps Complete Setup for Raspberry Pi                    ${NC}"
echo -e "${BLUE}====================================================================${NC}"

# Function to check if command exists
command_exists() {
    command -v "$1" >/dev/null 2>&1
}

# Define base directory
BASE_DIR="$HOME/tinymlops"
TEMP_REPO="$HOME/temp_tinymlops_repo"
REPO_URL="https://github.com/lukasgro63/tinymlops-poc.git"

# 1. Create base directory
echo -e "\n${YELLOW}[1/9] Creating base directory structure...${NC}"
mkdir -p "$BASE_DIR"
mkdir -p "$BASE_DIR/tinylcm"
mkdir -p "$BASE_DIR/src"
mkdir -p "$BASE_DIR/src/models"
mkdir -p "$BASE_DIR/src/data"
mkdir -p "$BASE_DIR/src/tinylcm_data/models"
mkdir -p "$BASE_DIR/src/tinylcm_data/data_logs"
mkdir -p "$BASE_DIR/src/tinylcm_data/inference_logs"
mkdir -p "$BASE_DIR/src/tinylcm_data/sync"
echo -e "${GREEN}✓ Directory structure created${NC}"

# 2. Check for required system packages and install if missing
echo -e "\n${YELLOW}[2/9] Checking required system packages...${NC}"
PACKAGES=("git" "python3" "python3-pip" "python3-venv")
MISSING_PACKAGES=()

for pkg in "${PACKAGES[@]}"; do
    if ! command_exists "$pkg"; then
        MISSING_PACKAGES+=("$pkg")
    fi
done

if [ ${#MISSING_PACKAGES[@]} -gt 0 ]; then
    echo -e "${YELLOW}Installing missing packages: ${MISSING_PACKAGES[*]}${NC}"
    sudo apt update
    sudo apt install -y "${MISSING_PACKAGES[@]}"
    echo -e "${GREEN}✓ Required packages installed${NC}"
else
    echo -e "${GREEN}✓ All required packages already installed${NC}"
fi

# 3. Install Raspberry Pi specific packages if on Raspberry Pi
echo -e "\n${YELLOW}[3/9] Checking for Raspberry Pi specific packages...${NC}"
if [ -f /proc/device-tree/model ] && grep -q "Raspberry Pi" /proc/device-tree/model; then
    echo -e "${YELLOW}Raspberry Pi detected, installing camera and GPIO libraries...${NC}"
    # Install picamera2 and related packages
    sudo apt install -y python3-picamera2 python3-libcamera
    # Install OpenCV
    sudo apt install -y python3-opencv
    # Install psutil for system metrics
    sudo apt install -y python3-psutil
    # Install GPIO libraries
    sudo apt install -y python3-gpiozero
    echo -e "${GREEN}✓ Raspberry Pi specific packages installed${NC}"
else
    echo -e "${YELLOW}Not running on Raspberry Pi, skipping Pi-specific packages${NC}"
fi

# 4. Clone the repository using sparse checkout to save space
echo -e "\n${YELLOW}[4/9] Cloning repository...${NC}"
if [ -d "$TEMP_REPO" ]; then
    rm -rf "$TEMP_REPO"
fi

git clone --filter=blob:none --sparse "$REPO_URL" "$TEMP_REPO"
cd "$TEMP_REPO"
git sparse-checkout set tinylcm example
echo -e "${GREEN}✓ Repository cloned${NC}"

# 5. Copy files to appropriate directories
echo -e "\n${YELLOW}[5/9] Copying files to appropriate directories...${NC}"
cp -r "$TEMP_REPO/tinylcm/"* "$BASE_DIR/tinylcm/"
cp -r "$TEMP_REPO/example/"* "$BASE_DIR/src/"
# Make scripts executable
chmod +x "$BASE_DIR/src/launch.sh"
chmod +x "$BASE_DIR/src/"*.py
# Create symbolic link for launch script
ln -sf "$BASE_DIR/src/launch.sh" "$BASE_DIR/launch.sh"
chmod +x "$BASE_DIR/launch.sh"
# Clean up temporary repository
cd "$HOME"
rm -rf "$TEMP_REPO"
echo -e "${GREEN}✓ Files copied to appropriate directories${NC}"

# 6. Create Python virtual environment if requested
echo -e "\n${YELLOW}[6/9] Python-Umgebung einrichten...${NC}"
read -p "Möchten Sie ein Python virtuelles Environment erstellen? (nicht empfohlen) (y/N) " CREATE_VENV
if [[ "$CREATE_VENV" =~ ^[Yy]$ ]]; then
    echo -e "${YELLOW}Erstelle virtuelles Environment mit Zugriff auf System-Packages...${NC}"
    # Wichtig: --system-site-packages erlaubt die Nutzung der systemeigenen Paketversionen für numpy, opencv etc.
    python3 -m venv "$BASE_DIR/venv" --system-site-packages
    
    # Erstellle das Aktivierungsskript
    cat > "$BASE_DIR/activate_venv.sh" << 'EOF'
#!/bin/bash
# Aktiviere das virtuelle Environment
source "$(dirname "$0")/venv/bin/activate"
echo "Virtuelles Environment aktiviert. 'deactivate' ausführen zum Beenden."
EOF
    chmod +x "$BASE_DIR/activate_venv.sh"
    
    # Aktiviere das virtuelle Environment
    source "$BASE_DIR/venv/bin/activate"
    echo -e "${GREEN}✓ Virtuelles Environment erstellt und aktiviert${NC}"
    
    # Installiere Python-Pakete im venv, nutze aber systemeigene Pakete wo möglich
    echo -e "${YELLOW}Installiere Python-Pakete...${NC}"
    pip install --upgrade pip
    pip install tflite-runtime requests psutil pyyaml
    # Installiere TinyLCM als editierbare Installation
    cd "$BASE_DIR/tinylcm"
    pip install -e .
    cd "$BASE_DIR"
else
    echo -e "${YELLOW}Installiere Python-Pakete system-weit...${NC}"
    # Für Raspberry Pi ist --break-system-packages erforderlich
    pip3 install tflite-runtime requests psutil pyyaml --break-system-packages
    # Installiere TinyLCM als editierbare Installation
    cd "$BASE_DIR/tinylcm"
    pip3 install -e . --break-system-packages
    cd "$BASE_DIR"
fi
echo -e "${GREEN}✓ Python-Umgebung eingerichtet${NC}"

# 7. Configure TinySphere connection
echo -e "\n${YELLOW}[7/9] Configuring TinySphere connection...${NC}"
CONFIG_FILE="$BASE_DIR/src/config.json"

# Read current settings if file exists
if [ -f "$CONFIG_FILE" ]; then
    CURRENT_URL=$(grep -o '"server_url": *"[^"]*"' "$CONFIG_FILE" | cut -d'"' -f4 || echo "http://localhost:8000")
    CURRENT_KEY=$(grep -o '"api_key": *"[^"]*"' "$CONFIG_FILE" | cut -d'"' -f4 || echo "test_key")
fi

# Ask for TinySphere server URL
read -p "Enter TinySphere server URL [${CURRENT_URL:-http://localhost:8000}]: " SERVER_URL
SERVER_URL=${SERVER_URL:-${CURRENT_URL:-http://localhost:8000}}

# Ask for TinySphere API key
read -p "Enter TinySphere API key [${CURRENT_KEY:-test_key}]: " API_KEY
API_KEY=${API_KEY:-${CURRENT_KEY:-test_key}}

# Update config.json with new values
if [ -f "$CONFIG_FILE" ]; then
    # Update existing file
    sed -i "s|\"server_url\": *\"[^\"]*\"|\"server_url\": \"$SERVER_URL\"|g" "$CONFIG_FILE"
    sed -i "s|\"api_key\": *\"[^\"]*\"|\"api_key\": \"$API_KEY\"|g" "$CONFIG_FILE"
else
    # Create new file if it doesn't exist
    cat > "$CONFIG_FILE" << EOF
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
        "server_url": "$SERVER_URL",
        "api_key": "$API_KEY",
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
EOF
fi
echo -e "${GREEN}✓ TinySphere connection configured${NC}"

# 8. Test TinySphere connection
echo -e "\n${YELLOW}[8/9] Testing TinySphere connection...${NC}"
cd "$BASE_DIR/src"
if python3 test_tinysphere_connection.py; then
    echo -e "${GREEN}✓ TinySphere connection successful${NC}"
else
    echo -e "${YELLOW}⚠ TinySphere connection test failed. The application will still work, but data will not be synced.${NC}"
    echo -e "${YELLOW}  You can update the configuration later in $CONFIG_FILE${NC}"
fi

# 9. Provide information about running the application
echo -e "\n${YELLOW}[9/9] Setup complete!${NC}"
echo -e "\n${BLUE}====================================================================${NC}"
echo -e "${BLUE}                      Setup Complete!                              ${NC}"
echo -e "${BLUE}====================================================================${NC}"
echo -e "\n${GREEN}Your TinyMLOps environment has been set up successfully.${NC}"
echo -e "\n${YELLOW}To run the application:${NC}"
if [[ "$CREATE_VENV" =~ ^[Yy]$ ]]; then
    echo -e "1. Activate the virtual environment:   ${GREEN}source $BASE_DIR/activate_venv.sh${NC}"
    echo -e "2. Run the application:                ${GREEN}$BASE_DIR/launch.sh${NC}"
else
    echo -e "1. Run the application:                ${GREEN}$BASE_DIR/launch.sh${NC}"
fi

echo -e "\n${YELLOW}Directory structure:${NC}"
echo -e "$BASE_DIR/"
echo -e "├── tinylcm/   # TinyLCM library"
echo -e "├── src/       # Application code"
echo -e "│   ├── models/               # ML models"
echo -e "│   ├── data/                 # Application data"
echo -e "│   ├── tinylcm_data/         # TinyLCM data"
echo -e "│   │   ├── models/           # Model storage"
echo -e "│   │   ├── data_logs/        # Data logs"
echo -e "│   │   └── inference_logs/   # Inference logs"
echo -e "│   └── launch.sh             # Launch script"
if [[ "$CREATE_VENV" =~ ^[Yy]$ ]]; then
    echo -e "├── venv/      # Python virtual environment"
    echo -e "├── activate_venv.sh  # Script to activate virtual environment"
fi
echo -e "└── launch.sh  # Symbolic link to launch script"

echo -e "\n${YELLOW}Important Notes:${NC}"
echo -e "1. You will need to add your model files to $BASE_DIR/src/models/"
echo -e "2. Make sure your camera is properly connected and enabled via 'sudo raspi-config'"
echo -e "3. To run in GUI mode (if you have a display): $BASE_DIR/launch.sh --gui"
echo -e "4. To use a different config file: $BASE_DIR/launch.sh --config path/to/config.json"

# Exit with success message
echo -e "\n${GREEN}Thank you for using TinyMLOps!${NC}"