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

# Function to check if a directory is already setup
is_already_setup() {
    local dir="$1"
    if [ -d "$dir" ] && [ -d "$dir/tinylcm" ] && [ -d "$dir/src" ]; then
        return 0  # True
    else
        return 1  # False
    fi
}

# Define base directory
BASE_DIR="$HOME/tinymlops"
TEMP_REPO="$HOME/temp_tinymlops_repo"
REPO_URL="https://github.com/lukasgro63/tinymlops-poc.git"

# Check if setup already exists
if is_already_setup "$BASE_DIR"; then
    echo -e "\n${YELLOW}TinyMLOps appears to be already set up in $BASE_DIR${NC}"
    read -p "Do you want to update the existing installation? (y/N) " UPDATE_EXISTING
    if [[ ! "$UPDATE_EXISTING" =~ ^[Yy]$ ]]; then
        echo -e "${GREEN}Setup aborted. Using existing installation.${NC}"
        exit 0
    fi
    echo -e "${YELLOW}Updating existing installation...${NC}"
fi

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
    # Check which packages are already installed
    PI_PACKAGES=("python3-picamera2" "python3-libcamera" "python3-opencv" "python3-psutil" "python3-gpiozero" "python3-numpy")
    MISSING_PI_PACKAGES=()
    
    for pkg in "${PI_PACKAGES[@]}"; do
        if ! dpkg -s "$pkg" >/dev/null 2>&1; then
            MISSING_PI_PACKAGES+=("$pkg")
        fi
    done
    
    if [ ${#MISSING_PI_PACKAGES[@]} -gt 0 ]; then
        echo -e "${YELLOW}Installing missing Raspberry Pi packages: ${MISSING_PI_PACKAGES[*]}${NC}"
        sudo apt update
        sudo apt install -y "${MISSING_PI_PACKAGES[@]}"
        echo -e "${GREEN}✓ Raspberry Pi specific packages installed${NC}"
    else
        echo -e "${GREEN}✓ All Raspberry Pi specific packages already installed${NC}"
    fi
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
chmod +x "$BASE_DIR/src/"*.py 2>/dev/null || true
# Create symbolic link for launch script
ln -sf "$BASE_DIR/src/launch.sh" "$BASE_DIR/launch.sh"
chmod +x "$BASE_DIR/launch.sh"
# Clean up temporary repository
cd "$HOME"
rm -rf "$TEMP_REPO"
echo -e "${GREEN}✓ Files copied to appropriate directories${NC}"

# 6. Python environment setup
echo -e "\n${YELLOW}[6/9] Python-Umgebung einrichten...${NC}"
read -p "Möchten Sie ein Python virtuelles Environment erstellen? (nicht empfohlen) (y/N) " CREATE_VENV
if [[ "$CREATE_VENV" =~ ^[Yy]$ ]]; then
    echo -e "${YELLOW}Erstelle virtuelles Environment mit Zugriff auf System-Packages...${NC}"
    # Wichtig: --system-site-packages erlaubt die Nutzung der systemeigenen Paketversionen für numpy, opencv etc.
    python3 -m venv "$BASE_DIR/venv" --system-site-packages
    
    # Erstelle das Aktivierungsskript
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
echo -e "\n${YELLOW}[7/9] TinySphere-Verbindung konfigurieren...${NC}"
CONFIG_FILE="$BASE_DIR/src/config.json"

# Read current settings if file exists
if [ -f "$CONFIG_FILE" ]; then
    CURRENT_URL=$(grep -o '"server_url": *"[^"]*"' "$CONFIG_FILE" | cut -d'"' -f4 || echo "http://localhost:8000")
    CURRENT_KEY=$(grep -o '"api_key": *"[^"]*"' "$CONFIG_FILE" | cut -d'"' -f4 || echo "test_key")
    CURRENT_DEVICE=$(grep -o '"device_id": *"[^"]*"' "$CONFIG_FILE" | cut -d'"' -f4 || echo "pi-stone-detector")
fi

# Ask for TinySphere server URL
read -p "TinySphere-Server URL eingeben [${CURRENT_URL:-http://localhost:8000}]: " SERVER_URL
SERVER_URL=${SERVER_URL:-${CURRENT_URL:-http://localhost:8000}}

# Ask for TinySphere API key
read -p "TinySphere API-Key eingeben [${CURRENT_KEY:-test_key}]: " API_KEY
API_KEY=${API_KEY:-${CURRENT_KEY:-test_key}}

# Ask for device ID
read -p "Geräte-ID eingeben [${CURRENT_DEVICE:-pi-stone-detector}]: " DEVICE_ID
DEVICE_ID=${DEVICE_ID:-${CURRENT_DEVICE:-pi-stone-detector}}

# Update config.json with new values
if [ -f "$CONFIG_FILE" ]; then
    # Update existing file
    sed -i "s|\"server_url\": *\"[^\"]*\"|\"server_url\": \"$SERVER_URL\"|g" "$CONFIG_FILE"
    sed -i "s|\"api_key\": *\"[^\"]*\"|\"api_key\": \"$API_KEY\"|g" "$CONFIG_FILE"
    sed -i "s|\"device_id\": *\"[^\"]*\"|\"device_id\": \"$DEVICE_ID\"|g" "$CONFIG_FILE"
    echo -e "${GREEN}✓ Bestehende Konfigurationsdatei aktualisiert${NC}"
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
        "device_id": "$DEVICE_ID"
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
    echo -e "${GREEN}✓ Neue Konfigurationsdatei erstellt${NC}"
fi

# 8. Test TinySphere connection
echo -e "\n${YELLOW}[8/9] TinySphere-Verbindung testen...${NC}"
cd "$BASE_DIR/src"
if python3 test_tinysphere_connection.py; then
    echo -e "${GREEN}✓ TinySphere-Verbindung erfolgreich${NC}"
else
    echo -e "${YELLOW}⚠ TinySphere-Verbindungstest fehlgeschlagen. Die Anwendung wird trotzdem funktionieren, aber Daten werden nicht synchronisiert.${NC}"
    echo -e "${YELLOW}  Sie können die Konfiguration später in $CONFIG_FILE aktualisieren${NC}"
fi

# 9. Provide information about running the application
echo -e "\n${YELLOW}[9/9] Setup abgeschlossen!${NC}"
echo -e "\n${BLUE}====================================================================${NC}"
echo -e "${BLUE}                      Setup Abgeschlossen!                         ${NC}"
echo -e "${BLUE}====================================================================${NC}"
echo -e "\n${GREEN}Ihre TinyMLOps-Umgebung wurde erfolgreich eingerichtet.${NC}"
echo -e "\n${YELLOW}Anwendung starten:${NC}"
if [[ "$CREATE_VENV" =~ ^[Yy]$ ]]; then
    echo -e "1. Virtuelles Environment aktivieren:  ${GREEN}source $BASE_DIR/activate_venv.sh${NC}"
    echo -e "2. Anwendung starten:                  ${GREEN}$BASE_DIR/launch.sh${NC}"
else
    echo -e "1. Anwendung starten:                  ${GREEN}$BASE_DIR/launch.sh${NC}"
fi

echo -e "\n${YELLOW}Verzeichnisstruktur:${NC}"
echo -e "$BASE_DIR/"
echo -e "├── tinylcm/   # TinyLCM-Bibliothek"
echo -e "├── src/       # Anwendungscode"
echo -e "│   ├── models/               # ML-Modelle"
echo -e "│   ├── data/                 # Anwendungsdaten"
echo -e "│   ├── tinylcm_data/         # TinyLCM-Daten"
echo -e "│   │   ├── models/           # Modellspeicher"
echo -e "│   │   ├── data_logs/        # Datenlogs"
echo -e "│   │   └── inference_logs/   # Inferenzlogs"
echo -e "│   └── launch.sh             # Start-Skript"
if [[ "$CREATE_VENV" =~ ^[Yy]$ ]]; then
    echo -e "├── venv/      # Python virtuelles Environment"
    echo -e "├── activate_venv.sh  # Skript zum Aktivieren des virtuellen Environments"
fi
echo -e "└── launch.sh  # Symbolischer Link zum Start-Skript"

echo -e "\n${YELLOW}Wichtige Hinweise:${NC}"
echo -e "1. Fügen Sie Ihre Modelldateien zu $BASE_DIR/src/models/ hinzu."
echo -e "2. Stellen Sie sicher, dass die Kamera richtig angeschlossen und über 'sudo raspi-config' aktiviert ist."
echo -e "3. Zum Ausführen im GUI-Modus (wenn ein Display angeschlossen ist): $BASE_DIR/launch.sh --gui"
echo -e "4. Um eine andere Konfigurationsdatei zu verwenden: $BASE_DIR/launch.sh --config path/to/config.json"

# Exit with success message
echo -e "\n${GREEN}Vielen Dank für die Verwendung von TinyMLOps!${NC}"