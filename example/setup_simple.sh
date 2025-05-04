#!/bin/bash
# Vereinfachtes Setup-Skript für TinyMLOps auf Raspberry Pi
# Dieses Skript automatisiert den gesamten Setup-Prozess ohne virtuelle Umgebungen

set -e  # Bei Fehlern abbrechen

# Farbcodes für die Ausgabe
RED='\033[0;31m'
GREEN='\033[0;32m'
YELLOW='\033[0;33m'
BLUE='\033[0;34m'
NC='\033[0m' # Keine Farbe

echo -e "${BLUE}====================================================================${NC}"
echo -e "${BLUE}       TinyMLOps Einfaches Setup für Raspberry Pi                   ${NC}"
echo -e "${BLUE}====================================================================${NC}"

# Verzeichnisse definieren
BASE_DIR="$HOME/tinymlops"
TEMP_DIR="$HOME/temp_tinymlops"
REPO_URL="https://github.com/lukasgro63/tinymlops-poc.git"

# 1. Alte Installation entfernen, falls vorhanden
echo -e "\n${YELLOW}[1/7] Entferne alte Installation, falls vorhanden...${NC}"
if [ -d "$BASE_DIR" ]; then
    echo -e "${YELLOW}Bestehende Installation gefunden in $BASE_DIR${NC}"
    echo -e "${YELLOW}Diese wird gesichert und neu installiert...${NC}"
    
    # Sichern der Konfigurationsdatei, falls sie existiert
    if [ -f "$BASE_DIR/src/config.json" ]; then
        mkdir -p "$HOME/tinymlops_backup"
        cp "$BASE_DIR/src/config.json" "$HOME/tinymlops_backup/config.json"
        echo -e "${GREEN}✓ Konfiguration gesichert in ~/tinymlops_backup/config.json${NC}"
    fi
    
    # Entfernen der alten Installation
    rm -rf "$BASE_DIR"
fi
echo -e "${GREEN}✓ Bereit für neue Installation${NC}"

# 2. Verzeichnisstruktur erstellen
echo -e "\n${YELLOW}[2/7] Erstelle Verzeichnisstruktur...${NC}"
mkdir -p "$BASE_DIR"
mkdir -p "$BASE_DIR/tinylcm"
mkdir -p "$BASE_DIR/src"
mkdir -p "$BASE_DIR/src/models"
mkdir -p "$BASE_DIR/src/data"
mkdir -p "$BASE_DIR/src/tinylcm_data/models"
mkdir -p "$BASE_DIR/src/tinylcm_data/data_logs"
mkdir -p "$BASE_DIR/src/tinylcm_data/inference_logs"
mkdir -p "$BASE_DIR/src/tinylcm_data/sync"
echo -e "${GREEN}✓ Verzeichnisstruktur erstellt${NC}"

# 3. Raspberry Pi Pakete installieren
echo -e "\n${YELLOW}[3/7] Installiere benötigte Pakete...${NC}"
if [ -f /proc/device-tree/model ] && grep -q "Raspberry Pi" /proc/device-tree/model; then
    echo -e "${YELLOW}Raspberry Pi erkannt, installiere Systempakete...${NC}"
    
    # Installiere Grundpakete
    sudo apt update
    sudo apt install -y git python3 python3-pip
    
    # Installiere Raspberry Pi spezifische Pakete
    sudo apt install -y python3-picamera2 python3-libcamera python3-opencv 
    sudo apt install -y python3-numpy python3-psutil python3-gpiozero
    echo -e "${GREEN}✓ Raspberry Pi Pakete installiert${NC}"
else
    echo -e "${YELLOW}Kein Raspberry Pi erkannt, überspringe Pi-spezifische Pakete${NC}"
    sudo apt update
    sudo apt install -y git python3 python3-pip python3-opencv python3-numpy
fi

# 4. Repository klonen
echo -e "\n${YELLOW}[4/7] Klone Repository und Dateien kopieren...${NC}"
# Temporäres Verzeichnis säubern, falls vorhanden
if [ -d "$TEMP_DIR" ]; then
    rm -rf "$TEMP_DIR"
fi

# Clone mit Sparse-Checkout
git clone --filter=blob:none --sparse "$REPO_URL" "$TEMP_DIR"
cd "$TEMP_DIR"
git sparse-checkout set tinylcm example

# Dateien kopieren
cp -r tinylcm/* "$BASE_DIR/tinylcm/"
cp -r example/* "$BASE_DIR/src/"

# Ausführbar machen
chmod +x "$BASE_DIR/src/launch.sh"
chmod +x "$BASE_DIR/src/"*.py 2>/dev/null || true

# Symbolischer Link für launch.sh - Wichtig: Mit absolutem Pfad
ln -sf "$BASE_DIR/src/launch.sh" "$BASE_DIR/launch.sh"
chmod +x "$BASE_DIR/launch.sh"
chmod +x "$BASE_DIR/src/launch.sh"

# Temporäres Verzeichnis entfernen
cd "$HOME"
rm -rf "$TEMP_DIR"
echo -e "${GREEN}✓ Repository geklont und Dateien kopiert${NC}"

# 5. Python-Pakete installieren
echo -e "\n${YELLOW}[5/7] Installiere Python-Pakete...${NC}"
pip3 install --break-system-packages tflite-runtime requests psutil pyyaml
cd "$BASE_DIR/tinylcm"
pip3 install -e . --break-system-packages
cd "$BASE_DIR"
echo -e "${GREEN}✓ Python-Pakete installiert${NC}"

# 6. Konfiguration anpassen
echo -e "\n${YELLOW}[6/7] Konfiguriere TinySphere-Verbindung...${NC}"
CONFIG_FILE="$BASE_DIR/src/config.json"

# Prüfen, ob wir eine gesicherte Konfiguration haben
BACKUP_CONFIG="$HOME/tinymlops_backup/config.json"
if [ -f "$BACKUP_CONFIG" ]; then
    echo -e "${YELLOW}Wiederherstellung der gesicherten Konfiguration...${NC}"
    cp "$BACKUP_CONFIG" "$CONFIG_FILE"
    echo -e "${GREEN}✓ Konfiguration wiederhergestellt${NC}"
else
    # Neue Konfiguration erstellen
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
        "server_url": "http://192.168.0.66:8000",
        "api_key": "test_key",
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
    echo -e "${GREEN}✓ Neue Konfigurationsdatei erstellt${NC}"
    echo -e "${YELLOW}WICHTIG: Bitte passen Sie die TinySphere-Verbindung in $CONFIG_FILE an${NC}"
fi

# 7. Test der TinySphere Verbindung
echo -e "\n${YELLOW}[7/7] Teste TinySphere-Verbindung...${NC}"
cd "$BASE_DIR/src"
if python3 test_tinysphere_connection.py; then
    echo -e "${GREEN}✓ TinySphere-Verbindung erfolgreich${NC}"
else
    echo -e "${YELLOW}⚠ TinySphere-Verbindungstest fehlgeschlagen.${NC}"
    echo -e "${YELLOW}  Die Anwendung funktioniert trotzdem, aber Daten werden nicht synchronisiert.${NC}"
    echo -e "${YELLOW}  Bitte aktualisieren Sie die Konfiguration in $CONFIG_FILE${NC}"
fi

# Abschluss
echo -e "\n${BLUE}====================================================================${NC}"
echo -e "${BLUE}                      Setup Abgeschlossen!                         ${NC}"
echo -e "${BLUE}====================================================================${NC}"
echo -e "\n${GREEN}Ihre TinyMLOps-Umgebung wurde erfolgreich installiert.${NC}"

echo -e "\n${YELLOW}Anwendung starten:${NC}"
echo -e "${GREEN}$BASE_DIR/launch.sh${NC}"

echo -e "\n${YELLOW}Verzeichnisstruktur:${NC}"
echo -e "$BASE_DIR/"
echo -e "├── tinylcm/   # TinyLCM-Bibliothek"
echo -e "├── src/       # Anwendungscode"
echo -e "│   ├── models/               # ML-Modelle"
echo -e "│   ├── data/                 # Anwendungsdaten"
echo -e "│   ├── tinylcm_data/         # TinyLCM-Daten"
echo -e "│   └── launch.sh             # Start-Skript"
echo -e "└── launch.sh  # Symbolischer Link zum Start-Skript"

echo -e "\n${YELLOW}Wichtige Hinweise:${NC}"
echo -e "1. Fügen Sie Ihre Modelldateien zu $BASE_DIR/src/models/ hinzu"
echo -e "2. Stellen Sie sicher, dass die Kamera über 'sudo raspi-config' aktiviert ist"
echo -e "3. GUI-Modus (wenn Display vorhanden): $BASE_DIR/launch.sh --gui"
echo -e "4. Alternative Konfiguration: $BASE_DIR/launch.sh --config andere_config.json"

echo -e "\n${GREEN}Vielen Dank für die Verwendung von TinyMLOps!${NC}"