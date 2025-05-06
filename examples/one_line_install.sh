#!/bin/bash
# One-Line Installer für TinyLCM Beispiele
# Dieses Skript kann direkt per curl ausgeführt werden
# Verwendung: curl -sSL https://raw.githubusercontent.com/lukasgrodmeier/tinymlops-poc/main/examples/one_line_install.sh | bash

set -e  # Bei Fehlern abbrechen

# Farbcodes für die Ausgabe
RED='\033[0;31m'
GREEN='\033[0;32m'
YELLOW='\033[0;33m'
BLUE='\033[0;34m'
NC='\033[0m' # Keine Farbe

echo -e "${BLUE}====================================================================${NC}"
echo -e "${BLUE}   TinyLCM One-Line Installer für Raspberry Pi Zero 2W              ${NC}"
echo -e "${BLUE}====================================================================${NC}"

# Verzeichnisse definieren
BASE_DIR="$HOME/tinylcm-examples"
TEMP_DIR="$HOME/temp_tinylcm"
REPO_URL="https://github.com/lukasgro63/tinymlops-poc.git"

# Überprüfen, ob wir auf einem Raspberry Pi laufen
IS_RASPBERRY_PI=false
if [ -f /proc/device-tree/model ] && grep -q "Raspberry Pi" /proc/device-tree/model; then
    IS_RASPBERRY_PI=true
    echo -e "${GREEN}Raspberry Pi erkannt: $(cat /proc/device-tree/model | tr -d '\0')${NC}"
else
    echo -e "${YELLOW}Kein Raspberry Pi erkannt. Installation wird fortgesetzt, aber einige Features könnten nicht funktionieren.${NC}"
fi

# 1. Alte Installation entfernen, falls vorhanden
echo -e "\n${YELLOW}[1/6] Prüfe auf bestehende Installation...${NC}"
if [ -d "$BASE_DIR" ]; then
    echo -e "${YELLOW}Bestehende Installation gefunden in $BASE_DIR${NC}"
    read -p "Möchten Sie diese entfernen und neu installieren? (j/n) " -n 1 -r
    echo
    if [[ ! $REPLY =~ ^[Jj]$ ]]; then
        echo -e "${RED}Installation abgebrochen.${NC}"
        exit 1
    fi
    
    # Sichern der Konfigurationsdatei, falls sie existiert
    if [ -f "$BASE_DIR/scenario1_monitoring_only/config_scenario1.json" ]; then
        mkdir -p "$HOME/tinylcm_backup"
        cp "$BASE_DIR/scenario1_monitoring_only/config_scenario1.json" "$HOME/tinylcm_backup/config_scenario1.json"
        echo -e "${GREEN}✓ Konfiguration gesichert in ~/tinylcm_backup/config_scenario1.json${NC}"
    fi
    
    # Entfernen der alten Installation
    rm -rf "$BASE_DIR"
fi
echo -e "${GREEN}✓ Bereit für neue Installation${NC}"

# 2. Benötigte Pakete installieren
echo -e "\n${YELLOW}[2/6] Installiere benötigte Pakete...${NC}"
echo -e "${YELLOW}Dies kann einige Minuten dauern, bitte warten...${NC}"

# Installiere Grundpakete
sudo apt update
sudo apt install -y git python3 python3-pip libopenjp2-7 libatlas-base-dev

# Installiere Raspberry Pi spezifische Pakete, wenn wir auf einem Pi sind
if [ "$IS_RASPBERRY_PI" = true ]; then
    sudo apt install -y python3-picamera2 python3-libcamera python3-opencv python3-numpy python3-psutil
    echo -e "${GREEN}✓ Raspberry Pi Pakete installiert${NC}"
else
    sudo apt install -y python3-opencv python3-numpy
    echo -e "${GREEN}✓ Allgemeine Pakete installiert${NC}"
fi

# 3. Repository klonen und Dateien kopieren
echo -e "\n${YELLOW}[3/6] Klone Repository...${NC}"
# Temporäres Verzeichnis säubern, falls vorhanden
if [ -d "$TEMP_DIR" ]; then
    rm -rf "$TEMP_DIR"
fi

# Clone Repository
git clone --depth=1 "$REPO_URL" "$TEMP_DIR"

# Erstelle Verzeichnisstruktur
mkdir -p "$BASE_DIR"

# Kopiere nur die relevanten Teile
echo -e "${YELLOW}Kopiere Dateien...${NC}"
cp -r "$TEMP_DIR/tinylcm" "$BASE_DIR/"
cp -r "$TEMP_DIR/examples/utils" "$BASE_DIR/"
cp -r "$TEMP_DIR/examples/scenario1_monitoring_only" "$BASE_DIR/"
cp -r "$TEMP_DIR/examples/assets" "$BASE_DIR/"

# Erstelle fehlende Verzeichnisse
mkdir -p "$BASE_DIR/scenario1_monitoring_only/logs"
mkdir -p "$BASE_DIR/scenario1_monitoring_only/state"
mkdir -p "$BASE_DIR/scenario1_monitoring_only/debug"
mkdir -p "$BASE_DIR/scenario1_monitoring_only/drift_images"
mkdir -p "$BASE_DIR/scenario1_monitoring_only/sync_data"

# Kopiere das Modell, wenn verfügbar
if [ -f "$TEMP_DIR/examples/assets/model/model.tflite" ]; then
    echo -e "${GREEN}✓ TFLite Modell gefunden und kopiert${NC}"
else
    echo -e "${YELLOW}⚠ Kein TFLite Modell gefunden. Bitte legen Sie ein model.tflite in $BASE_DIR/assets/model/ ab.${NC}"
fi

# Symbolischer Link für ausführbare Dateien
ln -sf "$BASE_DIR/scenario1_monitoring_only/main_scenario1.py" "$BASE_DIR/main_scenario1.py"
chmod +x "$BASE_DIR/scenario1_monitoring_only/main_scenario1.py"

# Temporäres Verzeichnis entfernen
cd "$HOME"
rm -rf "$TEMP_DIR"
echo -e "${GREEN}✓ Repository geklont und Dateien kopiert${NC}"

# 4. Python-Pakete installieren
echo -e "\n${YELLOW}[4/6] Installiere Python-Pakete...${NC}"
echo -e "${YELLOW}Dies kann mehrere Minuten dauern...${NC}"

# Installiere grundlegende Pakete
python3 -m pip install --break-system-packages numpy requests pillow

# Installiere tflite-runtime
python3 -m pip install --break-system-packages tflite-runtime

# Installiere OpenCV wenn nicht bereits installiert
if ! python3 -c "import cv2" 2>/dev/null; then
    python3 -m pip install --break-system-packages opencv-python-headless
fi

# Installiere TinyLCM als editable package
cd "$BASE_DIR/tinylcm"
python3 -m pip install -e . --break-system-packages
cd "$BASE_DIR"
echo -e "${GREEN}✓ Python-Pakete installiert${NC}"

# 5. Konfiguration, Fehlerbehebung und Launch-Skript erstellen
echo -e "\n${YELLOW}[5/6] Erstelle Konfiguration, behebe bekannte Fehler und erstelle Launch-Skript...${NC}"

# Konfiguration anpassen, wenn nötig
CONFIG_FILE="$BASE_DIR/scenario1_monitoring_only/config_scenario1.json"
BACKUP_CONFIG="$HOME/tinylcm_backup/config_scenario1.json"

if [ -f "$BACKUP_CONFIG" ]; then
    echo -e "${YELLOW}Wiederherstellung der gesicherten Konfiguration...${NC}"
    cp "$BACKUP_CONFIG" "$CONFIG_FILE"
    echo -e "${GREEN}✓ Konfiguration wiederhergestellt${NC}"
else
    # Bearbeite Server-URL entsprechend für lokalen Test
    sed -i 's/"server_url": "http:\/\/192.168.0.66:8000"/"server_url": "http:\/\/localhost:8000"/' "$CONFIG_FILE"
    echo -e "${GREEN}✓ Konfiguration angepasst${NC}"
fi

# Behebe bekannte Probleme im Code
echo -e "${YELLOW}Behebe bekannte Fehler im Code...${NC}"

# 1. Fix CircularImport Problem in data_structures.py
DATA_STRUCTURES_FILE="$BASE_DIR/tinylcm/core/data_structures.py"
if [ -f "$DATA_STRUCTURES_FILE" ]; then
    # Sichern der Originaldatei
    cp "$DATA_STRUCTURES_FILE" "${DATA_STRUCTURES_FILE}.bak"
    
    echo -e "${YELLOW}Korrigiere CircularImport-Problem in data_structures.py...${NC}"
    
    # Prüfe, ob die Datei bereits den TYPE_CHECKING-Import enthält
    if ! grep -q "TYPE_CHECKING" "$DATA_STRUCTURES_FILE"; then
        # Ersetze Import-Sektion
        sed -i '1s/^/from dataclasses import dataclass, field\nfrom enum import Enum\nfrom typing import Any, List, Optional, Dict, Union, TYPE_CHECKING, ForwardRef\nimport time\nimport numpy as np\nfrom uuid import uuid4\n\n# Forward reference for FeatureSample to resolve circular dependency\nif TYPE_CHECKING:\n    from typing import TypeAlias\n    FeatureSample: TypeAlias = "FeatureSample"\nelse:\n    FeatureSample = ForwardRef("FeatureSample")\n\n/' "$DATA_STRUCTURES_FILE"
        
        # Entferne alte Import-Zeilen
        sed -i '/^from dataclasses import dataclass, field$/d' "$DATA_STRUCTURES_FILE"
        sed -i '/^from enum import Enum$/d' "$DATA_STRUCTURES_FILE"
        sed -i '/^from typing import Any, List, Optional, Dict, Union$/d' "$DATA_STRUCTURES_FILE"
        sed -i '/^import time$/d' "$DATA_STRUCTURES_FILE"
        sed -i '/^import numpy as np$/d' "$DATA_STRUCTURES_FILE"
        sed -i '/^from uuid import uuid4$/d' "$DATA_STRUCTURES_FILE"
        
        # Füge Forward-Referenz-Auflösung am Ende hinzu
        echo -e "\n# Resolve the forward reference\nif not TYPE_CHECKING:\n    FeatureSample.__forward_arg__ = None" >> "$DATA_STRUCTURES_FILE"
        
        echo -e "${GREEN}✓ CircularImport-Problem in data_structures.py korrigiert${NC}"
    else
        echo -e "${GREEN}✓ data_structures.py ist bereits korrigiert${NC}"
    fi
fi

# 2. Korrigiere Importpfade in main_scenario1.py
MAIN_SCRIPT="$BASE_DIR/scenario1_monitoring_only/main_scenario1.py"
if [ -f "$MAIN_SCRIPT" ]; then
    # Sichern der Originaldatei
    cp "$MAIN_SCRIPT" "${MAIN_SCRIPT}.bak"
    
    echo -e "${YELLOW}Korrigiere Importpfade in main_scenario1.py...${NC}"
    
    # Modifiziere die Import-Sektion
    # Suche nach dem System-Pfad-Import und ersetze ihn
    if grep -q "sys.path.append" "$MAIN_SCRIPT"; then
        # Ersetze die Zeile mit einer verbesserten Variante
        sed -i 's/sys.path.append(os.path.dirname(os.path.dirname(os.path.abspath(__file__))))/sys.path.insert(0, os.path.dirname(os.path.dirname(os.path.abspath(__file__))))/' "$MAIN_SCRIPT"
        echo -e "${GREEN}✓ Importpfade in main_scenario1.py korrigiert${NC}"
    else
        echo -e "${YELLOW}Import-Pfad-Anweisung nicht gefunden - füge sie hinzu${NC}"
        # Füge die Zeile nach den Imports ein
        sed -i '/import numpy as np/a\
# Insert module path at the beginning of sys.path to ensure local modules are found first\
sys.path.insert(0, os.path.dirname(os.path.dirname(os.path.abspath(__file__))))' "$MAIN_SCRIPT"
        echo -e "${GREEN}✓ Importpfade in main_scenario1.py korrigiert${NC}"
    fi
fi

# 3. Erstelle ein verbessertes Launch-Skript
LAUNCH_SCRIPT="$BASE_DIR/launch.sh"
cat > "$LAUNCH_SCRIPT" <<EOF
#!/bin/bash
# Verbessertes Start-Skript für TinyLCM Beispiel Szenario 1

# Verwende die absolute Verzeichnis-Referenz
SCRIPT_DIR="\$(cd "\$(dirname "\${BASH_SOURCE[0]}")" && pwd)"
cd "\$SCRIPT_DIR"

# Stelle sicher, dass Python den korrekten Modul-Pfad hat
export PYTHONPATH="\$SCRIPT_DIR:\$PYTHONPATH"

echo "Starte TinyLCM Szenario 1 (Autonomes Monitoring)..."
cd "\$SCRIPT_DIR/scenario1_monitoring_only"
python3 main_scenario1.py "\$@"
EOF
chmod +x "$LAUNCH_SCRIPT"
echo -e "${GREEN}✓ Verbessertes Launch-Skript erstellt${NC}"

# Erstelle ein Test-Skript
TEST_SCRIPT="$BASE_DIR/test_tinysphere_connection.py"
cat > "$TEST_SCRIPT" <<EOF
#!/usr/bin/env python3
"""
Test-Skript für die TinySphere-Verbindung
"""

import json
import sys
import os
import requests
from pathlib import Path

# Lade die Konfiguration
config_path = Path("scenario1_monitoring_only/config_scenario1.json")
try:
    with open(config_path, 'r') as f:
        config = json.load(f)
    
    tinylcm_config = config.get("tinylcm", {})
    sync_config = tinylcm_config.get("sync_client", {})
    
    server_url = sync_config.get("server_url", "http://localhost:8000")
    api_key = sync_config.get("api_key", "tinylcm-demo-key")
    
    # Führe einen einfachen Verbindungstest durch
    headers = {"X-API-Key": api_key}
    print(f"Teste Verbindung zu {server_url}...")
    
    response = requests.get(f"{server_url}/api/status", headers=headers, timeout=10)
    if response.status_code == 200:
        print(f"✅ Verbindung erfolgreich! Server-Status: {response.json()}")
        sys.exit(0)
    else:
        print(f"❌ Verbindungsfehler. Server antwortete mit Status {response.status_code}: {response.text}")
        sys.exit(1)
except requests.RequestException as e:
    print(f"❌ Verbindungsfehler: {e}")
    sys.exit(1)
except Exception as e:
    print(f"❌ Fehler beim Testen der Verbindung: {e}")
    sys.exit(1)
EOF
chmod +x "$TEST_SCRIPT"

echo -e "${GREEN}✓ Launch-Skript und Test-Skript erstellt${NC}"

# 6. Kamera-Konfiguration prüfen und Hinweise anzeigen
echo -e "\n${YELLOW}[6/6] Prüfe Kamera-Konfiguration...${NC}"

if [ "$IS_RASPBERRY_PI" = true ]; then
    # Prüfe, ob die Kamera aktiviert ist
    if vcgencmd get_camera | grep -q "detected=1"; then
        echo -e "${GREEN}✓ Kamera ist aktiviert und bereit${NC}"
    else
        echo -e "${YELLOW}⚠ Kamera ist möglicherweise nicht aktiviert oder nicht angeschlossen.${NC}"
        echo -e "${YELLOW}  Bitte aktivieren Sie die Kamera mit 'sudo raspi-config' (Interface Options -> Camera)${NC}"
    fi
else
    echo -e "${YELLOW}⚠ Kein Raspberry Pi erkannt, Kamera-Konfiguration überspringen.${NC}"
fi

# Abschluss
echo -e "\n${BLUE}====================================================================${NC}"
echo -e "${BLUE}                      Installation Abgeschlossen!                   ${NC}"
echo -e "${BLUE}====================================================================${NC}"
echo -e "\n${GREEN}Ihre TinyLCM-Beispielumgebung wurde erfolgreich installiert in:${NC}"
echo -e "${GREEN}$BASE_DIR${NC}"

echo -e "\n${YELLOW}Anwendung starten:${NC}"
echo -e "${GREEN}$BASE_DIR/launch.sh${NC}"

echo -e "\n${YELLOW}TinySphere-Verbindung testen:${NC}"
echo -e "${GREEN}python3 $BASE_DIR/test_tinysphere_connection.py${NC}"

echo -e "\n${YELLOW}TinySphere-Server-URL in der Konfiguration anpassen (falls nötig):${NC}"
echo -e "${GREEN}nano $BASE_DIR/scenario1_monitoring_only/config_scenario1.json${NC}"

echo -e "\n${GREEN}Viel Spaß mit TinyLCM!${NC}"