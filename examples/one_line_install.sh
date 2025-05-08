#!/usr/bin/env bash
#
# TinyLCM One-Line Installer for Raspberry Pi Zero 2W (global, idempotent)
# Usage: curl -sSL https://raw.githubusercontent.com/lukasgro63/tinymlops-poc/main/examples/one_line_install.sh | bash

set -euo pipefail
: "${PYTHONPATH:=}"  # Ensure PYTHONPATH is always defined

# Color codes
RED='\033[0;31m'; GREEN='\033[0;32m'; YELLOW='\033[0;33m'; BLUE='\033[0;34m'; NC='\033[0m'

echo -e "${BLUE}====================================================================${NC}"
echo -e "${BLUE}   TinyLCM One-Line Installer for Raspberry Pi Zero 2W              ${NC}"
echo -e "${BLUE}====================================================================${NC}"

# Directories and repo
BASE_DIR="$HOME/tinymlops"
TEMP_DIR="$HOME/temp_tinymlops"
REPO_URL="https://github.com/lukasgro63/tinymlops-poc.git"
REQ_FILE="$BASE_DIR/examples/requirements.txt"

# 1) Clone only if not present
if [ ! -d "$BASE_DIR/tinylcm" ]; then
  echo -e "${YELLOW}[1/5] Klone tinylcm + examples…${NC}"
  rm -rf "$TEMP_DIR" "$BASE_DIR"
  mkdir -p "$TEMP_DIR" "$BASE_DIR"
  git clone --filter=blob:none --sparse "$REPO_URL" "$TEMP_DIR"
  cd "$TEMP_DIR"
  git sparse-checkout set tinylcm examples
  cp -r tinylcm "$BASE_DIR/"
  cp -r examples "$BASE_DIR/"
  rm -rf "$TEMP_DIR"
else
  echo -e "${GREEN}[1/5] Quelle bereits vorhanden, überspringe Clone${NC}"
fi

# 2) Install system packages (apt skips installed automatically)
echo -e "\n${YELLOW}[2/5] Installiere System-Pakete…${NC}"
sudo apt update
sudo apt install -y git python3 python3-pip libopenjp2-7 libatlas-base-dev \
                    python3-opencv python3-numpy python3-psutil

# 3) Create requirements.txt if missing
if [ ! -f "$REQ_FILE" ]; then
  echo -e "\n${YELLOW}[3/5] Erstelle requirements.txt…${NC}"
  mkdir -p "$(dirname "$REQ_FILE")"
  cat > "$REQ_FILE" <<EOL
numpy>=1.20.0
pillow>=8.0.0
requests>=2.25.0
tflite-runtime>=2.7.0
opencv-python-headless>=4.5.0
psutil>=5.8.0
EOL
  echo -e "${GREEN}✓ requirements.txt erstellt${NC}"
else
  echo -e "\n${GREEN}[3/5] requirements.txt vorhanden, überspringe Erstellung${NC}"
fi

# 4) Install Python dependencies (only missing)
echo -e "\n${YELLOW}[4/5] Installiere Python-Pakete…${NC}"
python3 -m pip install --user --disable-pip-version-check -r "$REQ_FILE"

# 5) Install tinylcm as editable if not already importable
echo -e "\n${YELLOW}[5/5] Prüfe tinylcm-Installation…${NC}"
if ! python3 - <<'PYCODE' 2>/dev/null
import tinylcm
print("OK")
PYCODE
then
  echo -e "${YELLOW}→ Installiere tinylcm als editable…${NC}"
  python3 -m pip install --user -e "$BASE_DIR/tinylcm"
else
  echo -e "${GREEN}✓ tinylcm bereits installierbar, überspringe${NC}"
fi

# 6) Create robust launch.sh
echo -e "\n${YELLOW}Erzeuge/aktualisiere launch.sh…${NC}"
cat > "$BASE_DIR/launch.sh" << 'EOF'
#!/usr/bin/env bash
set -euo pipefail
: "${PYTHONPATH:=}"

# Determine base directory
BASE="$(cd "$(dirname "${BASH_SOURCE[0]}")" && pwd)"
cd "$BASE"

# Export PYTHONPATH safely
export PYTHONPATH="$BASE:$BASE/examples:$BASE/examples/utils:$BASE/tinylcm${PYTHONPATH:+:$PYTHONPATH}"

# Gather scenario scripts
declare -a SCRIPTS NAMES
i=1
for d in "$BASE/examples"/scenario*; do
  [ -d "$d" ] || continue
  main=$(find "$d" -maxdepth 1 -type f -name 'main_*.py' | head -n1)
  if [ -n "$main" ]; then
    SCRIPTS+=("$main")
    NAMES+=("$(basename "$d")")
    printf "  [%d] %s\n" "$i" "${NAMES[-1]}"
    ((i++))
  fi
done

total=${#SCRIPTS[@]}
if (( total == 0 )); then
  echo "Error: No scenarios found in $BASE/examples"
  exit 1
fi

echo
read -p "Select scenario to run (1-${total}): " sel
if ! [[ "$sel" =~ ^[0-9]+$ ]] || (( sel < 1 || sel > total )); then
  echo "Invalid selection"
  exit 1
fi

chosen="${SCRIPTS[$((sel-1))]}"
echo
echo "▶ Running scenario: ${NAMES[$((sel-1))]}"
python3 "$chosen"
EOF

chmod +x "$BASE_DIR/launch.sh"

echo -e "\n${GREEN}✓ Installation abgeschlossen!${NC}"
echo -e "  Deine Beispiele liegen in: ${GREEN}$BASE_DIR${NC}"
echo -e "  Starte sie mit:         ${GREEN}$BASE_DIR/launch.sh${NC}"
