#!/bin/bash
# Simple One-Line Installer for TinyLCM Examples
# Usage: curl -sSL https://raw.githubusercontent.com/lukasgro63/tinymlops-poc/main/examples/one_line_install.sh | bash

# Exit on errors but don't exit on unbound variable
set -e

# Color codes for output
RED='\033[0;31m'
GREEN='\033[0;32m'
YELLOW='\033[0;33m'
BLUE='\033[0;34m'
NC='\033[0m' # No Color

echo -e "${BLUE}====================================================================${NC}"
echo -e "${BLUE}   TinyLCM One-Line Installer for Raspberry Pi Zero 2W              ${NC}"
echo -e "${BLUE}====================================================================${NC}"

# Define directories and repo
BASE_DIR="$HOME/tinymlops"
TEMP_DIR="$HOME/temp_tinymlops"
REPO_URL="https://github.com/lukasgro63/tinymlops-poc.git"

echo -e "\n${YELLOW}[1/4] Setting up directories and cloning repository...${NC}"
# Remove previous installation if it exists
if [ -d "$BASE_DIR" ]; then
  echo -e "${YELLOW}Existing installation found, removing...${NC}"
  rm -rf "$BASE_DIR"
fi

# Create fresh directory
mkdir -p "$BASE_DIR"

# Clone repository
echo -e "${YELLOW}Cloning repository...${NC}"
git clone --depth=1 "$REPO_URL" "$TEMP_DIR"

# Copy required directories
echo -e "${YELLOW}Copying files...${NC}"
cp -r "$TEMP_DIR/tinylcm" "$BASE_DIR/"
cp -r "$TEMP_DIR/examples" "$BASE_DIR/"

# Clean up
rm -rf "$TEMP_DIR"
echo -e "${GREEN}✓ Repository cloned and files copied${NC}"

echo -e "\n${YELLOW}[2/4] Installing required packages...${NC}"
# Install required packages
sudo apt update
sudo apt install -y git python3 python3-pip libopenjp2-7 libatlas-base-dev \
  python3-opencv python3-numpy python3-picamera2 python3-libcamera python3-psutil
echo -e "${GREEN}✓ System packages installed${NC}"

echo -e "\n${YELLOW}[3/4] Installing Python requirements...${NC}"
# Create a basic requirements file
cat > "$BASE_DIR/requirements.txt" <<EOL
numpy>=1.20.0
pillow>=8.0.0
requests>=2.25.0
tflite-runtime>=2.7.0
opencv-python-headless>=4.5.0
psutil>=5.8.0
EOL

# Install Python requirements
python3 -m pip install --user -r "$BASE_DIR/requirements.txt"

# Install tinylcm in development mode
cd "$BASE_DIR/tinylcm"
python3 -m pip install --user -e .
cd "$HOME"
echo -e "${GREEN}✓ Python packages installed${NC}"

echo -e "\n${YELLOW}[4/4] Creating launch script...${NC}"
# Create a launch script
cat > "$BASE_DIR/launch.sh" <<'EOL'
#!/bin/bash
# TinyLCM Launcher Script

# Set base directory
SCRIPT_DIR="$(cd "$(dirname "${BASH_SOURCE[0]}")" && pwd)"
cd "$SCRIPT_DIR"

# Setup PYTHONPATH
export PYTHONPATH="$SCRIPT_DIR:$SCRIPT_DIR/examples:$SCRIPT_DIR/examples/utils:$SCRIPT_DIR/tinylcm"

# Simple function to fix imports in main scripts
fix_imports() {
  local main_script="$1"
  local script_dir="$(dirname "$main_script")"
  
  # Add import fix at the top of the script 
  sed -i '3i # Add parent directory to path for imports\nimport os, sys\nsys.path.append(os.path.dirname(os.path.dirname(os.path.abspath(__file__))))' "$main_script"
}

# Find scenarios
echo -e "\033[0;34m===================================================================\033[0m"
echo -e "\033[0;34m                    TinyLCM Example Launcher                       \033[0m"
echo -e "\033[0;34m===================================================================\033[0m"

echo -e "\nAvailable scenarios:"

# Find all scenario directories and main scripts
declare -a SCENARIOS SCENARIO_NAMES
i=1
while read -r main_script; do
  if [ -n "$main_script" ]; then
    scenario_dir="$(dirname "$main_script")"
    scenario_name="$(basename "$scenario_dir")"
    
    # Fix imports in the script if needed
    if ! grep -q "sys.path.append" "$main_script"; then
      fix_imports "$main_script"
    fi
    
    SCENARIOS+=("$main_script")
    SCENARIO_NAMES+=("$scenario_name")
    echo -e "  \033[0;32m$i\033[0m. $scenario_name"
    i=$((i+1))
  fi
done < <(find "$SCRIPT_DIR/examples" -name "main_*.py" -type f | sort)

if [ ${#SCENARIOS[@]} -eq 0 ]; then
  echo -e "\033[0;31mNo scenarios found!\033[0m"
  exit 1
fi

# Ask user to select a scenario
echo -e "\nPlease select a scenario to run (1-${#SCENARIOS[@]}):"
read -r selection

# Validate selection
if ! [[ "$selection" =~ ^[0-9]+$ ]] || [ "$selection" -lt 1 ] || [ "$selection" -gt ${#SCENARIOS[@]} ]; then
  echo -e "\033[0;31mInvalid selection. Please enter a number between 1 and ${#SCENARIOS[@]}.\033[0m"
  exit 1
fi

# Get the selected scenario
selected_script="${SCENARIOS[$((selection-1))]}"
selected_name="${SCENARIO_NAMES[$((selection-1))]}"

echo -e "\nStarting \033[0;32m$selected_name\033[0m..."

# Run the selected scenario
cd "$(dirname "$selected_script")"
python3 "$(basename "$selected_script")"
EOL

chmod +x "$BASE_DIR/launch.sh"
echo -e "${GREEN}✓ Launch script created${NC}"

# Create a PYTHONPATH helper script
cat > "$BASE_DIR/setup_env.sh" <<'EOL'
#!/bin/bash
# Setup environment for TinyLCM

# Set Python path
export PYTHONPATH="$HOME/tinymlops:$HOME/tinymlops/examples:$HOME/tinymlops/examples/utils:$HOME/tinymlops/tinylcm"

echo "Environment variables set for TinyLCM"
echo "Your PYTHONPATH is now:"
echo "$PYTHONPATH"
EOL

chmod +x "$BASE_DIR/setup_env.sh"

# Conclusion
echo -e "\n${BLUE}====================================================================${NC}"
echo -e "${BLUE}                      Installation Complete!                       ${NC}"
echo -e "${BLUE}====================================================================${NC}"
echo -e "\n${GREEN}Your TinyLCM example environment has been installed successfully in:${NC}"
echo -e "${GREEN}$BASE_DIR${NC}"

echo -e "\n${YELLOW}To launch an example:${NC}"
echo -e "${GREEN}$BASE_DIR/launch.sh${NC}"

echo -e "\n${YELLOW}To ensure the correct Python path, first run:${NC}"
echo -e "${GREEN}source $BASE_DIR/setup_env.sh${NC}"

echo -e "\n${GREEN}Enjoy using TinyLCM!${NC}"