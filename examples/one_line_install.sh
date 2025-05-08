#!/bin/bash
# TinyLCM One-Line Installer for Raspberry Pi Zero 2W
# Usage: curl -sSL https://raw.githubusercontent.com/lukasgro63/tinymlops-poc/main/examples/one_line_install.sh | bash

# Safe PYTHONPATH initialization
PYTHONPATH=${PYTHONPATH:-}

# Color codes
RED='\033[0;31m'
GREEN='\033[0;32m'
YELLOW='\033[0;33m'
BLUE='\033[0;34m'
NC='\033[0m' # No Color

echo -e "${BLUE}====================================================================${NC}"
echo -e "${BLUE}   TinyLCM One-Line Installer for Raspberry Pi Zero 2W              ${NC}"
echo -e "${BLUE}====================================================================${NC}"

# Define directories
BASE_DIR="$HOME/tinymlops"
TEMP_DIR="$HOME/temp_tinymlops"
REPO_URL="https://github.com/lukasgro63/tinymlops-poc.git"

# 1. Clone repository
echo -e "\n${YELLOW}[1/4] Cloning repository...${NC}"
if [ -d "$BASE_DIR" ]; then
  echo -e "${YELLOW}Existing installation found in $BASE_DIR${NC}"
  read -p "Would you like to remove it and reinstall? (y/n) " -n 1 -r
  echo
  if [[ $REPLY =~ ^[Yy]$ ]]; then
    rm -rf "$BASE_DIR"
    echo -e "${GREEN}✓ Removed existing installation${NC}"
  else
    echo -e "${RED}Installation aborted.${NC}"
    exit 1
  fi
fi

# Create fresh directory
mkdir -p "$BASE_DIR"

# Clone repository
git clone --depth=1 "$REPO_URL" "$TEMP_DIR"

# Copy required directories
echo -e "${YELLOW}Copying files...${NC}"
cp -r "$TEMP_DIR/tinylcm" "$BASE_DIR/"
cp -r "$TEMP_DIR/examples" "$BASE_DIR/"

# Clean up
rm -rf "$TEMP_DIR"
echo -e "${GREEN}✓ Repository cloned and files copied${NC}"

# 2. Install required packages
echo -e "\n${YELLOW}[2/4] Installing required packages...${NC}"
sudo apt update
sudo apt install -y git python3 python3-pip libopenjp2-7 libatlas-base-dev \
  python3-opencv python3-numpy python3-picamera2 python3-libcamera python3-psutil
echo -e "${GREEN}✓ Required packages installed${NC}"

# 3. Install Python requirements
echo -e "\n${YELLOW}[3/4] Installing Python requirements...${NC}"

# Create requirements.txt file
cat > "$BASE_DIR/requirements.txt" <<EOL
numpy>=1.20.0
pillow>=8.0.0
requests>=2.25.0
tflite-runtime>=2.7.0
opencv-python-headless>=4.5.0
psutil>=5.8.0
EOL

# Install Python requirements
echo -e "${YELLOW}Installing Python packages (this may take a few minutes)...${NC}"
python3 -m pip install --user -r "$BASE_DIR/requirements.txt"

# Install tinylcm in development mode
cd "$BASE_DIR/tinylcm"
python3 -m pip install --user -e .
cd "$HOME"
echo -e "${GREEN}✓ Python packages installed${NC}"

# 4. Create directories and fix imports
echo -e "\n${YELLOW}[4/4] Setting up directories and fixing imports...${NC}"

# Create necessary directories for scenarios
for scenario_dir in $(find "$BASE_DIR/examples" -name "scenario*" -type d); do
  mkdir -p "$scenario_dir/logs"
  mkdir -p "$scenario_dir/state"
  mkdir -p "$scenario_dir/debug"
  mkdir -p "$scenario_dir/drift_images"
  mkdir -p "$scenario_dir/sync_data"
done

# Create __init__.py files in utils directory
mkdir -p "$BASE_DIR/examples/utils"
touch "$BASE_DIR/examples/utils/__init__.py"
touch "$BASE_DIR/examples/__init__.py"

# FIX THE IMPORTS IN PYTHON FILES
# The key issue is that main_scenario1.py is using 'from utils.camera_handler' instead of relative imports

# Find each main Python file and fix the imports
for main_file in $(find "$BASE_DIR/examples" -name "main_*.py"); do
  echo -e "${YELLOW}Fixing imports in $(basename "$main_file")...${NC}"
  
  # Create a backup
  cp "$main_file" "${main_file}.bak"
  
  # Fix the import statements
  sed -i 's/from utils\./from examples.utils./g' "$main_file"
  
  chmod +x "$main_file"
done

# Create a launch script
cat > "$BASE_DIR/launch.sh" <<'EOL'
#!/bin/bash
# TinyLCM Launcher Script

# Set base directory
SCRIPT_DIR="$(cd "$(dirname "${BASH_SOURCE[0]}")" && pwd)"
cd "$SCRIPT_DIR"

# Setup PYTHONPATH - this is crucial
export PYTHONPATH="$SCRIPT_DIR:$PYTHONPATH"

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

# Run the selected scenario directly with Python
cd "$SCRIPT_DIR"
python3 "$selected_script"
EOL

chmod +x "$BASE_DIR/launch.sh"
echo -e "${GREEN}✓ Launch script created${NC}"

# Create a simple symlink to make utils directory accessible
echo -e "${YELLOW}Creating symlink for utils...${NC}"
cd "$BASE_DIR/examples/scenario1_monitoring_only"
ln -sf ../utils utils
cd "$HOME"

# Conclusion
echo -e "\n${BLUE}====================================================================${NC}"
echo -e "${BLUE}                      Installation Complete!                       ${NC}"
echo -e "${BLUE}====================================================================${NC}"
echo -e "\n${GREEN}Your TinyLCM example environment has been installed successfully in:${NC}"
echo -e "${GREEN}$BASE_DIR${NC}"

echo -e "\n${YELLOW}To launch an example:${NC}"
echo -e "${GREEN}$BASE_DIR/launch.sh${NC}"

echo -e "\n${GREEN}Enjoy using TinyLCM!${NC}"