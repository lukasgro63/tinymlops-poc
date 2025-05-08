Du:
#!/bin/bash
# One-Line Installer for TinyLCM Examples
# This script can be executed directly via curl
# Usage: curl -sSL https://raw.githubusercontent.com/lukasgro63/tinymlops-poc/main/examples/one_line_install.sh | bash

set -e  # Exit on errors

# Color codes for output
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

# Check if we're running on a Raspberry Pi
IS_RASPBERRY_PI=false
if [ -f /proc/device-tree/model ] && grep -q "Raspberry Pi" /proc/device-tree/model; then
    IS_RASPBERRY_PI=true
    echo -e "${GREEN}Raspberry Pi detected: $(cat /proc/device-tree/model | tr -d '\0')${NC}"
else
    echo -e "${YELLOW}No Raspberry Pi detected. Installation will continue, but some features might not work.${NC}"
fi

# 1. Remove old installation if present
echo -e "\n${YELLOW}[1/5] Checking for existing installation...${NC}"
if [ -d "$BASE_DIR" ]; then
    echo -e "${YELLOW}Existing installation found in $BASE_DIR${NC}"
    read -p "Would you like to remove it and reinstall? (y/n) " -n 1 -r
    echo
    if [[ ! $REPLY =~ ^[Yy]$ ]]; then
        echo -e "${RED}Installation aborted.${NC}"
        exit 1
    fi
    
    # Backup configuration files if they exist
    if [ -d "$BASE_DIR/examples" ]; then
        mkdir -p "$HOME/tinymlops_backup/configs"
        find "$BASE_DIR/examples" -name "config*.json" -exec cp {} "$HOME/tinymlops_backup/configs/" \;
        echo -e "${GREEN}✓ Configurations backed up in ~/tinymlops_backup/configs/${NC}"
    fi
    
    # Remove old installation
    rm -rf "$BASE_DIR"
fi
echo -e "${GREEN}✓ Ready for new installation${NC}"

# 2. Install required packages
echo -e "\n${YELLOW}[2/5] Installing required packages...${NC}"
echo -e "${YELLOW}This may take a few minutes, please wait...${NC}"

# Install base packages
sudo apt update
sudo apt install -y git python3 python3-pip python3-venv libopenjp2-7 libatlas-base-dev

# Install Raspberry Pi specific packages if we're on a Pi
if [ "$IS_RASPBERRY_PI" = true ]; then
    sudo apt install -y python3-picamera2 python3-libcamera python3-opencv python3-numpy python3-psutil
    echo -e "${GREEN}✓ Raspberry Pi packages installed${NC}"
else
    sudo apt install -y python3-opencv python3-numpy
    echo -e "${GREEN}✓ General packages installed${NC}"
fi

# 3. Clone repository and copy files
echo -e "\n${YELLOW}[3/5] Cloning repository...${NC}"
# Clean up temporary directory if present
if [ -d "$TEMP_DIR" ]; then
    rm -rf "$TEMP_DIR"
fi

# Clone only the required parts of the repository
mkdir -p "$TEMP_DIR"
git clone --depth=1 --no-checkout "$REPO_URL" "$TEMP_DIR"
cd "$TEMP_DIR"
git sparse-checkout init --cone
git sparse-checkout set tinylcm examples
git checkout

# Create directory structure
mkdir -p "$BASE_DIR"

# Copy only the relevant parts
echo -e "${YELLOW}Copying files...${NC}"
cp -r "$TEMP_DIR/tinylcm" "$BASE_DIR/"
cp -r "$TEMP_DIR/examples" "$BASE_DIR/"

# Create necessary directories for all scenarios
find "$BASE_DIR/examples" -name "scenario*" -type d | while read scenario_dir; do
    mkdir -p "$scenario_dir/logs"
    mkdir -p "$scenario_dir/state"
    mkdir -p "$scenario_dir/debug"
    mkdir -p "$scenario_dir/drift_images"
    mkdir -p "$scenario_dir/sync_data"
    
    # Make main script executable if it exists
    main_script=$(find "$scenario_dir" -name "main_*.py" -type f)
    if [ -n "$main_script" ]; then
        chmod +x "$main_script"
    fi
done

# Verify model files
if [ -d "$BASE_DIR/examples/assets/model" ] && [ -f "$BASE_DIR/examples/assets/model/model.tflite" ]; then
    echo -e "${GREEN}✓ TFLite model found${NC}"
else
    echo -e "${YELLOW}⚠ No TFLite model found. Please place a model.tflite file in $BASE_DIR/examples/assets/model/${NC}"
fi

# Remove temporary directory
cd "$HOME"
rm -rf "$TEMP_DIR"
echo -e "${GREEN}✓ Repository cloned and files copied${NC}"

# 4. Install Python packages and tinylcm as a development package
echo -e "\n${YELLOW}[4/5] Installing Python packages...${NC}"
echo -e "${YELLOW}This may take several minutes...${NC}"

# Create a requirements.txt file if it doesn't exist
REQUIREMENTS_FILE="$BASE_DIR/examples/requirements.txt"
if [ ! -f "$REQUIREMENTS_FILE" ]; then
    echo -e "${YELLOW}Creating requirements.txt file...${NC}"
    cat > "$REQUIREMENTS_FILE" <<EOL
numpy>=1.20.0
pillow>=8.0.0
requests>=2.25.0
tflite-runtime>=2.7.0
opencv-python-headless>=4.5.0
psutil>=5.8.0
EOL
    echo -e "${GREEN}✓ requirements.txt created${NC}"
fi

# Install requirements
python3 -m pip install --break-system-packages -r "$REQUIREMENTS_FILE"

# Install tinylcm as editable package
cd "$BASE_DIR/tinylcm"
python3 -m pip install -e . --break-system-packages
cd "$BASE_DIR"

# Verify tinylcm installation
echo -e "${YELLOW}Verifying tinylcm module installation...${NC}"
if ! python3 -c "import tinylcm; print(f'TinyLCM Version: {tinylcm.__version__}')" 2>/dev/null; then
    echo -e "${YELLOW}tinylcm module not found. Adding to PYTHONPATH...${NC}"
    # Create PYTHONPATH helper file
    cat > "$BASE_DIR/set_pythonpath.sh" <<'EOL'
#!/bin/bash
# Helper script to set the correct PYTHONPATH
export PYTHONPATH="$HOME/tinymlops:$HOME/tinymlops/tinylcm:$PYTHONPATH"
EOL
    chmod +x "$BASE_DIR/set_pythonpath.sh"
    echo -e "${GREEN}✓ PYTHONPATH helper script created (use 'source set_pythonpath.sh' if you encounter issues)${NC}"
else
    echo -e "${GREEN}✓ tinylcm module correctly installed${NC}"
fi

echo -e "${GREEN}✓ Python packages installed${NC}"

# 5. Create launch script that lists and runs different scenarios
echo -e "\n${YELLOW}[5/5] Creating launch script...${NC}"

# Create helper wrapper file for scenarios
mkdir -p "$BASE_DIR/examples/utils/wrappers"

# Create a simpler solution - a Python path fixer script
cat > "$BASE_DIR/fix_paths.py" <<'EOL'
#!/usr/bin/env python3
import os
import sys

def fix_python_paths():
    """Fix Python import paths to find utils and other modules."""
    base_dir = os.path.dirname(os.path.dirname(os.path.abspath(__file__)))
    examples_dir = os.path.join(base_dir, "examples")
    utils_dir = os.path.join(examples_dir, "utils")
    tinylcm_dir = os.path.join(base_dir, "tinylcm")
    
    # Add paths to sys.path if not already there
    for path in [base_dir, examples_dir, utils_dir, tinylcm_dir]:
        if path not in sys.path:
            sys.path.insert(0, path)
    
    return True
EOL

# Create a wrapper script for each scenario
find "$BASE_DIR/examples" -name "scenario*" -type d | while read scenario_dir; do
    scenario_name=$(basename "$scenario_dir")
    main_script=$(find "$scenario_dir" -name "main_*.py" -type f)
    if [ -n "$main_script" ]; then
        script_basename=$(basename "$main_script" .py)
        wrapper_file="$BASE_DIR/examples/utils/wrappers/run_${scenario_name}.py"
        
        cat > "$wrapper_file" <<EOL
#!/usr/bin/env python3
"""
Wrapper script for ${scenario_name} to handle imports correctly
"""
import os
import sys

# Fix sys.path to include the necessary directories
sys.path.insert(0, "${BASE_DIR}")
sys.path.insert(0, "${BASE_DIR}/examples")
sys.path.insert(0, "${BASE_DIR}/examples/utils")
sys.path.insert(0, "${BASE_DIR}/tinylcm")

# Import and run the main script
sys.path.insert(0, "${scenario_dir}")
from ${script_basename} import main

if __name__ == "__main__":
    main()
EOL
        chmod +x "$wrapper_file"
        echo -e "${GREEN}✓ Created wrapper for ${scenario_name}${NC}"
    fi
done

LAUNCH_SCRIPT="$BASE_DIR/launch.sh"
cat > "$LAUNCH_SCRIPT" <<'EOL'
#!/bin/bash
# Launch script for TinyLCM examples

# Use absolute directory reference
SCRIPT_DIR="$(cd "$(dirname "${BASH_SOURCE[0]}")" && pwd)"
cd "$SCRIPT_DIR"

# Ensure local paths are in PYTHONPATH
export PYTHONPATH="$SCRIPT_DIR:$SCRIPT_DIR/examples:$SCRIPT_DIR/examples/utils:$SCRIPT_DIR/tinylcm:$PYTHONPATH"

# Find all scenario directories
SCENARIOS=()
SCENARIO_NAMES=()

echo -e "\033[0;34m===================================================================\033[0m"
echo -e "\033[0;34m                    TinyLCM Example Launcher                       \033[0m"
echo -e "\033[0;34m===================================================================\033[0m"

echo -e "\nAvailable scenarios:"

# Find all wrapper scripts
i=1
while read -r wrapper_script; do
    # Extract scenario name from the wrapper filename
    scenario_name=$(basename "$wrapper_script" .py | sed 's/run_//')
    
    SCENARIOS+=("$wrapper_script")
    SCENARIO_NAMES+=("$scenario_name")
    echo -e "  \033[0;32m$i\033[0m. $scenario_name"
    i=$((i+1))
done < <(find "$SCRIPT_DIR/examples/utils/wrappers" -name "run_*.py" | sort)

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

# Run the selected scenario wrapper
python3 "$selected_script"
EOL

chmod +x "$LAUNCH_SCRIPT"
echo -e "${GREEN}✓ Launch script created${NC}"

# Create a PYTHONPATH setup script to run at login
PROFILE_SCRIPT="/home/pi/.profile"
if [ -f "$PROFILE_SCRIPT" ]; then
    # Check if entry already exists
    if ! grep -q "PYTHONPATH.*tinymlops" "$PROFILE_SCRIPT"; then
        echo -e "\n# TinyLCM Python Path Setup" >> "$PROFILE_SCRIPT"
        echo "export PYTHONPATH=\"\$HOME/tinymlops:\$HOME/tinymlops/examples:\$HOME/tinymlops/examples/utils:\$HOME/tinymlops/tinylcm:\$PYTHONPATH\"" >> "$PROFILE_SCRIPT"
        echo -e "${GREEN}✓ Added PYTHONPATH to .profile for automatic setup on login${NC}"
    fi
fi

# Check camera configuration if on a Raspberry Pi
if [ "$IS_RASPBERRY_PI" = true ]; then
    echo -e "\n${YELLOW}Checking camera configuration...${NC}"
    # Check if the camera is enabled
    if vcgencmd get_camera | grep -q "detected=1"; then
        echo -e "${GREEN}✓ Camera is enabled and ready${NC}"
    else
        echo -e "${YELLOW}⚠ Camera may not be enabled or connected.${NC}"
        echo -e "${YELLOW}  Please enable the camera with 'sudo raspi-config' (Interface Options -> Camera)${NC}"
    fi
fi

# Conclusion
echo -e "\n${BLUE}====================================================================${NC}"
echo -e "${BLUE}                      Installation Complete!                       ${NC}"
echo -e "${BLUE}====================================================================${NC}"
echo -e "\n${GREEN}Your TinyLCM example environment has been installed successfully in:${NC}"
echo -e "${GREEN}$BASE_DIR${NC}"

echo -e "\n${YELLOW}To launch an example:${NC}"
echo -e "${GREEN}$BASE_DIR/launch.sh${NC}"

echo -e "\n${YELLOW}To ensure the correct Python path for this session:${NC}"
echo -e "${GREEN}source $BASE_DIR/set_pythonpath.sh${NC}"

echo -e "\n${YELLOW}To permanently set up the Python path, restart your session or run:${NC}"
echo -e "${GREEN}source ~/.profile${NC}"

echo -e "\n${GREEN}Enjoy using TinyLCM!${NC}"