#!/bin/bash
# TinyLCM One-Line Installer for Raspberry Pi Zero 2W
# Usage: curl -sSL https://raw.githubusercontent.com/lukasgro63/tinymlops-poc/main/examples/one_line_install.sh | bash

# Safe PYTHONPATH initialization for versions that have set -u
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

# 1. Setup repository
echo -e "\n${YELLOW}[1/4] Setting up repository...${NC}"
# Remove previous installation if requested
if [ -d "$BASE_DIR" ]; then
  echo -e "${YELLOW}Existing installation found in $BASE_DIR${NC}"
  read -p "Would you like to remove it and reinstall? (y/n) " -n 1 -r
  echo
  if [[ $REPLY =~ ^[Yy]$ ]]; then
    rm -rf "$BASE_DIR"
    echo -e "${GREEN}✓ Removed existing installation${NC}"
  else
    echo -e "${YELLOW}Using existing installation...${NC}"
  fi
else
  echo -e "${GREEN}No existing installation found${NC}"
fi

# Create directories if needed
mkdir -p "$BASE_DIR"

# Clone repository if it doesn't exist
if [ ! -d "$BASE_DIR/tinylcm" ] || [ ! -d "$BASE_DIR/examples" ]; then
  echo -e "${YELLOW}Cloning repository...${NC}"
  # Clean up temporary directory if present
  rm -rf "$TEMP_DIR"
  mkdir -p "$TEMP_DIR"
  
  # Clone repository
  git clone --depth=1 "$REPO_URL" "$TEMP_DIR"
  
  # Copy only the parts we need
  if [ ! -d "$BASE_DIR/tinylcm" ]; then
    cp -r "$TEMP_DIR/tinylcm" "$BASE_DIR/"
  fi
  if [ ! -d "$BASE_DIR/examples" ]; then
    cp -r "$TEMP_DIR/examples" "$BASE_DIR/"
  fi
  
  # Cleanup
  rm -rf "$TEMP_DIR"
  echo -e "${GREEN}✓ Repository cloned and files copied${NC}"
else
  echo -e "${GREEN}✓ Repository already exists${NC}"
fi

# 2. Install required packages
echo -e "\n${YELLOW}[2/4] Installing required packages...${NC}"
sudo apt update
sudo apt install -y git python3 python3-pip libopenjp2-7 libatlas-base-dev \
  python3-opencv python3-numpy python3-picamera2 python3-libcamera python3-psutil

echo -e "${GREEN}✓ Required packages installed${NC}"

# 3. Install Python requirements
echo -e "\n${YELLOW}[3/4] Installing Python requirements...${NC}"

# Create a directory for Python packages
mkdir -p "$HOME/.local/lib/python3.9/site-packages"

# Create __init__.py files in the utils directory
mkdir -p "$BASE_DIR/examples/utils"
touch "$BASE_DIR/examples/utils/__init__.py"

# Create requirements.txt file if it doesn't exist
if [ ! -f "$BASE_DIR/examples/requirements.txt" ]; then
  cat > "$BASE_DIR/examples/requirements.txt" <<EOL
numpy>=1.20.0
pillow>=8.0.0
requests>=2.25.0
tflite-runtime>=2.7.0
opencv-python-headless>=4.5.0
psutil>=5.8.0
EOL
fi

# Install Python requirements
echo -e "${YELLOW}Installing Python packages (this may take a few minutes)...${NC}"
python3 -m pip install --user -r "$BASE_DIR/examples/requirements.txt"

# Install tinylcm in development mode
cd "$BASE_DIR/tinylcm"
python3 -m pip install --user -e .
cd "$HOME"
echo -e "${GREEN}✓ Python packages installed${NC}"

# 4. Create directories and fix imports
echo -e "\n${YELLOW}[4/4] Setting up directories and scripts...${NC}"

# Create necessary directories for scenarios
for scenario_dir in $(find "$BASE_DIR/examples" -name "scenario*" -type d); do
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

# Create .pth file to add our directories to Python path
USER_SITE_DIR=$(python3 -m site --user-site)
mkdir -p "$USER_SITE_DIR"
echo "$BASE_DIR" > "$USER_SITE_DIR/tinymlops.pth"
echo "$BASE_DIR/examples" >> "$USER_SITE_DIR/tinymlops.pth"
echo "$BASE_DIR/examples/utils" >> "$USER_SITE_DIR/tinymlops.pth"
echo "$BASE_DIR/tinylcm" >> "$USER_SITE_DIR/tinymlops.pth"
echo -e "${GREEN}✓ Python path configured${NC}"

# Create a simple launch script
cat > "$BASE_DIR/launch.sh" <<EOL
#!/bin/bash
# TinyLCM Launcher Script

# Set base directory
SCRIPT_DIR="\$(cd "\$(dirname "\${BASH_SOURCE[0]}")" && pwd)"
cd "\$SCRIPT_DIR"

# Find scenarios
echo -e "\033[0;34m===================================================================\033[0m"
echo -e "\033[0;34m                    TinyLCM Example Launcher                       \033[0m"
echo -e "\033[0;34m===================================================================\033[0m"

echo -e "\nAvailable scenarios:"

# Find all scenario directories and main scripts
declare -a SCENARIOS SCENARIO_NAMES
i=1
while read -r main_script; do
  if [ -n "\$main_script" ]; then
    scenario_dir="\$(dirname "\$main_script")"
    scenario_name="\$(basename "\$scenario_dir")"
    
    SCENARIOS+=("\$main_script")
    SCENARIO_NAMES+=("\$scenario_name")
    echo -e "  \033[0;32m\$i\033[0m. \$scenario_name"
    i=\$((i+1))
  fi
done < <(find "\$SCRIPT_DIR/examples" -name "main_*.py" -type f | sort)

if [ \${#SCENARIOS[@]} -eq 0 ]; then
  echo -e "\033[0;31mNo scenarios found!\033[0m"
  exit 1
fi

# Ask user to select a scenario
echo -e "\nPlease select a scenario to run (1-\${#SCENARIOS[@]}):"
read -r selection

# Validate selection
if ! [[ "\$selection" =~ ^[0-9]+\$ ]] || [ "\$selection" -lt 1 ] || [ "\$selection" -gt \${#SCENARIOS[@]} ]; then
  echo -e "\033[0;31mInvalid selection. Please enter a number between 1 and \${#SCENARIOS[@]}.\033[0m"
  exit 1
fi

# Get the selected scenario
selected_script="\${SCENARIOS[\$((selection-1))]}"
selected_name="\${SCENARIO_NAMES[\$((selection-1))]}"

echo -e "\nStarting \033[0;32m\$selected_name\033[0m..."

# Force Python to look in the right places
export PYTHONPATH="\$SCRIPT_DIR:\$SCRIPT_DIR/examples:\$SCRIPT_DIR/examples/utils:\$SCRIPT_DIR/tinylcm:\$PYTHONPATH"

# Create a simple wrapper script
cat > "\$SCRIPT_DIR/run_scenario.py" <<PYCODE
#!/usr/bin/env python3
import os
import sys

# Add all necessary paths
sys.path.insert(0, "\$SCRIPT_DIR")
sys.path.insert(0, "\$SCRIPT_DIR/examples")
sys.path.insert(0, "\$SCRIPT_DIR/examples/utils")
sys.path.insert(0, "\$SCRIPT_DIR/tinylcm")
sys.path.insert(0, os.path.dirname("\$selected_script"))

# Import and run the main function
main_module = os.path.basename("\$selected_script")[:-3]
__import__(main_module).main()
PYCODE

chmod +x "\$SCRIPT_DIR/run_scenario.py"

# Run the scenario using the wrapper
python3 "\$SCRIPT_DIR/run_scenario.py"
EOL

chmod +x "$BASE_DIR/launch.sh"
echo -e "${GREEN}✓ Launch script created${NC}"

# Create a persistent PYTHONPATH script
cat > "$BASE_DIR/set_env.sh" <<EOL
#!/bin/bash
# Set environment for TinyLCM

export PYTHONPATH="$BASE_DIR:$BASE_DIR/examples:$BASE_DIR/examples/utils:$BASE_DIR/tinylcm:\$PYTHONPATH"
echo "Python path set for TinyLCM"
echo "Current PYTHONPATH: \$PYTHONPATH"
EOL

chmod +x "$BASE_DIR/set_env.sh"

# Conclusion
echo -e "\n${BLUE}====================================================================${NC}"
echo -e "${BLUE}                      Installation Complete!                       ${NC}"
echo -e "${BLUE}====================================================================${NC}"
echo -e "\n${GREEN}Your TinyLCM example environment has been installed successfully in:${NC}"
echo -e "${GREEN}$BASE_DIR${NC}"

echo -e "\n${YELLOW}To launch an example:${NC}"
echo -e "${GREEN}$BASE_DIR/launch.sh${NC}"

echo -e "\n${YELLOW}If you have import issues, first run:${NC}"
echo -e "${GREEN}source $BASE_DIR/set_env.sh${NC}"

echo -e "\n${GREEN}Enjoy using TinyLCM!${NC}"