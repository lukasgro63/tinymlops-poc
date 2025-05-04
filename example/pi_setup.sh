#!/bin/bash
# Setup script for Raspberry Pi TinyMLOps deployment
# This script organizes the directory structure and installs TinyLCM

set -e  # Exit on any error

# Color coding for output
RED='\033[0;31m'
GREEN='\033[0;32m'
YELLOW='\033[0;33m'
NC='\033[0m' # No Color

echo -e "${GREEN}=== TinyMLOps Setup for Raspberry Pi ===${NC}"

# Create base directory
BASE_DIR="$HOME/tinymlops"
echo -e "${YELLOW}Creating directory structure in ${BASE_DIR}${NC}"
mkdir -p "$BASE_DIR"
cd "$BASE_DIR"

# Setup tinylcm directory (library code)
echo -e "${YELLOW}Setting up TinyLCM library...${NC}"
mkdir -p tinylcm
# This directory will contain the entire TinyLCM library
# The user needs to copy the entire tinylcm directory from the repo

# Setup src directory (application code)
echo -e "${YELLOW}Setting up application directory...${NC}"
mkdir -p src
mkdir -p src/models
mkdir -p src/data

# Create necessary data directories for TinyLCM
echo -e "${YELLOW}Creating data directories...${NC}"
mkdir -p src/tinylcm_data/models
mkdir -p src/tinylcm_data/data_logs
mkdir -p src/tinylcm_data/inference_logs

# Create symbolic link for launch script
if [ -f "$BASE_DIR/src/launch.sh" ]; then
  echo -e "${YELLOW}Creating symbolic link for launch script...${NC}"
  ln -sf "$BASE_DIR/src/launch.sh" "$BASE_DIR/launch.sh"
fi

# Setup virtual environment if requested
read -p "Do you want to create a Python virtual environment? (y/n) " create_venv
if [[ $create_venv == "y" || $create_venv == "Y" ]]; then
  echo -e "${YELLOW}Creating Python virtual environment...${NC}"
  python3 -m venv "$BASE_DIR/venv"
  source "$BASE_DIR/venv/bin/activate"
  echo -e "${GREEN}Virtual environment created and activated${NC}"
  
  # Install requirements
  if [ -f "$BASE_DIR/src/requirements.txt" ]; then
    echo -e "${YELLOW}Installing requirements...${NC}"
    pip install -r "$BASE_DIR/src/requirements.txt"
  fi
  
  # Install TinyLCM as a local package
  echo -e "${YELLOW}Installing TinyLCM as a local package...${NC}"
  cd "$BASE_DIR/tinylcm"
  pip install -e .
  cd "$BASE_DIR"
fi

echo -e "${GREEN}=== Setup Complete ===${NC}"
echo -e ""
echo -e "${YELLOW}Next steps:${NC}"
echo -e "1. Copy the tinylcm directory from the repository to $BASE_DIR/tinylcm"
echo -e "2. Copy the example files to $BASE_DIR/src"
echo -e "3. If using a virtual environment, activate it with: source $BASE_DIR/venv/bin/activate"
echo -e "4. Run the application with: $BASE_DIR/launch.sh"
echo -e ""
echo -e "${YELLOW}Directory structure:${NC}"
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
echo -e "├── venv/      # Python virtual environment (optional)"
echo -e "└── launch.sh  # Symbolic link to launch script"