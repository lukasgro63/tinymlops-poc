#!/bin/bash
# Create symbolic link for launch.sh in the main directory

set -e  # Exit on any error

# Color coding for output
GREEN='\033[0;32m'
YELLOW='\033[0;33m'
RED='\033[0;31m'
NC='\033[0m' # No Color

echo -e "${YELLOW}Creating symbolic link for launch.sh...${NC}"

# Determine base directory (parent of the current directory)
BASE_DIR="$(dirname "$(pwd)")"
CURRENT_DIR="$(pwd)"

# Check if we're in the src directory
if [[ "$(basename "$CURRENT_DIR")" != "src" ]]; then
    echo -e "${RED}Error: This script should be run from the src directory${NC}"
    echo -e "${YELLOW}Current directory: $CURRENT_DIR${NC}"
    echo -e "${YELLOW}Please navigate to the src directory and try again${NC}"
    exit 1
fi

# Check if launch.sh exists in the current directory
if [[ ! -f "$CURRENT_DIR/launch.sh" ]]; then
    echo -e "${RED}Error: launch.sh not found in the current directory${NC}"
    exit 1
fi

# Make sure launch.sh is executable
chmod +x "$CURRENT_DIR/launch.sh"

# Create symbolic link in the parent directory
ln -sf "$CURRENT_DIR/launch.sh" "$BASE_DIR/launch.sh"

if [[ $? -eq 0 ]]; then
    echo -e "${GREEN}Symbolic link created successfully at $BASE_DIR/launch.sh${NC}"
    echo -e "${YELLOW}You can now run the application from the main directory with: ./launch.sh${NC}"
else
    echo -e "${RED}Failed to create symbolic link${NC}"
    exit 1
fi