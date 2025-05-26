#!/bin/bash
# Run the enhanced analysis script with the virtual environment

SCRIPT_DIR="$(cd "$(dirname "${BASH_SOURCE[0]}")" && pwd)"
VENV_PYTHON="$SCRIPT_DIR/venv/bin/python"

if [ ! -f "$VENV_PYTHON" ]; then
    echo "Virtual environment not found. Creating it now..."
    python3 -m venv "$SCRIPT_DIR/venv"
    "$SCRIPT_DIR/venv/bin/pip" install -r "$SCRIPT_DIR/requirements.txt"
fi

echo "Running enhanced analysis..."
"$VENV_PYTHON" "$SCRIPT_DIR/compare_scenarios_enhanced.py"