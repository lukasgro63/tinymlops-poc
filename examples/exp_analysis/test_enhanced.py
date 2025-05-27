#!/usr/bin/env python3
"""Test the enhanced analysis script"""

import subprocess
import sys
from pathlib import Path

# Get the virtual environment python
venv_python = Path(__file__).parent / "venv" / "bin" / "python"

if not venv_python.exists():
    print("Virtual environment not found. Please create it first.")
    sys.exit(1)

# Run the enhanced script
script_path = Path(__file__).parent / "compare_scenarios_enhanced.py"

print("Running enhanced analysis script...")
result = subprocess.run([str(venv_python), str(script_path)], capture_output=True, text=True)

print("STDOUT:")
print(result.stdout)

if result.stderr:
    print("\nSTDERR:")
    print(result.stderr)

print(f"\nReturn code: {result.returncode}")