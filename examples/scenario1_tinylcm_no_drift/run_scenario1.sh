#!/bin/bash
# Run Scenario 1 - TinyLCM without Drift Detection

echo "Starting Scenario 1 - TinyLCM without Drift Detection"
echo "===================================================="

# Create necessary directories
mkdir -p logs

# Run the scenario
python3 main_scenario1.py --config config_scenario1.json

echo "Scenario 1 completed. Check logs/ directory for performance metrics."