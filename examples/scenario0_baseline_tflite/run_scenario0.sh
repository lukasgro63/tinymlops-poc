#!/bin/bash
# Run Scenario 0 - Baseline TFLite Performance Test

echo "Starting Scenario 0 - Baseline TFLite Performance Test"
echo "======================================================"

# Create necessary directories
mkdir -p logs

# Run the scenario
python3 main_scenario0.py --config config_scenario0.json

echo "Scenario 0 completed. Check logs/ directory for performance metrics."