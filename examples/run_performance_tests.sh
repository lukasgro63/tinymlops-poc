#!/bin/bash
# Run all performance test scenarios

echo "TinyLCM Performance Evaluation"
echo "=============================="
echo "Running 3 scenarios with 5 Hz inference rate (200ms interval)"
echo ""

# Create a timestamp for this test run
TIMESTAMP=$(date +"%Y%m%d_%H%M%S")
LOG_DIR="performance_logs_${TIMESTAMP}"

echo "Test run: ${TIMESTAMP}"
echo "Results will be saved to: ${LOG_DIR}"
echo ""

# Scenario 0: Baseline TFLite
echo "Starting Scenario 0 - Baseline TFLite (no TinyLCM)..."
cd scenario0_baseline_tflite
python3 main_scenario0.py --config config_scenario0.json
echo "Scenario 0 completed."
echo ""

# Wait between scenarios
sleep 5

# Scenario 1: TinyLCM without drift detection
echo "Starting Scenario 1 - TinyLCM without drift detection..."
cd ../scenario1_tinylcm_no_drift
python3 main_scenario1.py --config config_scenario1.json
echo "Scenario 1 completed."
echo ""

# Wait between scenarios
sleep 5

# Scenario 2: Full TinyLCM with drift detection
echo "Starting Scenario 2 - Full TinyLCM with drift detection..."
cd ../scenario2_drift_objects
python3 main_scenario2.py --config config_scenario2_performance.json
echo "Scenario 2 completed."
echo ""

# Collect all logs
cd ..
mkdir -p "${LOG_DIR}"
echo "Collecting performance logs..."

# Copy logs from each scenario
cp -r scenario0_baseline_tflite/logs/* "${LOG_DIR}/" 2>/dev/null
cp -r scenario1_tinylcm_no_drift/logs/* "${LOG_DIR}/" 2>/dev/null
cp -r scenario2_drift_objects/logs/* "${LOG_DIR}/" 2>/dev/null

# Run analysis
echo ""
echo "Running performance analysis..."
python3 analyze_performance.py --log-dir "${LOG_DIR}"

echo ""
echo "Performance evaluation complete!"
echo "Results saved in: ${LOG_DIR}"