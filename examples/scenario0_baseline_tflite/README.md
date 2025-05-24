# Scenario 0 - Baseline TFLite Performance Test

This scenario provides a performance baseline by running pure TFLite inference without any TinyLCM components. It's designed to measure the raw inference performance and resource usage for comparison with TinyLCM-enabled scenarios.

## Purpose

- Establish baseline performance metrics for TFLite inference
- Measure CPU usage, memory consumption, and inference time without TinyLCM
- Provide comparison data for evaluating TinyLCM overhead

## Key Features

- Direct TFLite inference without feature extraction or drift detection
- Manual performance logging with detailed metrics:
  - Inference time (ms)
  - CPU usage (%)
  - Memory usage (MB)
  - Prediction results
- Consistent settings with Scenario 2 for fair comparison

## Usage

```bash
python main_scenario0.py --config config_scenario0.json
```

## Performance Metrics

The script logs the following metrics to `logs/performance_scenario0_*.json`:

- **System Info**: CPU count, total memory, versions
- **Per-Inference Metrics**:
  - Inference time in milliseconds
  - CPU percentage usage
  - Memory usage in MB
  - Prediction class and confidence
- **Summary Statistics**:
  - Average, min, max inference times
  - Average CPU and memory usage
  - Total runtime and inference count

## Configuration

The configuration file uses the same structure as other scenarios but only includes necessary components:

- Camera settings (resolution, framerate)
- Model paths (reuses Scenario 2 model)
- Inference interval (2000ms to match Scenario 2)

## Output

Performance data is saved in JSON format with one entry per line for easy parsing:

```json
{"timestamp": "2024-01-23T10:00:00", "type": "inference", "inference_time_ms": 150.5, "cpu_percent": 45.2, "memory_mb": 125.3, ...}
```

A summary is logged at the end with aggregate statistics for analysis.