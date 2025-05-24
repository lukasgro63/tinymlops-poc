# Scenario 1 - TinyLCM without Drift Detection

This scenario evaluates the performance of TinyLCM's core components (feature extraction, transformation, and KNN classification) WITHOUT drift detection. It provides a middle ground between pure TFLite inference (Scenario 0) and full TinyLCM with drift detection (Scenario 2).

## Purpose

- Measure the overhead of TinyLCM's core inference pipeline
- Isolate the performance impact of feature extraction and KNN classification
- Provide comparison data between baseline TFLite and full TinyLCM

## Key Features

- TinyLCM feature extraction using MobileNetV2
- StandardScaler + PCA feature transformation (256D)
- KNN classifier for predictions
- **NO drift detection components**
- Detailed performance logging with timing breakdown

## Usage

```bash
python main_scenario1.py --config config_scenario1.json
```

## Performance Metrics

The script logs detailed metrics to `logs/performance_scenario1_*.json`:

- **System Info**: CPU count, total memory, versions
- **Per-Inference Metrics**:
  - Total inference time (ms)
  - Feature extraction time (ms)
  - KNN inference time (ms)
  - CPU percentage usage
  - Memory usage in MB
  - Prediction class and confidence
- **Summary Statistics**:
  - Average times for each component
  - CPU and memory usage statistics
  - Total runtime and inference count

## Configuration

Uses the same model and settings as Scenario 2 but with drift detection disabled:

- Feature extractor: TFLite MobileNetV2 (1280D features)
- Feature transformer: StandardScaler + PCA (256D)
- Classifier: Lightweight KNN (k=5, max 200 samples)
- Inference interval: 2000ms (matching other scenarios)

## Comparison Points

This scenario helps identify:

1. **vs Scenario 0**: Overhead of TinyLCM feature extraction and KNN
2. **vs Scenario 2**: Performance cost of drift detection components

## Output Format

Performance data uses the same JSON format as Scenario 0 with additional timing breakdowns:

```json
{
  "timestamp": "2024-01-23T10:00:00",
  "type": "inference",
  "total_time_ms": 250.5,
  "feature_extraction_time_ms": 180.2,
  "knn_inference_time_ms": 70.3,
  "cpu_percent": 55.2,
  "memory_mb": 145.3,
  "prediction": {
    "class": "lego",
    "confidence": 0.92,
    "knn_samples": 200
  }
}
```