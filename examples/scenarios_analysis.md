# TinyLCM Scenario Analysis

## Current Status

The `scenario1_tinylcm_inferencepipeline` is working successfully with the following key components:

1. **KNN State Loading**: The pre-trained KNN state is loaded successfully:
   ```
   2025-05-11 15:10:00,083 - tinylcm_scenario1 - INFO - LightweightKNN initialisiert mit 100 Samples aus Zustand: ../assets/initial_states/knn_initial_state_RG.json
   ```

2. **TinySphere Connection**: Connection to TinySphere server is working:
   ```
   2025-05-11 15:10:00,259 - utils.sync_client - INFO - Server connection status: Connected
   ```

3. **Drift Detection**: The PageHinkleyFeatureMonitor successfully detects drift:
   ```
   2025-05-11 15:10:46,504 - tinylcm_scenario1 - WARNING - DRIFT DETECTED by Unknown: Drift detected
   ```

4. **Drift Reporting**: Drift events are sent to TinySphere:
   ```
   2025-05-11 15:10:47,764 - tinylcm_scenario1 - INFO - Drift event sent using extended client: Success
   ```

## Analysis of Drift Detection

### How PageHinkleyFeatureMonitor Works

The PageHinkleyFeatureMonitor monitors changes in feature statistics over time. It works by:

1. During the warm-up phase, it calculates a reference mean of the feature statistic
2. For each new sample, it computes the deviation from the reference mean
3. It maintains a cumulative sum of deviations and a running minimum
4. Drift is signaled when the difference between the cumulative sum and minimum exceeds the lambda threshold

### Current Implementation

The current implementation uses a `safe_feature_extractor` function that extracts the first feature from the feature vector:

```python
def safe_feature_extractor(features):
    """Extract a feature safely with boundary checking."""
    if features is None:
        return 0.0

    # Convert to numpy array if needed
    if not isinstance(features, np.ndarray):
        try:
            features = np.array(features)
        except:
            return 0.0

    # Check if feature index is in range
    if feature_index < 0 or feature_index >= features.size:
        # Use last feature if index out of range
        if features.size > 0:
            return float(features.flatten()[-1])
        return 0.0

    # Get the feature at the specified index
    try:
        return float(features.flatten()[feature_index])
    except:
        return 0.0
```

### Observed Issues

The logs show very large values for the feature statistics:

```
2025-05-11 15:10:46,510 - tinylcm_scenario1 - INFO -   statistic_value: 128000000000000.0
2025-05-11 15:10:46,511 - tinylcm_scenario1 - INFO -   reference_mean: 16161616161625.787
```

This suggests that either:
1. The feature values from the TFLite model are extremely large
2. There could be a scaling issue (possibly floating point vs. integer representation)
3. The feature extraction process might be accessing values not intended for direct comparison

### Drift Detection Parameters

The drift detector is configured with:
- `delta`: 0.005 (small bias parameter)
- `lambda_param`: 50 (threshold for detection)
- `min_samples`: 50 (warm-up sample count)
- `warmup_samples`: 100 (reference update interval)

## Inference Pipeline

The inference pipeline is working correctly:
1. Images are captured from the camera
2. Features are extracted using the TFLite feature extractor
3. Classification is performed using the pre-trained KNN classifier
4. Drift detection monitors feature statistics for changes

## Potential Improvements

1. **Feature Normalization**: Ensure feature values are properly normalized to avoid numerical issues

2. **Alternative Statistics**: Instead of monitoring single feature values, consider monitoring:
   - Feature vector norms
   - Mean/variance of feature groups
   - Distance to cluster centroids

3. **Multiple Detectors**: Add multiple drift detectors (like in the monitoring-only scenario) for more robust detection

4. **Adaptation Logic**: After detecting drift, implement adaptation logic to update the KNN classifier

5. **Visualization**: Add visualization of drift events and feature values over time to better understand detection patterns

## Next Steps

1. Analyze the feature values from the TFLite model
2. Tune drift detector parameters for better sensitivity
3. Implement visualization tools for drift events
4. Test with real-world drift scenarios (different lighting, objects, etc.)
5. Add adaptive capabilities (through QuarantineBuffer and HeuristicAdapter)