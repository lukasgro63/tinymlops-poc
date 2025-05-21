# KNN-based Drift Detection in TinyMLOps

## Introduction

Drift detection is a critical challenge in deployed machine learning systems, particularly for resource-constrained edge devices operating in environments where ground truth labels are unavailable. The TinyMLOps framework addresses this challenge through a novel approach to drift detection using k-nearest neighbors (KNN) distance monitoring, which is particularly suited for resource-constrained devices like the Raspberry Pi Zero 2W.

This document explains in detail how the `KNNDistanceMonitor` component in TinyLCM works, its theoretical foundations, implementation details, and practical applications.

## Theoretical Foundation

### KNN Distance as a Proxy for Distribution Shift

The core insight behind KNN-based drift detection is that the distance to nearest neighbors in feature space is a powerful indicator of how well a new sample aligns with the known data distribution. When new samples come from the same distribution as the training data, they typically have close nearest neighbors. Conversely, samples from a different distribution (indicating drift) tend to have larger distances to their nearest neighbors.

This principle can be formalized as follows:

For a feature vector $x$ and a reference set of feature vectors $\{x_1, x_2, ..., x_n\}$, let $d_k(x)$ be the distance to the $k$-th nearest neighbor of $x$ in the reference set. Then:

- If $x$ is from the same distribution as the reference set, $d_k(x)$ should be approximately within the range of distances observed in the reference set.
- If $x$ is from a different distribution (drift), $d_k(x)$ will tend to be significantly larger than the typical distances in the reference set.

### Page-Hinkley Test for Change Detection

To detect significant and sustained increases in nearest neighbor distances (rather than reacting to temporary fluctuations), the `KNNDistanceMonitor` implements the Page-Hinkley test, a sequential analysis technique from statistical process control.

The Page-Hinkley test works as follows:

1. Let $X_t$ be the observed value at time $t$ (in our case, the average distance to $k$ nearest neighbors)
2. Let $\mu_{ref}$ be the reference mean (average distance during normal operation)
3. Let $\delta$ be a small positive value that allows for acceptable small increases

We compute:
- Deviation: $d_t = X_t - (\mu_{ref} + \delta)$
- Cumulative sum: $m_t = \sum_{i=1}^{t} d_i$
- Running minimum: $M_t = \min_{i \leq t} m_i$
- Test statistic: $PH_t = m_t - M_t$

Drift is detected when $PH_t > \lambda$, where $\lambda$ is a threshold value.

This test has several desirable properties:
- It detects sustained shifts rather than temporary spikes
- It adapts to the scale of the data through the reference mean
- It has a low computational and memory footprint
- It allows configuration of sensitivity via $\delta$ and $\lambda$ parameters

## Implementation in TinyLCM

The `KNNDistanceMonitor` class in TinyLCM implements this approach with several extensions and optimizations for resource-constrained devices:

### Key Components

1. **Distance Extraction**: The monitor extracts distances to nearest neighbors from the KNN classifier:
   ```python
   def _extract_distances(self, record: Dict[str, Any]) -> Optional[List[float]]:
       """Extract distance information from the record."""
       # Try multiple sources in order of preference
       if '_knn_distances' in record:
           return list(record['_knn_distances'])
       
       if 'classifier' in record and hasattr(record['classifier'], '_last_distances'):
           distances = record['classifier']._last_distances
           if distances is not None:
               return list(distances)
       
       # Additional fallbacks...
   ```

2. **Reference Statistics**: The monitor maintains reference statistics about normal distance values:
   ```python
   def _initialize_reference(self):
       """Initialize reference statistics from collected samples."""
       self.reference_mean = np.mean(self.warm_up_values)
       self.reference_std = np.std(self.warm_up_values)
       
       # Set control limits
       self.ucl = self.reference_mean + self.threshold * self.reference_std
       self.lcl = max(0, self.reference_mean - self.threshold * self.reference_std)
   ```

3. **Page-Hinkley Update**: The monitor implements the Page-Hinkley test for drift detection:
   ```python
   # Calculate the deviation from the reference (plus delta)
   deviation = avg_distance - (self.reference_mean + self.delta)
   
   # Track cumulative sum of deviations 
   self.cumulative_sum += deviation
   
   # Track minimum cumulative sum seen so far
   self.minimum_sum = min(self.minimum_sum, self.cumulative_sum)
   
   # Calculate the Page-Hinkley test statistic
   ph_value = self.cumulative_sum - self.minimum_sum
   
   # Compare to threshold
   if ph_value > self.lambda_threshold:
       self.drift_detected = True
   ```

4. **Adaptive Thresholds**: The monitor can adjust thresholds based on the observed standard deviation:
   ```python
   if self.use_adaptive_thresholds and self.reference_std > 0:
       # Set thresholds based on reference_std
       self.delta = self.reference_std * self.adaptive_delta_std_multiplier
       self.lambda_threshold = self.reference_std * self.adaptive_lambda_std_multiplier
   ```

5. **Cooldown Mechanism**: To prevent rapid-fire drift detections, a cooldown period is implemented:
   ```python
   if self.in_cooldown_period and self.samples_since_last_drift < self.drift_cooldown_period:
       # We're in cooldown - suppress new drift detections
       self.drift_detected = False
   ```

6. **Reference Update**: The monitor gradually updates reference statistics to track normal evolution of the data:
   ```python
   if self.should_update_reference():
       # Using the rolling update formula for mean
       self.reference_mean = (self.reference_update_factor * self.reference_mean + 
                             (1 - self.reference_update_factor) * avg_distance)
   ```

### Configuration Parameters

The `KNNDistanceMonitor` is highly configurable to adapt to different scenarios:

1. **Page-Hinkley Parameters**:
   - `delta`: Magnitude parameter for small allowable fluctuations
   - `lambda_threshold`: Threshold for drift detection

2. **Reference Management**:
   - `warm_up_samples`: Number of samples to establish initial reference
   - `reference_update_interval`: Frequency of reference updates
   - `reference_update_factor`: Weight of new observations in updates
   - `pause_reference_update_during_drift`: Whether to pause updates during drift

3. **Adaptive Thresholds**:
   - `use_adaptive_thresholds`: Whether to use standard deviation for thresholds
   - `adaptive_delta_std_multiplier`: Factor for delta threshold
   - `adaptive_lambda_std_multiplier`: Factor for lambda threshold

4. **Operational Control**:
   - `drift_cooldown_period`: Samples to wait before detecting another drift
   - `high_confidence_threshold`: Confidence threshold for stable predictions
   - `stable_known_classes`: Classes considered stable (to reduce false positives)

Example configuration from `config_scenario2.json`:
```json
{
  "type": "KNNDistanceMonitor",
  "delta": 2, 
  "lambda_threshold": 10,
  "exit_threshold_factor": 0.6,
  "high_confidence_threshold": 0.78,
  "stable_known_classes": ["lego", "stone", "leaf", "negative"],
  "reference_stats_path": "./initial_state/knn_reference_stats.json",
  "warm_up_samples": 0,
  "reference_update_interval": 10,
  "reference_update_factor": 0.05,
  "pause_reference_update_during_drift": true,
  "drift_cooldown_period": 2
}
```

## Integration with InferencePipeline

The `KNNDistanceMonitor` is integrated into the `InferencePipeline` to provide autonomous drift detection during inference:

```python
# Initialize drift detectors
drift_detectors = []

# Create KNN distance monitor
knn_distance_monitor = KNNDistanceMonitor(
    delta=detector_config.get("delta", 0.1),
    lambda_threshold=detector_config.get("lambda_threshold", 5.0),
    exit_threshold_factor=detector_config.get("exit_threshold_factor", 0.7),
    # Additional parameters...
)

# Register drift callback
knn_distance_monitor.register_callback(on_drift_detected)

# Add to detector list
drift_detectors.append(knn_distance_monitor)

# Initialize InferencePipeline with the detectors
pipeline = InferencePipeline(
    feature_extractor=feature_extractor,
    classifier=classifier,
    autonomous_monitors=drift_detectors,
    operational_monitor=operational_monitor,
    data_logger=data_logger
)
```

## Responding to Drift

When drift is detected, the `on_drift_detected` callback is triggered, which can perform several actions:

1. **Logging and Notification**:
   ```python
   logger.warning(f"DRIFT DETECTED by {detector_name} (drift_type={drift_type}): {reason}")
   ```

2. **Drift Event Capture**:
   ```python
   # Save the current frame if drift image saving is enabled
   if config["tinylcm"]["features"]["save_drift_images"] and current_frame is not None:
       # Create filename with timestamp and event ID
       event_id = f"event_{timestamp.replace(' ', '_').replace(':', '-')}_{uuid.uuid4().hex[:8]}"
       image_filename = f"{event_id}.jpg"
       
       # Save the image
       image_path = date_dir / image_filename
       cv2.imwrite(str(image_path), rgb_frame)
   ```

3. **Drift Event Packaging**:
   ```python
   # Prepare drift info for sending
   drift_data = {
       "detector_name": detector_name,
       "reason": reason,
       "metrics": metrics,
       "drift_type": drift_type,
       "timestamp": datetime.now().isoformat()
   }
   
   # Use the specialized method
   success = sync_client.create_and_send_drift_event_package(
       detector_name=detector_name,
       reason=reason,
       metrics=metrics,
       sample=current_sample_obj,
       image_path=str(image_path) if image_path else None
   )
   ```

## Practical Applications and Benefits

### Resource Efficiency

The KNN-based drift detection approach is particularly well-suited for resource-constrained devices:

1. **Low memory footprint**: Only requires storing reference statistics and current state
2. **Minimal computation**: Reuses distances already computed by the KNN classifier
3. **Incremental updates**: Reference statistics are updated incrementally
4. **Configurable parameters**: Sensitivity can be tuned to balance detection vs. false positives

### Drift Types Detected

This approach can detect several types of drift:

1. **Novel class detection**: When a new, previously unseen class appears
2. **Domain shift**: When the feature distribution shifts gradually over time
3. **Feature corruption**: When sensor issues cause abnormal feature values
4. **Model degradation**: When the model's feature space loses discriminative power

### Integration with Adaptive Pipeline

In the `AdaptivePipeline`, the drift detection is integrated with adaptation mechanisms:

1. **Quarantine Buffer**: Samples that trigger drift are stored in a quarantine buffer
2. **Heuristic Adaptation**: Quarantined samples are analyzed for potential adaptation
3. **State Management**: Snapshots are created before adaptation for potential rollback
4. **Server Validation**: Drift events are sent to TinySphere for validation when connectivity is available

## Practical Example

Consider a scenario where a Raspberry Pi with a camera is running the TinyLCM framework to detect objects (lego, stone, leaf, negative). The system is deployed and working well. Then, a new object (e.g., a toy car) appears in front of the camera.

The sequence would be:

1. The object's image is captured and processed through the feature extractor
2. Features are transformed using StandardScalerPCA
3. The KNN classifier attempts to classify the object, but the nearest neighbors are relatively far away
4. The `KNNDistanceMonitor` detects that the average distance to neighbors is unusually high
5. The Page-Hinkley test accumulates these deviations until crossing the threshold
6. Drift is detected, and the image is saved and packaged
7. The drift event package is queued for synchronization with TinySphere
8. When connectivity is available, the package is sent to TinySphere
9. TinySphere processes the drift event, stores the image, and updates dashboards
10. An operator can review the drift event and provide feedback for model improvement

## Comparison with Other Drift Detection Methods

The KNN-based approach has several advantages over other common drift detection methods:

1. **Compared to Distribution-based methods (e.g., KL divergence)**:
   - More sample-efficient (works with individual samples)
   - Lower computational requirements
   - Does not require building explicit distributions
   - More sensitive to localized shifts in feature space

2. **Compared to Confidence-based methods**:
   - Detects drift even when confidence remains high
   - Not directly tied to model output, providing an independent check
   - Can detect novel classes that the model confidently misclassifies

3. **Compared to Error-rate methods**:
   - Does not require ground truth labels
   - Can operate fully autonomously
   - Provides early detection before performance degrades

## Conclusion

The KNN-based drift detection approach in TinyMLOps provides an elegant solution to the challenge of autonomous drift detection on resource-constrained devices. By leveraging the nearest neighbor distances as a proxy for distribution shift, it enables edge devices to identify when they are operating outside their known data distribution.

This capability is fundamental to the autonomous and adaptive nature of TinyMLOps, allowing devices to maintain performance even in changing environments without continuous connectivity or immediate supervised feedback. The implementation balances detection sensitivity with resource efficiency, making it practical for deployment on devices like the Raspberry Pi Zero 2W.

In conjunction with the adaptive pipeline, quarantine buffer, and server-side validation, this approach forms a comprehensive solution for maintaining ML model performance at the edge in the face of evolving data distributions and operational conditions.