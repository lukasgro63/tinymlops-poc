# Adaptive Workflow in TinyMLOps

## Introduction

Machine learning models deployed on edge devices face unique challenges. The operational environment can shift over time, new types of data can emerge, and physical conditions can change. Traditional MLOps approaches require continuous connectivity to cloud servers and access to ground truth labels for maintaining model performance. These requirements are often impractical for edge deployments in real-world settings.

TinyMLOps introduces a novel adaptive workflow that enables ML models to evolve and maintain performance on resource-constrained edge devices, with or without connectivity to a central server. This document explains the adaptive workflow in detail, focusing on how the system handles concept drift, performs on-device adaptations, and synchronizes with the server when available.

## Overview of the Adaptive Workflow

The adaptive workflow in TinyMLOps consists of four main components:

1. **Autonomous Drift Detection**: Detecting when the input data distribution shifts without requiring ground truth labels
2. **Sample Quarantine**: Storing potentially drifted samples for analysis and adaptation
3. **Heuristic Adaptation**: Using unsupervised and semi-supervised techniques to adapt the model on-device
4. **Server-Assisted Validation**: Optional external validation and correction when connectivity is available

These components work together to create a flexible system that can operate in different modes based on connectivity availability and application requirements.

## Autonomous Drift Detection

Drift detection is the entry point to the adaptive workflow. Without reliable identification of distribution shifts, adaptation can be misapplied, potentially degrading model performance.

### Types of Drift Monitored

The TinyLCM framework implements multiple drift detection strategies:

1. **Feature Distribution Drift**: Detected by the `FeatureMonitor` class, monitors statistical properties of feature vectors
2. **KNN Distance Drift**: Detected by the `KNNDistanceMonitor` class, tracks distances to nearest neighbors
3. **Confidence Drift**: Detected by the `EWMAConfidenceMonitor` class, monitors prediction confidence
4. **Feature Value Drift**: Detected by the `PageHinkleyFeatureMonitor` class, uses Page-Hinkley test on feature values

Each detection strategy is designed to be:
- Label-free: No ground truth required
- Computationally efficient: Suitable for resource-constrained devices
- Adaptive: Can update reference statistics over time

### Drift Event Handling

When drift is detected, the `on_drift_detected` callback is triggered, which:

1. Logs the drift event with detailed information
2. Captures the current input (e.g., image) for later analysis
3. Creates a drift event package for synchronization with the server
4. Optionally adds the sample to a quarantine buffer for on-device adaptation

```python
def on_drift_detected(drift_info: Dict[str, Any], *args) -> None:
    """Callback function for drift detection events."""
    
    # Log the drift detection with more visibility
    logger.warning(f"DRIFT DETECTED by {detector_name} (drift_type={drift_type}): {reason}")
    
    # Save the current frame for drift visualization
    if config["tinylcm"]["features"]["save_drift_images"] and current_frame is not None:
        # Create directory structure for compatibility with tinysphere bucket format
        image_path = date_dir / image_filename
        cv2.imwrite(str(image_path), rgb_frame)
    
    # Send drift event information to server (create a custom package)
    if sync_client:
        success = sync_client.create_and_send_drift_event_package(
            detector_name=detector_name,
            reason=reason,
            metrics=metrics,
            sample=current_sample_obj,
            image_path=str(image_path) if image_path else None
        )
```

## Sample Quarantine

The quarantine buffer acts as a staging area for samples that exhibit drift characteristics. It provides several critical functions:

1. **Sample Storage**: Maintains a buffer of potentially drifted samples
2. **Metadata Tracking**: Stores drift information, predictions, and confidence scores
3. **Expiration Management**: Implements time-based or count-based expiration to manage buffer size
4. **Prioritization**: Allows prioritizing samples based on drift magnitude or recency

### Quarantine Buffer Implementation

The `QuarantineBuffer` class manages samples that exhibit drift:

```python
class QuarantineBuffer:
    """Buffer for storing samples flagged by autonomous detectors.
    
    This buffer stores samples that have been flagged by autonomous drift detectors
    for later analysis, validation, or adaptation. It supports various strategies for
    managing the buffer size and determining which samples to quarantine.
    """
    
    def __init__(
        self,
        max_size: int = 100,
        max_age: Optional[int] = 86400,  # 1 day in seconds
        quarantine_strategy: str = "all"
    ):
        """Initialize the quarantine buffer."""
        self.max_size = max_size
        self.max_age = max_age
        self.quarantine_strategy = quarantine_strategy
        
        # Buffer storage
        self.samples = {}  # Dict of sample_id -> QuarantinedSample
        self.drift_types = defaultdict(int)  # Count of each drift type
```

### Sample Evaluation

The buffer evaluates whether samples should be quarantined based on configurable strategies:

```python
def should_quarantine(self, drift_info: Dict[str, Any], sample: FeatureSample) -> bool:
    """Determine if a sample should be quarantined based on drift info."""
    # Different strategies for different use cases
    if self.quarantine_strategy == "all":
        # Quarantine all samples with detected drift
        return True
    
    elif self.quarantine_strategy == "high_confidence":
        # Only quarantine high-confidence drift detections
        confidence = drift_info.get("confidence", 0.0)
        threshold = drift_info.get("threshold", 0.0)
        
        if confidence and threshold:
            return confidence > threshold * 1.5  # Significantly above threshold
    
    # Default to not quarantining
    return False
```

## Heuristic Adaptation

The heuristic adaptation component is one of the most innovative aspects of TinyMLOps. It enables on-device adaptation without requiring ground truth labels by implementing unsupervised and semi-supervised learning techniques.

### Heuristic Adapter

The `HeuristicAdapter` class analyzes quarantined samples to identify patterns and assign pseudo-labels:

```python
class HeuristicAdapter:
    """On-device heuristic adaptation based on quarantined samples."""
    
    def __init__(
        self,
        quarantine_buffer: QuarantineBuffer,
        min_cluster_size: int = 5,
        variance_threshold: float = 0.1,
        k_representatives: int = 3,
        use_numpy: bool = True
    ):
        """Initialize the heuristic adapter."""
        self.quarantine_buffer = quarantine_buffer
        self.min_cluster_size = min_cluster_size
        self.variance_threshold = variance_threshold
        self.k_representatives = k_representatives
        self.use_numpy = use_numpy
        
        # Track known classes to avoid conflicts
        self.known_classes = set()
```

### Pseudo-Labeling Process

The core of the heuristic adapter is the pseudo-labeling process, which involves:

1. **Clustering**: Grouping similar samples to identify potential class patterns
2. **Pattern Analysis**: Analyzing cluster properties to determine if they represent a new concept
3. **Label Assignment**: Assigning pseudo-labels to samples based on cluster membership
4. **Confidence Estimation**: Providing a confidence score for each pseudo-label

```python
def apply_pseudo_labels(self, min_confidence: float = 0.7) -> List[FeatureSample]:
    """Apply pseudo-labels to quarantined samples based on heuristic analysis."""
    # Get all quarantined samples
    quarantined_samples = self.quarantine_buffer.get_all_samples()
    
    if not quarantined_samples:
        return []
    
    # Extract features from samples
    features = []
    for sample in quarantined_samples:
        if hasattr(sample, 'features') and sample.features is not None:
            features.append(sample.features)
    
    # Cluster samples and analyze patterns
    clusters = self._cluster_samples(features_array)
    
    # Analyze each cluster and assign pseudo-labels
    adapted_samples = []
    for cluster_id, sample_indices in clusters.items():
        # Apply pseudo-labels to high-confidence samples
        # [Implementation details omitted for brevity]
    
    return adapted_samples
```

## Server-Assisted Validation

While the heuristic adaptation provides autonomous on-device adaptation, TinyMLOps also supports server-assisted validation for higher-quality adaptation when connectivity is available.

### Synchronization Process

The synchronization process connects edge devices with the TinySphere server:

```python
def trigger_server_sync(self) -> Dict[str, Any]:
    """Manually trigger synchronization with the TinySphere server."""
    # Check if server sync is enabled
    if not self.enable_server_sync or not self.sync_client:
        return {"success": False, "reason": "Server sync not enabled"}
    
    try:
        # Sync quarantine buffer
        quarantine_sync_result = self.sync_client.sync_quarantine()
        
        # Sync drift events
        drift_sync_result = self.sync_client.sync_drift_events()
        
        # Process validation results
        validation_results = []
        if quarantine_sync_result:
            validation_count = quarantine_sync_result.get("validation_results_received", 0)
            if validation_count > 0:
                validation_results = self.quarantine_buffer.get_validated_samples()
                
                # Apply validations to the classifier
                for sample in validation_results:
                    if sample.label is not None:
                        self.provide_feedback(
                            features=sample.features,
                            label=sample.label,
                            is_validated_label=True,
                            sample_id=sample.sample_id,
                            timestamp=sample.timestamp
                        )
        
        return {
            "success": True,
            "validation_results_processed": len(validation_results)
        }
    except Exception as e:
        return {"success": False, "error": str(e)}
```

## State Management and Rollback

To ensure safe adaptation, TinyMLOps implements state management with rollback capabilities:

### State Snapshotting

Before making adaptations, the system creates snapshots that can be used for rollback:

```python
def create_snapshot(self, reason: str, metadata: Optional[Dict[str, Any]] = None) -> Optional[str]:
    """Create a snapshot of the current state."""
    if not self.state_manager:
        return None
    
    try:
        # Get component states
        component_states = {}
        
        # Get classifier state
        if hasattr(self.classifier, "get_state"):
            component_states["classifier"] = self.classifier.get_state()
        
        # Get handler state
        if hasattr(self.handler, "get_state"):
            component_states["handler"] = self.handler.get_state()
        
        # Create the snapshot
        snapshot_id = self.state_manager.create_snapshot(reason, metadata)
        return snapshot_id
    except Exception as e:
        return None
```

## Operational Modes

The adaptive workflow supports multiple operational modes to address different deployment scenarios:

### 1. Monitoring-Only Mode

In this mode, the system only detects drift without performing adaptation:

```python
# Configuration
config = {
    "enable_autonomous_detection": True,
    "enable_quarantine": False,
    "enable_heuristic_adaptation": False,
    "enable_server_sync": True
}

# Setup
pipeline = InferencePipeline(
    feature_extractor=feature_extractor,
    classifier=classifier,
    autonomous_monitors=drift_detectors,
    operational_monitor=operational_monitor,
    data_logger=data_logger
)
```

### 2. Autonomous Mode

In this mode, the system performs on-device adaptation without server assistance:

```python
# Configuration
config = {
    "enable_autonomous_detection": True,
    "enable_quarantine": True,
    "enable_heuristic_adaptation": True,
    "enable_server_sync": False
}

# Setup
pipeline = AdaptivePipeline(
    feature_extractor=feature_extractor,
    classifier=classifier,
    autonomous_monitors=drift_detectors,
    quarantine_buffer=quarantine_buffer,
    heuristic_adapter=heuristic_adapter,
    state_manager=state_manager
)
```

### 3. Server-Assisted Mode

In this mode, the system relies on server validation for adaptation:

```python
# Configuration
config = {
    "enable_autonomous_detection": True,
    "enable_quarantine": True,
    "enable_heuristic_adaptation": False,
    "enable_server_sync": True,
    "external_validation": True
}

# Setup
pipeline = AdaptivePipeline(
    feature_extractor=feature_extractor,
    classifier=classifier,
    autonomous_monitors=drift_detectors,
    quarantine_buffer=quarantine_buffer,
    sync_client=sync_client,
    state_manager=state_manager
)
```

### 4. Hybrid Mode

In this mode, the system performs on-device adaptation but also uses server validation when available:

```python
# Configuration
config = {
    "enable_autonomous_detection": True,
    "enable_quarantine": True,
    "enable_heuristic_adaptation": True,
    "enable_server_sync": True,
    "external_validation": True
}

# Setup
pipeline = AdaptivePipeline(
    feature_extractor=feature_extractor,
    classifier=classifier,
    autonomous_monitors=drift_detectors,
    quarantine_buffer=quarantine_buffer,
    heuristic_adapter=heuristic_adapter,
    sync_client=sync_client,
    state_manager=state_manager
)
```

## Example Workflow Scenario

Consider a scenario where a TinyLCM-powered edge device is deployed for object recognition:

1. **Initial State**:
   - Device has a KNN classifier trained on classes: "lego", "stone", "leaf", "negative"
   - KNNDistanceMonitor is configured for drift detection
   - AdaptivePipeline is set up in hybrid mode

2. **Novel Object Appears**:
   - A toy car (not in the training set) appears in front of the camera
   - Feature extraction and transformation processes the image
   - KNN classifier attempts to classify but nearest neighbors are distant
   - KNNDistanceMonitor detects drift via Page-Hinkley test
   - Image is saved and drift event is created
   - Sample is added to quarantine buffer

3. **On-Device Adaptation**:
   - As more images of the toy car are captured, the quarantine buffer accumulates samples
   - HeuristicAdapter clusters the quarantined samples
   - A new cluster forms with similar toy car images
   - Pseudo-label "unknown_class_1" is assigned to the cluster
   - State snapshot is created before adaptation
   - Representative samples are used to update the KNN classifier
   - Adaptation event is logged

4. **Server Synchronization (when available)**:
   - Device connects to TinySphere server
   - Drift events and images are uploaded
   - Quarantined samples are synchronized
   - Server processes the drift event
   - Human expert reviews the drift event on dashboard
   - Expert provides label "toy_car" for the new class
   - Validation result is sent back to device

5. **Validated Adaptation**:
   - Device receives validation result
   - State snapshot is created before applying validation
   - KNN classifier is updated with validated label
   - Pseudo-label "unknown_class_1" is replaced with "toy_car"
   - Adaptation event is logged
   - Future toy car instances are correctly classified

## Performance and Resource Considerations

The adaptive workflow is designed with resource constraints in mind:

1. **Memory Efficiency**:
   - Quarantine buffer has configurable size limits
   - Samples can expire based on age
   - State snapshots use incremental storage

2. **Computational Efficiency**:
   - Heuristic adaptation runs periodically, not on every sample
   - Clustering is optimized for resource constraints
   - Representative samples limit the growth of the KNN dataset

3. **Power Efficiency**:
   - Synchronization is opportunistic to save power
   - Processing intensive tasks can be scheduled during charging
   - Adaptation frequency can be configured based on battery status

## Conclusion

The adaptive workflow in TinyMLOps represents a significant advancement in edge AI capabilities. By combining autonomous drift detection, on-device heuristic adaptation, and opportunistic server synchronization, it enables ML models to maintain performance over time on resource-constrained devices, even in challenging deployment environments.

This approach provides several key benefits:

1. **Autonomy**: Edge devices can detect and respond to distribution shifts without requiring continuous connectivity
2. **Adaptability**: Models can evolve to handle new classes and changing environments
3. **Resource Efficiency**: All components are designed to work within the constraints of edge devices
4. **Robustness**: State management and rollback capabilities ensure safe adaptation
5. **Flexibility**: Different operational modes accommodate various deployment scenarios

The result is a comprehensive solution for maintaining ML model performance at the edge in the face of evolving data distributions and operational conditions.