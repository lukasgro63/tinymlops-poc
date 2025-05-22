# Adaptive Pipeline with KNN Distance Monitor - Class Diagram

This document provides a class diagram for the adaptive machine learning pipeline in TinyLCM, including the KNN Distance Monitor, Quarantine Buffer, and Heuristic Adapter components for autonomous drift detection and adaptation.

## Overview

The KNN Distance Monitor implements autonomous drift detection by monitoring distances to nearest neighbors in the feature space. When drift is detected, samples are sent to a QuarantineBuffer for analysis by the HeuristicAdapter, which performs clustering-based pseudo-labeling for on-device adaptation without requiring ground truth labels.

## Class Diagram

```plantuml
@startuml KNN Distance Monitor Class Diagram
!theme plain
skinparam backgroundColor white
skinparam classBackgroundColor white
skinparam classBorderColor black
skinparam direction left to right

' Feature Processing Classes
class TFLiteFeatureExtractor {
    +extract_features(image)
    +model_path: str
}

class StandardScalerPCATransformer {
    +transform(features)
    +model_path: str
}

' Classification Class
class LightweightKNN {
    +predict_proba(features)
    +_last_distances: List[float]
    +k: int
}

' Drift Detection Classes
abstract class AutonomousDriftDetector {
    +update(record): bool
    +drift_detected: bool
}

class KNNDistanceMonitor {
    +delta: float
    +lambda_threshold: float
    +reference_mean: float
    +cumulative_sum: float
    +minimum_sum: float
    +update(record): Tuple[bool, Dict]
}

' Adaptation System Classes
class QuarantineBuffer {
    +buffer: deque
    +max_size: int
    +add_sample(sample): void
    +get_samples(): List[QuarantinedSample]
}

class HeuristicAdapter {
    +min_cluster_size: int
    +variance_threshold: float
    +analyze_quarantine(): List[AdaptationAction]
    +generate_pseudo_labels(): Dict
}

' True class relationships from code
AutonomousDriftDetector <|-- KNNDistanceMonitor
KNNDistanceMonitor ..> LightweightKNN : depends on
HeuristicAdapter --> QuarantineBuffer : uses
KNNDistanceMonitor ..> QuarantineBuffer : sends samples to

note bottom of KNNDistanceMonitor
**Page-Hinkley Test:**
deviation = avg_distance - (reference_mean + delta)
cumulative_sum += deviation
ph_value = cumulative_sum - minimum_sum
drift_detected = (ph_value > lambda_threshold)
end note

note bottom of HeuristicAdapter
**Clustering Heuristic:**
1. Group quarantined samples by prediction
2. Calculate intra-group feature dispersion
3. Assign pseudo-labels to coherent clusters
4. Select representative samples
end note

@enduml
```

## Processing Pipeline

The complete workflow shows how images are processed through the feature extraction and transformation pipeline before reaching the KNN classifier and drift detection:

**Raw Image** → **TFLiteFeatureExtractor** (MobileNetV2, 1280D) → **StandardScalerPCATransformer** (256D) → **LightweightKNN** → **KNNDistanceMonitor**

The KNNDistanceMonitor applies the Page-Hinkley test as an embedded algorithm (not a separate class) to detect significant increases in neighbor distances, which indicates potential drift or unknown objects.