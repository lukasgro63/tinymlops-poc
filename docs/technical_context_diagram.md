# Technical Context Diagram for TinyMLOps

Below is a detailed technical context diagram in PlantUML representing the architecture of the TinyMLOps system.

```plantuml
@startuml TinyMLOps_Technical_Context

!define ICONURL https://raw.githubusercontent.com/tupadr3/plantuml-icon-font-sprites/v2.4.0

skinparam componentStyle rectangle
skinparam defaultTextAlignment center
skinparam rectangle {
  RoundCorner 15
  BackgroundColor #F0F8FF
  BorderColor #4682B4
  BorderThickness 1
}

skinparam component {
  BorderColor #4682B4
  BackgroundColor #F0F8FF
}

skinparam database {
  BorderColor #4682B4
  BackgroundColor #F0F8FF
}

' Main components as rectangles
rectangle "TinyLCM Core Library" as TinyLCM {
  component "Feature Extractors" as FeatureExtractors
  component "Feature Transformers" as FeatureTransformers
  component "Classifiers" as Classifiers
  component "Drift Detectors" as DriftDetectors
  component "Inference Pipeline" as Pipeline
  component "Adaptive Pipeline" as AdaptivePipeline
  component "KNN State Manager" as StateManager
  component "Data Logger" as DataLogger
  component "Sync Client" as SyncClient
  component "Adaptation Tracker" as AdaptationTracker
  component "Operational Monitor" as OpMonitor
  component "Quarantine Buffer" as QuarantineBuffer
  component "Heuristic Adapter" as HeuristicAdapter
}

' Edge Device (example implementation)
rectangle "Edge Device (Raspberry Pi Zero 2W)" as EdgeDevice {
  component "TFLite Model" as TFLiteModel
  component "Camera/Sensors" as Camera
  component "Config (JSON)" as Config
  component "Example Implementation\n(main_scenario2.py)" as MainImpl
}

' TinySphere Server Components
rectangle "TinySphere Server" as TinySphere {
  component "FastAPI REST API" as API
  component "Device Management" as DeviceManagement
  component "Drift Management" as DriftManagement
  component "Package Processor" as PackageProcessor
  component "Dashboard" as Dashboard
  database "PostgreSQL DB" as PostgreSQL
  rectangle "MinIO Storage" as MinIO {
    database "Drift Images" as DriftBucket
    database "Prediction Images" as PredictionBucket
    database "Operational Logs" as LogsBucket
  }
}

' External Components
cloud "External Services" as External {
  component "MLflow" as MLflow
}

' Relationships between TinyLCM components
FeatureExtractors --> FeatureTransformers : "extracts and\ntransforms features"
FeatureTransformers --> Pipeline : "provides processed\nfeatures"
Pipeline --> Classifiers : "uses for\nprediction"
Pipeline --> DriftDetectors : "uses for\ndrift detection"
Pipeline --> OpMonitor : "reports metrics"
Pipeline --> DataLogger : "logs data"
AdaptivePipeline --|> Pipeline : "extends"
AdaptivePipeline --> StateManager : "manages model\nversions"
AdaptivePipeline --> AdaptationTracker : "tracks adaptation\nevents"
AdaptivePipeline --> QuarantineBuffer : "stores samples\nwith drift"
AdaptivePipeline --> HeuristicAdapter : "adapts model with\npseudo-labels"
SyncClient --> TinySphere : "synchronizes with\nserver when available"

' Edge Device relationships
EdgeDevice --> TinyLCM : "uses"
MainImpl --> Config : "reads"
Camera --> MainImpl : "provides\ninput data"
TFLiteModel --> MainImpl : "provides\nmodel for inference"
MainImpl --> FeatureExtractors : "uses for\nfeature extraction"
MainImpl --> DriftDetectors : "uses for\ndrift detection"
MainImpl --> SyncClient : "uses for\nserver communication"

' TinySphere relationships
API --> DeviceManagement : "routes\ndevice requests"
API --> DriftManagement : "routes\ndrift requests"
API --> PackageProcessor : "processes\nuploaded packages"
DeviceManagement --> PostgreSQL : "stores\ndevice info"
DriftManagement --> PostgreSQL : "stores\ndrift events"
DriftManagement --> DriftBucket : "stores\ndrift images"
PackageProcessor --> MinIO : "stores\nartifacts"
Dashboard --> PostgreSQL : "reads\ndata for visualizations"
MLflow --> TinySphere : "provides\nmodel registry"

' Main data flows
TinyLCM ..> TinySphere : "1. Device registration\n2. Drift event packages\n3. Prediction images\n4. Operational metrics"
TinySphere ..> TinyLCM : "1. Validation results\n2. Model updates\n3. Configuration changes"

' Optional components with dotted lines
HeuristicAdapter -[dashed]-> QuarantineBuffer : "processes quarantined\nsamples"
AdaptivePipeline -[dashed]-> SyncClient : "optional sync\nwith server"

' Legend
legend right
  **TinyMLOps System Architecture**
  This diagram shows the technical components of the TinyMLOps system:
  1. **TinyLCM**: Core library for on-device ML lifecycle management
  2. **Edge Device**: Raspberry Pi running TinyLCM applications
  3. **TinySphere**: Server platform for centralized management
endlegend

@enduml
```

## Component Descriptions

### TinyLCM Core Library

The core library providing on-device ML lifecycle management capabilities:

1. **Feature Extractors**: Extract features from raw input data (e.g., TFLiteFeatureExtractor)
2. **Feature Transformers**: Transform high-dimensional features (e.g., StandardScalerPCATransformer)
3. **Classifiers**: Lightweight classifiers for inference (e.g., LightweightKNN)
4. **Drift Detectors**: Autonomous monitors for detecting data/model drift (e.g., KNNDistanceMonitor)
5. **Inference Pipeline**: Base pipeline for non-adaptive inference with drift detection
6. **Adaptive Pipeline**: Extended pipeline with adaptation capabilities
7. **KNN State Manager**: Manages model state and provides versioning
8. **Data Logger**: Records inputs, outputs, and operational metrics
9. **Sync Client**: Handles communication with TinySphere server
10. **Adaptation Tracker**: Tracks adaptation events and model changes
11. **Operational Monitor**: Collects system and operational metrics
12. **Quarantine Buffer**: Stores samples flagged by autonomous detectors
13. **Heuristic Adapter**: On-device pseudo-labeling for quarantined samples

### Edge Device Implementation

The example implementation running on a resource-constrained device:

1. **TFLite Model**: Optimized model for edge inference
2. **Camera/Sensors**: Input devices for data collection
3. **Config (JSON)**: Configuration file for system parameters
4. **Example Implementation**: Main application code (e.g., main_scenario2.py)

### TinySphere Server

The centralized server platform for management and validation:

1. **FastAPI REST API**: Provides endpoints for device communication
2. **Device Management**: Handles device registration and status
3. **Drift Management**: Processes and analyzes drift events
4. **Package Processor**: Extracts and processes uploaded packages
5. **Dashboard**: Web interface for visualization and control
6. **PostgreSQL DB**: Stores structured data (devices, drift events, etc.)
7. **MinIO Storage**: Object storage for artifacts:
   - Drift Images Bucket: Stores images related to drift events
   - Prediction Images Bucket: Stores prediction results
   - Operational Logs Bucket: Stores device operational logs

### External Services

1. **MLflow**: Integration for experiment tracking and model registry

## Data Flows

1. **Edge Device to TinySphere**:
   - Device registration
   - Drift event packages
   - Prediction images
   - Operational metrics

2. **TinySphere to Edge Device**:
   - Validation results
   - Model updates
   - Configuration changes

## Key Features

1. **Autonomous Drift Detection**: Edge devices can detect distribution shifts without ground truth
2. **On-device Adaptation**: Devices can adapt their models using heuristic approaches
3. **Opportunistic Synchronization**: Devices sync with the server when connectivity is available
4. **Comprehensive State Management**: Versioning provides rollback capabilities
5. **Flexible Operational Modes**: Support for different levels of autonomy and server assistance