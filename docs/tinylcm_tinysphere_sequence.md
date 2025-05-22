# TinyLCM ↔ TinySphere Integration Sequence Diagram

This document provides a sequence diagram illustrating the complete integration workflow between TinyLCM (autonomous edge ML) and TinySphere (optional MLOps platform), demonstrating how edge autonomy is enhanced—not compromised—by cloud capabilities.

## Integration Sequence Diagram

```plantuml
@startuml TinyLCM-TinySphere Integration Sequence
!theme plain
skinparam backgroundColor white
skinparam participantBackgroundColor white
skinparam participantBorderColor black
skinparam sequenceArrowColor black

participant "Edge Device\n(TinyLCM)" as Edge
participant "SyncClient" as Sync
participant "TinySphere API" as API
participant "Data Hub" as DataHub
participant "Drift Hub" as DriftHub
participant "MLflow" as MLflow
participant "MinIO Storage" as MinIO
participant "Human Operator" as Human

== Phase 1: Autonomous Edge Operation ==
Edge -> Edge: Autonomous inference & drift detection
Edge -> Edge: Local quarantine buffer management
Edge -> Edge: Heuristic adaptation (pseudo-labeling)
Edge -> Edge: Local state snapshots & versioning

note over Edge: Device operates completely\nautonomously - no server needed

== Phase 2: Optional TinySphere Integration ==
Edge -> Sync: create_sync_package() with logs/metrics/drift_data
Sync -> Sync: Package data into TAR archive
Sync -> API: POST /upload/package (with retry logic)

alt Network Available
    API -> API: Extract package contents
    API -> DataHub: Process operational logs & metrics
    API -> DriftHub: Process drift events & samples
    API -> MLflow: Register models & experiments
    API -> MinIO: Store artifacts & features
    API -> Edge: Upload success confirmation
else Network Unavailable
    Sync -> Sync: Queue package for later upload
    Edge -> Edge: Continue autonomous operation
    note over Edge: Graceful degradation -\nfull functionality maintained
end

== Phase 3: Data Hub Analytics ==
DataHub -> MinIO: Store prediction images & features
DataHub -> DataHub: Generate device performance metrics
DataHub -> API: Update device status & analytics

== Phase 4: MLflow Model Tracking ==
MLflow -> MLflow: Create experiment for device
MLflow -> MLflow: Log model parameters & metrics
MLflow -> MinIO: Store model artifacts
MLflow -> MLflow: Register model version

== Phase 5: Drift Hub Ground Truth Validation ==
DriftHub -> Human: Present drift events for validation
Human -> DriftHub: Provide ground truth labels
DriftHub -> DriftHub: Update validation status
DriftHub -> API: Create feedback for device

== Phase 6: Feedback Loop ==
API -> MinIO: Store validated ground truth
API -> Edge: Send feedback package (when connected)
Edge -> Edge: Update local models with feedback
Edge -> Edge: Improve drift detection accuracy

== Phase 7: Enterprise Dashboard ==
API -> API: Aggregate fleet analytics
API -> Human: Real-time dashboard updates
Human -> API: Fleet management operations

note over Edge, Human: TinyLCM maintains autonomy\nwhile TinySphere provides\nenterprise MLOps capabilities

@enduml
```

## Sequence Analysis

### Phase 1: Pure Autonomy
- **Complete Independence**: TinyLCM operates without any server dependency
- **Local Intelligence**: Drift detection, adaptation, and state management
- **Offline Resilience**: Full functionality during network outages

### Phase 2: Optional Enhancement
- **Graceful Integration**: SyncClient provides seamless cloud connectivity
- **Resilient Design**: Network failures don't interrupt autonomous operation
- **Batch Processing**: Efficient data packaging and transmission

### Phase 3-4: Enterprise Features
- **Data Hub**: Centralized analytics and performance monitoring
- **MLflow Integration**: Experiment tracking and model versioning
- **Scalable Storage**: MinIO provides S3-compatible artifact management

### Phase 5-6: Ground Truth Loop
- **Human Validation**: Manual verification of drift detection accuracy
- **Feedback Integration**: Validated labels improve edge model performance
- **Continuous Learning**: Closed-loop improvement without breaking autonomy

### Phase 7: Fleet Management
- **Enterprise Dashboard**: Real-time monitoring of device fleets
- **Centralized Control**: Management operations across multiple devices
- **Analytics Pipeline**: Comprehensive insights into system performance

## Key Integration Principles

### 1. Autonomous First Design
TinyLCM provides complete ML capabilities without external dependencies:
- Local inference with TFLite models
- Autonomous drift detection using statistical tests
- Heuristic adaptation through clustering and pseudo-labeling
- Local state management with versioning and rollback

### 2. Optional Enhancement Pattern
TinySphere enhances but never replaces core functionality:
- Network connectivity is optional, not required
- Cloud features are additive, not substitutive
- Edge devices degrade gracefully during outages
- Full autonomy maintained at all times

### 3. Enterprise Scalability
Integration enables enterprise-grade MLOps capabilities:
- Fleet management for hundreds of devices
- Centralized monitoring and analytics
- Ground truth validation workflows
- Continuous improvement feedback loops

This sequence demonstrates how TinyLCM achieves **true edge autonomy** while enabling **optional enterprise enhancement** through TinySphere, creating a complete MLOps platform suitable for both isolated edge deployments and large-scale industrial IoT applications.