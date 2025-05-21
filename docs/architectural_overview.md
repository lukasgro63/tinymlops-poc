# TinyMLOps Architectural Overview

## Introduction

TinyMLOps is a framework designed to address the challenges of deploying and managing adaptive Machine Learning (ML) models on resource-constrained edge devices. The system consists of three main components:

1. **TinyLCM** (TinyML Lifecycle Management): Core library for on-device ML management
2. **Edge Device Applications**: Example implementations running on devices like Raspberry Pi Zero 2W
3. **TinySphere**: Server platform for centralized management and validation

This document provides a comprehensive architectural overview of the TinyMLOps system, explaining how these components interact to enable autonomous and adaptive ML lifecycle management in resource-constrained environments.

## Design Philosophy

TinyMLOps is designed around several key principles:

1. **Edge Autonomy**: Edge devices should operate independently with minimal reliance on cloud connectivity
2. **Resource Efficiency**: All components must function within the constraints of low-power devices
3. **Drift Resilience**: The system must detect and adapt to data distribution shifts without ground truth
4. **Opportunistic Validation**: External validation should be used when available but not required
5. **Incremental Adaptation**: Models should evolve gradually as new data becomes available

## System Architecture

### 1. TinyLCM (Core Library)

TinyLCM is the foundation of the TinyMLOps ecosystem, providing the core functionality for on-device ML lifecycle management.

#### 1.1 Pipeline Architecture

The pipeline architecture in TinyLCM provides a structured approach to ML inference and adaptation:

- **InferencePipeline**: Base pipeline for non-adaptive inference with drift detection
  - Handles feature extraction, classification, monitoring, and drift detection
  - Provides a unified interface for processing input data
  - Manages operational monitoring and data logging
  - Designed for resource efficiency on edge devices

- **AdaptivePipeline**: Extended pipeline with adaptation capabilities
  - Inherits all functionality from InferencePipeline
  - Adds capabilities for model adaptation and state management
  - Integrates with QuarantineBuffer for storing samples with detected drift
  - Includes HeuristicAdapter for on-device pseudo-labeling
  - Provides different operational modes (monitoring-only, autonomous, server-assisted)

#### 1.2 Feature Processing

Feature processing in TinyLCM follows a two-stage approach:

- **Feature Extraction**:
  - TFLiteFeatureExtractor: Extracts high-dimensional feature vectors from raw input data
  - Leverages TensorFlow Lite for efficient inference on resource-constrained devices
  - Configurable to extract features from specific layers of the model

- **Feature Transformation**:
  - StandardScalerPCATransformer: Applies standardization and PCA dimensionality reduction
  - Reduces high-dimensional feature vectors to manageable size (e.g., 1280D → 256D)
  - Improves efficiency of downstream tasks like KNN computation
  - Preserves essential information for drift detection

#### 1.3 Drift Detection

TinyLCM implements multiple autonomous drift detection strategies:

- **KNNDistanceMonitor**:
  - Monitors distances to nearest neighbors in feature space
  - Uses Page-Hinkley test to detect significant increases in distances
  - Particularly effective for detecting unknown objects or classes
  - Configurable thresholds and cooldown periods

- **FeatureMonitor**:
  - Tracks statistical properties of feature vectors
  - Detects changes in feature distribution
  - Uses control charts to identify out-of-distribution samples

- **PageHinkleyFeatureMonitor**:
  - Implements Page-Hinkley test on feature statistics
  - Efficient algorithm for detecting sustained changes in data distribution
  - Low memory footprint for resource-constrained devices

- **EWMAFeatureMonitor**:
  - Uses Exponentially Weighted Moving Average to detect distributional shifts
  - Adaptive to gradual changes in feature statistics

#### 1.4 Classification and Adaptation

- **LightweightKNN**:
  - K-Nearest Neighbors classifier optimized for resource-constrained devices
  - Maintains a buffer of training samples with their timestamps
  - Supports different distance metrics (euclidean, cosine, manhattan)
  - Provides confidence scores based on neighbor distances

- **Adaptive Handlers**:
  - ActiveHandler: Aggressively adapts to new labeled samples
  - PassiveHandler: Conservatively updates the model
  - HybridHandler: Balances between stability and adaptability
  - Each handler implements different strategies for model adaptation

#### 1.5 State Management and Persistence

- **AdaptiveStateManager**:
  - Manages versioning of model state
  - Provides rollback capabilities when adaptation degrades performance
  - Persists state to disk for recovery after device reboot

- **AdaptationTracker**:
  - Records adaptation events and model changes
  - Tracks metrics before and after adaptation
  - Enables analysis of adaptation effectiveness

#### 1.6 Synchronization and Communication

- **SyncClient**:
  - Manages communication with TinySphere server
  - Implements package-based data exchange
  - Handles intermittent connectivity scenarios
  - Supports opportunistic synchronization when connectivity is available

- **ExtendedSyncClient**:
  - Extends SyncClient with additional capabilities
  - Creates and sends specialized packages (drift events, metrics, logs)
  - Handles device information updates
  - Supports geolocation tracking

### 2. Edge Device Applications

#### 2.1 Scenario 2: Drift Detection for Multi-class Object Recognition

This example implementation (`main_scenario2.py`) demonstrates autonomous drift detection for multi-class object recognition:

- **Application Flow**:
  1. Initialization: Loads configuration, sets up camera, initializes TinyLCM components
  2. Feature Processing: Extracts features using TFLite model, transforms with StandardScalerPCA
  3. Classification: Uses LightweightKNN with pre-loaded state
  4. Drift Detection: Monitors KNN distances to detect distribution shifts
  5. Synchronization: Periodically syncs with TinySphere server when available
  6. Logging: Records metrics, drift events, and images for analysis

- **Configuration**:
  - JSON-based configuration (`config_scenario2.json`)
  - Configurable parameters for camera, model, feature extraction, transformation
  - Drift detector settings (thresholds, update intervals, etc.)
  - Synchronization settings (server URL, API key, intervals)

- **Hardware Integration**:
  - Camera access via OpenCV
  - TensorFlow Lite for efficient model inference
  - Designed for Raspberry Pi Zero 2W with limited resources

- **Operational Monitoring**:
  - Tracks inference time, memory usage, CPU load
  - Records metrics for performance analysis
  - Logs device operating conditions

### 3. TinySphere Server

TinySphere is the server-side platform that provides centralized management, monitoring, and validation capabilities for the TinyMLOps ecosystem.

#### 3.1 API Architecture

- **FastAPI Backend**:
  - RESTful API endpoints for device communication
  - Asynchronous request handling for scalability
  - JSON-based data exchange format
  - Routers organized by domain (devices, packages, drift, etc.)

- **Key Endpoints**:
  - Device Registration and Management
  - Package Upload and Processing
  - Drift Event Analysis
  - Model Registry and Deployment
  - Operational Metrics Collection

#### 3.2 Data Processing Pipeline

- **Package Processor**:
  - Extracts incoming packages from edge devices
  - Processes different package types (drift events, metrics, logs)
  - Routes data to appropriate services

- **Drift Service**:
  - Analyzes drift events from devices
  - Validates on-device heuristic adaptations
  - Provides feedback for correction

- **Operational Logs Service**:
  - Processes device operational logs
  - Extracts metrics and events
  - Enables performance monitoring

#### 3.3 Storage Architecture

- **PostgreSQL Database**:
  - Stores structured data (devices, drift events, metrics)
  - Tracking of device status and history
  - Querying for dashboard visualizations

- **MinIO (S3-compatible) Storage**:
  - Drift Images Bucket: Stores images related to drift events
  - Prediction Images Bucket: Stores images from device predictions
  - Operational Logs Bucket: Stores device logs and metrics

#### 3.4 Integration with External Systems

- **MLflow Integration**:
  - Experiment tracking
  - Model registry
  - Model versioning and deployment

## Key Workflows

### 1. Autonomous Drift Detection Workflow

1. **Feature Processing**:
   - Input data captured by device sensors
   - Features extracted by TFLiteFeatureExtractor
   - Features transformed by StandardScalerPCATransformer

2. **Inference and Monitoring**:
   - LightweightKNN makes prediction
   - KNNDistanceMonitor analyzes neighbor distances
   - Page-Hinkley test detects significant increase in distances

3. **Drift Response**:
   - Drift event triggered when threshold exceeded
   - Image and metadata captured
   - Drift package created for synchronization

4. **Synchronization**:
   - Package queued for transmission
   - Sent to TinySphere when connectivity available
   - Server processes drift event

5. **Server-side Analysis**:
   - TinySphere validates drift detection
   - Analyzes drift images and metadata
   - Updates dashboard with new drift events

### 2. On-device Adaptation Workflow

1. **Quarantine Process**:
   - Samples with detected drift quarantined
   - QuarantineBuffer manages quarantined samples
   - Samples grouped by feature similarity

2. **Heuristic Adaptation**:
   - HeuristicAdapter analyzes quarantined samples
   - Clustering identifies patterns in similar samples
   - Pseudo-labels assigned based on cluster characteristics

3. **Model Update**:
   - AdaptivePipeline updates model with pseudo-labeled samples
   - AdaptationTracker records adaptation events
   - AdaptiveStateManager creates snapshot for potential rollback

4. **Performance Monitoring**:
   - Impact of adaptation measured
   - Rollback triggered if performance degraded
   - Metrics recorded for analysis

### 3. Server-Assisted Validation Workflow

1. **Device Synchronization**:
   - Device connects to TinySphere server
   - Uploads pending packages (drift events, images, logs)
   - Requests validation of quarantined samples

2. **Server Processing**:
   - Server processes uploaded packages
   - Analyzes drift events and images
   - Validates heuristic adaptations

3. **Feedback Loop**:
   - Server sends validation results
   - Device applies validated labels
   - Model updated with high-confidence labels

4. **Fleet-wide Analysis**:
   - Server aggregates data across devices
   - Identifies common drift patterns
   - Updates dashboards with insights

## Implementation Details

### TinyLCM Code Organization

- **Core Module**:
  - `base.py`: Core interfaces and abstract classes
  - `pipeline.py`: InferencePipeline and AdaptivePipeline implementations
  - `data_structures.py`: Data containers for samples and events

- **Feature Processing**:
  - `feature_extractors/`: Components for feature extraction
  - `feature_transformers/`: Components for feature transformation

- **Drift Detection**:
  - `drift_detection/`: Autonomous drift detectors
  - `drift_detection/base.py`: Base class for all detectors
  - `drift_detection/features.py`: Feature-based drift detectors

- **Classification and Adaptation**:
  - `classifiers/`: Lightweight classifier implementations
  - `handlers/`: Adaptation strategy implementations
  - `heuristics/`: Heuristic adaptation components

- **State Management**:
  - `state_manager.py`: Version control for model state
  - `adaptation_tracker.py`: Tracking adaptation events

- **Monitoring and Logging**:
  - `data_logger/`: Components for data logging
  - `operational_monitor/`: Monitoring system resources

- **Synchronization**:
  - `client/`: Communication with TinySphere server
  - `client/sync_interface.py`: Interface for synchronization
  - `client/sync_client.py`: Implementation of synchronization logic

### TinySphere Code Organization

- **API Module**:
  - `main.py`: FastAPI application setup
  - `routes/`: API endpoint implementations
  - `dependencies/`: Shared dependencies (DB, etc.)
  - `middleware/`: Request processing middleware

- **Database Module**:
  - `models.py`: Database model definitions
  - `migrations/`: Alembic migrations for schema changes

- **Services Module**:
  - `device_service.py`: Device management
  - `drift_service.py`: Drift event processing
  - `package_service.py`: Package management
  - `notification_service.py`: Notification handling

- **Importer Module**:
  - `package_processor.py`: Package extraction and processing
  - `extractors/`: Package format extractors
  - `transformers/`: Data transformation components

## Deployment Architecture

### Edge Device Deployment

- **Hardware Requirements**:
  - Raspberry Pi Zero 2W (or similar)
  - Camera module
  - Storage for model and state

- **Software Requirements**:
  - Python 3.7+
  - TensorFlow Lite
  - OpenCV
  - NumPy

- **Deployment Steps**:
  1. Install dependencies
  2. Copy TinyLCM library
  3. Configure application settings
  4. Initialize model state
  5. Run example implementation

### TinySphere Server Deployment

- **Infrastructure Requirements**:
  - Web server
  - PostgreSQL database
  - MinIO object storage

- **Containerization**:
  - Docker containers for backend, frontend, and services
  - Docker Compose for local deployment
  - Kubernetes for production deployment (optional)

- **Deployment Steps**:
  1. Set up PostgreSQL database
  2. Configure MinIO buckets
  3. Run database migrations
  4. Start API server
  5. Deploy frontend application

## Performance Considerations

### Edge Device Optimizations

- **Memory Efficiency**:
  - Lightweight implementations of algorithms
  - Incremental processing of data
  - Careful management of object lifecycle

- **Computational Efficiency**:
  - TensorFlow Lite quantized models
  - Dimensionality reduction for feature vectors
  - Optimized KNN implementation

- **Storage Efficiency**:
  - Compact state representation
  - Configurable logging levels
  - Data buffer management

### Server Scalability

- **Database Scaling**:
  - Indexed queries for performance
  - Connection pooling
  - Periodic cleanup of old data

- **API Performance**:
  - Asynchronous request handling
  - Performance monitoring middleware
  - Rate limiting for device endpoints

## Security Considerations

- **Device Authentication**:
  - API key-based authentication
  - Unique device identifiers
  - Secure storage of credentials

- **Data Privacy**:
  - Configurable data collection settings
  - Minimization of sensitive data transmission
  - Data anonymization where appropriate

- **Server Security**:
  - CORS configuration
  - Input validation
  - Rate limiting and brute force protection

## Future Extensions

1. **Federated Learning**: Aggregate insights across devices without centralizing data
2. **Advanced Adaptation Strategies**: More sophisticated on-device adaptation techniques
3. **Multi-modal Support**: Extend beyond image-based models to other sensor types
4. **Edge-to-Edge Communication**: Enable device collaboration without server mediation
5. **Compression Techniques**: Further optimize data exchange between devices and server

## Conclusion

The TinyMLOps ecosystem provides a comprehensive solution for deploying and managing ML models on resource-constrained edge devices. By combining on-device autonomy with opportunistic server assistance, it enables robust ML lifecycle management even in challenging connectivity environments.

The modular architecture allows for flexible deployment scenarios, from fully autonomous edge operation to server-assisted adaptation, making it suitable for a wide range of IoT and edge AI applications.