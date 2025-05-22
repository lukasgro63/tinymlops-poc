# TinyLCM Complete Workflow Analysis

This document provides a comprehensive analysis of the complete TinyLCM workflow from local development to production deployment with monitoring and MLOps integration.

## Overview

The TinyLCM system implements a complete MLOps pipeline for edge devices, spanning from local model development to production deployment with full monitoring and model management capabilities. The workflow integrates multiple components: local development, GitHub distribution, edge deployment, MLflow experiment tracking, TinySphere server management, and MinIO artifact storage.

## Workflow Components Analysis

### 1. Local Development Phase

**Location**: Developer's local machine
**Key Scripts**: `/examples/model_training/train_tflite_object_model.py`

#### Process:
1. **Model Training**:
   - Uses MobileNetV2 transfer learning for object classification
   - Trains on custom dataset in `/examples/assets/training_images/`
   - Creates TFLite quantized model optimized for Raspberry Pi Zero 2W
   - Outputs: `model_object.tflite`, `labels_object.txt`

2. **Feature Processing Pipeline Creation**:
   - Generates `StandardScalerPCATransformer` for feature dimensionality reduction (1280D → 256D)
   - Creates `feature_processor.pkl` for deployment
   - Ensures compatibility with TinyLCM drift detection system

3. **Configuration Generation**:
   - Creates scenario-specific JSON configs (e.g., `config_scenario2.json`)
   - Defines feature extractors, transformers, classifiers, and drift detectors
   - Specifies model paths and hyperparameters

#### Example Configuration:
```json
{
  "feature_extractor": {
    "type": "TFLiteFeatureExtractor",
    "model_path": "./model/model_object.tflite",
    "feature_layer_index": 0
  },
  "feature_transformation": {
    "type": "StandardScalerPCA", 
    "model_path": "./model/feature_processor.pkl"
  },
  "adaptive_classifier": {
    "type": "LightweightKNN",
    "k": 5,
    "distance_metric": "euclidean"
  },
  "drift_detectors": [{
    "type": "KNNDistanceMonitor",
    "delta": 0.1,
    "lambda_threshold": 5.0
  }]
}
```

### 2. GitHub Repository & Distribution

**Repository**: `https://github.com/lukasgro63/tinymlops-poc`

#### Structure for Deployment:
```
tinymlops-poc/
├── tinylcm/                    # Core library
├── examples/
│   ├── one_line_install.sh     # Device installation script
│   ├── scenario2_drift_objects/
│   │   ├── config_scenario2.json
│   │   ├── main_scenario2.py
│   │   └── model/              # Trained models
│   │       ├── model_object.tflite
│   │       ├── feature_processor.pkl
│   │       └── labels_object.txt
│   └── utils/                  # Device utilities
└── uploads/                    # Device data uploads
```

#### Distribution Process:
1. **Model Artifacts**: Trained models and configs pushed to GitHub
2. **Version Control**: Git tags for model versions and deployments
3. **Public Access**: Raw GitHub URLs for direct device downloads

### 3. Device Deployment

**Target**: Raspberry Pi Zero 2W
**Installation Script**: `/examples/one_line_install.sh`

#### One-Line Installation Process:
```bash
curl -sSL https://raw.githubusercontent.com/lukasgro63/tinymlops-poc/main/examples/one_line_install.sh | bash
```

#### Installation Steps:
1. **Repository Cloning**:
   ```bash
   git clone --depth=1 "$REPO_URL" "$TEMP_DIR"
   ```

2. **Selective File Copy**:
   ```bash
   cp -r "$TEMP_DIR/tinylcm" "$BASE_DIR/"
   cp -r "$TEMP_DIR/examples" "$BASE_DIR/"
   ```

3. **Dependency Installation**:
   ```bash
   python3 -m pip install --break-system-packages -r requirements.txt
   python3 -m pip install --break-system-packages -e tinylcm/
   ```

4. **Directory Structure Creation**:
   ```bash
   mkdir -p scenario*/logs scenario*/state scenario*/debug 
   mkdir -p scenario*/drift_images scenario*/sync_data
   ```

5. **Cleanup**: Removes unnecessary files, keeps only library + models

#### Runtime Execution:
```bash
cd ~/tinymlops/examples/scenario2_drift_objects/
python3 main_scenario2.py
```

### 4. MLflow Integration

**Service**: `tinysphere/api/services/mlflow_service.py`
**Database**: PostgreSQL (mlflow-postgres container)
**Artifact Store**: MinIO S3-compatible storage

#### MLflow Workflow:
1. **Experiment Tracking**:
   - Device uploads training packages via SyncClient
   - MLflowService processes packages asynchronously
   - Extracts models, metrics, logs, and artifacts

2. **Model Registry**:
   - Models registered with versioning
   - Metadata tracking (accuracy, drift performance, etc.)
   - Stage management (staging, production, archived)

3. **Artifact Management**:
   - Models stored in MinIO (`mlflow` bucket)
   - Features and preprocessors versioned
   - Training data and validation sets archived

#### Code Example:
```python
class MLflowService:
    def __init__(self):
        self.mlflow_uri = "http://mlflow:5000"
        mlflow.set_tracking_uri(self.mlflow_uri)
        
    def process_package(self, package_path):
        # Extract models, metrics, logs
        transformers = [
            ModelTransformer(),
            MetricsTransformer(), 
            DriftTransformer(),
            LogsTransformer()
        ]
```

### 5. TinySphere Integration

**Server**: FastAPI application with React frontend
**Database**: PostgreSQL (tinysphere-postgres container)
**Model Registry**: Separate from MLflow for production models

#### TinySphere Components:

1. **Package Processing**:
   ```python
   # tinysphere/importer/package_processor.py
   class PackageImporter:
       def extract_and_process(self, package_path):
           # Extract .tar packages from devices
           # Process drift events, logs, images
           # Store in PostgreSQL + MinIO
   ```

2. **Model Management**:
   - Imports models from MLflow
   - Maintains separate production model registry
   - Handles model deployment to device fleets

3. **Device Fleet Management**:
   - Device registration and status tracking
   - Performance monitoring and alerting
   - Geolocation and connectivity tracking

### 6. Monitoring & Data Pipeline

**On-Device Monitoring**: Continuous data collection
**Data Upload**: SyncClient batched uploads
**Storage**: MinIO buckets for different data types

#### Data Flow:
1. **Device Data Collection**:
   ```python
   # tinylcm/client/sync_client.py
   class SyncClient:
       def upload_package(self, package_path):
           # Upload logs, metrics, drift images
           # Batch processing for efficiency
   ```

2. **MinIO Bucket Structure**:
   ```
   minio/
   ├── mlflow/              # MLflow artifacts
   ├── drift/               # Drift detection data
   ├── drift-samples/       # Quarantined samples
   ├── drift-features/      # Feature vectors
   ├── prediction-images/   # Prediction samples
   └── data-logs/           # Operational logs
   ```

3. **Data Processing Pipeline**:
   ```python
   # tinysphere/importer/transformers/
   transformers = [
       DriftEventsTransformer(),     # Process drift images
       OperationalLogsTransformer(), # System metrics
       PredictionImagesTransformer() # Prediction samples
   ]
   ```

### 7. Infrastructure Components

**Docker Compose Setup**: `docker-compose.yml`

#### Service Architecture:
```yaml
services:
  postgres:           # TinySphere database
  mlflow-postgres:    # MLflow tracking database  
  minio:              # S3-compatible object storage
  minio-setup:        # Bucket initialization
  mlflow:             # MLflow tracking server
  tinysphere-backend: # TinySphere API
  tinysphere-frontend:# React dashboard
  nginx:              # Reverse proxy
```

#### Network Communication:
```
Device (Pi Zero 2W) ←→ TinySphere API ←→ PostgreSQL
                                    ←→ MinIO
                                    ←→ MLflow Server ←→ MLflow PostgreSQL
```

## Complete Workflow Sequence

### Phase 1: Development & Training
1. Developer trains model locally using `train_tflite_object_model.py`
2. Model and config files generated in `examples/scenario2_drift_objects/model/`
3. Changes committed and pushed to GitHub repository

### Phase 2: Device Deployment  
1. Device executes one-line install script via curl
2. Repository cloned, unnecessary files removed
3. TinyLCM library and models installed locally
4. Device starts running inference with adaptive monitoring

### Phase 3: Production Operation
1. Device performs inference with continuous drift detection
2. Adaptation events and monitoring data collected locally
3. Periodic upload of packages to TinySphere via SyncClient
4. TinySphere processes packages and updates databases

### Phase 4: MLOps Management
1. TinySphere imports data to MLflow for experiment tracking
2. Models versioned and stored with performance metrics
3. Production models managed through TinySphere registry
4. Dashboard provides real-time monitoring and fleet management

This end-to-end workflow enables complete MLOps lifecycle management for edge AI deployments with autonomous adaptation capabilities.