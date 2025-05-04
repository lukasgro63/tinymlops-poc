# TinyLCM: Tiny Lifecycle Manager

TinyLCM is a lightweight edge computing library for Machine Learning Lifecycle Management on resource-constrained devices. It provides tools for managing ML models, monitoring inference processes, detecting data drift, logging data, and synchronizing with a central server.

## Features

- **Model Management**: Version tracking, storage, and loading of ML models
- **Inference Monitoring**: Track latency, accuracy, and resource consumption
- **Drift Detection**: Detect data and model drift using statistical methods
- **Data Logging**: Store input/output data and metadata for analysis
- **TinySphere Integration**: Sync with central server for model updates and data upload

## Installation

### Installation on Development Environment

```bash
# Clone the repository
git clone https://github.com/lukasgro63/tinymlops-poc.git
cd tinymlops-poc/tinylcm

# Install as editable package
pip install -e .
```

### Installation on Raspberry Pi

When deploying on a Raspberry Pi or other edge device, we recommend using the pi_setup.sh script provided in the example directory:

```bash
# Create directory structure and prepare environment
bash pi_setup.sh

# Install dependencies
pip3 install numpy tflite-runtime requests --break-system-packages

# Install TinyLCM as a package
cd ~/tinymlops/tinylcm
pip3 install -e . --break-system-packages

# Make sure launch script is executable
chmod +x ~/tinymlops/src/launch.sh
chmod +x ~/tinymlops/launch.sh
```

## Architecture

TinyLCM consists of several core components:

1. **Feature Extractors**: Transform input data into feature vectors
2. **Classifiers**: Make predictions and adapt to new data
3. **Handlers**: Manage adaptation strategies (active, passive, hybrid)
4. **Drift Detectors**: Detect changes in data distribution or model performance
5. **State Manager**: Persist and load adaptive model states
6. **Data Logger**: Record input/output data and metadata
7. **Sync Client**: Communicate with TinySphere server

## Quick Start

```python
import numpy as np
from tinylcm.core.pipeline import AdaptivePipeline
from tinylcm.core.feature_extractors.base import NullFeatureExtractor
from tinylcm.core.classifiers.knn import LightweightKNN
from tinylcm.core.handlers.hybrid import HybridHandler

# Initialize components
feature_extractor = NullFeatureExtractor()
classifier = LightweightKNN(n_neighbors=3)
handler = HybridHandler(classifier)

# Create pipeline
pipeline = AdaptivePipeline(
    feature_extractor=feature_extractor,
    handler=handler,
    model_storage_dir="./models",
    data_log_dir="./data_logs"
)

# Inference with adaptation
input_data = np.array([1.0, 2.0, 3.0, 4.0])
prediction = pipeline.predict(input_data)
print(f"Prediction: {prediction}")

# Provide feedback (ground truth) for adaptation
pipeline.adapt(input_data, "class_a")
```

## Synchronization with TinySphere

```python
from tinylcm.client.sync_client import SyncClient

# Initialize sync client
sync_client = SyncClient(
    server_url="http://your-server:8000",
    api_key="your-api-key",
    device_id="device-001",
    sync_dir="./sync_data"
)

# Register device with TinySphere
sync_client.register_device()

# Sync all pending packages
results = sync_client.sync_all_pending_packages()
print(f"Synced {sum(1 for r in results if r['success'])} packages")
```

## Troubleshooting

### Common Issues on Raspberry Pi

1. **Permission Errors**:
   ```bash
   # Make scripts executable
   chmod +x ~/tinymlops/src/launch.sh
   chmod +x ~/tinymlops/launch.sh
   ```

2. **File Not Found Errors**:
   ```bash
   # Make sure you're in the right directory
   cd ~/tinymlops/src
   ```

3. **Import Errors**:
   ```bash
   # Check TinyLCM installation
   pip3 list | grep tinylcm
   
   # Reinstall if needed
   cd ~/tinymlops/tinylcm
   pip3 install -e . --break-system-packages
   ```

## Contributing

Contributions to TinyLCM are welcome! Please follow these steps:

1. Fork the repository
2. Create a feature branch
3. Make your changes
4. Run tests
5. Submit a pull request

## License

TinyLCM is released under the MIT License.