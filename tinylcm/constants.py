from enum import Enum
from typing import Dict, Any, Final

VERSION: Final[str] = "0.2.0"

class ModelFormat(str, Enum):
    TFLITE = "tflite"
    ONNX = "onnx"
    PYTORCH = "pytorch"
    PICKLE = "pkl"
    JSON = "json"
    UNKNOWN = "unknown"

MODEL_FORMAT_TFLITE: Final[str] = ModelFormat.TFLITE.value
MODEL_FORMAT_ONNX: Final[str] = ModelFormat.ONNX.value
MODEL_FORMAT_PYTORCH: Final[str] = ModelFormat.PYTORCH.value
MODEL_FORMAT_PICKLE: Final[str] = ModelFormat.PICKLE.value

class Status(str, Enum):
    RUNNING = "RUNNING"
    COMPLETED = "COMPLETED"
    FAILED = "FAILED"
    ABORTED = "ABORTED"
    PENDING = "PENDING"
    UNKNOWN = "UNKNOWN"

STATUS_RUNNING: Final[str] = Status.RUNNING.value
STATUS_COMPLETED: Final[str] = Status.COMPLETED.value
STATUS_FAILED: Final[str] = Status.FAILED.value
STATUS_ABORTED: Final[str] = Status.ABORTED.value

class DataType(str, Enum):
    IMAGE = "image"
    TEXT = "text"
    SENSOR = "sensor"
    JSON = "json"
    AUDIO = "audio"
    VIDEO = "video"
    BINARY = "binary"
    FEATURE = "feature"

DATA_TYPE_IMAGE: Final[str] = DataType.IMAGE.value
DATA_TYPE_TEXT: Final[str] = DataType.TEXT.value
DATA_TYPE_SENSOR: Final[str] = DataType.SENSOR.value
DATA_TYPE_JSON: Final[str] = DataType.JSON.value
DATA_TYPE_FEATURE: Final[str] = DataType.FEATURE.value

class AdaptationStrategy(str, Enum):
    PASSIVE = "passive"
    ACTIVE = "active"
    HYBRID = "hybrid"
    HEURISTIC = "heuristic"
    EXTERNAL = "external"
    NONE = "none"

ADAPTATION_STRATEGY_PASSIVE: Final[str] = AdaptationStrategy.PASSIVE.value
ADAPTATION_STRATEGY_ACTIVE: Final[str] = AdaptationStrategy.ACTIVE.value
ADAPTATION_STRATEGY_HYBRID: Final[str] = AdaptationStrategy.HYBRID.value
ADAPTATION_STRATEGY_HEURISTIC: Final[str] = AdaptationStrategy.HEURISTIC.value
ADAPTATION_STRATEGY_EXTERNAL: Final[str] = AdaptationStrategy.EXTERNAL.value

class FileFormat(str, Enum):
    JSON = "json"
    CSV = "csv"
    TXT = "txt"
    PNG = "png"
    JPG = "jpg"
    JPEG = "jpeg"
    YAML = "yaml"
    HDF5 = "h5"
    PICKLE = "pkl"

FILE_FORMAT_JSON: Final[str] = FileFormat.JSON.value
FILE_FORMAT_CSV: Final[str] = FileFormat.CSV.value
FILE_FORMAT_TXT: Final[str] = FileFormat.TXT.value
FILE_FORMAT_PNG: Final[str] = FileFormat.PNG.value
FILE_FORMAT_JPG: Final[str] = FileFormat.JPG.value
FILE_FORMAT_JPEG: Final[str] = FileFormat.JPEG.value

DEFAULT_BASE_DIR: Final[str] = "tinylcm_data"
DEFAULT_INFERENCE_DIR: Final[str] = f"{DEFAULT_BASE_DIR}/inference_logs"
DEFAULT_OPERATION_DIR: Final[str] = f"{DEFAULT_BASE_DIR}/operation_logs"  # For OperationalMonitor
DEFAULT_DATA_DIR: Final[str] = f"{DEFAULT_BASE_DIR}/data_logs"
DEFAULT_SYNC_DIR: Final[str] = f"{DEFAULT_BASE_DIR}/sync"
DEFAULT_LOG_DIR: Final[str] = f"{DEFAULT_BASE_DIR}/logs"
DEFAULT_ADAPTIVE_STATES_DIR: Final[str] = f"{DEFAULT_BASE_DIR}/adaptive_states"
DEFAULT_ADAPTATION_LOGS_DIR: Final[str] = f"{DEFAULT_BASE_DIR}/adaptation_logs"
DEFAULT_QUARANTINE_DIR: Final[str] = f"{DEFAULT_BASE_DIR}/quarantine"
DEFAULT_HEURISTIC_DIR: Final[str] = f"{DEFAULT_BASE_DIR}/heuristic_logs"
# Legacy paths maintained for compatibility with existing data
DEFAULT_MODELS_DIR: Final[str] = f"{DEFAULT_BASE_DIR}/models"  # Now use ADAPTIVE_STATES_DIR
DEFAULT_TRAINING_DIR: Final[str] = f"{DEFAULT_BASE_DIR}/training_runs"  # Now use ADAPTATION_LOGS_DIR
DEFAULT_DRIFT_DIR: Final[str] = f"{DEFAULT_BASE_DIR}/drift_detector"  # Now integrated into adaptive handlers

DEFAULT_LOG_FILE: Final[str] = f"{DEFAULT_LOG_DIR}/tinylcm.log"
DEFAULT_CONFIG_FILE: Final[str] = "tinylcm_config.json"
DEFAULT_ACTIVE_MODEL_LINK: Final[str] = "active_model"

DEFAULT_MAX_STORAGE: Final[int] = 1024 * 1024 * 1024  # 1GB
DEFAULT_MEMORY_ENTRIES: Final[int] = 1000
DEFAULT_LOG_INTERVAL: Final[int] = 100
DEFAULT_BUFFER_SIZE: Final[int] = 50

DEFAULT_CONFIG: Final[Dict[str, Dict[str, Any]]] = {
    "storage": {
        "base_dir": DEFAULT_BASE_DIR,
        "max_storage_bytes": DEFAULT_MAX_STORAGE,
        "cleanup_threshold": 0.9  # Cleanup when storage is 90% full
    },
    "operational_monitor": {
        "storage_dir": DEFAULT_OPERATION_DIR,
        "log_interval": DEFAULT_LOG_INTERVAL,
        "memory_window_size": DEFAULT_MEMORY_ENTRIES,
        "confidence_threshold": 0.3,  # Alert on confidence below this
        "latency_threshold_ms": 100,  # Alert on latency above this (ms)
        "collect_system_metrics": True,
        "system_metrics_interval": 10.0  # Seconds between system metric collections
    },
    "inference_monitor": {
        "storage_dir": DEFAULT_INFERENCE_DIR,
        "log_interval": DEFAULT_LOG_INTERVAL,
        "memory_window_size": DEFAULT_MEMORY_ENTRIES,
        "confidence_threshold": 0.3,  # Alert on confidence below this
        "latency_threshold_ms": 100  # Alert on latency above this (ms)
    },
    "data_logger": {
        "storage_dir": DEFAULT_DATA_DIR,
        "buffer_size": DEFAULT_BUFFER_SIZE,
        "max_entries": 10000,
        "image_format": "jpg"
    },
    "sync": {
        "storage_dir": DEFAULT_SYNC_DIR,
        "auto_sync": False,
        "sync_interval_seconds": 3600,
        "max_retry": 3
    },
    "logging": {
        "log_dir": DEFAULT_LOG_DIR,
        "level": "INFO",
        "format": "%(asctime)s - %(name)s - %(levelname)s - %(message)s"
    },
    "state_manager": {
        "storage_dir": DEFAULT_ADAPTIVE_STATES_DIR,
        "max_states": 10,
        "enable_integrity_check": True,
        "compression": True
    },
    "adaptation_tracker": {
        "storage_dir": DEFAULT_ADAPTATION_LOGS_DIR,
        "max_events_in_memory": 100,
        "log_interval": 20,
        "export_format": "json"
    },
    "quarantine": {
        "storage_dir": DEFAULT_QUARANTINE_DIR,
        "max_size": 1000,
        "auto_persist": True,
        "persist_interval": 10,
        "quarantine_strategy": "HIGH_UNCERTAINTY",
        "confidence_threshold": 0.6,
        "outlier_threshold": 3.0
    },
    "heuristic": {
        "storage_dir": DEFAULT_HEURISTIC_DIR,
        "strategy": "HYBRID",
        "min_cluster_size": 5,
        "min_samples_for_adaptation": 10,
        "confidence_threshold": 0.7,
        "distance_threshold": 0.8,
        "max_new_classes": 3,
        "clustering_method": "SIMPLE_DISTANCE",
        "adaptation_cooldown": 100,
        "enable_rollback": True
    },
    "drift_detection": {
        "ewma_confidence": {
            "enabled": True,
            "lambda_param": 0.1,
            "threshold_factor": 3.0,
            "drift_window": 5,
            "training_size": 30
        },
        "page_hinkley_confidence": {
            "enabled": False,
            "delta": 0.05,
            "lambda_param": 10.0,
            "alpha": 0.9999
        },
        "distribution": {
            "enabled": True,
            "window_size": 50,
            "threshold": 0.25,
            "method": "block"
        },
        "feature": {
            "enabled": True,
            "window_size": 100,
            "threshold": 3.0,
            "reference_size": 50,
            "max_features": 50,
            "distance_metric": "euclidean"
        }
    },
    "adaptive": {
        "default_strategy": "hybrid",
        "max_training_samples": 100,
        "cusum_threshold": 5.0,
        "cusum_delta": 0.25,
        "batch_size": 30,
        "feature_layer_index": -2,
        "k_neighbors": 5,
        "distance_metric": "euclidean",
        "state_storage_dir": DEFAULT_ADAPTIVE_STATES_DIR,
        "adaptation_logs_dir": DEFAULT_ADAPTATION_LOGS_DIR,
        "max_saved_states": 10,
        "max_events_in_memory": 100,
        "enable_condensing": False,
        "condensing_method": "class_balanced",
        "use_numpy": True,
        "baseline_accuracy": 0.9,
        # New configuration for autonomous components
        "enable_autonomous_detection": True,
        "enable_quarantine": True,
        "enable_heuristic_adaptation": False,
        "external_validation": False,
        "quarantine_check_interval": 50,
        "heuristic_confidence_threshold": 0.7,
        "validation_timeout_seconds": 86400  # 24 hours
    },
    # Resource-constrained configuration for Pi Zero
    "adaptive_pi_zero": {
        "default_strategy": "hybrid",
        "max_training_samples": 50,
        "cusum_threshold": 5.0,
        "cusum_delta": 0.25,
        "batch_size": 15,
        "feature_layer_index": -2,
        "k_neighbors": 3,
        "distance_metric": "manhattan",
        "state_storage_dir": DEFAULT_ADAPTIVE_STATES_DIR,
        "adaptation_logs_dir": DEFAULT_ADAPTATION_LOGS_DIR,
        "max_saved_states": 5,
        "max_events_in_memory": 50,
        "enable_condensing": True,
        "condensing_method": "class_balanced",
        "use_numpy": False,
        "baseline_accuracy": 0.8,
        # Lightweight autonomous settings for Pi Zero
        "enable_autonomous_detection": True,
        "enable_quarantine": True,
        "enable_heuristic_adaptation": True,  # For true edge autonomy
        "external_validation": False,
        "quarantine_check_interval": 100,
        "heuristic_confidence_threshold": 0.8
    },
    # Legacy configurations (kept for backward compatibility)
    "legacy": {
        "model_manager_dir": DEFAULT_MODELS_DIR,
        "training_tracker_dir": DEFAULT_TRAINING_DIR,
        "drift_detector_dir": DEFAULT_DRIFT_DIR
    }
}
