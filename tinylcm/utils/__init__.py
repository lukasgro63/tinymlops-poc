"""
Utility functions and helpers for the TinyLCM library.

This module provides various utility functionalities:
- config: Configuration management
- file_utils: File operations helpers
- metrics: Metrics calculation and statistics utilities
- versioning: Version management and comparison utilities
- logging: Logging utilities
- errors: Custom exception types
"""

from tinylcm.utils.config import Config, get_config, set_global_config, load_config
from tinylcm.utils.file_utils import (
    ensure_dir,
    get_file_size,
    load_json,
    save_json,
    list_files,
    safe_remove,
    stream_read,
    stream_write,
    stream_read_jsonl,
    stream_write_jsonl
)
from tinylcm.utils.metrics import (
    MetricsCalculator,
    Timer,
    MovingAverage
)
from tinylcm.utils.versioning import (
    generate_timestamp_version,
    generate_incremental_version,
    calculate_file_hash,
    calculate_content_hash,
    create_version_info,
    compare_versions,
    get_version_diff
)
from tinylcm.utils.logging import setup_logger
from tinylcm.utils.errors import (
    TinyLCMError,
    ModelError, ModelNotFoundError, ModelIntegrityError,
    StorageError, StorageAccessError, StorageWriteError,
    ConfigError,
    DataLoggerError,
    MonitoringError,
    InvalidInputError,
    SyncError
)

__all__ = [
    # Configuration
    "Config",
    "get_config",
    "set_global_config",
    "load_config",

    # File utilities
    "ensure_dir",
    "get_file_size",
    "load_json",
    "save_json",
    "list_files",
    "safe_remove",
    "stream_read",
    "stream_write",
    "stream_read_jsonl",
    "stream_write_jsonl",

    # Metrics utilities
    "MetricsCalculator",
    "Timer",
    "MovingAverage",

    # Versioning utilities
    "generate_timestamp_version",
    "generate_incremental_version",
    "calculate_file_hash",
    "calculate_content_hash",
    "create_version_info",
    "compare_versions",
    "get_version_diff",
    
    # Logging utilities
    "setup_logger",
    
    # Error classes
    "TinyLCMError",
    "ModelError", "ModelNotFoundError", "ModelIntegrityError",
    "StorageError", "StorageAccessError", "StorageWriteError",
    "ConfigError",
    "DataLoggerError",
    "MonitoringError", 
    "InvalidInputError",
    "SyncError",
    "ConnectionError"
]