import gzip
import json
import os
import shutil
import tempfile
import time
import uuid
import zipfile
from pathlib import Path
from typing import Any, Callable, Dict, List, Optional, Union

from tinylcm.constants import DEFAULT_SYNC_DIR
from tinylcm.utils.config import Config, get_config
from tinylcm.utils.errors import SyncError
from tinylcm.utils.file_utils import ensure_dir, load_json, save_json
from tinylcm.utils.logging import setup_logger
from tinylcm.utils.versioning import calculate_file_hash


class SyncPackage:
    def __init__(self, package_id: str, device_id: str, package_type: str, work_dir: Union[str, Path], compression: Optional[str] = None, description: Optional[str] = None):
        self.logger = setup_logger(f"{__name__}.{self.__class__.__name__}")
        self.package_id = package_id
        self.device_id = device_id
        self.package_type = package_type
        self.description = description
        self.work_dir = Path(work_dir)
        self.compression = compression
        self.creation_time = time.time()
        self.files = []
        self.is_finalized = False
        ensure_dir(self.work_dir)
        self.logger.debug(f"Created sync package: {package_id}")
    
    def add_file(self, file_path: Union[str, Path], file_type: Optional[str] = None, metadata: Optional[Dict[str, Any]] = None) -> None:
        if self.is_finalized:
            raise SyncError(f"Cannot add file to finalized package: {self.package_id}")
        src_path = Path(file_path)
        if not src_path.exists():
            raise FileNotFoundError(f"File not found: {file_path}")
        rel_path = src_path.name
        dst_path = self.work_dir / rel_path
        shutil.copy2(src_path, dst_path)
        file_entry = {
            "original_path": str(src_path),
            "package_path": rel_path,
            "file_type": file_type or "unknown",
            "size_bytes": os.path.getsize(dst_path),
            "timestamp": time.time(),
            "hash": calculate_file_hash(dst_path),
            "metadata": metadata or {}
        }
        self.files.append(file_entry)
        self.logger.debug(f"Added file to package {self.package_id}: {rel_path}")
    
    def finalize(self, output_path: Union[str, Path]) -> str:
        if self.is_finalized:
            raise SyncError(f"Package already finalized: {self.package_id}")
        if not self.files:
            raise SyncError(f"Cannot finalize empty package: {self.package_id}")
        out_path = Path(output_path)
        ensure_dir(out_path.parent)
        if self.compression == "gzip":
            self._finalize_gzip(out_path)
        elif self.compression == "zip":
            self._finalize_zip(out_path)
        else:
            self._finalize_tar(out_path)
        self.is_finalized = True
        self.logger.info(f"Finalized package {self.package_id} to {out_path}")
        return str(out_path)
    
    def _finalize_gzip(self, output_path: Path) -> None:
        import tarfile
        with tarfile.open(output_path, "w:gz") as tar:
            for file_entry in self.files:
                file_path = self.work_dir / file_entry["package_path"]
                arcname = file_entry["package_path"]
                tar.add(file_path, arcname=arcname)
    
    def _finalize_zip(self, output_path: Path) -> None:
        with zipfile.ZipFile(output_path, "w", zipfile.ZIP_DEFLATED) as zipf:
            for file_entry in self.files:
                file_path = self.work_dir / file_entry["package_path"]
                zipf.write(file_path, arcname=file_entry["package_path"])
    
    def _finalize_tar(self, output_path: Path) -> None:
        import tarfile
        with tarfile.open(output_path, "w") as tar:
            for file_entry in self.files:
                file_path = self.work_dir / file_entry["package_path"]
                arcname = file_entry["package_path"]
                tar.add(file_path, arcname=arcname)
    
    def get_metadata(self) -> Dict[str, Any]:
        return {
            "package_id": self.package_id,
            "device_id": self.device_id,
            "package_type": self.package_type,
            "description": self.description,
            "creation_time": self.creation_time,
            "files": self.files,
            "compression": self.compression,
            "is_finalized": self.is_finalized
        }


class SyncInterface:
    def __init__(self, sync_dir: Optional[Union[str, Path]] = None, config: Optional[Config] = None):
        self.config = config or get_config()
        component_config = self.config.get_component_config("sync")
        self.logger = setup_logger(f"{__name__}.{self.__class__.__name__}")
        self.sync_dir = Path(sync_dir or component_config.get("storage_dir", DEFAULT_SYNC_DIR))
        self.packages_dir = ensure_dir(self.sync_dir / "packages")
        self.history_dir = ensure_dir(self.sync_dir / "history")
        self.active_packages = {}
        self.logger.info(f"Initialized sync interface with storage at: {self.sync_dir}")
    
    def create_package(self, device_id: str, package_type: str, description: Optional[str] = None, compression: Optional[str] = None) -> str:
        package_id = str(uuid.uuid4())
        package_dir = ensure_dir(self.packages_dir / f"tmp_{package_id}")
        package = SyncPackage(package_id=package_id, device_id=device_id, package_type=package_type, work_dir=package_dir, compression=compression, description=description)
        self.active_packages[package_id] = package
        self.logger.debug(f"Created package: {package_id}")
        return package_id
    
    def add_file_to_package(self, package_id: str, file_path: Union[str, Path], file_type: Optional[str] = None, metadata: Optional[Dict[str, Any]] = None) -> None:
        package = self._get_package(package_id)
        package.add_file(file_path, file_type, metadata)
    
    def add_directory_to_package(self, package_id: str, directory_path: Union[str, Path], recursive: bool = False, file_type: Optional[str] = None) -> int:
        package = self._get_package(package_id)
        dir_path = Path(directory_path)
        if not dir_path.exists() or not dir_path.is_dir():
            raise FileNotFoundError(f"Directory not found: {directory_path}")
        count = 0
        if recursive:
            for root, _, files in os.walk(dir_path):
                for file in files:
                    file_path = Path(root) / file
                    try:
                        package.add_file(file_path, file_type)
                        count += 1
                    except Exception as e:
                        self.logger.warning(f"Failed to add file {file_path}: {e}")
        else:
            for file_path in dir_path.iterdir():
                if file_path.is_file():
                    try:
                        package.add_file(file_path, file_type)
                        count += 1
                    except Exception as e:
                        self.logger.warning(f"Failed to add file {file_path}: {e}")
        self.logger.debug(f"Added {count} files from directory {directory_path} to package {package_id}")
        return count
    
    def create_package_from_components(self, device_id: str, model_manager=None, inference_monitor=None, data_logger=None, training_tracker=None, compression: Optional[str] = None) -> str:
        if not any([model_manager, inference_monitor, data_logger, training_tracker]):
            raise SyncError("At least one component must be provided")
        package_id = self.create_package(device_id=device_id, package_type="components", description="Package with component data", compression=compression)
        if model_manager:
            try:
                model_path = model_manager.load_model()
                try:
                    model_meta = model_manager.get_active_model_metadata()
                    if hasattr(model_meta, "items"):
                        safe_meta = {}
                        for key, value in model_meta.items():
                            if isinstance(value, (str, int, float, bool, list, dict)) or value is None:
                                safe_meta[key] = value
                    else:
                        safe_meta = {"note": "Model metadata extraction limited"}
                except Exception as meta_err:
                    self.logger.warning(f"Error extracting model metadata: {meta_err}")
                    safe_meta = {}
                self.add_file_to_package(package_id=package_id, file_path=model_path, file_type="model", metadata=safe_meta)
                self.logger.debug(f"Added model from ModelManager to package {package_id}")
            except Exception as e:
                self.logger.warning(f"Failed to add model data to package: {e}")
        if inference_monitor:
            try:
                metrics_path = inference_monitor.export_metrics(format="json")
                self.add_file_to_package(package_id=package_id, file_path=metrics_path, file_type="metrics")
                self.logger.debug(f"Added metrics from InferenceMonitor to package {package_id}")
            except Exception as e:
                self.logger.warning(f"Failed to add metrics data to package: {e}")
        if data_logger:
            try:
                log_path = data_logger.export_to_csv()
                self.add_file_to_package(package_id=package_id, file_path=log_path, file_type="data_log")
                self.logger.debug(f"Added data log from DataLogger to package {package_id}")
            except Exception as e:
                self.logger.warning(f"Failed to add data log to package: {e}")
        if training_tracker:
            try:
                runs = training_tracker.list_runs()
                if runs:
                    with tempfile.TemporaryDirectory() as temp_dir:
                        most_recent_run = runs[0]
                        export_dir = os.path.join(temp_dir, "runs")
                        training_tracker.export_to_mlflow_format(most_recent_run["run_id"], export_dir)
                        self.add_directory_to_package(package_id=package_id, directory_path=export_dir, recursive=True, file_type="training_run")
                        self.logger.debug(f"Added training run from TrainingTracker to package {package_id}")
            except Exception as e:
                self.logger.warning(f"Failed to add training data to package: {e}")
        return package_id
    
    def finalize_package(self, package_id: str) -> str:
        package = self._get_package(package_id)
        timestamp = int(time.time())
        output_file = f"{package_id}_{timestamp}"
        if package.compression == "gzip":
            output_file += ".tar.gz"
        elif package.compression == "zip":
            output_file += ".zip"
        else:
            output_file += ".tar"
        output_path = self.packages_dir / output_file
        package_path = package.finalize(output_path)
        metadata_path = self.packages_dir / f"{package_id}.meta.json"
        save_json(package.get_metadata(), metadata_path)
        del self.active_packages[package_id]
        package_dir = self.packages_dir / f"tmp_{package_id}"
        if package_dir.exists():
            shutil.rmtree(package_dir)
        self.logger.info(f"Finalized package {package_id} to {package_path}")
        return package_path
    
    def list_packages(self, filter_func: Optional[Callable[[Dict[str, Any]], bool]] = None, include_synced: bool = False) -> List[Dict[str, Any]]:
        packages = []
        metadata_files = list(self.packages_dir.glob("*.meta.json"))
        for meta_file in metadata_files:
            try:
                metadata = load_json(meta_file)
                package_id = metadata.get("package_id")
                if not package_id:
                    continue
                synced_meta_path = self.history_dir / f"{package_id}.sync.json"
                if synced_meta_path.exists():
                    sync_meta = load_json(synced_meta_path)
                    metadata["sync_status"] = sync_meta.get("status")
                    metadata["sync_time"] = sync_meta.get("sync_time")
                    metadata["server_id"] = sync_meta.get("server_id")
                    if not include_synced:
                        continue
                if filter_func and not filter_func(metadata):
                    continue
                packages.append(metadata)
            except Exception as e:
                self.logger.warning(f"Failed to load package metadata from {meta_file}: {e}")
        return packages
    
    def mark_as_synced(self, package_id: str, sync_time: float, server_id: str, status: str) -> None:
        meta_path = self.packages_dir / f"{package_id}.meta.json"
        if not meta_path.exists():
            raise SyncError(f"Package metadata not found: {package_id}")
        sync_meta = {
            "package_id": package_id,
            "sync_time": sync_time,
            "server_id": server_id,
            "status": status
        }
        sync_meta_path = self.history_dir / f"{package_id}.sync.json"
        save_json(sync_meta, sync_meta_path)
        self.logger.info(f"Marked package {package_id} as synced with server {server_id}")
    
    def _get_package(self, package_id: str) -> SyncPackage:
        if package_id not in self.active_packages:
            raise SyncError(f"Package not found or not active: {package_id}")
        return self.active_packages[package_id]