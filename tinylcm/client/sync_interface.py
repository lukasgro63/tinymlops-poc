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
    
    def add_file_to_package(self, package_id: str, file_path: Union[str, Path], file_type: Optional[str] = None, 
                        metadata: Optional[Dict[str, Any]] = None, wait_for_file: bool = True,
                        max_retries: int = 5, retry_delay: float = 0.5) -> bool:
        """Add a file to a package with retry logic.
        
        Args:
            package_id: ID of the package to add the file to
            file_path: Path to the file to add
            file_type: Type of the file
            metadata: Additional metadata
            wait_for_file: Whether to wait for the file to exist if it doesn't
            max_retries: Maximum number of retries when waiting for the file
            retry_delay: Delay between retries in seconds
            
        Returns:
            True if the file was added successfully, False otherwise
        """
        package = self._get_package(package_id)
        src_path = Path(file_path)
        
        # Wait for the file to exist if requested
        if wait_for_file and not src_path.exists():
            self.logger.debug(f"File {file_path} doesn't exist yet, waiting... ({max_retries} retries with {retry_delay}s delay)")
            for attempt in range(max_retries):
                time.sleep(retry_delay)
                if src_path.exists():
                    self.logger.debug(f"File {file_path} now exists after {attempt+1} retries")
                    break
            else:
                self.logger.warning(f"File {file_path} still doesn't exist after {max_retries} retries")
                return False
        
        try:
            package.add_file(file_path, file_type, metadata)
            return True
        except FileNotFoundError:
            self.logger.warning(f"File not found: {file_path}")
            return False
        except Exception as e:
            self.logger.warning(f"Error adding file {file_path} to package: {e}")
            return False
    
    def add_directory_to_package(self, package_id: str, directory_path: Union[str, Path], 
                            recursive: bool = False, file_type: Optional[str] = None,
                            max_retries: int = 3, retry_delay: float = 0.5) -> int:
        """Add all files from a directory to a package.
        
        Args:
            package_id: ID of the package to add files to
            directory_path: Path to the directory containing files to add
            recursive: Whether to recursively add files from subdirectories
            file_type: Type of the files
            max_retries: Maximum number of retries if directory isn't accessible
            retry_delay: Delay between retries in seconds
            
        Returns:
            Number of files successfully added
            
        Raises:
            FileNotFoundError: If the directory doesn't exist after retries
        """
        package = self._get_package(package_id)
        dir_path = Path(directory_path)
        
        # Retry if directory doesn't exist or isn't accessible yet
        for attempt in range(max_retries):
            if dir_path.exists() and dir_path.is_dir():
                break
            self.logger.debug(f"Directory {directory_path} doesn't exist or is not a directory yet, waiting... (attempt {attempt+1}/{max_retries})")
            time.sleep(retry_delay)
        
        # Final check after retries
        if not dir_path.exists() or not dir_path.is_dir():
            self.logger.warning(f"Directory not found after {max_retries} retries: {directory_path}")
            raise FileNotFoundError(f"Directory not found: {directory_path}")
        
        count = 0
        try:
            if recursive:
                for root, _, files in os.walk(dir_path):
                    for file in files:
                        file_path = Path(root) / file
                        try:
                            # Skip hidden files or system files
                            if file.startswith('.') or file.startswith('~'):
                                continue
                            
                            package.add_file(file_path, file_type)
                            count += 1
                        except Exception as e:
                            self.logger.warning(f"Failed to add file {file_path}: {e}")
            else:
                for file_path in dir_path.iterdir():
                    if file_path.is_file():
                        try:
                            # Skip hidden files
                            if file_path.name.startswith('.') or file_path.name.startswith('~'):
                                continue
                            
                            package.add_file(file_path, file_type)
                            count += 1
                        except Exception as e:
                            self.logger.warning(f"Failed to add file {file_path}: {e}")
            
            self.logger.debug(f"Added {count} files from directory {directory_path} to package {package_id}")
            
            # Return early if some files were added
            if count > 0:
                return count
                
            # Directory was found but no files were added
            self.logger.warning(f"No files were added from directory: {directory_path}")
            return 0
            
        except Exception as e:
            self.logger.error(f"Error adding files from directory {directory_path}: {e}")
            return count  # Return the count of files added before the error
    
    def create_package_from_components(self, device_id: str, 
                             adaptive_pipeline=None, state_manager=None, 
                             adaptation_tracker=None, inference_monitor=None, 
                             data_logger=None, compression: Optional[str] = None,
                             model_path: Optional[str] = None, model_labels_path: Optional[str] = None) -> str:
        """
        Create a package from various TinyLCM components.
        
        Args:
            device_id: Unique identifier for the device
            adaptive_pipeline: Optional AdaptivePipeline instance
            state_manager: Optional StateManager instance
            adaptation_tracker: Optional AdaptationTracker instance
            inference_monitor: Optional InferenceMonitor instance
            data_logger: Optional DataLogger instance
            compression: Compression type to use ("gzip", "zip", or None)
            model_path: Optional path to the model file to include
            model_labels_path: Optional path to the model labels file to include
            
        Returns:
            ID of the created package
        """
        if not any([adaptive_pipeline, state_manager, adaptation_tracker, inference_monitor, data_logger, model_path]):
            raise SyncError("At least one component must be provided")
        
        # Determine the package type based on content
        package_type = "components"  # Default to components
        description = "Package with multiple component data types"
        
        # Use a more specific package type if the package only contains one type of data
        if model_path and not any([inference_monitor, data_logger]):
            package_type = "model"
            description = "Model package"
        elif inference_monitor and not any([model_path, data_logger]):
            package_type = "metrics"
            description = "Metrics package"
        elif data_logger and not any([model_path, inference_monitor]):
            package_type = "data_log"
            description = "Data log package"
        
        # For combinations of different data types, use the combined type
        if model_path and inference_monitor and not data_logger:
            package_type = "model_metrics"
            description = "Model with metrics package"
        elif model_path and data_logger and not inference_monitor:
            package_type = "model_logs"
            description = "Model with logs package"
        elif inference_monitor and data_logger and not model_path:
            package_type = "metrics_logs"
            description = "Metrics and logs package"
            
        package_id = self.create_package(device_id=device_id, package_type=package_type, 
                                      description=description, 
                                      compression=compression)
        
        # Handle adaptive components
        if adaptive_pipeline:
            try:
                # Save the current state if a state manager is provided
                if state_manager:
                    state_id = adaptive_pipeline.save_state()
                    state_files = list(Path(state_manager.storage_dir).glob(f"{state_id}.*"))
                    for state_file in state_files:
                        self.add_file_to_package(
                            package_id=package_id, 
                            file_path=state_file, 
                            file_type="adaptive_state",
                            metadata={"state_id": state_id}
                        )
                    self.logger.debug(f"Added adaptive state from AdaptivePipeline to package {package_id}")
                # If no state manager, try to serialize the pipeline stats directly
                else:
                    with tempfile.NamedTemporaryFile(suffix=".json", delete=False) as tmp:
                        tmp_path = tmp.name
                        stats = adaptive_pipeline.get_statistics()
                        json.dump(stats, tmp)
                    self.add_file_to_package(
                        package_id=package_id, 
                        file_path=tmp_path, 
                        file_type="adaptive_stats"
                    )
                    os.unlink(tmp_path)
                    self.logger.debug(f"Added adaptive stats to package {package_id}")
            except Exception as e:
                self.logger.warning(f"Failed to add adaptive pipeline data to package: {e}")
                
        # Handle adaptation tracker with improved directory handling
        if adaptation_tracker:
            try:
                # Create a unique temp directory to avoid collisions
                unique_id = str(uuid.uuid4())
                temp_dir = tempfile.mkdtemp(prefix=f"adaptation_{unique_id}_")
                try:
                    # Use a unique subdirectory for export to avoid "Directory not empty" errors
                    export_dir = os.path.join(temp_dir, f"adaptation_{int(time.time())}")
                    os.makedirs(export_dir, exist_ok=True)
                    
                    # Export to MLflow format if possible
                    if hasattr(adaptation_tracker, "export_to_mlflow"):
                        adaptation_tracker.export_to_mlflow(export_dir)
                        # Check if directory has content
                        if any(os.scandir(export_dir)):
                            file_count = self.add_directory_to_package(
                                package_id=package_id,
                                directory_path=export_dir,
                                recursive=True,
                                file_type="adaptation_logs"
                            )
                            self.logger.debug(f"Added {file_count} adaptation files to package {package_id}")
                        else:
                            self.logger.warning(f"Adaptation export directory is empty: {export_dir}")
                    # Otherwise try to get events and metrics directly
                    else:
                        events_file = os.path.join(export_dir, "events.json")
                        metrics_file = os.path.join(export_dir, "metrics.json")
                        
                        # Get events and metrics
                        events = adaptation_tracker.get_events()
                        metrics = adaptation_tracker.get_metrics()
                        
                        # Save to files
                        with open(events_file, "w") as f:
                            json.dump([e.to_dict() if hasattr(e, "to_dict") else e for e in events], f)
                        with open(metrics_file, "w") as f:
                            json.dump(metrics, f)
                        
                        # Add individual files rather than directory
                        success1 = self.add_file_to_package(
                            package_id=package_id, 
                            file_path=events_file, 
                            file_type="adaptation_events"
                        )
                        success2 = self.add_file_to_package(
                            package_id=package_id, 
                            file_path=metrics_file, 
                            file_type="adaptation_metrics"
                        )
                        
                        if success1 and success2:
                            self.logger.debug(f"Added adaptation events and metrics to package {package_id}")
                        else:
                            self.logger.warning(f"Failed to add some adaptation files to package {package_id}")
                finally:
                    # Clean up temporary directory
                    try:
                        shutil.rmtree(temp_dir)
                    except Exception as cleanup_err:
                        self.logger.warning(f"Failed to clean up temporary directory: {cleanup_err}")
            except Exception as e:
                self.logger.warning(f"Failed to add adaptation data to package: {e}")
        
        # Handle inference monitor with improved file waiting
        if inference_monitor:
            try:
                # Export metrics in blocking mode to ensure file is created immediately
                metrics_path = inference_monitor.export_metrics(format="json", blocking=True)
                self.logger.debug(f"Exported metrics to {metrics_path} in blocking mode")
                
                # Add the metrics file to the package
                success = self.add_file_to_package(
                    package_id=package_id, 
                    file_path=metrics_path, 
                    file_type="metrics",
                    wait_for_file=True,
                    max_retries=5,  # Fewer retries needed since we used blocking mode
                    retry_delay=0.5
                )
                
                if success:
                    self.logger.debug(f"Successfully added metrics from InferenceMonitor to package {package_id}")
                else:
                    self.logger.warning(f"Failed to add metrics data to package: File not available after retries")
            except Exception as e:
                self.logger.warning(f"Failed to add metrics data to package: {e}")
        
        # Handle data logger with improved file waiting
        if data_logger:
            try:
                # Export data log in blocking mode to ensure file is created immediately
                log_path = data_logger.export_to_csv(blocking=True)
                self.logger.debug(f"Exported data log to {log_path} in blocking mode")
                
                # Add with fewer retries since we used blocking mode
                success = self.add_file_to_package(
                    package_id=package_id, 
                    file_path=log_path, 
                    file_type="data_log",
                    wait_for_file=True,
                    max_retries=5,
                    retry_delay=0.5
                )
                if success:
                    self.logger.debug(f"Added data log from DataLogger to package {package_id}")
                else:
                    self.logger.warning(f"Failed to add data log to package: File not available after retries")
            except Exception as e:
                self.logger.warning(f"Failed to add data log to package: {e}")
        
        # Include model file if provided and only if it has changed
        if model_path:
            try:
                from tinylcm.utils.versioning import calculate_file_hash
                model_file = Path(model_path)
                if model_file.exists():
                    # Calculate model file hash
                    model_hash = calculate_file_hash(model_file, algorithm="sha256")
                    self.logger.debug(f"Calculated hash for model {model_path}: {model_hash}")
                    
                    # Check if we have a record of this hash in the sync history
                    model_hash_file = self.sync_dir / "model_hashes.json"
                    hash_history = {}
                    
                    # Load existing hash history if available
                    if model_hash_file.exists():
                        try:
                            with open(model_hash_file, 'r') as f:
                                hash_history = json.load(f)
                        except Exception as e:
                            self.logger.warning(f"Failed to load model hash history: {e}")
                    
                    # Check if model has changed
                    model_key = f"{device_id}_{model_file.name}"
                    if model_key in hash_history and hash_history[model_key]["hash"] == model_hash:
                        self.logger.info(f"Model {model_path} hasn't changed (hash: {model_hash}), skipping inclusion in package")
                        
                        # Add a placeholder metadata file with hash info
                        model_meta_path = None
                        try:
                            model_meta = {
                                "model_format": model_file.suffix.lstrip('.'),
                                "timestamp": time.time(),
                                "device_id": device_id,
                                "model_hash": model_hash,
                                "params": {
                                    "model_file": model_file.name
                                },
                                "metrics": {},  # No metrics available yet
                                "unchanged": True  # Flag to indicate model hasn't changed
                            }
                            
                            # Create temporary metadata file
                            with tempfile.NamedTemporaryFile(mode='w', suffix='.json', delete=False) as tmp:
                                model_meta_path = tmp.name
                                # Import TinyLCMJSONEncoder to handle numpy arrays
                                from tinylcm.utils.file_utils import TinyLCMJSONEncoder
                                json.dump(model_meta, tmp, cls=TinyLCMJSONEncoder)
                            
                            # Add metadata file only (without the actual model)
                            if model_meta_path:
                                self.add_file_to_package(
                                    package_id=package_id,
                                    file_path=model_meta_path,
                                    file_type="model_metadata",
                                    metadata={"model_path": str(model_path), "model_hash": model_hash}
                                )
                                os.unlink(model_meta_path)
                                self.logger.debug(f"Added model metadata (without model file) to package {package_id}")
                        except Exception as meta_err:
                            self.logger.warning(f"Failed to create model metadata: {meta_err}")
                            if model_meta_path and os.path.exists(model_meta_path):
                                try:
                                    os.unlink(model_meta_path)
                                except:
                                    pass
                    else:
                        # Model has changed or is new, include it in the package
                        self.logger.info(f"Model {model_path} has changed or is new, including in package")
                        
                        # Update hash history
                        hash_history[model_key] = {
                            "hash": model_hash,
                            "timestamp": time.time(),
                            "device_id": device_id,
                            "path": str(model_file)
                        }
                        
                        # Save updated hash history
                        try:
                            with open(model_hash_file, 'w') as f:
                                json.dump(hash_history, f, indent=2)
                        except Exception as e:
                            self.logger.warning(f"Failed to save model hash history: {e}")
                        
                        # Create model metadata with timestamp and hash
                        model_meta_path = None
                        try:
                            model_meta = {
                                "model_format": model_file.suffix.lstrip('.'),
                                "timestamp": time.time(),
                                "device_id": device_id,
                                "model_hash": model_hash,
                                "params": {
                                    "model_file": model_file.name
                                },
                                "metrics": {}  # No metrics available yet
                            }
                            
                            # Create temporary metadata file
                            with tempfile.NamedTemporaryFile(mode='w', suffix='.json', delete=False) as tmp:
                                model_meta_path = tmp.name
                                # Import TinyLCMJSONEncoder to handle numpy arrays
                                from tinylcm.utils.file_utils import TinyLCMJSONEncoder
                                json.dump(model_meta, tmp, cls=TinyLCMJSONEncoder)
                            
                            # Add model file
                            self.add_file_to_package(
                                package_id=package_id,
                                file_path=model_path,
                                file_type="model_file"
                            )
                            self.logger.debug(f"Added model file {model_path} to package {package_id}")
                            
                            # Add metadata file
                            if model_meta_path:
                                self.add_file_to_package(
                                    package_id=package_id,
                                    file_path=model_meta_path,
                                    file_type="model_metadata",
                                    metadata={"model_path": str(model_path), "model_hash": model_hash}
                                )
                                os.unlink(model_meta_path)
                                self.logger.debug(f"Added model metadata to package {package_id}")
                        except Exception as meta_err:
                            self.logger.warning(f"Failed to create model metadata: {meta_err}")
                            if model_meta_path and os.path.exists(model_meta_path):
                                try:
                                    os.unlink(model_meta_path)
                                except:
                                    pass
                else:
                    self.logger.warning(f"Model file not found: {model_path}")
            except Exception as e:
                self.logger.warning(f"Failed to add model file to package: {e}")
        
        # Include model labels if provided
        if model_labels_path:
            try:
                labels_file = Path(model_labels_path)
                if labels_file.exists():
                    self.add_file_to_package(
                        package_id=package_id,
                        file_path=model_labels_path,
                        file_type="model_labels"
                    )
                    self.logger.debug(f"Added model labels file {model_labels_path} to package {package_id}")
                else:
                    self.logger.warning(f"Model labels file not found: {model_labels_path}")
            except Exception as e:
                self.logger.warning(f"Failed to add model labels file to package: {e}")
                
        return package_id
    
    def finalize_package(self, package_id: str) -> str:
        """Finalize a package and prepare it for sending.
        
        Args:
            package_id: ID of the package to finalize
            
        Returns:
            Path to the finalized package file
            
        Raises:
            SyncError: If the package is empty or cannot be finalized
        """
        package = self._get_package(package_id)
        timestamp = int(time.time())
        
        # Check if the package has any files
        if not package.files:
            self.logger.warning(f"Attempting to finalize empty package {package_id}")
            # Create a minimal dummy file to allow finalization
            dummy_file = tempfile.NamedTemporaryFile(delete=False, suffix=".json", dir=package.work_dir)
            dummy_data = {
                "package_id": package_id,
                "timestamp": timestamp,
                "message": "Empty package - created minimal file to allow finalization",
                "device_id": package.device_id,
                "warning": "No actual data was included in this package"
            }
            with open(dummy_file.name, 'w') as f:
                json.dump(dummy_data, f)
            
            try:
                package.add_file(
                    dummy_file.name, 
                    file_type="empty_package_marker",
                    metadata={"is_empty_marker": True}
                )
                self.logger.info(f"Added minimal marker file to empty package {package_id} to allow finalization")
            except Exception as e:
                self.logger.error(f"Failed to add empty package marker: {e}")
                os.unlink(dummy_file.name)
                raise SyncError(f"Cannot finalize empty package {package_id} and failed to add marker: {e}")
        
        # Now finalize with the normal flow
        output_file = f"{package_id}_{timestamp}"
        if package.compression == "gzip":
            output_file += ".tar.gz"
        elif package.compression == "zip":
            output_file += ".zip"
        else:
            output_file += ".tar"
        output_path = self.packages_dir / output_file
        
        try:
            package_path = package.finalize(output_path)
            metadata_path = self.packages_dir / f"{package_id}.meta.json"
            save_json(package.get_metadata(), metadata_path)
            del self.active_packages[package_id]
            package_dir = self.packages_dir / f"tmp_{package_id}"
            if package_dir.exists():
                shutil.rmtree(package_dir)
            self.logger.info(f"Finalized package {package_id} to {package_path}")
            return package_path
        except Exception as e:
            self.logger.error(f"Failed to finalize package {package_id}: {e}")
            # Clean up if finalization fails
            try:
                if output_path.exists():
                    output_path.unlink()
            except Exception as cleanup_err:
                self.logger.warning(f"Error cleaning up failed package file: {cleanup_err}")
            raise
    
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