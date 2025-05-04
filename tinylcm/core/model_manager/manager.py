import os
import shutil
import time
import uuid
from pathlib import Path
from typing import Any, Dict, List, Optional, Union

from tinylcm.constants import DEFAULT_ACTIVE_MODEL_LINK, DEFAULT_MODELS_DIR
from tinylcm.utils.config import Config, get_config
from tinylcm.utils.file_utils import ensure_dir
from tinylcm.utils.logging import setup_logger
from tinylcm.utils.errors import ModelNotFoundError, ModelIntegrityError, StorageError
from tinylcm.utils.versioning import calculate_file_hash

from tinylcm.core.model_manager.storage import ModelStorageStrategy, FileSystemModelStorage
from tinylcm.core.model_manager.metadata import ModelMetadataProvider, JSONFileMetadataProvider

class ModelManager:
    def __init__(self, storage_dir: Optional[Union[str, Path]] = None, storage_strategy: Optional[ModelStorageStrategy] = None, metadata_provider: Optional[ModelMetadataProvider] = None, config: Optional[Config] = None):
        self.config = config or get_config()
        self.logger = setup_logger(f"{__name__}.{self.__class__.__name__}")
        self.storage_dir = Path(storage_dir or self.config.get("model_manager", "storage_dir", DEFAULT_MODELS_DIR))
        self.models_dir = ensure_dir(self.storage_dir / "models")
        self.metadata_dir = ensure_dir(self.storage_dir / "metadata")
        self.storage_strategy = storage_strategy or FileSystemModelStorage()
        self.metadata_provider = metadata_provider or JSONFileMetadataProvider()

    def save_model(self, model_path: Union[str, Path], model_format: str, version: Optional[str] = None, description: Optional[str] = None, tags: Optional[List[str]] = None, metrics: Optional[Dict[str, float]] = None, set_active: bool = False) -> str:
        model_id = str(uuid.uuid4())
        storage_meta = self.storage_strategy.save_model(model_path=model_path, model_id=model_id, target_dir=self.models_dir)
        metadata = {"model_id": model_id, "model_format": model_format, "version": version or f"v_{int(time.time())}", "description": description or "", "tags": tags or [], "metrics": metrics or {}, "timestamp": time.time(), "is_active": set_active, "md5_hash": storage_meta["md5_hash"], "filename": storage_meta["filename"]}
        self.metadata_provider.save_metadata(model_id=model_id, metadata=metadata, metadata_dir=self.metadata_dir)
        if set_active:
            self.set_active_model(model_id)
        return model_id

    def load_model(self, model_id: Optional[str] = None) -> str:
        if model_id is None:
            active_link = Path(self.storage_dir) / DEFAULT_ACTIVE_MODEL_LINK
            if not active_link.exists():
                raise ModelNotFoundError("No active model set")
            if active_link.is_symlink():
                target = os.readlink(active_link)
                if os.path.isabs(target):
                    model_dir = Path(target)
                else:
                    model_dir = (active_link.parent / target).resolve()
                model_id = model_dir.name
            else:
                try:
                    with open(active_link, "r", encoding='utf-8') as f:
                        model_id = f.read().strip()
                except Exception as e:
                    raise ModelNotFoundError(f"Failed to read active model reference: {str(e)}")
        model_dir = Path(self.models_dir) / model_id
        if not model_dir.exists():
            raise ModelNotFoundError(f"Model not found: {model_id}")
        model_files = list(model_dir.glob("*"))
        if not model_files:
            raise ModelNotFoundError(f"No model files found in directory: {model_dir}")
        return str(model_files[0])

    def get_model_metadata(self, model_id: Optional[str] = None) -> Dict[str, Any]:
        if model_id is None:
            active_link = Path(self.storage_dir) / DEFAULT_ACTIVE_MODEL_LINK
            if not active_link.exists() or not active_link.is_symlink():
                raise ModelNotFoundError("No active model set")
            model_id = os.path.basename(os.readlink(active_link))
        return self.metadata_provider.load_metadata(model_id=model_id, metadata_dir=self.metadata_dir)

    def list_models(self, tag: Optional[str] = None, model_format: Optional[str] = None, version: Optional[str] = None) -> List[Dict[str, Any]]:
        def filter_func(metadata: Dict[str, Any]) -> bool:
            if tag is not None and tag not in metadata.get("tags", []):
                return False
            if model_format is not None and metadata.get("model_format") != model_format:
                return False
            if version is not None and metadata.get("version") != version:
                return False
            return True
        return self.metadata_provider.list_metadata(metadata_dir=self.metadata_dir, filter_func=filter_func)

    def set_active_model(self, model_id: str) -> bool:
        model_dir = Path(self.models_dir) / model_id
        if not model_dir.exists():
            raise ModelNotFoundError(f"Model not found: {model_id}")
        all_models = self.list_models()
        for metadata in all_models:
            current_id = metadata["model_id"]
            metadata["is_active"] = (current_id == model_id)
            self.metadata_provider.save_metadata(model_id=current_id, metadata=metadata, metadata_dir=self.metadata_dir)
        active_link = Path(self.storage_dir) / DEFAULT_ACTIVE_MODEL_LINK
        try:
            if active_link.exists():
                if active_link.is_symlink():
                    os.unlink(active_link)
                elif active_link.is_file():
                    os.remove(active_link)
                elif active_link.is_dir():
                    shutil.rmtree(active_link)
        except Exception as e:
            self.logger.warning(f"Could not remove existing link/file at {active_link}: {e}")
        try:
            os.symlink(str(model_dir), str(active_link), target_is_directory=True)
        except (OSError, AttributeError) as e:
            self.logger.warning(f"Could not create symlink, using fallback mechanism: {e}")
            try:
                with open(active_link, "w", encoding='utf-8') as f:
                    f.write(model_id)
                return True
            except Exception as fallback_error:
                raise StorageError(f"Failed to create active model reference: {str(fallback_error)}")
        return True

    def delete_model(self, model_id: str, force: bool = False) -> bool:
        try:
            metadata = self.get_model_metadata(model_id)
            is_active = metadata.get("is_active", False)
            if is_active and not force:
                raise ValueError(f"Cannot delete active model (model_id={model_id}). Set force=True to override.")
            deleted_files = self.storage_strategy.delete_model(model_id=model_id, models_dir=self.models_dir)
            deleted_metadata = self.metadata_provider.delete_metadata(model_id=model_id, metadata_dir=self.metadata_dir)
            if is_active:
                active_link = Path(self.storage_dir) / DEFAULT_ACTIVE_MODEL_LINK
                if active_link.exists() and active_link.is_symlink():
                    active_link.unlink()
            return deleted_files and deleted_metadata
        except Exception as e:
            if "No active model set" in str(e):
                pass
            else:
                raise
        deleted_files = self.storage_strategy.delete_model(model_id=model_id, models_dir=self.models_dir)
        deleted_metadata = self.metadata_provider.delete_metadata(model_id=model_id, metadata_dir=self.metadata_dir)
        return deleted_files and deleted_metadata

    def add_tag(self, model_id: str, tag: str) -> bool:
        try:
            metadata = self.get_model_metadata(model_id)
            if "tags" not in metadata:
                metadata["tags"] = []
            if tag not in metadata["tags"]:
                metadata["tags"].append(tag)
                self.metadata_provider.save_metadata(model_id=model_id, metadata=metadata, metadata_dir=self.metadata_dir)
            return True
        except Exception as e:
            self.logger.error(f"Error adding tag to model {model_id}: {str(e)}")
            return False

    def remove_tag(self, model_id: str, tag: str) -> bool:
        try:
            metadata = self.get_model_metadata(model_id)
            if "tags" in metadata and tag in metadata["tags"]:
                metadata["tags"].remove(tag)
                self.metadata_provider.save_metadata(model_id=model_id, metadata=metadata, metadata_dir=self.metadata_dir)
            return True
        except Exception as e:
            self.logger.error(f"Error removing tag from model {model_id}: {str(e)}")
            return False

    def update_metrics(self, model_id: str, metrics: Dict[str, float]) -> bool:
        try:
            metadata = self.get_model_metadata(model_id)
            if "metrics" not in metadata:
                metadata["metrics"] = {}
            metadata["metrics"].update(metrics)
            self.metadata_provider.save_metadata(model_id=model_id, metadata=metadata, metadata_dir=self.metadata_dir)
            return True
        except Exception as e:
            self.logger.error(f"Error updating metrics for model {model_id}: {str(e)}")
            return False

    def verify_model_integrity(self, model_id: str) -> bool:
        try:
            metadata = self.get_model_metadata(model_id)
            stored_hash = metadata.get("md5_hash")
            if not stored_hash:
                return False
            model_path = self.load_model(model_id)
            current_hash = calculate_file_hash(model_path)
            match = current_hash == stored_hash
            if not match:
                self.logger.warning(f"Model integrity check failed for {model_id}: hash mismatch")
                self.logger.debug(f"Expected: {stored_hash}, Got: {current_hash}")
            return match
        except Exception as e:
            self.logger.error(f"Error verifying model integrity for {model_id}: {str(e)}")
            if isinstance(e, ModelNotFoundError):
                raise
            raise ModelIntegrityError(f"Failed to verify model integrity: {str(e)}")