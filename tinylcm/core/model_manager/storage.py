import shutil
from abc import ABC, abstractmethod
from pathlib import Path
from typing import Any, Dict, Union

from tinylcm.utils.errors import ModelNotFoundError, StorageError
from tinylcm.utils.file_utils import ensure_dir
from tinylcm.utils.versioning import calculate_file_hash

class ModelStorageStrategy(ABC):
    @abstractmethod
    def save_model(self, model_path: Union[str, Path], model_id: str, target_dir: Union[str, Path]) -> Dict[str, Any]:
        pass

    @abstractmethod
    def load_model(self, model_id: str, models_dir: Union[str, Path]) -> str:
        pass

    @abstractmethod
    def delete_model(self, model_id: str, models_dir: Union[str, Path]) -> bool:
        pass

class FileSystemModelStorage(ModelStorageStrategy):
    def save_model(self, model_path: Union[str, Path], model_id: str, target_dir: Union[str, Path]) -> Dict[str, Any]:
        source_path = Path(model_path)
        model_dir = Path(target_dir) / model_id
        ensure_dir(model_dir)
        target_path = model_dir / source_path.name
        try:
            shutil.copy2(source_path, target_path)
        except Exception as e:
            raise StorageError(f"Failed to copy model file: {str(e)}")
        md5_hash = calculate_file_hash(target_path)
        return {"filename": source_path.name, "path": str(target_path), "md5_hash": md5_hash}

    def load_model(self, model_id: str, models_dir: Union[str, Path]) -> str:
        model_dir = Path(models_dir) / model_id
        if not model_dir.exists():
            raise ModelNotFoundError(f"Model directory not found: {model_dir}")
        model_files = [f for f in model_dir.iterdir() if f.is_file()]
        if not model_files:
            raise ModelNotFoundError(f"No model file found in directory: {model_dir}")
        return str(model_files[0])

    def delete_model(self, model_id: str, models_dir: Union[str, Path]) -> bool:
        model_dir = Path(models_dir) / model_id
        if not model_dir.exists():
            return False
        try:
            shutil.rmtree(model_dir)
            return True
        except Exception as e:
            raise StorageError(f"Failed to delete model directory: {str(e)}")