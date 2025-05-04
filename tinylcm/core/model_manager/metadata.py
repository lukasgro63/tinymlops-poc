from abc import ABC, abstractmethod
from pathlib import Path
from typing import Any, Dict, List, Optional, Union, Callable

from tinylcm.utils.errors import ModelNotFoundError, StorageError
from tinylcm.utils.file_utils import load_json, save_json
from tinylcm.utils.logging import setup_logger

class ModelMetadataProvider(ABC):
    @abstractmethod
    def save_metadata(self, model_id: str, metadata: Dict[str, Any], metadata_dir: Union[str, Path]) -> None:
        pass

    @abstractmethod
    def load_metadata(self, model_id: str, metadata_dir: Union[str, Path]) -> Dict[str, Any]:
        pass

    @abstractmethod
    def list_metadata(self, metadata_dir: Union[str, Path], filter_func: Optional[Callable[[Dict[str, Any]], bool]] = None) -> List[Dict[str, Any]]:
        pass

    @abstractmethod
    def delete_metadata(self, model_id: str, metadata_dir: Union[str, Path]) -> bool:
        pass

class JSONFileMetadataProvider(ModelMetadataProvider):
    def __init__(self):
        self.logger = setup_logger(f"{__name__}.{self.__class__.__name__}")

    def save_metadata(self, model_id: str, metadata: Dict[str, Any], metadata_dir: Union[str, Path]) -> None:
        metadata_path = Path(metadata_dir) / f"{model_id}.json"
        try:
            save_json(metadata, metadata_path)
        except Exception as e:
            raise StorageError(f"Failed to save metadata: {str(e)}")

    def load_metadata(self, model_id: str, metadata_dir: Union[str, Path]) -> Dict[str, Any]:
        metadata_path = Path(metadata_dir) / f"{model_id}.json"
        if not metadata_path.exists():
            raise ModelNotFoundError(f"Metadata file not found: {metadata_path}")
        try:
            return load_json(metadata_path)
        except Exception as e:
            raise StorageError(f"Failed to load metadata: {str(e)}")

    def list_metadata(self, metadata_dir: Union[str, Path], filter_func: Optional[Callable[[Dict[str, Any]], bool]] = None) -> List[Dict[str, Any]]:
        metadata_list = []
        metadata_dir_path = Path(metadata_dir)
        if not metadata_dir_path.exists():
            return []
        for metadata_file in metadata_dir_path.glob("*.json"):
            try:
                metadata = load_json(metadata_file)
                if filter_func is None or filter_func(metadata):
                    metadata_list.append(metadata)
            except Exception as e:
                self.logger.error(f"Error loading metadata from {metadata_file}: {str(e)}")
        return metadata_list

    def delete_metadata(self, model_id: str, metadata_dir: Union[str, Path]) -> bool:
        metadata_path = Path(metadata_dir) / f"{model_id}.json"
        if not metadata_path.exists():
            return False
        try:
            metadata_path.unlink()
            return True
        except Exception as e:
            raise StorageError(f"Failed to delete metadata file: {str(e)}")