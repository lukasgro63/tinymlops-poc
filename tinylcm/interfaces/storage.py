from abc import ABC, abstractmethod
from typing import Any, Dict, Generator, List, Optional, Union, TypeVar, Generic

T = TypeVar('T')

class StorageBackend(ABC):
    @abstractmethod
    def save(self, data: Any, path: str) -> str:
        pass
    
    @abstractmethod
    def load(self, path: str) -> Any:
        pass
    
    @abstractmethod
    def delete(self, path: str) -> bool:
        pass
    
    @abstractmethod
    def exists(self, path: str) -> bool:
        pass

class ModelStorageBackend(StorageBackend):
    @abstractmethod
    def get_metadata(self, model_id: str) -> Dict[str, Any]:
        pass
    
    @abstractmethod
    def list_models(self, **filters) -> List[Dict[str, Any]]:
        pass

class StreamingStorageBackend(StorageBackend):
    @abstractmethod
    def stream_read(self, path: str, chunk_size: int = 1024) -> Generator[Any, None, None]:
        pass
    
    @abstractmethod
    def stream_write(self, data_generator: Any, path: str) -> str:
        pass