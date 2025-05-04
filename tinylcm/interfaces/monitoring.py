from abc import ABC, abstractmethod
from typing import Any, Dict, List, Optional, Tuple

class AnomalyDetector(ABC):
    @abstractmethod
    def check_for_anomalies(self, record: Dict[str, Any], context: Dict[str, Any]) -> Tuple[bool, List[str]]:
        pass

class MetricsProvider(ABC):
    @abstractmethod
    def get_metrics(self) -> Dict[str, Any]:
        pass

class MetricsConsumer(ABC):
    @abstractmethod
    def consume_metrics(self, metrics: Dict[str, Any]) -> None:
        pass

class DataStreamProcessor(ABC):
    @abstractmethod
    def process_record(self, record: Dict[str, Any]) -> Dict[str, Any]:
        pass
    
    @abstractmethod
    def process_batch(self, records: List[Dict[str, Any]]) -> List[Dict[str, Any]]:
        pass

class DriftDetector(ABC):
    @abstractmethod
    def create_reference_distribution(self, data: Dict[str, Any]) -> None:
        pass
    
    @abstractmethod
    def update(self, record: Dict[str, Any]) -> bool:
        pass
    
    @abstractmethod
    def check_for_drift(self) -> Dict[str, Any]:
        pass
    
    @abstractmethod
    def reset(self) -> None:
        pass