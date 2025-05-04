from abc import ABC, abstractmethod
from typing import Any, Dict, List, Optional, Tuple

# AnomalyDetector class has been removed as it's no longer needed.
# Anomaly detection is now handled by the adaptive components.

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
    """
    DEPRECATED: This interface is deprecated. Please use the new adaptive handlers instead.
    
    For drift detection, use one of:
    - tinylcm.core.handlers.ActiveHandler
    - tinylcm.core.handlers.HybridHandler 
    
    These handlers integrate drift detection with accuracy-based CUSUM monitoring.
    """
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
        
        
class AdaptiveComponent(ABC):
    """Interface for components that can participate in adaptive learning."""
    
    @abstractmethod
    def get_state(self) -> Dict[str, Any]:
        """Get the serializable state of the component."""
        pass
    
    @abstractmethod
    def set_state(self, state: Dict[str, Any]) -> None:
        """Restore the component state from the provided state dictionary."""
        pass