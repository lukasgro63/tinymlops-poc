from dataclasses import dataclass, field
from enum import Enum
from typing import Any, List, Optional, Dict, Union, TYPE_CHECKING, ForwardRef
import time
import numpy as np
from uuid import uuid4

# Forward reference for FeatureSample to resolve circular dependency
if TYPE_CHECKING:
    from typing import TypeAlias
    FeatureSample: TypeAlias = "FeatureSample"
else:
    FeatureSample = ForwardRef("FeatureSample")


class QuarantineStatus(str, Enum):
    """Status of a sample in the quarantine buffer."""
    PENDING = "pending"      # Awaiting analysis
    PROCESSED = "processed"  # Processed by heuristic adapter
    SYNCED = "synced"        # Sent to server for validation
    VALIDATED = "validated"  # Received validation from server
    ADAPTED = "adapted"      # Used for adaptation
    REJECTED = "rejected"    # Rejected by validation


@dataclass
class QuarantinedSample:
    """A sample that has been quarantined due to potential drift."""
    sample: FeatureSample
    reason: str
    status: QuarantineStatus = QuarantineStatus.PENDING
    quarantine_time: float = field(default_factory=time.time)
    processed_time: Optional[float] = None
    
    def to_dict(self) -> Dict[str, Any]:
        """Convert to dictionary for serialization."""
        return {
            "sample": self.sample.to_dict() if self.sample else None,
            "reason": self.reason,
            "status": self.status.value if isinstance(self.status, QuarantineStatus) else self.status,
            "quarantine_time": self.quarantine_time,
            "processed_time": self.processed_time
        }
    
    @classmethod
    def from_dict(cls, data: Dict[str, Any]) -> 'QuarantinedSample':
        """Create from dictionary after deserialization."""
        sample_dict = data.get("sample")
        sample = FeatureSample.from_dict(sample_dict) if sample_dict else None
        status_value = data.get("status", QuarantineStatus.PENDING.value)
        status = QuarantineStatus(status_value) if isinstance(status_value, str) else status_value
        
        return cls(
            sample=sample,
            reason=data.get("reason", "Unknown"),
            status=status,
            quarantine_time=data.get("quarantine_time", time.time()),
            processed_time=data.get("processed_time")
        )


@dataclass
class AdaptationAction:
    """An action resulting from heuristic adaptation."""
    sample_id: str
    features: np.ndarray
    potential_label: Any
    original_label: Any
    reason: str
    confidence: float = 0.0
    timestamp: float = field(default_factory=time.time)
    
    def to_dict(self) -> Dict[str, Any]:
        """Convert to dictionary for serialization."""
        return {
            "sample_id": self.sample_id,
            "features": self.features.tolist() if isinstance(self.features, np.ndarray) else self.features,
            "potential_label": self.potential_label,
            "original_label": self.original_label,
            "reason": self.reason,
            "confidence": self.confidence,
            "timestamp": self.timestamp
        }
    
    @classmethod
    def from_dict(cls, data: Dict[str, Any]) -> 'AdaptationAction':
        """Create from dictionary after deserialization."""
        features = data.get("features")
        if features and not isinstance(features, np.ndarray):
            features = np.array(features)
            
        return cls(
            sample_id=data.get("sample_id", ""),
            features=features,
            potential_label=data.get("potential_label"),
            original_label=data.get("original_label"),
            reason=data.get("reason", "Unknown"),
            confidence=data.get("confidence", 0.0),
            timestamp=data.get("timestamp", time.time())
        )


@dataclass
class FeatureSample:
    """Represents a feature vector with associated metadata.
    
    This is the core data structure used for adaptive learning in TinyLCM.
    It encapsulates feature vectors extracted from models along with their
    metadata such as labels, predictions, and timestamps.
    """
    features: np.ndarray
    label: Optional[Any] = None
    prediction: Optional[Any] = None
    timestamp: Optional[float] = None
    sample_id: Optional[str] = None
    metadata: Dict[str, Any] = field(default_factory=dict)
    
    def __post_init__(self):
        """Initialize defaults for unspecified fields."""
        if self.timestamp is None:
            self.timestamp = time.time()
        
        if self.sample_id is None:
            self.sample_id = str(uuid4())
            
    def is_misclassified(self) -> bool:
        """Check if the prediction doesn't match the label."""
        return (self.label is not None and 
                self.prediction is not None and 
                self.label != self.prediction)
    
    def to_dict(self) -> Dict[str, Any]:
        """Convert the sample to a dictionary for serialization."""
        return {
            "sample_id": self.sample_id,
            "features": self.features.tolist() if isinstance(self.features, np.ndarray) else self.features,
            "label": self.label,
            "prediction": self.prediction,
            "timestamp": self.timestamp,
            "metadata": self.metadata
        }
    
    @classmethod
    def from_dict(cls, data: Dict[str, Any]) -> 'FeatureSample':
        """Create a FeatureSample from a dictionary."""
        features = data["features"]
        if not isinstance(features, np.ndarray):
            features = np.array(features)
            
        return cls(
            features=features,
            label=data.get("label"),
            prediction=data.get("prediction"),
            timestamp=data.get("timestamp"),
            sample_id=data.get("sample_id"),
            metadata=data.get("metadata", {})
        )


@dataclass
class AdaptationEvent:
    """Represents an adaptation event in the system.
    
    This is used to track when and why the adaptive model was updated,
    including the performance metrics and drift detection information.
    """
    event_type: str  # 'passive', 'active', 'hybrid'
    timestamp: float = field(default_factory=time.time)
    event_id: str = field(default_factory=lambda: str(uuid4()))
    samples_added: int = 0
    samples_removed: int = 0
    drift_detected: bool = False
    drift_point_index: Optional[int] = None
    drift_magnitude: Optional[float] = None
    performance_before: Dict[str, float] = field(default_factory=dict)
    performance_after: Dict[str, float] = field(default_factory=dict)
    metadata: Dict[str, Any] = field(default_factory=dict)
    
    def to_dict(self) -> Dict[str, Any]:
        """Convert the event to a dictionary for serialization."""
        return {
            "event_id": self.event_id,
            "event_type": self.event_type,
            "timestamp": self.timestamp,
            "samples_added": self.samples_added,
            "samples_removed": self.samples_removed,
            "drift_detected": self.drift_detected,
            "drift_point_index": self.drift_point_index,
            "drift_magnitude": self.drift_magnitude,
            "performance_before": self.performance_before,
            "performance_after": self.performance_after,
            "metadata": self.metadata
        }
    
    @classmethod
    def from_dict(cls, data: Dict[str, Any]) -> 'AdaptationEvent':
        """Create an AdaptationEvent from a dictionary."""
        return cls(
            event_id=data.get("event_id", str(uuid4())),
            event_type=data["event_type"],
            timestamp=data.get("timestamp", time.time()),
            samples_added=data.get("samples_added", 0),
            samples_removed=data.get("samples_removed", 0),
            drift_detected=data.get("drift_detected", False),
            drift_point_index=data.get("drift_point_index"),
            drift_magnitude=data.get("drift_magnitude"),
            performance_before=data.get("performance_before", {}),
            performance_after=data.get("performance_after", {}),
            metadata=data.get("metadata", {})
        )


@dataclass
class AdaptiveState:
    """Represents the state of an adaptive model.
    
    This encapsulates all the information needed to serialize and deserialize
    the state of an adaptive learning pipeline, including the classifier,
    feature extractor, and handler states.
    """
    classifier_state: Dict[str, Any] = field(default_factory=dict)
    handler_state: Dict[str, Any] = field(default_factory=dict)
    extractor_state: Dict[str, Any] = field(default_factory=dict)
    samples: List[Dict[str, Any]] = field(default_factory=list)
    creation_timestamp: float = field(default_factory=time.time)
    version: str = "1.0.0"
    metadata: Dict[str, Any] = field(default_factory=dict)
    
    def to_dict(self) -> Dict[str, Any]:
        """Convert the state to a dictionary for serialization."""
        return {
            "classifier_state": self.classifier_state,
            "handler_state": self.handler_state,
            "extractor_state": self.extractor_state,
            "samples": self.samples,
            "creation_timestamp": self.creation_timestamp,
            "version": self.version,
            "metadata": self.metadata
        }
    
    @classmethod
    def from_dict(cls, data: Dict[str, Any]) -> 'AdaptiveState':
        """Create an AdaptiveState from a dictionary."""
        return cls(
            classifier_state=data.get("classifier_state", {}),
            handler_state=data.get("handler_state", {}),
            extractor_state=data.get("extractor_state", {}),
            samples=data.get("samples", []),
            creation_timestamp=data.get("creation_timestamp", time.time()),
            version=data.get("version", "1.0.0"),
            metadata=data.get("metadata", {})
        )


# Resolve the forward reference
if not TYPE_CHECKING:
    FeatureSample.__forward_arg__ = None