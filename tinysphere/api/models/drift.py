# api/models/drift.py
from datetime import datetime
from typing import Dict, List, Optional, Any

from pydantic import BaseModel, Field

from tinysphere.db.models import DriftStatus, DriftType


class DriftEventBase(BaseModel):
    device_id: str
    drift_type: str  # Intentionally not an enum to allow flexible client input
    drift_score: float
    detector_name: str
    model_id: Optional[str] = None
    description: Optional[str] = None
    timestamp: datetime = Field(default_factory=datetime.utcnow)
    event_metadata: Optional[Dict[str, Any]] = Field(default=None, alias="metadata")
    
    def _normalize_drift_type(self) -> str:
        """
        Normalize drift_type to lowercase to match PostgreSQL enum values
        
        This function ensures the drift_type is in a consistent format.
        It particularly handles the knn_distance variations, which is the most common drift type
        detected in the system.
        """
        if not self.drift_type:
            return "unknown"
        
        # Convert to lowercase
        drift_type_lower = self.drift_type.lower()
        
        # Handle common variations of KNN distance detector names
        if drift_type_lower in ["knn_distance", "knndistance", "knn", "knn distance", "knn-distance"]:
            return "knn_distance"
            
        # Handle valid type values from db.models.DriftType
        if drift_type_lower in ["confidence", "distribution", "feature", "outlier", "custom", "unknown"]:
            return drift_type_lower
        
        # Return the normalized value
        return drift_type_lower
        
    def get_normalized_data(self) -> Dict[str, Any]:
        """Get model data with normalized drift_type"""
        if hasattr(self, "model_dump"):
            data = self.model_dump()  # Pydantic v2
        else:
            data = self.dict()        # Pydantic v1
            
        # Replace drift_type with normalized version
        data["drift_type"] = self._normalize_drift_type()
        
        return data


class DriftEventCreate(DriftEventBase):
    samples: Optional[List[Dict[str, Any]]] = None


class DriftEventResponse(DriftEventBase):
    event_id: str
    status: str
    received_at: datetime
    resolved_at: Optional[datetime] = None
    resolution_notes: Optional[str] = None
    sample_count: int
    validation_count: int
    metadata: Optional[Dict[str, Any]] = Field(default=None, alias="event_metadata")


class DriftEventUpdateStatus(BaseModel):
    status: str
    resolution_notes: Optional[str] = None


class DriftSampleBase(BaseModel):
    drift_event_id: str
    prediction: str
    confidence: float
    drift_score: float
    feature_path: Optional[str] = None
    raw_data_path: Optional[str] = None
    timestamp: datetime = Field(default_factory=datetime.utcnow)
    sample_metadata: Optional[Dict[str, Any]] = Field(default=None, alias="metadata")


class DriftSampleCreate(DriftSampleBase):
    pass


class DriftSampleResponse(DriftSampleBase):
    sample_id: str
    true_label: Optional[str] = None
    status: str = "pending"  # pending or validated


class DriftValidationBase(BaseModel):
    drift_sample_id: str
    is_valid_drift: bool
    true_label: Optional[str] = None
    validated_by: str = "human"
    validation_notes: Optional[str] = None


class DriftValidationCreate(DriftValidationBase):
    pass


class DriftValidationResponse(DriftValidationBase):
    validation_id: str
    drift_event_id: str
    validated_at: datetime
    is_acknowledged: bool
    acknowledged_at: Optional[datetime] = None
    notes: Optional[str] = Field(default=None, alias="validation_notes")
    validation_metadata: Optional[Dict[str, Any]] = Field(default=None, alias="metadata")


class DriftAcknowledgement(BaseModel):
    validation_ids: List[str]


class DriftStatistics(BaseModel):
    total_events: int
    total_open: int
    total_resolved: int
    by_type: Dict[str, int]
    by_status: Dict[str, int]
    by_day: List[Dict[str, Any]]
    recent_events: List[DriftEventResponse]