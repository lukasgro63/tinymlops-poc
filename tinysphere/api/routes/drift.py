# api/routes/drift.py
from datetime import datetime, timedelta
from typing import List, Optional, Dict, Any

from fastapi import APIRouter, Depends, HTTPException, Query
from sqlalchemy.orm import Session

from tinysphere.api.dependencies.db import get_db
from tinysphere.api.models.drift import (
    DriftEventCreate, DriftEventResponse, DriftSampleResponse,
    DriftValidationCreate, DriftValidationResponse, DriftEventUpdateStatus,
    DriftAcknowledgement, DriftStatistics
)
from tinysphere.api.services.drift_service import DriftService
from tinysphere.api.services.device_service import DeviceService
from tinysphere.db.models import DriftStatus, DriftType

router = APIRouter()

# Drift events endpoints
@router.get("/events", response_model=List[DriftEventResponse])
def get_drift_events(
    skip: int = 0,
    limit: int = 100,
    device_id: Optional[str] = None,
    status: Optional[str] = None,
    drift_type: Optional[str] = None,
    start_date: Optional[datetime] = None,
    end_date: Optional[datetime] = None,
    db: Session = Depends(get_db)
):
    """
    Get drift events with optional filtering.
    """
    events = DriftService.get_drift_events(
        db, skip, limit, device_id, status, drift_type, start_date, end_date
    )
    
    # Add computed fields
    result = []
    for event in events:
        event_dict = {
            "event_id": event.event_id,
            "device_id": event.device_id,
            "drift_type": event.drift_type.value,
            "drift_score": event.drift_score,
            "detector_name": event.detector_name,
            "model_id": event.model_id,
            "description": event.description,
            "status": event.status.value,
            "timestamp": event.timestamp,
            "received_at": event.received_at,
            "resolved_at": event.resolved_at,
            "resolution_notes": event.resolution_notes,
            "sample_count": len(event.samples),
            "validation_count": len(event.validations)
        }
        result.append(event_dict)
    
    return result

@router.get("/events/{event_id}", response_model=DriftEventResponse)
def get_drift_event(event_id: str, db: Session = Depends(get_db)):
    """
    Get a specific drift event by ID.
    """
    event = DriftService.get_drift_event_by_id(db, event_id)
    if not event:
        raise HTTPException(status_code=404, detail="Drift event not found")
    
    # Add computed fields
    return {
        "event_id": event.event_id,
        "device_id": event.device_id,
        "drift_type": event.drift_type.value,
        "drift_score": event.drift_score,
        "detector_name": event.detector_name,
        "model_id": event.model_id,
        "description": event.description,
        "status": event.status.value,
        "timestamp": event.timestamp,
        "received_at": event.received_at,
        "resolved_at": event.resolved_at,
        "resolution_notes": event.resolution_notes,
        "sample_count": len(event.samples),
        "validation_count": len(event.validations)
    }

@router.post("/events", response_model=DriftEventResponse)
def create_drift_event(drift_event: DriftEventCreate, db: Session = Depends(get_db)):
    """
    Create a new drift event.
    """
    # Check if device exists
    device = DeviceService.get_device_by_id(db, drift_event.device_id)
    if not device:
        raise HTTPException(status_code=404, detail="Device not found")
    
    # Convert Pydantic model to dict
    if hasattr(drift_event, "model_dump"):
        event_data = drift_event.model_dump()  # Pydantic v2
    else:
        event_data = drift_event.dict()        # Pydantic v1
    
    # Process drift event
    event = DriftService.process_drift_event(db, drift_event.device_id, event_data)
    
    # Return response
    return {
        "event_id": event.event_id,
        "device_id": event.device_id,
        "drift_type": event.drift_type.value,
        "drift_score": event.drift_score,
        "detector_name": event.detector_name,
        "model_id": event.model_id,
        "description": event.description,
        "status": event.status.value,
        "timestamp": event.timestamp,
        "received_at": event.received_at,
        "resolved_at": event.resolved_at,
        "resolution_notes": event.resolution_notes,
        "sample_count": len(event.samples),
        "validation_count": len(event.validations)
    }

@router.patch("/events/{event_id}/status", response_model=DriftEventResponse)
def update_drift_event_status(
    event_id: str, 
    status_update: DriftEventUpdateStatus, 
    db: Session = Depends(get_db)
):
    """
    Update the status of a drift event.
    """
    if hasattr(status_update, "model_dump"):
        update_data = status_update.model_dump()
    else:
        update_data = status_update.dict()
    
    updated_event = DriftService.update_drift_event_status(
        db, 
        event_id, 
        update_data["status"],
        update_data.get("resolution_notes")
    )
    
    if not updated_event:
        raise HTTPException(status_code=404, detail="Drift event not found")
    
    # Return response
    return {
        "event_id": updated_event.event_id,
        "device_id": updated_event.device_id,
        "drift_type": updated_event.drift_type.value,
        "drift_score": updated_event.drift_score,
        "detector_name": updated_event.detector_name,
        "model_id": updated_event.model_id,
        "description": updated_event.description,
        "status": updated_event.status.value,
        "timestamp": updated_event.timestamp,
        "received_at": updated_event.received_at,
        "resolved_at": updated_event.resolved_at,
        "resolution_notes": updated_event.resolution_notes,
        "sample_count": len(updated_event.samples),
        "validation_count": len(updated_event.validations)
    }

# Drift samples endpoints
@router.get("/events/{event_id}/samples", response_model=List[DriftSampleResponse])
def get_drift_samples(event_id: str, db: Session = Depends(get_db)):
    """
    Get all samples for a drift event.
    """
    event = DriftService.get_drift_event_by_id(db, event_id)
    if not event:
        raise HTTPException(status_code=404, detail="Drift event not found")
    
    result = []
    for sample in event.samples:
        # Find validation status for this sample
        status = "pending"
        true_label = None
        
        for validation in sample.validations:
            if validation.true_label is not None:
                true_label = validation.true_label
                status = "validated"
                break
        
        sample_dict = {
            "sample_id": sample.sample_id,
            "drift_event_id": sample.drift_event_id,
            "prediction": sample.prediction,
            "confidence": sample.confidence,
            "drift_score": sample.drift_score,
            "feature_path": sample.feature_path,
            "raw_data_path": sample.raw_data_path,
            "timestamp": sample.timestamp,
            "metadata": sample.metadata,
            "true_label": true_label,
            "status": status
        }
        result.append(sample_dict)
    
    return result

@router.get("/samples/{sample_id}", response_model=DriftSampleResponse)
def get_drift_sample(sample_id: str, db: Session = Depends(get_db)):
    """
    Get a specific drift sample by ID.
    """
    # Find sample
    sample = db.query(db.models.DriftSample).filter(db.models.DriftSample.sample_id == sample_id).first()
    if not sample:
        raise HTTPException(status_code=404, detail="Drift sample not found")
    
    # Find validation status for this sample
    status = "pending"
    true_label = None
    
    for validation in sample.validations:
        if validation.true_label is not None:
            true_label = validation.true_label
            status = "validated"
            break
    
    return {
        "sample_id": sample.sample_id,
        "drift_event_id": sample.drift_event_id,
        "prediction": sample.prediction,
        "confidence": sample.confidence,
        "drift_score": sample.drift_score,
        "feature_path": sample.feature_path,
        "raw_data_path": sample.raw_data_path,
        "timestamp": sample.timestamp,
        "metadata": sample.metadata,
        "true_label": true_label,
        "status": status
    }

# Validation endpoints
@router.post("/samples/{sample_id}/validate", response_model=DriftValidationResponse)
def validate_drift_sample(
    sample_id: str, 
    validation: DriftValidationCreate, 
    db: Session = Depends(get_db)
):
    """
    Validate a drift sample with human feedback.
    """
    if hasattr(validation, "model_dump"):
        validation_data = validation.model_dump()
    else:
        validation_data = validation.dict()
    
    result = DriftService.validate_drift_sample(db, sample_id, validation_data)
    if not result:
        raise HTTPException(status_code=404, detail="Drift sample not found")
    
    return {
        "validation_id": result.validation_id,
        "drift_event_id": result.drift_event_id,
        "drift_sample_id": result.drift_sample_id,
        "is_valid_drift": result.is_valid_drift,
        "true_label": result.true_label,
        "validated_by": result.validated_by,
        "validated_at": result.validated_at,
        "is_acknowledged": result.is_acknowledged,
        "acknowledged_at": result.acknowledged_at,
        "notes": result.validation_notes
    }

@router.get("/validations/pending", response_model=List[Dict[str, Any]])
def get_pending_validations(device_id: Optional[str] = None, db: Session = Depends(get_db)):
    """
    Get list of validations pending device acknowledgment.
    """
    return DriftService.get_pending_validations(db, device_id)

@router.post("/validations/acknowledge")
def acknowledge_validations(ack: DriftAcknowledgement, db: Session = Depends(get_db)):
    """
    Mark validations as acknowledged by device.
    """
    if hasattr(ack, "model_dump"):
        ack_data = ack.model_dump()
    else:
        ack_data = ack.dict()
    
    validation_ids = ack_data["validation_ids"]
    results = []
    
    for validation_id in validation_ids:
        success = DriftService.acknowledge_validation(db, validation_id)
        results.append({"validation_id": validation_id, "success": success})
    
    return {"results": results}

# Statistics endpoints
@router.get("/statistics", response_model=DriftStatistics)
def get_drift_statistics(
    device_id: Optional[str] = None,
    days: int = 30,
    db: Session = Depends(get_db)
):
    """
    Get statistics about drift events.
    """
    return DriftService.get_drift_statistics(db, device_id, days)

@router.get("/devices/{device_id}/metrics")
def get_device_drift_metrics(device_id: str, db: Session = Depends(get_db)):
    """
    Get drift metrics for a specific device.
    """
    device = DeviceService.get_device_by_id(db, device_id)
    if not device:
        raise HTTPException(status_code=404, detail="Device not found")
    
    return DriftService.get_device_drift_metrics(db, device_id)