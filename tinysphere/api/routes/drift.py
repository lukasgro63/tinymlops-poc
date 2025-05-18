# api/routes/drift.py
import json
import logging
from datetime import datetime, timedelta
from typing import List, Optional, Dict, Any

logger = logging.getLogger(__name__)

from fastapi import APIRouter, Depends, HTTPException, Query, File, UploadFile, Form
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
    sort_order: Optional[str] = "desc",  # Neuer Parameter für Sortierreihenfolge
    db: Session = Depends(get_db)
):
    """
    Get drift events with optional filtering.
    """
    events = DriftService.get_drift_events(
        db, skip, limit, device_id, status, drift_type, start_date, end_date, sort_order
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
            "validation_count": len(event.validations),
            "metadata": event.event_metadata
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
        "validation_count": len(event.validations),
        "metadata": event.event_metadata
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
    
    # Get normalized data with consistent drift_type cases
    event_data = drift_event.get_normalized_data()
    
    # Check for KNN distance detector
    detector_name = event_data.get("detector_name", "").lower()
    description = event_data.get("description", "").lower()
    
    # Special handling for KNN distance detectors
    if ("knn" in detector_name or "distance" in detector_name or 
        "neighbor_distance" in description):
        
        # First check which drift types are actually valid in the database
        from sqlalchemy import text
        db_conn = db.connection()
        try:
            # Import needed to ensure text is available
            from sqlalchemy.sql import text
            
            # Query the database for valid drift type enum values using a more reliable approach
            valid_types = [row[0] for row in db.execute(
                text("SELECT enumlabel FROM pg_enum JOIN pg_type ON pg_enum.enumtypid = pg_type.oid WHERE pg_type.typname = 'drifttype'")
            ).fetchall()]
            
            logger.info(f"Available drift types in database: {valid_types}")
            
            # Use knn_distance if it's a valid enum value (case-sensitive check)
            if "KNN_DISTANCE" in valid_types:
                event_data["drift_type"] = "KNN_DISTANCE"  # Use string instead of enum
                logger.info("Using 'KNN_DISTANCE' enum value")
            elif "knn_distance" in valid_types:
                event_data["drift_type"] = "knn_distance"  # Use string instead of enum
                logger.info("Using 'knn_distance' enum value")
            # Otherwise use a safe fallback value
            elif "DISTRIBUTION" in valid_types:
                event_data["drift_type"] = "DISTRIBUTION"  # Use string instead of enum
                logger.info("Using 'DISTRIBUTION' as safe fallback for KNN distance drift")
            elif "distribution" in valid_types:
                event_data["drift_type"] = "distribution"  # Use string instead of enum
                logger.info("Using 'distribution' as safe fallback for KNN distance drift")
            else:
                # Default to CONFIDENCE if nothing else works
                if "CONFIDENCE" in valid_types:
                    event_data["drift_type"] = "CONFIDENCE"  # Use string instead of enum
                    logger.info("Using 'CONFIDENCE' as safe fallback for KNN distance drift")
                else:
                    event_data["drift_type"] = valid_types[0]  # Use first available enum value
                    logger.info(f"Using '{valid_types[0]}' as safe fallback for KNN distance drift")
                
        except Exception as e:
            # On error, fall back to CONFIDENCE
            logger.error(f"Error checking valid drift type values: {e}")
            event_data["drift_type"] = DriftType.CONFIDENCE
            logger.info("Error checking enum values, using DriftType.CONFIDENCE as fallback")
            
        # Store the original drift type in metadata
        if "metadata" not in event_data or event_data["metadata"] is None:
            event_data["metadata"] = {}
        event_data["metadata"]["original_drift_type"] = "knn_distance"
        event_data["metadata"]["drift_type_display"] = "KNN Distance"
    
    # Log the normalized drift type
    if isinstance(event_data["drift_type"], DriftType):
        logger.info(f"Creating drift event with drift_type enum: {event_data['drift_type'].name}")
    else:
        logger.info(f"Creating drift event with drift_type string: '{event_data['drift_type']}'")
    
    # Process drift event
    event = DriftService.process_drift_event(db, drift_event.device_id, event_data)
    
    # Determine the drift type to return to the client
    drift_type_to_use = event.drift_type.value  # Default to the database value
    
    # If we have metadata with the original drift type, use that instead
    if event.event_metadata and isinstance(event.event_metadata, dict):
        original_type = event.event_metadata.get("original_drift_type")
        if original_type:
            drift_type_to_use = original_type
            logger.info(f"Using original drift type from metadata: {drift_type_to_use}")
    
    # Return response with the correct drift type
    return {
        "event_id": event.event_id,
        "device_id": event.device_id,
        "drift_type": drift_type_to_use,  # Use the determined drift type
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
        "validation_count": len(event.validations),
        "metadata": event.event_metadata
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
        "validation_count": len(updated_event.validations),
        "metadata": updated_event.event_metadata
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

        # Handle potentially null values to satisfy Pydantic validation
        prediction = sample.prediction or ""  # Provide empty string if prediction is None
        confidence = sample.confidence or 0.0  # Provide 0.0 if confidence is None
        drift_score = sample.drift_score or 0.0  # Provide 0.0 if drift_score is None

        sample_dict = {
            "sample_id": sample.sample_id,
            "drift_event_id": sample.drift_event_id,
            "prediction": prediction,
            "confidence": confidence,
            "drift_score": drift_score,
            "feature_path": sample.feature_path,
            "raw_data_path": sample.raw_data_path,
            "timestamp": sample.timestamp or datetime.now(),  # Provide current time if timestamp is None
            "metadata": sample.sample_metadata or {},  # Provide empty dict if metadata is None
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

    # Handle potentially null values to satisfy Pydantic validation
    prediction = sample.prediction or ""  # Provide empty string if prediction is None
    confidence = sample.confidence or 0.0  # Provide 0.0 if confidence is None
    drift_score = sample.drift_score or 0.0  # Provide 0.0 if drift_score is None

    return {
        "sample_id": sample.sample_id,
        "drift_event_id": sample.drift_event_id,
        "prediction": prediction,
        "confidence": confidence,
        "drift_score": drift_score,
        "feature_path": sample.feature_path,
        "raw_data_path": sample.raw_data_path,
        "timestamp": sample.timestamp or datetime.now(),
        "metadata": sample.sample_metadata or {},  # Corrected to use sample_metadata instead of metadata
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
    drift_type: Optional[str] = None,  # Allow filtering statistics by drift type
    db: Session = Depends(get_db)
):
    """
    Get statistics about drift events.
    
    Args:
        device_id: Optional filter by device
        days: Number of days to include (default 30)
        drift_type: Optional filter by drift type (will be converted to lowercase)
    """
    # Log the request parameters for debugging
    logger.info(f"Getting drift statistics with device_id={device_id}, days={days}, drift_type={drift_type}")
    
    # Add drift_type parameter
    return DriftService.get_drift_statistics(db, device_id, days, drift_type=drift_type)

@router.post("/repair", status_code=200)
def repair_drift_events(db: Session = Depends(get_db)):
    """
    Repair drift events with unknown type that can be inferred from descriptions.
    
    This is an admin operation that searches for drift events with UNKNOWN type
    and attempts to assign the correct type based on the event description or detector name.
    """
    logger.info("Starting drift events repair process")
    results = DriftService.repair_drift_types(db)
    logger.info(f"Repair process completed: {results['repaired_count']} events repaired")
    return results

@router.get("/diagnose", status_code=200)
def diagnose_drift_enum_problems(db: Session = Depends(get_db)):
    """
    Diagnose problems with drift enum values in the database.
    
    This is an admin operation that checks the database for valid enum values and
    compares them to the expected values. It can also try to add missing values to
    the enum type.
    
    Returns:
        Dictionary with diagnostic information
    """
    logger.info("Starting drift enum diagnosis")
    results = DriftService.diagnose_drift_enum_problems(db)
    logger.info(f"Diagnosis completed: {len(results.get('problems', []))} problems found")
    return results

@router.get("/devices/{device_id}/metrics")
def get_device_drift_metrics(device_id: str, db: Session = Depends(get_db)):
    """
    Get drift metrics for a specific device.
    """
    device = DeviceService.get_device_by_id(db, device_id)
    if not device:
        raise HTTPException(status_code=404, detail="Device not found")

    return DriftService.get_device_drift_metrics(db, device_id)

@router.post("/events/package")
async def create_drift_event_with_data(
    device_id: str = Form(...),
    detector_name: str = Form(...),
    drift_score: float = Form(None),
    description: str = Form(None),
    drift_type: str = Form("UNKNOWN"),
    feature_file: Optional[UploadFile] = File(None),
    image_file: Optional[UploadFile] = File(None),
    metadata: Optional[str] = Form(None),
    db: Session = Depends(get_db)
):
    """
    Create a drift event with optional feature and image data.

    Args:
        device_id: ID of the device reporting drift
        detector_name: Name of the detector that triggered the event
        drift_score: Score/magnitude of the drift
        description: Description of the drift event
        drift_type: Type of drift (CONFIDENCE, FEATURE, CONCEPT, etc.)
        feature_file: Optional file containing feature data
        image_file: Optional image file associated with the drift
        metadata: Optional JSON string containing additional metadata

    Returns:
        Created drift event
    """
    # Check if device exists or register it
    device = DeviceService.get_device_by_id(db, device_id)
    if not device:
        # Auto-register the device
        DeviceService.register_device(db, {"device_id": device_id})

    # Parse metadata if provided
    event_metadata = {}
    if metadata:
        try:
            event_metadata = json.loads(metadata)
        except json.JSONDecodeError:
            raise HTTPException(status_code=400, detail="Invalid metadata JSON")

    # Create a base event object for normalization
    base_event = DriftEventBase(
        device_id=device_id,
        detector_name=detector_name,
        drift_score=drift_score or 0.0,
        description=description,
        drift_type=drift_type,
        event_metadata=event_metadata,
        timestamp=datetime.now()
    )
    
    # Get normalized data (especially drift_type in correct case)
    event_data = base_event.get_normalized_data()
    
    # Log the normalized drift type
    logger.info(f"Creating drift event with normalized drift_type: '{event_data['drift_type']}' (original: '{drift_type}')")

    # Process the drift event
    event = DriftService.process_drift_event(db, device_id, event_data)

    # Handle feature file if provided
    if feature_file:
        try:
            # Save feature file
            feature_data = await feature_file.read()
            feature_path = f"drift_features/{event.event_id}_{feature_file.filename}"

            # Process as sample
            sample_data = {
                "feature_path": feature_path,
                "drift_score": drift_score,
                "metadata": {
                    "filename": feature_file.filename
                }
            }

            DriftService.add_drift_sample(db, event.event_id, sample_data)
        except Exception as e:
            logger.error(f"Error processing feature file: {e}")

    # Handle image file if provided
    if image_file:
        try:
            # Save image file with organized structure
            device_folder = device_id
            # Use the actual drift type from the event to ensure consistency
            drift_type_folder = event.drift_type.value if event.drift_type else "unknown"
            date_folder = datetime.now().strftime("%Y%m%d")
            filename = f"event_{event.event_id}_{image_file.filename}"
            logger.info(f"Storing drift image in folder: {drift_type_folder}")

            # Create structured path for the image (device_id/drift_type/date/filename)
            image_path = f"{device_folder}/{drift_type_folder}/{date_folder}/{filename}"

            # Full path including bucket name
            full_path = f"drift/{image_path}"

            # Read image data
            image_data = await image_file.read()

            # Update the event with image info
            event.event_metadata = event.event_metadata or {}
            event.event_metadata["image_path"] = full_path

            # Create sample data with image info
            sample_data = {
                "drift_score": drift_score,
                "raw_data_path": image_path,  # Store organized path
                "metadata": {
                    "filename": image_file.filename,
                    "drift_type": drift_type,
                    "event_id": event.event_id
                }
            }

            # Add as drift sample to maintain association with the event
            DriftService.add_drift_sample(db, event.event_id, sample_data)

            db.commit()
            logger.info(f"Processed drift image: {full_path}")
        except Exception as e:
            logger.error(f"Error processing image file: {e}")

    # Return the created event
    return {
        "event_id": event.event_id,
        "device_id": event.device_id,
        "drift_type": event.drift_type.value,
        "drift_score": event.drift_score,
        "detector_name": event.detector_name,
        "description": event.description,
        "status": event.status.value,
        "timestamp": event.timestamp,
        "received_at": event.received_at,
        "sample_count": len(event.samples),
        "validation_count": len(event.validations),
        "metadata": event.event_metadata
    }