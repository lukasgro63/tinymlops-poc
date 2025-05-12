# api/routes/prediction_images.py
from typing import List, Optional, Dict, Any
from fastapi import APIRouter, Depends, HTTPException, Query

from tinysphere.api.dependencies.db import get_db
from tinysphere.api.services.prediction_images_service import PredictionImagesService

router = APIRouter()
prediction_images_service = PredictionImagesService()

@router.get("/devices", response_model=List[str])
def list_device_ids():
    """List all device IDs that have prediction images."""
    return prediction_images_service.list_device_ids()

@router.get("/devices/{device_id}/types", response_model=List[str])
def list_prediction_types(device_id: str):
    """List all prediction types for a specific device."""
    return prediction_images_service.list_prediction_types(device_id)

@router.get("/devices/{device_id}/types/{prediction_type}/dates", response_model=List[str])
def list_dates(device_id: str, prediction_type: str):
    """List all dates with prediction images for a device and prediction type."""
    return prediction_images_service.list_dates(device_id, prediction_type)

@router.get("/list", response_model=Dict[str, Any])
def list_images(
    device_id: Optional[str] = None,
    prediction_type: Optional[str] = None,
    date: Optional[str] = None,
    limit: int = Query(100, ge=1, le=1000),
    offset: int = Query(0, ge=0)
):
    """
    List prediction images with optional filtering.
    
    Args:
        device_id: Filter by device ID
        prediction_type: Filter by prediction type
        date: Filter by date (YYYYMMDD format)
        limit: Maximum number of images to return
        offset: Number of images to skip
    """
    return prediction_images_service.list_images(
        device_id=device_id,
        prediction_type=prediction_type,
        date=date,
        limit=limit,
        offset=offset
    )

@router.get("/url/{image_key:path}", response_model=Dict[str, str])
def get_image_url(image_key: str):
    """
    Generate a pre-signed URL for accessing a specific image.
    
    Args:
        image_key: Full S3 key of the image (device_id/prediction_type/date/filename)
    """
    url = prediction_images_service.get_image_url(image_key)
    if not url:
        raise HTTPException(status_code=404, detail="Image not found or error generating URL")
    
    return {"url": url}