# api/routes/prediction_images.py
from typing import List, Optional, Dict, Any
import logging
import fastapi
from fastapi import APIRouter, Depends, HTTPException, Query

from tinysphere.api.dependencies.db import get_db
from tinysphere.api.services.prediction_images_service import PredictionImagesService

logger = logging.getLogger(__name__)
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
    Generate a URL for accessing a specific image through the API.
    
    Args:
        image_key: Full S3 key of the image (device_id/prediction_type/date/filename)
    """
    url = prediction_images_service.get_image_url(image_key)
    if not url:
        raise HTTPException(status_code=404, detail="Image not found or error generating URL")
    
    # Return the API proxy URL
    return {"url": url}

@router.get("/image/{image_key:path}", response_class=fastapi.responses.StreamingResponse)
async def get_image(image_key: str, download: bool = False):
    """
    Stream an image directly from MinIO through the API.
    
    Args:
        image_key: Full S3 key of the image (device_id/prediction_type/date/filename)
        download: Set to true to download the file instead of viewing it
    """
    try:
        # Get the image from MinIO
        response = prediction_images_service.s3_client.get_object(
            Bucket=prediction_images_service.bucket_name,
            Key=image_key
        )
        
        # Get filename from the image key
        filename = image_key.split("/")[-1]
        
        # Determine content type based on file extension
        content_type = "image/jpeg"  # Default
        if image_key.lower().endswith(".png"):
            content_type = "image/png"
        elif image_key.lower().endswith(".gif"):
            content_type = "image/gif"
        
        # Set content disposition based on whether we're downloading or viewing
        content_disposition = f'attachment; filename="{filename}"' if download else f'inline; filename="{filename}"'
        
        # Stream the image back
        return fastapi.responses.StreamingResponse(
            response["Body"],
            media_type=content_type,
            headers={
                "Content-Disposition": content_disposition
            }
        )
    except Exception as e:
        logger.error(f"Error retrieving image {image_key}: {e}")
        raise HTTPException(status_code=404, detail="Image not found")