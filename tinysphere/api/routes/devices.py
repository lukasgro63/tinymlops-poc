# api/routes/devices.py
from typing import List, Optional

from fastapi import APIRouter, Depends, HTTPException
from sqlalchemy.orm import Session

from tinysphere.api.dependencies.db import get_db
from tinysphere.api.models.device import (Device, DeviceRegistrationRequest,
                                          DeviceRegistrationResponse)
from tinysphere.api.services.device_service import DeviceService
from tinysphere.api.services.notification_service import NotificationService

router = APIRouter()

@router.get("/", response_model=List[Device])
def get_devices(skip: int = 0, limit: int = 100, db: Session = Depends(get_db)):
    try:
        from tinysphere.api.models.base import format_datetime_with_z
        
        devices = DeviceService.get_all_devices(db, skip=skip, limit=limit)
        # Ensure all datetime fields are properly formatted with Z suffix
        for device in devices:
            if device.registration_time:
                device.registration_time = format_datetime_with_z(device.registration_time)
            if device.last_sync_time:
                device.last_sync_time = format_datetime_with_z(device.last_sync_time)
        
        return devices
    except Exception as e:
        print(f"Error formatting device dates: {e}")
        return DeviceService.get_all_devices(db, skip=skip, limit=limit)

# Zuerst alle spezifischen Routen
@router.get("/platforms")
def get_device_platforms(db: Session = Depends(get_db)):
    """Get distribution of device platforms."""
    try:
        return DeviceService.get_device_platforms(db)
    except Exception as e:
        raise HTTPException(status_code=500, detail=f"Error fetching platform distribution: {str(e)}")

@router.get("/metrics")
def get_device_metrics(device_id: Optional[str] = None, db: Session = Depends(get_db)):
    """Get aggregated device metrics."""
    try:
        return DeviceService.get_device_metrics(db, device_id)
    except Exception as e:
        raise HTTPException(status_code=500, detail=f"Error fetching device metrics: {str(e)}")

@router.get("/trends")
def get_device_connectivity_trends(days: int = 7, db: Session = Depends(get_db)):
    """Get device connectivity trends over time."""
    try:
        return DeviceService.get_device_connectivity_trends(db, days)
    except Exception as e:
        raise HTTPException(status_code=500, detail=f"Error fetching connectivity trends: {str(e)}")

@router.get("/top")
def get_top_devices(limit: int = 5, db: Session = Depends(get_db)):
    """Get top devices by package count."""
    try:
        return DeviceService.get_top_devices(db, limit)
    except Exception as e:
        raise HTTPException(status_code=500, detail=f"Error fetching top devices: {str(e)}")

@router.get("/{device_id}", response_model=Device)
def get_device(device_id: str, db: Session = Depends(get_db)):
    db_device = DeviceService.get_device_by_id(db, device_id=device_id)
    if db_device is None:
        raise HTTPException(status_code=404, detail="Device not found")
        
    try:
        from tinysphere.api.models.base import format_datetime_with_z
        
        # Format datetime fields correctly
        if db_device.registration_time:
            db_device.registration_time = format_datetime_with_z(db_device.registration_time)
        if db_device.last_sync_time:
            db_device.last_sync_time = format_datetime_with_z(db_device.last_sync_time)
    except Exception as e:
        print(f"Error formatting device dates: {e}")
        
    return db_device

@router.post("/register", response_model=DeviceRegistrationResponse)
def register_device(registration: DeviceRegistrationRequest, db: Session = Depends(get_db)):
    try:
        # For Pydantic v2 compatibility
        if hasattr(registration, "model_dump"):
            device_data = registration.model_dump()
        else:
            device_data = registration.dict()
            
        device = DeviceService.register_device(db, device_data)
        
        # Create notification for device registration/connection
        NotificationService.notify_device_connected(
            db, 
            device_id=device.device_id,
            hostname=device.hostname
        )
        
        return DeviceRegistrationResponse(
            device_id=device.device_id,
            status="success",
            registered=True,
            message="Device registered successfully"
        )
    except Exception as e:
        import logging
        logger = logging.getLogger(__name__)
        logger.error(f"Registration error: {str(e)}")
        
        return DeviceRegistrationResponse(
            device_id=registration.device_id,
            status="error",
            registered=False,
            message=f"Registration failed: {str(e)}"
        )