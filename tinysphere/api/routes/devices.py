# api/routes/devices.py
from typing import List, Optional

from fastapi import APIRouter, Depends, HTTPException
from sqlalchemy.orm import Session

from tinysphere.api.dependencies.db import get_db
from tinysphere.api.models.device import (Device, DeviceRegistrationRequest,
                                          DeviceRegistrationResponse, DeviceLocation)
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
        
@router.get("/debug-locations")
def get_debug_device_locations(db: Session = Depends(get_db)):
    """Debug endpoint to get raw location data from database."""
    try:
        from sqlalchemy import text
        
        # Direct SQL query to get location data
        results = db.execute(text("SELECT device_id, hostname, latitude, longitude, geo_accuracy FROM devices WHERE latitude IS NOT NULL")).fetchall()
        
        # Convert to list of dictionaries
        locations = []
        for row in results:
            locations.append({
                "device_id": row[0],
                "hostname": row[1],
                "latitude": float(row[2]),
                "longitude": float(row[3]),
                "geo_accuracy": float(row[4]) if row[4] is not None else None
            })
            
        return {
            "count": len(locations),
            "locations": locations
        }
    except Exception as e:
        import logging
        logger = logging.getLogger(__name__)
        logger.error(f"Error fetching debug locations: {str(e)}")
        import traceback
        logger.error(traceback.format_exc())
        traceback.print_exc()
        raise HTTPException(status_code=500, detail=f"Error fetching debug locations: {str(e)}")

# Locations endpoint muss VOR dem /{device_id} Endpoint definiert werden!
@router.get("/locations")
def get_all_device_locations(limit: int = 100, offset: int = 0, db: Session = Depends(get_db)):
    """Get location data for all devices with valid coordinates."""
    try:
        from sqlalchemy import text
        
        # EXAKT dieselbe Implementierung wie der Debug-Endpunkt
        results = db.execute(text("SELECT device_id, hostname, latitude, longitude, geo_accuracy FROM devices WHERE latitude IS NOT NULL")).fetchall()
        
        # Convert to list of dictionaries
        locations = []
        for row in results:
            locations.append({
                "device_id": row[0],
                "hostname": row[1],
                "latitude": float(row[2]),
                "longitude": float(row[3]),
                "geo_accuracy": float(row[4]) if row[4] is not None else None
            })
            
        # Einziger Unterschied: "total" statt "count" für Frontend-Kompatibilität
        return {
            "total": len(locations),
            "locations": locations
        }
    except Exception as e:
        import traceback
        traceback.print_exc()
        raise HTTPException(status_code=500, detail=f"Error fetching device locations: {str(e)}")

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

@router.get("/{device_id}/location", status_code=200)
def get_device_location(device_id: str, db: Session = Depends(get_db)):
    """Get location data for a specific device.
    
    Simplified implementation similar to the working debug endpoint.
    """
    try:
        # Debug info for device location request
        import logging
        logger = logging.getLogger(__name__)
        logger.info(f"Device location request for device: {device_id} - SIMPLE IMPLEMENTATION")
        
        # Use direct SQL to get device location
        from sqlalchemy import text
        from datetime import datetime, timezone
        
        # Query directly with SQL to bypass any ORM issues - using the same approach as the debug endpoint
        sql_result = db.execute(text(
            f"SELECT device_id, hostname, latitude, longitude, geo_accuracy, is_active, last_sync_time "
            f"FROM devices WHERE device_id = '{device_id}' AND latitude IS NOT NULL"
        )).fetchone()
        
        if not sql_result:
            logger.warning(f"No location data found for device {device_id}")
            raise HTTPException(
                status_code=404, 
                detail=f"Device not found or location data not available for device: {device_id}"
            )
        
        # Parse SQL result
        result_device_id = sql_result[0]
        hostname = sql_result[1]
        latitude = sql_result[2]
        longitude = sql_result[3]
        geo_accuracy = sql_result[4]
        is_active = sql_result[5]
        last_sync_time = sql_result[6]
        
        logger.info(f"Found location for device {device_id}: lat={latitude}, lon={longitude}, accuracy={geo_accuracy}")
        
        # Get current UTC time for last_update field
        current_time = datetime.now(timezone.utc).isoformat()
        
        # Create the location object
        location = {
            "device_id": result_device_id,
            "name": hostname or result_device_id,
            "location_name": hostname or result_device_id,  # Add location_name for frontend compatibility
            "latitude": float(latitude),
            "longitude": float(longitude),
            "geo_accuracy": float(geo_accuracy) if geo_accuracy is not None else None,
            "accuracy": float(geo_accuracy) if geo_accuracy is not None else None,
            "is_active": bool(is_active) if is_active is not None else True,
            "last_update": last_sync_time.isoformat() if last_sync_time else current_time
        }
        
        # Log the location data that's being returned
        logger.info(f"Returning location data for device {device_id}: {location}")
        
        return location
    except HTTPException:
        raise
    except Exception as e:
        import logging, traceback
        logger = logging.getLogger(__name__)
        logger.error(f"Error fetching device location: {str(e)}")
        logger.error(traceback.format_exc())
        traceback.print_exc()
        raise HTTPException(status_code=500, detail=f"Error fetching device location: {str(e)}")