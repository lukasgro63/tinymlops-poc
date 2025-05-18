# api/models/device.py
from datetime import datetime
from typing import Dict, Optional, Any, List

from pydantic import BaseModel, Field


class DeviceBase(BaseModel):
    device_id: str
    hostname: Optional[str] = None
    ip_address: Optional[str] = None
    platform: Optional[str] = None
    platform_version: Optional[str] = None  # Added for OS version
    device_model: Optional[str] = None      # Added for device model
    python_version: Optional[str] = None
    tinylcm_version: Optional[str] = None
    device_info: Optional[Dict[str, Any]] = None
    latitude: Optional[float] = None
    longitude: Optional[float] = None
    geo_accuracy: Optional[float] = None


class DeviceCreate(DeviceBase):
    registration_time: Optional[datetime] = Field(default_factory=lambda: datetime.now())


class DeviceUpdate(BaseModel):
    hostname: Optional[str] = None
    ip_address: Optional[str] = None
    platform: Optional[str] = None
    platform_version: Optional[str] = None  # Added for OS version
    device_model: Optional[str] = None      # Added for device model
    python_version: Optional[str] = None
    tinylcm_version: Optional[str] = None
    device_info: Optional[Dict[str, Any]] = None
    is_active: Optional[bool] = None
    last_sync_time: Optional[datetime] = None
    latitude: Optional[float] = None
    longitude: Optional[float] = None
    geo_accuracy: Optional[float] = None
    
    class Config:
        from_attributes = True


class Device(DeviceBase):
    id: int
    registration_time: datetime
    last_sync_time: Optional[datetime] = None
    is_active: bool
    
    class Config:
        from_attributes = True


# Request models
class DeviceRegistrationRequest(BaseModel):
    device_id: str
    hostname: Optional[str] = None
    ip_address: Optional[str] = None
    registration_time: Optional[datetime] = None
    device_info: Optional[Dict[str, Any]] = None
    latitude: Optional[float] = None
    longitude: Optional[float] = None
    geo_accuracy: Optional[float] = None


# Response models
class DeviceRegistrationResponse(BaseModel):
    device_id: str
    status: str
    registered: bool
    message: str
    
    
class DeviceLocation(BaseModel):
    device_id: str
    name: str
    location_name: str  # Added for frontend compatibility
    latitude: float
    longitude: float
    accuracy: Optional[float] = None
    geo_accuracy: Optional[float] = None  # Added for API consistency
    is_active: bool = True
    last_update: Optional[str] = None  # Added for frontend compatibility

    class Config:
        # Allow validating the JSON response from the backend
        from_attributes = True
        # Allow extra fields that might be in the database
        extra = "allow"