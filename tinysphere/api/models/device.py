# api/models/device.py
from datetime import datetime
from typing import Dict, Optional, Any, List

from pydantic import BaseModel, Field


class DeviceBase(BaseModel):
    device_id: str
    hostname: Optional[str] = None
    ip_address: Optional[str] = None
    platform: Optional[str] = None
    python_version: Optional[str] = None
    tinylcm_version: Optional[str] = None
    device_info: Optional[Dict[str, Any]] = None


class DeviceCreate(DeviceBase):
    registration_time: Optional[datetime] = Field(default_factory=lambda: datetime.now())


class DeviceUpdate(BaseModel):
    hostname: Optional[str] = None
    ip_address: Optional[str] = None
    platform: Optional[str] = None
    python_version: Optional[str] = None
    tinylcm_version: Optional[str] = None
    device_info: Optional[Dict[str, Any]] = None
    is_active: Optional[bool] = None
    last_sync_time: Optional[datetime] = None
    
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


# Response models
class DeviceRegistrationResponse(BaseModel):
    device_id: str
    status: str
    registered: bool
    message: str