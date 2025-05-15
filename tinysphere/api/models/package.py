# api/models/package.py
from datetime import datetime
from typing import Dict, Optional, Any

from pydantic import BaseModel


class PackageBase(BaseModel):
    package_id: str
    device_id: str
    package_type: str
    filename: str
    file_path: str
    file_hash: Optional[str] = None
    file_size: Optional[int] = None
    package_metadata: Optional[Dict[str, Any]] = None


class PackageCreate(PackageBase):
    uploaded_at: Optional[datetime] = None
    is_processed: bool = False
    processing_status: str = "pending"
    processing_error: Optional[str] = None


class PackageUpdate(BaseModel):
    file_hash: Optional[str] = None
    file_size: Optional[int] = None
    processed_at: Optional[datetime] = None
    is_processed: Optional[bool] = None
    processing_status: Optional[str] = None
    processing_error: Optional[str] = None
    package_metadata: Optional[Dict[str, Any]] = None
    
    class Config:
        from_attributes = True


class Package(PackageBase):
    id: int
    uploaded_at: datetime
    processed_at: Optional[datetime] = None
    is_processed: bool
    processing_status: str
    processing_error: Optional[str] = None
    
    class Config:
        from_attributes = True


class PackageResponse(Package):
    pass


class PackageUploadResponse(BaseModel):
    package_id: str
    status: str
    message: str
    uploaded: bool