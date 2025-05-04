# db/models.py
from datetime import datetime, timezone
from enum import Enum as PyEnum

from sqlalchemy import (JSON, Boolean, Column, DateTime, Enum, Float,
                        ForeignKey, Integer, String, Text)
from sqlalchemy.ext.declarative import declarative_base
from sqlalchemy.orm import relationship

from tinysphere.api.dependencies.db import Base


class NotificationType(str, PyEnum):
    """Typen von Benachrichtigungen"""
    INFO = "info"
    WARNING = "warning"
    ERROR = "error"
    SUCCESS = "success"


class Device(Base):
    __tablename__ = "devices"
    
    id = Column(Integer, primary_key=True, index=True)
    device_id = Column(String(255), unique=True, index=True, nullable=False)
    hostname = Column(String(255))
    ip_address = Column(String(100))
    platform = Column(String(255))
    python_version = Column(String(50))
    tinylcm_version = Column(String(50))
    registration_time = Column(DateTime, default=lambda: datetime.now(datetime.UTC))
    last_sync_time = Column(DateTime, nullable=True)
    is_active = Column(Boolean, default=True)
    device_info = Column(JSON, nullable=True)
    metrics = relationship("DeviceMetric", back_populates="device", cascade="all, delete-orphan")

class DeviceMetric(Base):
    __tablename__ = "device_metrics"
    
    id = Column(Integer, primary_key=True, index=True)
    device_id = Column(String(255), ForeignKey("devices.device_id"), nullable=False)
    timestamp = Column(DateTime, default=lambda: datetime.now(timezone.utc))
    metric_type = Column(String(50))  # "inference_time", "cpu_usage", "memory_usage"
    value = Column(Float)
    
    device = relationship("Device", back_populates="metrics")

class Package(Base):
    __tablename__ = "packages"
    
    id = Column(Integer, primary_key=True, index=True)
    package_id = Column(String(255), unique=True, index=True, nullable=False)
    device_id = Column(String(255), ForeignKey("devices.device_id"), nullable=False)
    package_type = Column(String(100), nullable=False)
    filename = Column(String(255), nullable=False)
    file_path = Column(String(512), nullable=False)
    file_hash = Column(String(255), nullable=True)
    file_size = Column(Integer, nullable=True)
    uploaded_at = Column(DateTime, default=datetime.utcnow)
    processed_at = Column(DateTime, nullable=True)
    is_processed = Column(Boolean, default=False)
    processing_status = Column(String(50), default="pending")
    processing_error = Column(Text, nullable=True)
    package_metadata = Column(JSON, nullable=True)


class Notification(Base):
    """Modell für Benachrichtigungen im System"""
    __tablename__ = "notifications"
    
    id = Column(Integer, primary_key=True, index=True)
    message = Column(String(512), nullable=False)
    notification_type = Column(Enum(NotificationType), default=NotificationType.INFO, nullable=False)
    source = Column(String(100), nullable=True)  # Quelle der Benachrichtigung (z.B. "device", "package", "system")
    source_id = Column(String(255), nullable=True)  # ID der Quelle (z.B. device_id, package_id)
    details = Column(JSON, nullable=True)  # Zusätzliche Details zur Benachrichtigung
    created_at = Column(DateTime, default=lambda: datetime.now(timezone.utc), nullable=False)
    read = Column(Boolean, default=False)  # Ist die Benachrichtigung gelesen worden?
    read_at = Column(DateTime, nullable=True)  # Wann wurde die Benachrichtigung gelesen?