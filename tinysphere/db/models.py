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


class DriftStatus(str, PyEnum):
    """Status of a drift event"""
    PENDING = "pending"
    VALIDATED = "validated"
    REJECTED = "rejected"
    RESOLVED = "resolved"
    IGNORED = "ignored"


class DriftType(str, PyEnum):
    """Types of drift detection"""
    CONFIDENCE = "confidence"
    DISTRIBUTION = "distribution"
    FEATURE = "feature"
    OUTLIER = "outlier"
    CUSTOM = "custom"
    KNN_DISTANCE = "knn_distance"
    UNKNOWN = "unknown"
    
    def __str__(self) -> str:
        """Ensure the string representation is always the value, not the name"""
        return self.value
        
    def __repr__(self) -> str:
        """Ensure proper representation for debugging"""
        return f"{self.__class__.__name__}.{self.name} [value='{self.value}']"
    
    @classmethod
    def match_value(cls, value_str: str) -> 'DriftType':
        """Match a string to an enum value, case-insensitive
        
        Args:
            value_str: String to match against enum values
            
        Returns:
            The matching enum, or UNKNOWN if no match found
        """
        if not value_str:
            return cls.UNKNOWN
            
        value_lower = value_str.lower()
        
        # Direct match with lowercase values
        for enum_value in cls:
            if enum_value.value.lower() == value_lower:
                return enum_value
                
        # Special handling for knn variations
        if "knn" in value_lower or "distance" in value_lower:
            return cls.KNN_DISTANCE
            
        # Default
        return cls.UNKNOWN


class Device(Base):
    __tablename__ = "devices"
    
    id = Column(Integer, primary_key=True, index=True)
    device_id = Column(String(255), unique=True, index=True, nullable=False)
    hostname = Column(String(255))
    ip_address = Column(String(100))
    platform = Column(String(255))
    # Added for OS version and device model
    platform_version = Column(String(255), nullable=True)   # Added for OS version (e.g., "bookworm") 
    device_model = Column(String(255), nullable=True)       # Added for device model (e.g., "Pi Zero")
    python_version = Column(String(50))
    tinylcm_version = Column(String(50))
    registration_time = Column(DateTime, default=lambda: datetime.now(timezone.utc))
    last_sync_time = Column(DateTime, nullable=True)
    is_active = Column(Boolean, default=True)
    device_info = Column(JSON, nullable=True)
    # Geolocation fields
    latitude = Column(Float, nullable=True)
    longitude = Column(Float, nullable=True)
    geo_accuracy = Column(Float, nullable=True)
    metrics = relationship("DeviceMetric", back_populates="device", cascade="all, delete-orphan")
    drift_events = relationship("DriftEvent", back_populates="device", cascade="all, delete-orphan")


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
    uploaded_at = Column(DateTime, default=lambda: datetime.now(timezone.utc))
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


class DriftEvent(Base):
    """Model for tracking drift detection events from edge devices"""
    __tablename__ = "drift_events"
    
    id = Column(Integer, primary_key=True, index=True)
    event_id = Column(String(255), unique=True, index=True, nullable=False)
    device_id = Column(String(255), ForeignKey("devices.device_id"), nullable=False)
    model_id = Column(String(255), nullable=True)
    # Explicitly use a string name for the enum type to avoid SQLAlchemy issues
    drift_type = Column(Enum(DriftType, name="drifttype", values_callable=lambda obj: [e.value for e in obj]), 
                      default=DriftType.UNKNOWN, nullable=False)
    drift_score = Column(Float, nullable=True)
    detector_name = Column(String(255), nullable=True)
    
    # Event metadata
    timestamp = Column(DateTime, default=lambda: datetime.now(timezone.utc), nullable=False)
    received_at = Column(DateTime, default=lambda: datetime.now(timezone.utc), nullable=False)
    metrics_before = Column(JSON, nullable=True)  # Performance metrics before drift
    metrics_after = Column(JSON, nullable=True)   # Performance metrics after drift (if adapted)
    
    # Event status
    status = Column(Enum(DriftStatus), default=DriftStatus.PENDING, nullable=False)
    resolved_at = Column(DateTime, nullable=True)
    resolution_notes = Column(Text, nullable=True)
    
    # Additional data
    description = Column(Text, nullable=True)
    event_metadata = Column(JSON, nullable=True)  # Renamed from 'metadata' which is reserved in SQLAlchemy
    
    # Relationships
    device = relationship("Device", back_populates="drift_events")
    samples = relationship("DriftSample", back_populates="drift_event", cascade="all, delete-orphan")
    validations = relationship("DriftValidation", back_populates="drift_event", cascade="all, delete-orphan")


class DriftSample(Base):
    """Model for storing individual samples that triggered drift detection"""
    __tablename__ = "drift_samples"
    
    id = Column(Integer, primary_key=True, index=True)
    sample_id = Column(String(255), unique=True, index=True, nullable=False)
    drift_event_id = Column(String(255), ForeignKey("drift_events.event_id"), nullable=False)
    
    # Sample data
    prediction = Column(String(255), nullable=True)
    confidence = Column(Float, nullable=True)
    drift_score = Column(Float, nullable=True)
    
    # File storage info
    feature_path = Column(String(512), nullable=True)  # Path to feature vector in MinIO
    raw_data_path = Column(String(512), nullable=True)  # Path to raw input data in MinIO
    
    # Metadata
    timestamp = Column(DateTime, nullable=True)
    sample_metadata = Column(JSON, nullable=True)  # Renamed from 'metadata' which is reserved in SQLAlchemy
    
    # Relationships
    drift_event = relationship("DriftEvent", back_populates="samples")
    validations = relationship("DriftValidation", back_populates="drift_sample", cascade="all, delete-orphan")


class DriftValidation(Base):
    """Model for storing human validation of drift samples"""
    __tablename__ = "drift_validations"
    
    id = Column(Integer, primary_key=True, index=True)
    validation_id = Column(String(255), unique=True, index=True, nullable=False)
    drift_event_id = Column(String(255), ForeignKey("drift_events.event_id"), nullable=False)
    drift_sample_id = Column(String(255), ForeignKey("drift_samples.sample_id"), nullable=True)
    
    # Validation data
    is_valid_drift = Column(Boolean, nullable=True)  # Is this a valid drift case?
    true_label = Column(String(255), nullable=True)  # Correct label if provided
    
    # User and timestamp
    validated_by = Column(String(255), nullable=True)
    validated_at = Column(DateTime, default=lambda: datetime.now(timezone.utc), nullable=False)
    
    # Comments and metadata
    validation_notes = Column(Text, nullable=True)
    is_acknowledged = Column(Boolean, default=False)  # Whether device has acknowledged this validation
    acknowledged_at = Column(DateTime, nullable=True)
    validation_metadata = Column(JSON, nullable=True)  # Renamed from 'metadata' which is reserved in SQLAlchemy
    
    # Relationships
    drift_event = relationship("DriftEvent", back_populates="validations")
    drift_sample = relationship("DriftSample", back_populates="validations")