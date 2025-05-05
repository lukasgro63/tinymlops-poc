# api/services/drift_service.py
from datetime import datetime, timedelta, timezone
from typing import Any, Dict, List, Optional, Union
import uuid
import logging
import json
from sqlalchemy import func, desc
from sqlalchemy.orm import Session

from tinysphere.db.models import Device, DriftEvent, DriftSample, DriftValidation, DriftStatus, DriftType
from tinysphere.api.services.notification_service import NotificationService
from tinysphere.api.services.device_service import DeviceService

logger = logging.getLogger(__name__)

class DriftService:
    @staticmethod
    def get_drift_events(
        db: Session, 
        skip: int = 0, 
        limit: int = 100, 
        device_id: Optional[str] = None,
        status: Optional[str] = None,
        drift_type: Optional[str] = None,
        start_date: Optional[datetime] = None,
        end_date: Optional[datetime] = None
    ) -> List[DriftEvent]:
        """
        Get drift events with optional filtering.
        
        Args:
            db: Database session
            skip: Number of records to skip (pagination)
            limit: Max number of records to return
            device_id: Filter by device
            status: Filter by drift status
            drift_type: Filter by drift type
            start_date: Filter by minimum date
            end_date: Filter by maximum date
            
        Returns:
            List of drift events
        """
        query = db.query(DriftEvent)
        
        # Apply filters
        if device_id:
            query = query.filter(DriftEvent.device_id == device_id)
        
        if status:
            try:
                status_enum = getattr(DriftStatus, status.upper())
                query = query.filter(DriftEvent.status == status_enum)
            except (AttributeError, KeyError):
                # Invalid status, ignore filter
                pass
        
        if drift_type:
            try:
                drift_type_enum = getattr(DriftType, drift_type.upper())
                query = query.filter(DriftEvent.drift_type == drift_type_enum)
            except (AttributeError, KeyError):
                # Invalid drift type, ignore filter
                pass
        
        if start_date:
            query = query.filter(DriftEvent.timestamp >= start_date)
            
        if end_date:
            query = query.filter(DriftEvent.timestamp <= end_date)
        
        # Order by timestamp (newest first)
        query = query.order_by(desc(DriftEvent.timestamp))
        
        # Apply pagination
        query = query.offset(skip).limit(limit)
        
        return query.all()
    
    @staticmethod
    def get_drift_event_by_id(db: Session, event_id: str) -> Optional[DriftEvent]:
        """Get a drift event by ID."""
        return db.query(DriftEvent).filter(DriftEvent.event_id == event_id).first()
    
    @staticmethod
    def process_drift_event(db: Session, device_id: str, event_data: Dict[str, Any]) -> DriftEvent:
        """
        Process a drift event received from a device.
        
        Args:
            db: Database session
            device_id: ID of the device reporting drift
            event_data: Drift event data from the device
            
        Returns:
            Created DriftEvent
        """
        # Check if device exists
        device = DeviceService.get_device_by_id(db, device_id)
        if not device:
            # Should not normally happen as should be authenticated already
            logger.warning(f"Received drift event from unknown device: {device_id}")
            # Create minimal device record
            device_data = {"device_id": device_id}
            DeviceService.register_device(db, device_data)
        
        # Generate event ID if not provided
        event_id = event_data.get("event_id", str(uuid.uuid4()))
        
        # Extract drift type
        drift_type_str = event_data.get("drift_type", "unknown").lower()
        try:
            drift_type = getattr(DriftType, drift_type_str.upper())
        except (AttributeError, KeyError):
            drift_type = DriftType.UNKNOWN
        
        # Create drift event
        drift_event = DriftEvent(
            event_id=event_id,
            device_id=device_id,
            model_id=event_data.get("model_id"),
            drift_type=drift_type,
            drift_score=event_data.get("drift_score"),
            detector_name=event_data.get("detector_name"),
            timestamp=event_data.get("timestamp", datetime.now(timezone.utc)),
            metrics_before=event_data.get("performance_before"),
            metrics_after=event_data.get("performance_after"),
            description=event_data.get("description"),
            metadata=event_data.get("metadata", {})
        )
        
        db.add(drift_event)
        db.commit()
        db.refresh(drift_event)
        
        # Process samples if provided
        samples = event_data.get("samples", [])
        for sample_data in samples:
            DriftService.add_drift_sample(db, drift_event.event_id, sample_data)
        
        # Create notification
        NotificationService.create_notification(
            db=db,
            message=f"Drift detected on device {device_id} ({drift_type_str})",
            notification_type="warning",
            source="drift_event",
            source_id=event_id,
            details={
                "drift_type": drift_type_str,
                "drift_score": event_data.get("drift_score"),
                "device_id": device_id
            }
        )
        
        logger.info(f"Processed drift event {event_id} from device {device_id}")
        return drift_event
    
    @staticmethod
    def add_drift_sample(db: Session, event_id: str, sample_data: Dict[str, Any]) -> DriftSample:
        """
        Add a sample to a drift event.
        
        Args:
            db: Database session
            event_id: ID of the drift event
            sample_data: Sample data
            
        Returns:
            Created DriftSample
        """
        # Generate sample ID if not provided
        sample_id = sample_data.get("sample_id", str(uuid.uuid4()))
        
        # Parse timestamp
        timestamp = None
        if "timestamp" in sample_data:
            try:
                if isinstance(sample_data["timestamp"], (int, float)):
                    timestamp = datetime.fromtimestamp(sample_data["timestamp"], tz=timezone.utc)
                elif isinstance(sample_data["timestamp"], str):
                    timestamp = datetime.fromisoformat(sample_data["timestamp"])
            except (ValueError, TypeError):
                timestamp = datetime.now(timezone.utc)
        
        # Create sample
        drift_sample = DriftSample(
            sample_id=sample_id,
            drift_event_id=event_id,
            prediction=sample_data.get("prediction"),
            confidence=sample_data.get("confidence"),
            drift_score=sample_data.get("drift_score"),
            feature_path=sample_data.get("feature_path"),
            raw_data_path=sample_data.get("raw_data_path"),
            timestamp=timestamp,
            metadata=sample_data.get("metadata", {})
        )
        
        db.add(drift_sample)
        db.commit()
        db.refresh(drift_sample)
        
        return drift_sample
    
    @staticmethod
    def validate_drift_sample(
        db: Session, 
        sample_id: str, 
        validation_data: Dict[str, Any]
    ) -> Optional[DriftValidation]:
        """
        Validate a drift sample with human feedback.
        
        Args:
            db: Database session
            sample_id: ID of the sample to validate
            validation_data: Validation data including true_label and is_valid_drift
            
        Returns:
            Created DriftValidation or None if sample not found
        """
        # Get sample
        sample = db.query(DriftSample).filter(DriftSample.sample_id == sample_id).first()
        if not sample:
            logger.warning(f"Attempted to validate non-existent sample: {sample_id}")
            return None
        
        # Create validation
        validation = DriftValidation(
            validation_id=str(uuid.uuid4()),
            drift_event_id=sample.drift_event_id,
            drift_sample_id=sample_id,
            is_valid_drift=validation_data.get("is_valid_drift"),
            true_label=validation_data.get("true_label"),
            validated_by=validation_data.get("validated_by"),
            validation_notes=validation_data.get("notes"),
            metadata=validation_data.get("metadata", {})
        )
        
        db.add(validation)
        
        # Update event status if needed
        event = sample.drift_event
        if event.status == DriftStatus.PENDING:
            event.status = DriftStatus.VALIDATED
        
        db.commit()
        db.refresh(validation)
        
        return validation
    
    @staticmethod
    def get_pending_validations(db: Session, device_id: Optional[str] = None) -> List[Dict[str, Any]]:
        """
        Get list of validations pending device acknowledgment.
        
        Args:
            db: Database session
            device_id: Optional filter by device
            
        Returns:
            List of pending validations
        """
        query = db.query(DriftValidation).filter(
            DriftValidation.is_acknowledged == False
        ).join(
            DriftEvent, DriftValidation.drift_event_id == DriftEvent.event_id
        )
        
        if device_id:
            query = query.filter(DriftEvent.device_id == device_id)
        
        # Order by validation time (oldest first)
        query = query.order_by(DriftValidation.validated_at)
        
        validations = query.all()
        
        # Convert to dict for API response
        result = []
        for validation in validations:
            # Get sample if available
            sample = None
            if validation.drift_sample_id:
                sample = db.query(DriftSample).filter(
                    DriftSample.sample_id == validation.drift_sample_id
                ).first()
            
            # Build validation data
            validation_data = {
                "validation_id": validation.validation_id,
                "drift_event_id": validation.drift_event_id,
                "sample_id": validation.drift_sample_id,
                "device_id": validation.drift_event.device_id,
                "is_valid_drift": validation.is_valid_drift,
                "true_label": validation.true_label,
                "validated_at": validation.validated_at.isoformat(),
                "validated_by": validation.validated_by,
                "notes": validation.validation_notes
            }
            
            # Add sample info if available
            if sample:
                validation_data["sample_info"] = {
                    "prediction": sample.prediction,
                    "confidence": sample.confidence,
                    "drift_score": sample.drift_score,
                    "feature_path": sample.feature_path,
                    "raw_data_path": sample.raw_data_path
                }
            
            result.append(validation_data)
        
        return result
    
    @staticmethod
    def acknowledge_validation(db: Session, validation_id: str) -> bool:
        """
        Mark a validation as acknowledged by device.
        
        Args:
            db: Database session
            validation_id: ID of the validation
            
        Returns:
            True if successful, False otherwise
        """
        validation = db.query(DriftValidation).filter(
            DriftValidation.validation_id == validation_id
        ).first()
        
        if not validation:
            logger.warning(f"Attempted to acknowledge non-existent validation: {validation_id}")
            return False
        
        validation.is_acknowledged = True
        validation.acknowledged_at = datetime.now(timezone.utc)
        
        db.commit()
        logger.info(f"Validation {validation_id} acknowledged")
        return True
    
    @staticmethod
    def update_drift_event_status(
        db: Session, 
        event_id: str, 
        status: str,
        resolution_notes: Optional[str] = None
    ) -> Optional[DriftEvent]:
        """
        Update the status of a drift event.
        
        Args:
            db: Database session
            event_id: ID of the drift event
            status: New status (pending, validated, rejected, resolved, ignored)
            resolution_notes: Optional notes about resolution
            
        Returns:
            Updated DriftEvent or None if not found
        """
        event = DriftService.get_drift_event_by_id(db, event_id)
        if not event:
            return None
        
        try:
            status_enum = getattr(DriftStatus, status.upper())
            event.status = status_enum
            
            # If resolving or ignoring, set resolved_at
            if status_enum in (DriftStatus.RESOLVED, DriftStatus.IGNORED):
                event.resolved_at = datetime.now(timezone.utc)
            
            # Update notes if provided
            if resolution_notes:
                event.resolution_notes = resolution_notes
            
            db.commit()
            db.refresh(event)
            return event
            
        except (AttributeError, KeyError):
            logger.warning(f"Invalid drift status: {status}")
            return None
    
    @staticmethod
    def get_drift_statistics(
        db: Session, 
        device_id: Optional[str] = None,
        days: int = 30
    ) -> Dict[str, Any]:
        """
        Get statistics about drift events.
        
        Args:
            db: Database session
            device_id: Optional filter by device
            days: Number of days to include in time series
            
        Returns:
            Dictionary of statistics
        """
        start_date = datetime.now(timezone.utc) - timedelta(days=days)
        
        # Base query
        query = db.query(DriftEvent)
        if device_id:
            query = query.filter(DriftEvent.device_id == device_id)
        
        # Total counts
        total_events = query.count()
        
        status_counts = {}
        for status in DriftStatus:
            count = query.filter(DriftEvent.status == status).count()
            status_counts[status.value] = count
        
        type_counts = {}
        for drift_type in DriftType:
            count = query.filter(DriftEvent.drift_type == drift_type).count()
            if count > 0:  # Only include types with events
                type_counts[drift_type.value] = count
        
        # Time series data (events per day)
        daily_counts = (
            query.filter(DriftEvent.timestamp >= start_date)
            .with_entities(
                func.date_trunc('day', DriftEvent.timestamp).label('day'),
                func.count().label('count')
            )
            .group_by(func.date_trunc('day', DriftEvent.timestamp))
            .all()
        )
        
        # Format time series
        time_series = []
        current_date = start_date
        end_date = datetime.now(timezone.utc)
        
        # Create a lookup for the query results
        daily_map = {day.strftime('%Y-%m-%d'): count for day, count in daily_counts}
        
        # Generate series with all days
        while current_date <= end_date:
            date_str = current_date.strftime('%Y-%m-%d')
            time_series.append({
                'date': date_str,
                'count': daily_map.get(date_str, 0)
            })
            current_date += timedelta(days=1)
        
        # Compile final statistics
        stats = {
            'total_events': total_events,
            'status_counts': status_counts,
            'type_counts': type_counts,
            'time_series': time_series
        }
        
        return stats
    
    @staticmethod
    def get_device_drift_metrics(db: Session, device_id: str) -> Dict[str, Any]:
        """
        Get drift metrics for a specific device.
        
        Args:
            db: Database session
            device_id: ID of the device
            
        Returns:
            Dictionary of drift metrics
        """
        # Get counts
        total_events = db.query(DriftEvent).filter(
            DriftEvent.device_id == device_id
        ).count()
        
        pending_count = db.query(DriftEvent).filter(
            DriftEvent.device_id == device_id,
            DriftEvent.status == DriftStatus.PENDING
        ).count()
        
        resolved_count = db.query(DriftEvent).filter(
            DriftEvent.device_id == device_id,
            DriftEvent.status.in_([DriftStatus.RESOLVED, DriftStatus.VALIDATED])
        ).count()
        
        # Get latest drift event
        latest_event = db.query(DriftEvent).filter(
            DriftEvent.device_id == device_id
        ).order_by(desc(DriftEvent.timestamp)).first()
        
        latest_event_info = None
        if latest_event:
            latest_event_info = {
                'event_id': latest_event.event_id,
                'drift_type': latest_event.drift_type.value,
                'drift_score': latest_event.drift_score,
                'timestamp': latest_event.timestamp.isoformat(),
                'status': latest_event.status.value
            }
        
        # Return metrics
        return {
            'total_events': total_events,
            'pending_count': pending_count,
            'resolved_count': resolved_count,
            'latest_event': latest_event_info
        }