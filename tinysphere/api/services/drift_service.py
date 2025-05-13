"""
Drift Service API Implementation
--------------------------------
Handles all drift-related database operations and business logic.
"""

import json
import logging
import uuid
from datetime import datetime, timedelta, timezone
from typing import Dict, List, Optional, Any, Tuple, Union

from sqlalchemy import func, desc
from sqlalchemy.orm import Session, joinedload
from sqlalchemy.exc import SQLAlchemyError

from tinysphere.db.models import (
    DriftEvent, DriftSample, DriftValidation, Device,
    DriftStatus, DriftType
)

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
        end_date: Optional[datetime] = None,
        sort_order: Optional[str] = "desc"
    ) -> List[DriftEvent]:
        """
        Get drift events with optional filtering.
        
        Args:
            db: Database session
            skip: Number of records to skip (pagination)
            limit: Maximum number of records to return
            device_id: Optional filter by device ID
            status: Optional filter by status
            drift_type: Optional filter by drift type
            start_date: Optional filter by start date
            end_date: Optional filter by end date
            sort_order: Sort order ("asc" or "desc")
            
        Returns:
            List of DriftEvent objects
        """
        try:
            # Before operations, rollback any existing transaction to ensure clean state
            db.rollback()
            
            logger.info(f"Getting drift events with filters: device_id={device_id}, status={status}, drift_type={drift_type}")
            
            query = db.query(DriftEvent).options(
                joinedload(DriftEvent.samples),
                joinedload(DriftEvent.validations)
            )
            
            # Apply filters
            if device_id:
                query = query.filter(DriftEvent.device_id == device_id)
                
            if status:
                status_enum = DriftService._get_status_enum(status)
                if status_enum:
                    query = query.filter(DriftEvent.status == status_enum)
                    logger.info(f"Filtering by status enum: {status_enum}")
                else:
                    logger.warning(f"Invalid status filter: {status}")
                    
            if drift_type:
                try:
                    # Convert to lowercase for consistency
                    drift_type_lower = drift_type.lower()
                    logger.info(f"Will filter events by drift_type: {drift_type_lower}")
                    
                    # Use text() for safer SQL construction
                    from sqlalchemy import text
                    
                    # Create a new query that uses raw SQL to filter by drift_type
                    # This bypasses SQLAlchemy's enum conversion completely
                    sql = """
                    SELECT * FROM drift_events 
                    WHERE drift_type::text = :drift_type
                    """
                    
                    # Add optional device filter
                    if device_id:
                        sql += " AND device_id = :device_id"
                        
                    # Add optional status filter
                    if status:
                        status_enum = DriftService._get_status_enum(status)
                        if status_enum:
                            sql += " AND status = :status"
                    
                    # Add date filters if specified
                    if start_date:
                        sql += " AND timestamp >= :start_date"
                    
                    if end_date:
                        sql += " AND timestamp <= :end_date"
                    
                    # Create the params dict with all values
                    params = {
                        "drift_type": drift_type_lower,
                    }
                    
                    if device_id:
                        params["device_id"] = device_id
                    
                    if status and status_enum:
                        params["status"] = status_enum.value
                        
                    if start_date:
                        params["start_date"] = start_date
                        
                    if end_date:
                        params["end_date"] = end_date
                    
                    # Create a subquery
                    filtered_events = db.query(DriftEvent).from_statement(
                        text(sql).params(**params)
                    ).subquery()
                    
                    # Replace our main query with one based on the filtered subquery
                    query = db.query(DriftEvent).select_entity_from(filtered_events)
                    
                    logger.info(f"Applied raw SQL filter for drift_type: {drift_type_lower}")
                except Exception as filter_error:
                    # If there's an error with the raw SQL approach, log it
                    logger.error(f"Error applying drift type filter: {filter_error}")
                    
                    # Fallback to using the enum - less reliable but may work for some drift types
                    drift_type_enum = DriftService._get_drift_type_enum(drift_type)
                    if drift_type_enum:
                        query = query.filter(DriftEvent.drift_type == drift_type_enum)
                        logger.info(f"Fallback: filtering by drift type enum: {drift_type_enum.value}")
                    else:
                        logger.warning(f"Invalid drift type filter: {drift_type} - ignoring filter")
                    
            if start_date:
                query = query.filter(DriftEvent.timestamp >= start_date)
                
            if end_date:
                query = query.filter(DriftEvent.timestamp <= end_date)
                
            # Apply sorting
            if sort_order and sort_order.lower() == "asc":
                query = query.order_by(DriftEvent.timestamp)
            else:
                query = query.order_by(desc(DriftEvent.timestamp))
                
            # Apply pagination
            query = query.offset(skip).limit(limit)
            
            # Execute query
            result = query.all()
            logger.info(f"Found {len(result)} drift events matching filters")
            
            return result
            
        except Exception as e:
            db.rollback()
            logger.error(f"Error getting drift events: {e}")
            return []
    
    @staticmethod
    def get_drift_event_by_id(db: Session, event_id: str) -> Optional[DriftEvent]:
        """
        Get drift event by ID.
        
        Args:
            db: Database session
            event_id: Event ID
            
        Returns:
            DriftEvent object or None if not found
        """
        return db.query(DriftEvent).options(
            joinedload(DriftEvent.samples),
            joinedload(DriftEvent.validations)
        ).filter(DriftEvent.event_id == event_id).first()
    
    @staticmethod
    def _get_status_enum(status_str: str) -> Optional[DriftStatus]:
        """
        Convert status string to enum.
        
        Args:
            status_str: Status string
            
        Returns:
            DriftStatus enum
        """
        if not status_str:
            return None
            
        # Try to match by value (lowercase)
        status_lower = status_str.lower()
        for status_enum in DriftStatus:
            if status_enum.value.lower() == status_lower:
                return status_enum
                
        # Try to match by name (uppercase)
        try:
            return getattr(DriftStatus, status_str.upper())
        except (AttributeError, KeyError):
            return None
    
    @staticmethod
    def _get_drift_type_enum(drift_type_str: str):
        """
        Safely convert a drift type string to the corresponding enum object.
        Always handles case sensitivity correctly.
        
        Args:
            drift_type_str: String representation of drift type
            
        Returns:
            DriftType enum object or None if not found
        """
        if not drift_type_str:
            return None
        
        # Always convert to lowercase first for consistent comparison
        drift_type_lower = drift_type_str.lower()
        
        # Direct mapping of lowercase strings to enum values
        mapping = {
            "confidence": DriftType.CONFIDENCE,
            "distribution": DriftType.DISTRIBUTION,
            "feature": DriftType.FEATURE,
            "outlier": DriftType.OUTLIER,
            "custom": DriftType.CUSTOM,
            "knn_distance": DriftType.KNN_DISTANCE,
            "unknown": DriftType.UNKNOWN
        }
        
        # Special handling for KNN_DISTANCE (with underscore)
        if drift_type_lower == "knn_distance":
            return DriftType.KNN_DISTANCE
            
        # Handle both database value and enum name formats
        if drift_type_lower in mapping:
            return mapping[drift_type_lower]
            
        # Try to match by name pattern (KNN_DISTANCE → knn_distance)
        if drift_type_str.upper() == "KNN_DISTANCE":
            return DriftType.KNN_DISTANCE
            
        # Final fallback for any remaining formats
        for dt_enum in DriftType:
            if dt_enum.value.lower() == drift_type_lower:
                return dt_enum
                
        return None
    
    @staticmethod
    def _infer_drift_type_from_description(description: str, detector_name: str):
        """
        Infer the drift type from the event description or detector name.
        
        Args:
            description: Event description text
            detector_name: Name of the detector
            
        Returns:
            DriftType enum object or None if can't determine
        """
        if not description and not detector_name:
            return None
            
        # Convert all inputs to lowercase for easier matching
        desc_lower = description.lower() if description else ""
        detector_lower = detector_name.lower() if detector_name else ""
        
        # Check for KNN distance drift
        if "neighbor_distance" in desc_lower or "knn" in desc_lower or "distance" in detector_lower:
            return DriftType.KNN_DISTANCE
            
        # Check for confidence drift
        if "confidence" in desc_lower or "ewma" in desc_lower:
            return DriftType.CONFIDENCE
            
        # Check for feature drift
        if "feature" in desc_lower:
            return DriftType.FEATURE
            
        # Check for distribution drift
        if "distribution" in desc_lower:
            return DriftType.DISTRIBUTION
            
        # Check for outlier
        if "outlier" in desc_lower:
            return DriftType.OUTLIER
            
        return None
    
    @staticmethod
    def process_drift_event(db: Session, device_id: str, event_data: Dict[str, Any]) -> DriftEvent:
        """
        Process a drift event from a device.
        
        Args:
            db: Database session
            device_id: ID of the device that sent the event
            event_data: Dictionary containing drift event data
            
        Returns:
            Created DriftEvent object
        """
        try:
            # Extract drift type
            drift_type_str = event_data.get("drift_type", "unknown")
            description = event_data.get("reason", event_data.get("description", ""))
            detector_name = event_data.get("detector_name", "")
            
            # Log the drift type we're trying to process
            logger.info(f"Processing drift event with type: {drift_type_str}, detector: {detector_name}")
            
            # First, try to convert the drift type string directly
            drift_type_enum = DriftService._get_drift_type_enum(drift_type_str)
            
            # If that doesn't work, try to infer from the description or detector name
            if not drift_type_enum or drift_type_enum == DriftType.UNKNOWN:
                inferred_type = DriftService._infer_drift_type_from_description(description, detector_name)
                if inferred_type:
                    logger.info(f"Inferred drift type {inferred_type.value} from description: '{description}'")
                    drift_type_enum = inferred_type
            
            # If all else fails, default to UNKNOWN
            if not drift_type_enum:
                logger.warning(f"Could not determine drift type from input: {drift_type_str}")
                drift_type_enum = DriftType.UNKNOWN
                
            # KNN_DISTANCE special case handling
            if drift_type_enum == DriftType.KNN_DISTANCE:
                logger.info(f"Using KNN_DISTANCE drift type (value: '{drift_type_enum.value}') for consistency")
            
            # Create the drift event object
            event = DriftEvent(
                event_id=event_data.get("event_id", str(uuid.uuid4())),
                device_id=device_id,
                drift_type=drift_type_enum,
                drift_score=event_data.get("drift_score", 0.0),
                detector_name=detector_name,
                model_id=event_data.get("model_id"),
                description=description,
                status=DriftStatus.PENDING,
                timestamp=event_data.get("timestamp", datetime.now(timezone.utc)),
                received_at=datetime.now(timezone.utc),
                event_metadata=event_data.get("metadata")
            )
            
            # Add to database
            db.add(event)
            
            # Process any samples included with the event
            samples = event_data.get("samples", [])
            for sample_data in samples:
                DriftService.add_drift_sample(db, event.event_id, sample_data, commit=False)
            
            # Commit the transaction
            db.commit()
            
            # Log successful processing
            logger.info(f"Successfully processed drift event {event.event_id} of type {drift_type_enum.value}")
            
            return event
        except Exception as e:
            db.rollback()
            logger.error(f"Error processing drift event: {e}")
            raise
    
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
            event_id: ID of the event
            status: New status
            resolution_notes: Optional notes about resolution
            
        Returns:
            Updated DriftEvent or None if event not found
        """
        event = db.query(DriftEvent).filter(DriftEvent.event_id == event_id).first()
        if not event:
            return None
            
        status_enum = DriftService._get_status_enum(status)
        if not status_enum:
            raise ValueError(f"Invalid status: {status}")
            
        event.status = status_enum
        
        if status_enum in [DriftStatus.RESOLVED, DriftStatus.REJECTED]:
            event.resolved_at = datetime.now(timezone.utc)
            
        if resolution_notes:
            event.resolution_notes = resolution_notes
            
        db.commit()
        
        return event
    
    @staticmethod
    def add_drift_sample(
        db: Session, 
        event_id: str, 
        sample_data: Dict[str, Any],
        commit: bool = True
    ) -> DriftSample:
        """
        Add a sample to a drift event.
        
        Args:
            db: Database session
            event_id: ID of the drift event
            sample_data: Dictionary with sample data
            commit: Whether to commit the transaction
            
        Returns:
            Created DriftSample object
        """
        sample = DriftSample(
            sample_id=sample_data.get("sample_id", str(uuid.uuid4())),
            drift_event_id=event_id,
            prediction=sample_data.get("prediction"),
            confidence=sample_data.get("confidence"),
            drift_score=sample_data.get("drift_score"),
            feature_path=sample_data.get("feature_path"),
            raw_data_path=sample_data.get("raw_data_path"),
            timestamp=sample_data.get("timestamp", datetime.now(timezone.utc)),
            sample_metadata=sample_data.get("metadata")
        )
        
        db.add(sample)
        
        if commit:
            db.commit()
            
        return sample
    
    @staticmethod
    def validate_drift_sample(
        db: Session, 
        sample_id: str, 
        validation_data: Dict[str, Any]
    ) -> Optional[DriftValidation]:
        """
        Validate a drift sample with ground truth label.
        
        Args:
            db: Database session
            sample_id: ID of the sample to validate
            validation_data: Validation data including true label
            
        Returns:
            Created DriftValidation object or None if sample not found
        """
        sample = db.query(DriftSample).filter(DriftSample.sample_id == sample_id).first()
        if not sample:
            return None
            
        # Create validation
        validation = DriftValidation(
            validation_id=str(uuid.uuid4()),
            drift_event_id=sample.drift_event_id,
            drift_sample_id=sample_id,
            is_valid_drift=validation_data.get("is_valid_drift", True),
            true_label=validation_data.get("true_label"),
            validated_by=validation_data.get("validated_by", "user"),
            validated_at=datetime.now(timezone.utc),
            validation_notes=validation_data.get("notes")
        )
        
        db.add(validation)
        
        # Update parent event status if needed
        if validation_data.get("update_event_status", False):
            event = db.query(DriftEvent).filter(DriftEvent.event_id == sample.drift_event_id).first()
            if event and event.status == DriftStatus.PENDING:
                status = DriftStatus.VALIDATED if validation.is_valid_drift else DriftStatus.REJECTED
                event.status = status
                event.resolved_at = datetime.now(timezone.utc)
                event.resolution_notes = validation_data.get("notes")
        
        db.commit()
        
        return validation
    
    @staticmethod
    def get_pending_validations(db: Session, device_id: Optional[str] = None) -> List[Dict[str, Any]]:
        """
        Get list of validations pending device acknowledgment.
        
        Args:
            db: Database session
            device_id: Optional device ID to filter by
            
        Returns:
            List of validation details
        """
        # Query for unacknowledged validations
        query = db.query(DriftValidation).filter(DriftValidation.is_acknowledged == False)
        
        # Join to get linked sample and event
        query = query.join(DriftSample, DriftValidation.drift_sample_id == DriftSample.sample_id)
        query = query.join(DriftEvent, DriftValidation.drift_event_id == DriftEvent.event_id)
        
        # If device ID provided, filter to only that device
        if device_id:
            query = query.filter(DriftEvent.device_id == device_id)
            
        results = []
        for validation in query.all():
            # Get the sample this validation is for
            sample = db.query(DriftSample).filter(DriftSample.sample_id == validation.drift_sample_id).first()
            if not sample:
                continue
                
            results.append({
                "validation_id": validation.validation_id,
                "drift_event_id": validation.drift_event_id,
                "drift_sample_id": validation.drift_sample_id,
                "is_valid_drift": validation.is_valid_drift,
                "true_label": validation.true_label,
                "validated_by": validation.validated_by,
                "validated_at": validation.validated_at,
                "notes": validation.validation_notes,
                "device_id": sample.drift_event.device_id,
                "sample_info": {
                    "prediction": sample.prediction,
                    "confidence": sample.confidence,
                    "drift_score": sample.drift_score,
                    "feature_path": sample.feature_path,
                    "raw_data_path": sample.raw_data_path
                }
            })
            
        return results
    
    @staticmethod
    def acknowledge_validation(db: Session, validation_id: str) -> bool:
        """
        Mark a validation as acknowledged by device.
        
        Args:
            db: Database session
            validation_id: ID of the validation to acknowledge
            
        Returns:
            True if successful, False if validation not found
        """
        validation = db.query(DriftValidation).filter(DriftValidation.validation_id == validation_id).first()
        if not validation:
            return False
            
        validation.is_acknowledged = True
        validation.acknowledged_at = datetime.now(timezone.utc)
        
        db.commit()
        return True
    
    @staticmethod
    def get_drift_statistics(
        db: Session, 
        device_id: Optional[str] = None,
        days: int = 30,
        drift_type: Optional[str] = None  # Add optional drift_type filter
    ) -> Dict[str, Any]:
        """
        Get statistics about drift events.
        
        Args:
            db: Database session
            device_id: Optional filter by device
            days: Number of days to include in time series
            drift_type: Optional filter by drift type
            
        Returns:
            Dictionary of statistics
        """
        try:
            # Before operations, rollback any existing transaction
            db.rollback()
            
            # Start a new transaction
            db.begin()
            
            start_date = datetime.now(timezone.utc) - timedelta(days=days)
            
            # Base query
            query = db.query(DriftEvent)
            if device_id:
                query = query.filter(DriftEvent.device_id == device_id)
                
            # Apply drift type filter if provided - but ONLY using raw SQL
            if drift_type:
                try:
                    # Convert to lowercase for consistency
                    drift_type_lower = drift_type.lower()
                    logger.info(f"Will filter statistics by drift_type: {drift_type_lower}")
                    
                    # Use text() for safer SQL construction
                    from sqlalchemy import text
                    
                    # Create a new query that uses raw SQL to filter by drift_type
                    # This bypasses SQLAlchemy's enum conversion completely
                    sql = """
                    SELECT * FROM drift_events 
                    WHERE drift_type::text = :drift_type
                    """
                    
                    # Add optional device filter
                    if device_id:
                        sql += " AND device_id = :device_id"
                    
                    # Create a subquery
                    filtered_events = db.query(DriftEvent).from_statement(
                        text(sql).params(
                            drift_type=drift_type_lower,
                            device_id=device_id if device_id else None
                        )
                    ).subquery()
                    
                    # Replace our main query with one based on the filtered subquery
                    query = db.query(DriftEvent).select_entity_from(filtered_events)
                    
                    logger.info(f"Applied raw SQL filter for drift_type: {drift_type_lower}")
                except Exception as filter_error:
                    # If there's an error with the raw SQL approach, log and continue without filtering
                    logger.error(f"Error applying drift type filter: {filter_error}")
                    logger.info("Continuing without drift type filter")
            
            # Total counts
            total_events = query.count()
            
            # Status counts
            status_counts = {}
            for status in DriftStatus:
                count = query.filter(DriftEvent.status == status).count()
                status_counts[status.value] = count
            
            # Calculate open vs resolved events
            total_open = status_counts.get("pending", 0) + status_counts.get("ignored", 0)
            total_resolved = status_counts.get("resolved", 0) + status_counts.get("rejected", 0) + status_counts.get("validated", 0)
            
            # Type counts - hard-coded to avoid all enum issues
            type_counts = {}
            
            # Try all possible drift types directly with specific SQL queries
            # This is the most reliable approach bypassing SQLAlchemy enum handling
            drift_types_to_check = [
                "knn_distance", "confidence", "distribution", 
                "feature", "outlier", "custom", "unknown"
            ]
            
            logger.info(f"Checking drift type counts directly with SQL")
            
            # Check each drift type explicitly
            from sqlalchemy import text
            for drift_type_name in drift_types_to_check:
                try:
                    # Build query for this specific drift type
                    count_sql = f"SELECT COUNT(*) FROM drift_events WHERE drift_type::text = '{drift_type_name}'"
                    
                    # Add device filter if needed
                    if device_id:
                        count_sql += f" AND device_id = '{device_id}'"
                        
                    # Add date filter if needed
                    if start_date:
                        count_sql += f" AND timestamp >= '{start_date}'"
                    
                    # Execute directly as text
                    count_result = db.execute(text(count_sql)).scalar() or 0
                    
                    # Only add non-zero counts
                    if count_result > 0:
                        type_counts[drift_type_name] = count_result
                        logger.info(f"Found {count_result} events of type '{drift_type_name}'")
                except Exception as type_error:
                    logger.error(f"Error counting drift type {drift_type_name}: {type_error}")
            
            # If we failed to get any counts, set a minimal valid result
            if not type_counts:
                logger.warning("No drift type counts found, using empty set")
                type_counts = {"unknown": 0}
            
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
            
            # Get recent events for preview
            recent_events = []
            for event in query.order_by(desc(DriftEvent.timestamp)).limit(5).all():
                # Count samples and validations for this event directly using the imported models
                # This avoids using db.models which doesn't exist
                sample_count = db.query(DriftSample).filter(
                    DriftSample.drift_event_id == event.event_id
                ).count()
                
                validation_count = db.query(DriftValidation).filter(
                    DriftValidation.drift_event_id == event.event_id
                ).count()
                
                # Adding the required fields (sample_count and validation_count)
                recent_events.append({
                    "event_id": event.event_id,
                    "device_id": event.device_id,
                    "drift_type": event.drift_type.value.lower(),  # Ensure lowercase consistency
                    "drift_score": event.drift_score,
                    "detector_name": event.detector_name,
                    "model_id": event.model_id,
                    "description": event.description,
                    "status": event.status.value,
                    "timestamp": event.timestamp,
                    "received_at": event.received_at,
                    "resolved_at": event.resolved_at,
                    "resolution_notes": event.resolution_notes,
                    "sample_count": sample_count,  # Add required field
                    "validation_count": validation_count  # Add required field
                })
            
            result = {
                "total_events": total_events,
                "total_open": total_open,
                "total_resolved": total_resolved,
                "by_status": status_counts,
                "by_type": type_counts,
                "by_day": time_series,
                "recent_events": recent_events
            }
            
            db.commit()
            return result
            
        except Exception as e:
            db.rollback()
            logger.error(f"Error getting drift statistics: {e}")
            
            # Return empty data in case of error
            return {
                "total_events": 0,
                "total_open": 0,
                "total_resolved": 0,
                "by_status": {},
                "by_type": {},
                "by_day": [],
                "recent_events": [],
                "error": str(e)
            }
    
    @staticmethod
    def get_device_drift_metrics(db: Session, device_id: str) -> Dict[str, Any]:
        """
        Get drift-related metrics for a specific device.
        
        Args:
            db: Database session
            device_id: Device ID
            
        Returns:
            Dictionary of metrics
        """
        # Init results structure
        results = {
            "total_events": 0,
            "by_status": {},
            "by_type": {},
            "first_event": None,
            "last_event": None
        }
        
        # Get basic counts
        query = db.query(DriftEvent).filter(DriftEvent.device_id == device_id)
        results["total_events"] = query.count()
        
        # Get status breakdown
        for status in DriftStatus:
            count = query.filter(DriftEvent.status == status).count()
            if count > 0:
                results["by_status"][status.value] = count
        
        # Get type breakdown
        for drift_type in DriftType:
            try:
                # Use consistent lowercase keys for the frontend
                count = query.filter(DriftEvent.drift_type == drift_type).count()
                if count > 0:
                    results["by_type"][drift_type.value.lower()] = count
            except Exception as type_error:
                logger.error(f"Error counting drift type {drift_type}: {type_error}")
        
        # Get first and last events
        first_event = query.order_by(DriftEvent.timestamp).first()
        if first_event:
            results["first_event"] = {
                "event_id": first_event.event_id,
                "timestamp": first_event.timestamp,
                "type": first_event.drift_type.value
            }
            
        last_event = query.order_by(desc(DriftEvent.timestamp)).first()
        if last_event:
            results["last_event"] = {
                "event_id": last_event.event_id,
                "timestamp": last_event.timestamp,
                "type": last_event.drift_type.value
            }
            
        return results

    @staticmethod
    def repair_drift_types(db: Session) -> Dict[str, Any]:
        """
        Repair drift types for all events that can be inferred from descriptions.
        This function repairs both UNKNOWN drift types and corrects KNN_DISTANCE issues.
        
        Args:
            db: Database session
            
        Returns:
            Dictionary with repair results
        """
        from sqlalchemy import text
        
        try:
            # Make sure we have a clean transaction
            db.rollback()
            
            # First, get a count of all drift events
            total_count = db.execute(text("SELECT COUNT(*) FROM drift_events")).scalar() or 0
            logger.info(f"Found {total_count} total drift events to check")
            
            # Get UNKNOWN events directly from the database
            unknown_events = []
            try:
                unknown_results = db.execute(
                    text("SELECT id, event_id, description, detector_name FROM drift_events WHERE drift_type = 'unknown'")
                ).fetchall()
                
                for row in unknown_results:
                    unknown_events.append({
                        "id": row[0],
                        "event_id": row[1],
                        "description": row[2],
                        "detector_name": row[3],
                        "current_type": "unknown"
                    })
                    
                logger.info(f"Found {len(unknown_events)} events with 'unknown' drift type")
            except Exception as unknown_error:
                logger.error(f"Error getting unknown events: {unknown_error}")
            
            # Also check for events with knn distance descriptions but wrong type
            knn_mismatched = []
            try:
                # Look for events that have "distance" or "knn" in description but aren't knn_distance type
                mismatched_results = db.execute(
                    text("""
                        SELECT id, event_id, description, detector_name, drift_type 
                        FROM drift_events 
                        WHERE 
                            (description ILIKE '%distance%' OR description ILIKE '%knn%' OR detector_name ILIKE '%distance%') 
                            AND drift_type != 'knn_distance'
                    """)
                ).fetchall()
                
                for row in mismatched_results:
                    knn_mismatched.append({
                        "id": row[0],
                        "event_id": row[1],
                        "description": row[2],
                        "detector_name": row[3],
                        "current_type": row[4]
                    })
                    
                logger.info(f"Found {len(knn_mismatched)} events with KNN descriptions but wrong type")
            except Exception as knn_error:
                logger.error(f"Error getting KNN mismatched events: {knn_error}")
            
            # Combine both sets of events to repair
            events_to_check = unknown_events + knn_mismatched
            logger.info(f"Checking {len(events_to_check)} events for repair")
            
            # Count how many we repair
            repaired_count = 0
            repaired_events = []
            
            # Process each event
            for event in events_to_check:
                # Try to infer drift type from description and detector name
                inferred_type = DriftService._infer_drift_type_from_description(
                    event.get("description", ""),
                    event.get("detector_name", "")
                )
                
                # Only update if we inferred a valid type
                if inferred_type and inferred_type != DriftType.UNKNOWN:
                    try:
                        # New type (always lowercase for DB)
                        new_type = inferred_type.value.lower()
                        old_type = event.get("current_type", "unknown")
                        
                        # Update directly with raw SQL to avoid enum issues
                        db.execute(
                            text("UPDATE drift_events SET drift_type = :new_type WHERE id = :id"),
                            {
                                "id": event["id"],
                                "new_type": new_type
                            }
                        )
                        
                        repaired_count += 1
                        
                        logger.info(f"Repaired event {event['event_id']}: {old_type} -> {new_type}")
                        repaired_events.append({
                            "event_id": event["event_id"],
                            "old_type": old_type,
                            "new_type": new_type,
                            "description": event.get("description", ""),
                            "detector_name": event.get("detector_name", "")
                        })
                    except Exception as update_error:
                        logger.error(f"Error updating event {event['event_id']}: {update_error}")
            
            # Commit the changes if we repaired any events
            if repaired_count > 0:
                db.commit()
                logger.info(f"Successfully repaired {repaired_count} drift events")
            
            return {
                "total_events": total_count,
                "repaired_count": repaired_count,
                "repaired_events": repaired_events
            }
            
        except Exception as e:
            logger.error(f"Error repairing drift types: {e}")
            db.rollback()
            return {
                "error": str(e),
                "total_events": 0,
                "repaired_count": 0,
                "repaired_events": []
            }