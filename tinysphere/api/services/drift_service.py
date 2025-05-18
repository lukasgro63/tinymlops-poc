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
                    # Use the simplest possible approach, avoiding complex SQL
                    drift_type_lower = drift_type.lower()
                    logger.info(f"Will filter events by drift_type: {drift_type_lower} (simple Python filtering)")
                    
                    # Get all events that match other filters
                    filtered_query = db.query(DriftEvent.id, DriftEvent.drift_type)
                    
                    # Apply standard filters
                    if device_id:
                        filtered_query = filtered_query.filter(DriftEvent.device_id == device_id)
                        
                    if status:
                        status_enum = DriftService._get_status_enum(status)
                        if status_enum:
                            filtered_query = filtered_query.filter(DriftEvent.status == status_enum)
                    
                    if start_date:
                        filtered_query = filtered_query.filter(DriftEvent.timestamp >= start_date)
                    
                    if end_date:
                        filtered_query = filtered_query.filter(DriftEvent.timestamp <= end_date)
                    
                    # Execute the query
                    all_events = filtered_query.all()
                    
                    # Filter in Python based on string comparison
                    matching_ids = []
                    for event_id, event_drift_type in all_events:
                        if event_drift_type.value.lower() == drift_type_lower:
                            matching_ids.append(event_id)
                    
                    # Return empty list if no matches
                    if not matching_ids:
                        logger.warning(f"No events found with drift_type: {drift_type_lower}")
                        return []
                    
                    # Filter the original query by these IDs
                    query = query.filter(DriftEvent.id.in_(matching_ids))
                    logger.info(f"Found {len(matching_ids)} events with drift_type: {drift_type_lower}")
                    
                except Exception as filter_error:
                    # Log the error and continue without filtering
                    logger.error(f"Error applying drift type filter: {filter_error}")
                    logger.warning(f"Continuing without drift type filter due to error")
                    
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
        UNIVERSAL WORKAROUND: All drift types are mapped to UNKNOWN for database compatibility.
        The original type is preserved in metadata.
        
        Args:
            drift_type_str: String representation of drift type
            
        Returns:
            Always returns DriftType.UNKNOWN
        """
        if not drift_type_str:
            return DriftType.UNKNOWN
        
        # Always convert to lowercase for logging
        drift_type_lower = drift_type_str.lower() if drift_type_str else "unknown"
        
        # Log the original requested type
        if drift_type_lower != "unknown":
            logger.warning(f"UNIVERSAL WORKAROUND: Mapping '{drift_type_str}' to UNKNOWN for database compatibility")
            
        # No complex mappings - always return UNKNOWN
        # The actual type will be preserved in metadata
        return DriftType.UNKNOWN
    
    @staticmethod
    def _infer_drift_type_from_description(description: str, detector_name: str):
        """
        Infer the logical drift type from the event description or detector name.
        This method is now used only to determine the logical drift type to store in metadata.
        The database type is always UNKNOWN.
        
        Args:
            description: Event description text
            detector_name: Name of the detector
            
        Returns:
            String representation of the detected drift type
        """
        if not description and not detector_name:
            return "unknown"
            
        # Convert all inputs to lowercase for easier matching
        desc_lower = description.lower() if description else ""
        detector_lower = detector_name.lower() if detector_name else ""
        
        # Check for KNN distance drift
        if "neighbor_distance" in desc_lower or "knn" in desc_lower or "distance" in detector_lower:
            return "knn_distance"
            
        # Check for confidence drift
        if "confidence" in desc_lower or "ewma" in desc_lower:
            return "confidence"
            
        # Check for feature drift
        if "feature" in desc_lower:
            return "feature"
            
        # Check for distribution drift
        if "distribution" in desc_lower:
            return "distribution"
            
        # Check for outlier
        if "outlier" in desc_lower:
            return "outlier"
            
        # Default to unknown if we can't determine
        return "unknown"
    
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
            # Extract relevant information
            drift_type = event_data.get("drift_type", "unknown")
            description = event_data.get("reason", event_data.get("description", ""))
            detector_name = event_data.get("detector_name", "")
            
            # Handle if drift_type is already an enum
            if isinstance(drift_type, DriftType):
                # We already have the correct enum type, use it directly
                drift_type_str = drift_type.value
                logger.info(f"Using provided DriftType enum: {drift_type.name}, value: {drift_type.value}")
            else:
                # It's a string, normalize it
                drift_type_str = str(drift_type).lower() if drift_type else "unknown"
                logger.info(f"Converting string drift_type '{drift_type_str}' to appropriate enum")
            
            # Log the original drift type we received
            logger.info(f"Processing drift event with original type: {drift_type_str}, detector: {detector_name}")
            
            # STEP 1: Determine the logical drift type (to store in metadata)
            logical_drift_type = "unknown"
            
            # First check if we already have a DriftType enum
            if isinstance(drift_type, DriftType):
                # Already have the enum, use its string value for logical type
                logical_drift_type = drift_type.value
                logger.info(f"Using provided DriftType enum value for logical_drift_type: {logical_drift_type}")
            # Then check the detector_name and description for KNN distance signs
            else:
                detector_name_lower = detector_name.lower() if detector_name else ""
                description_lower = description.lower() if description else ""
                
                # KNN distance detector is our primary drift detector in the system
                if ("knndistance" in detector_name_lower or 
                    "knn distance" in detector_name_lower or
                    "knn" in detector_name_lower or
                    "neighbor_distance" in description_lower or
                    "distance" in detector_name_lower):
                    logical_drift_type = "knn_distance"
                    logger.info(f"Detected KNN-based drift from detector name: {detector_name}")
                # If a drift_type was explicitly provided, use that as the logical type
                elif drift_type_str and drift_type_str.lower() != "unknown":
                    logical_drift_type = drift_type_str.lower()
                # Otherwise, try to infer from description/detector
                else:
                    # Infer logical type from description/detector
                    inferred_type = DriftService._infer_drift_type_from_description(description, detector_name)
                    if inferred_type != "unknown":
                        logical_drift_type = inferred_type
                        logger.info(f"Inferred logical drift type: {logical_drift_type} from description/detector")
            
            # STEP 2: Prepare the appropriate enum value based on logical_drift_type
            # This will be set properly in the next step
            drift_type_enum = None
            logger.info(f"Will set appropriate database enum for logical drift type: {logical_drift_type}")
            
            # STEP 3: Ensure metadata exists and store the logical type there
            if event_data.get("metadata") is None:
                event_data["metadata"] = {}
                
            metadata = event_data.get("metadata", {})
            if isinstance(metadata, dict):
                # Store the logical drift type
                metadata["original_drift_type"] = logical_drift_type
                
                # Add a display name for better readability
                if logical_drift_type == "knn_distance":
                    metadata["drift_type_display"] = "KNN Distance"
                else:
                    # Just capitalize the first letter of the type
                    metadata["drift_type_display"] = logical_drift_type.capitalize()
                    
            logger.info(f"Stored logical drift type '{logical_drift_type}' in metadata for later reference")
            
            # We'll determine the enum based on logical_drift_type later
            logger.info(f"Will determine appropriate database enum based on logical_drift_type: '{logical_drift_type}'")
            
            # Create event with properly-handled enum to avoid SQLAlchemy enum issues
            try:
                # Simple validation that we have a logical drift type
                if not logical_drift_type or logical_drift_type == "unknown":
                    logger.warning(f"No specific drift type could be determined, using 'confidence' as fallback")
                    logical_drift_type = "confidence"
                
                # Diagnose-Ausgabe for the incoming event
                logger.info(f"Using logical drift type for mapping: '{logical_drift_type}'")
                
                # IMPROVED HANDLING: Map the drift type to the appropriate enum value
                # Based on the logical drift type, select the appropriate enum
                final_db_type = None
                
                # For KNN distance events, use a safe fallback
                try:
                    # First try to check the actual enum values in PostgreSQL
                    valid_enum_values = []
                    enum_check = db.execute(
                        text("SELECT enumlabel FROM pg_enum JOIN pg_type ON pg_enum.enumtypid = pg_type.oid WHERE pg_type.typname = 'drifttype'")
                    ).fetchall()
                    
                    if enum_check:
                        valid_enum_values = [val[0] for val in enum_check]
                        logger.info(f"Found valid enum values for drift_type: {valid_enum_values}")
                    
                    # If KNN_DISTANCE is a valid enum, we can use it directly
                    if logical_drift_type.lower() in ['knn_distance', 'knn', 'knndistance', 'knn distance']:
                        if "KNN_DISTANCE" in valid_enum_values:
                            final_db_type = DriftType.KNN_DISTANCE
                            logger.info("Using DriftType.KNN_DISTANCE as it exists in the database")
                        else:
                            # Otherwise fallback to a guaranteed enum
                            final_db_type = DriftType.CONFIDENCE
                            logger.info("Using DriftType.CONFIDENCE as safe fallback for KNN distance drift (KNN_DISTANCE not in database)")
                    # For confidence events
                    elif logical_drift_type.lower() in ['confidence']:
                        final_db_type = DriftType.CONFIDENCE
                    # For other known types
                    elif logical_drift_type.lower() in ['distribution']:
                        final_db_type = DriftType.DISTRIBUTION
                    elif logical_drift_type.lower() in ['feature']:
                        final_db_type = DriftType.FEATURE
                    elif logical_drift_type.lower() in ['outlier']:
                        final_db_type = DriftType.OUTLIER
                    elif logical_drift_type.lower() in ['custom']:
                        final_db_type = DriftType.CUSTOM
                    else:
                        # Default to a known valid enum for unknown types
                        final_db_type = DriftType.CONFIDENCE
                except Exception as e:
                    # On error, just use a safe enum
                    logger.error(f"Error checking valid enum values: {e}")
                    final_db_type = DriftType.CONFIDENCE
                    logger.info("Using DriftType.CONFIDENCE due to error checking valid enum values")
                
                # Speichern des tatsächlichen Werts im Event-Metadata, um die Originalinformation zu behalten
                if event_data.get("metadata") is None:
                    event_data["metadata"] = {}
                
                metadata = event_data.get("metadata", {})
                if isinstance(metadata, dict):
                    metadata["original_drift_type"] = logical_drift_type
                    
                    # Für KNN-basierte Events auch einen lesbaren Namen hinzufügen
                    if logical_drift_type.lower() in ['knn_distance', 'knn', 'knndistance']:
                        metadata["drift_type_display"] = "KNN Distance"
                    else:
                        # Ersten Buchstaben groß für bessere Lesbarkeit
                        metadata["drift_type_display"] = logical_drift_type.capitalize()
                
                # Diagnose-Ausgabe
                logger.info(f"Using drift_type={final_db_type.name} (value='{final_db_type.value}') in database")
                
                # EMERGENCY FIX: Create a simpler version of this data as a new event
                # Store it with minimal fields to avoid enum issues
                
                # Prepare the data 
                event_id = event_data.get("event_id", str(uuid.uuid4()))
                
                # Prepare metadata with the important information
                emergency_metadata = {
                    "original_data": event_data,  # Store full original data
                    "logical_drift_type": logical_drift_type,
                    "drift_type_display": logical_drift_type.capitalize() if logical_drift_type != "knn_distance" else "KNN Distance"
                }
                
                # Add any existing metadata
                if event_data.get("metadata"):
                    if isinstance(event_data["metadata"], dict):
                        emergency_metadata.update(event_data["metadata"])
                    elif isinstance(event_data["metadata"], str):
                        try:
                            emergency_metadata["original_metadata"] = json.loads(event_data["metadata"])
                        except:
                            emergency_metadata["original_metadata_str"] = event_data["metadata"]
                
                # Make sure metadata is properly formatted
                metadata_json = json.dumps(emergency_metadata)
                
                try:
                    logger.info("EMERGENCY FALLBACK: Directly inserting drift event with UPDATE statement")
                    
                    # Import text from SQLAlchemy (should be already imported above)
                    from sqlalchemy import text
                    
                    # Try a query to check what enum values exist in the database
                    try:
                        # First try to check the actual enum values in PostgreSQL
                        check_result = db.execute(text("SELECT enumlabel FROM pg_enum JOIN pg_type ON pg_enum.enumtypid = pg_type.oid WHERE pg_type.typname = 'drifttype'")).fetchall()
                        if check_result:
                            # Extract the available enum values
                            available_enum_values = [val[0] for val in check_result]
                            logger.info(f"Found PostgreSQL enum values for drifttype: {available_enum_values}")
                            
                            # Convert logical drift type to uppercase for case-sensitive matching
                            logical_drift_type_upper = logical_drift_type.upper() if logical_drift_type else "UNKNOWN"
                            
                            # Use the most appropriate value for our logical drift_type
                            if logical_drift_type == "knn_distance" and ("KNN_DISTANCE" in available_enum_values or "knn_distance" in available_enum_values):
                                # The KNN distance type exists in the database - use case-sensitive match
                                if "KNN_DISTANCE" in available_enum_values:
                                    sample_drift_type = "KNN_DISTANCE"
                                else:
                                    sample_drift_type = "knn_distance"
                                logger.info(f"Using actual {sample_drift_type} enum value")
                            # Check for standard types in both upper and lower case
                            elif "DISTRIBUTION" in available_enum_values:
                                sample_drift_type = "DISTRIBUTION"
                            elif "distribution" in available_enum_values:
                                sample_drift_type = "distribution"
                            elif "CONFIDENCE" in available_enum_values:
                                sample_drift_type = "CONFIDENCE"
                            elif "confidence" in available_enum_values:
                                sample_drift_type = "confidence"
                            elif "UNKNOWN" in available_enum_values:
                                sample_drift_type = "UNKNOWN"
                            elif "unknown" in available_enum_values:
                                sample_drift_type = "unknown"
                            else:
                                # Use the first valid value
                                sample_drift_type = available_enum_values[0]
                                
                            logger.info(f"Selected valid enum value from database: {sample_drift_type}")
                        else:
                            # Default to distribution which should exist based on migration files
                            sample_drift_type = "distribution"
                            logger.info(f"No enum values found, using default: {sample_drift_type}")
                    except Exception as check_error:
                        logger.error(f"Error checking for valid drift_type enum values: {check_error}")
                        # Default to distribution which should exist based on migration files
                        sample_drift_type = "distribution"
                        logger.info(f"Using fallback drift_type: {sample_drift_type}")
                    
                    # First get a valid status value from the database
                    status_value = "PENDING"  # Default value
                    try:
                        # Get actual valid status value from database
                        status_result = db.execute(text(
                            "SELECT enumlabel FROM pg_enum JOIN pg_type ON pg_enum.enumtypid = pg_type.oid WHERE pg_type.typname = 'driftstatus' LIMIT 1"
                        )).fetchone()
                        
                        if status_result:
                            status_value = status_result[0]
                            logger.info(f"Using database-confirmed status value: {status_value}")
                    except Exception as status_err:
                        logger.error(f"Error getting valid status value: {status_err}")
                    
                    # Use direct SQL with text() function to handle SQL properly
                    # Use a more reliable approach with parameterized query
                    insert_query = f"""
                    INSERT INTO drift_events 
                    (event_id, device_id, model_id, drift_score, detector_name, drift_type, timestamp, received_at, status, description, event_metadata) 
                    VALUES 
                    (:event_id, :device_id, NULL, :drift_score, :detector_name, 
                     :drift_type, 
                     :timestamp, :received_at, 
                     '{status_value}', :description, :metadata)
                    """

                    # Prepare parameters with proper escaping
                    params = {
                        "event_id": event_id,
                        "device_id": device_id,
                        "drift_score": event_data.get('drift_score', 0.0),
                        "detector_name": detector_name,
                        "drift_type": sample_drift_type,  # This is a validated drift_type that exists in the database
                        "timestamp": event_data.get("timestamp", datetime.now(timezone.utc).isoformat()),
                        "received_at": datetime.now(timezone.utc).isoformat(),
                        "description": description,
                        "metadata": metadata_json
                    }
                    
                    # Execute with parameterized query for safety
                    db.execute(text(insert_query), params)
                    
                    logger.info(f"EMERGENCY FALLBACK: Successfully inserted event {event_id} with direct SQL")
                                        
                    # Create object without saving - just for returning
                    event = DriftEvent(
                        event_id=event_id,
                        device_id=device_id,
                        drift_score=event_data.get("drift_score", 0.0),
                        detector_name=detector_name,
                        model_id=event_data.get("model_id"),
                        description=description,
                        status=DriftStatus.PENDING,
                        timestamp=event_data.get("timestamp", datetime.now(timezone.utc)),
                        received_at=datetime.now(timezone.utc),
                        event_metadata=event_data.get("metadata")
                    )
                    
                    logger.info(f"Created DriftEvent object successfully with direct SQL insert - bypassing enum type issue")
                except Exception as sql_error:
                    logger.error(f"Error with direct SQL insert: {sql_error}")
                    raise
            except Exception as create_error:
                logger.error(f"Error creating DriftEvent object: {create_error}")
                raise
            
            # Skip adding to database through ORM - we're using raw SQL
            # db.add(event)  # commented out to avoid duplicated entry
            
            # Skip sample insertion completely for now
            # We'll just include sample information in the metadata
            samples = event_data.get("samples", [])
            if samples:
                logger.info(f"Samples data is stored in metadata - skipping separate sample table insertion")
                # The samples are already stored in the event_metadata emergency_metadata['original_data']['samples']
            
            # Commit the transaction
            db.commit()
            logger.info(f"Transaction committed successfully for drift event {event_id}")
            
            # Log successful processing
            logger.info(f"Successfully processed drift event {event_id} of type 'distribution' (overridden)")
            
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
                
            # Apply drift type filter - simple approach avoiding complex SQL
            if drift_type:
                try:
                    # Get all events that match other standard filters
                    filtered_query = db.query(DriftEvent.id, DriftEvent.drift_type)
                    
                    # Apply basic filters first
                    if device_id:
                        filtered_query = filtered_query.filter(DriftEvent.device_id == device_id)
                    
                    if start_date:
                        filtered_query = filtered_query.filter(DriftEvent.timestamp >= start_date)
                    
                    # Execute query and filter in Python
                    all_events = filtered_query.all()
                    drift_type_lower = drift_type.lower()
                    
                    # Find matching events by string comparison
                    matching_ids = []
                    for event_id, event_drift_type in all_events:
                        if event_drift_type.value.lower() == drift_type_lower:
                            matching_ids.append(event_id)
                    
                    # Apply filter to main query if we found matches
                    if matching_ids:
                        logger.info(f"Found {len(matching_ids)} events with drift_type '{drift_type_lower}'")
                        query = query.filter(DriftEvent.id.in_(matching_ids))
                    else:
                        logger.warning(f"No events found with drift_type '{drift_type_lower}'")
                    
                except Exception as filter_error:
                    # Log error and continue without filtering
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
            
            # Type counts - using metadata to get the actual logical type
            type_counts = {}
            
            try:
                # First, get all events with standard query
                event_query = db.query(DriftEvent)
                
                # Apply filters
                if device_id:
                    event_query = event_query.filter(DriftEvent.device_id == device_id)
                    
                if start_date:
                    event_query = event_query.filter(DriftEvent.timestamp >= start_date)
                
                # Get all matching events
                events = event_query.all()
                
                # Manually count by extracting the logical type from metadata
                for event in events:
                    # Default to the database drift type
                    drift_type_name = event.drift_type.value.lower()
                    
                    # Try to get the actual type from metadata
                    if event.event_metadata and isinstance(event.event_metadata, dict):
                        original_type = event.event_metadata.get("original_drift_type")
                        if original_type:
                            drift_type_name = original_type
                    
                    # Increment the count for this type
                    if drift_type_name in type_counts:
                        type_counts[drift_type_name] += 1
                    else:
                        type_counts[drift_type_name] = 1
                
                # Log the counts
                for dtype, count in type_counts.items():
                    logger.info(f"Found {count} events of logical type '{dtype}'")
                
            except Exception as type_error:
                logger.error(f"Error counting drift types: {type_error}")
            
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
                
                # Determine correct drift type - either from metadata or standard value
                drift_type_value = event.drift_type.value.lower()  # Default to enum value
                
                # Check if this is a KNN_DISTANCE event stored as CUSTOM
                if event.drift_type == DriftType.CUSTOM and event.event_metadata:
                    try:
                        if isinstance(event.event_metadata, dict) and event.event_metadata.get("original_drift_type") == "knn_distance":
                            drift_type_value = "knn_distance"
                            logger.info(f"Restored original KNN_DISTANCE type for event {event.event_id} from metadata")
                    except Exception as metadata_err:
                        logger.error(f"Error processing event metadata: {metadata_err}")
                
                # Extract proper drift type from metadata if available
                drift_type_value = "unknown"  # Default value
                
                # Check metadata for original drift type
                if event.event_metadata and isinstance(event.event_metadata, dict):
                    original_type = event.event_metadata.get("original_drift_type")
                    if original_type:
                        drift_type_value = original_type
                
                # Adding the required fields with the correct drift type
                recent_events.append({
                    "event_id": event.event_id,
                    "device_id": event.device_id,
                    "drift_type": drift_type_value,  # Use drift type from metadata when available
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
        
        # Get type breakdown including special handling for KNN_DISTANCE events
        
        # 1. Get standard drift types
        for drift_type in DriftType:
            try:
                # Use consistent lowercase keys for the frontend
                count = query.filter(DriftEvent.drift_type == drift_type).count()
                if count > 0:
                    # Store regular types
                    results["by_type"][drift_type.value.lower()] = count
            except Exception as type_error:
                logger.error(f"Error counting drift type {drift_type}: {type_error}")
                
        # 2. Special handling for KNN_DISTANCE events stored as CUSTOM with metadata
        try:
            # Look for CUSTOM events that have KNN_DISTANCE in their metadata
            knn_count = 0
            custom_events = query.filter(DriftEvent.drift_type == DriftType.CUSTOM).all()
            
            for event in custom_events:
                if event.event_metadata and isinstance(event.event_metadata, dict):
                    if event.event_metadata.get("original_drift_type") == "knn_distance":
                        knn_count += 1
                        
                        # Remove this event from the CUSTOM count since we're counting it as KNN_DISTANCE
                        if "custom" in results["by_type"]:
                            results["by_type"]["custom"] -= 1
                            
                            # Remove custom entirely if count is now zero
                            if results["by_type"]["custom"] == 0:
                                del results["by_type"]["custom"]
            
            # Add KNN_DISTANCE count if we found any
            if knn_count > 0:
                # If knn_distance already exists (unlikely but possible), add to it
                if "knn_distance" in results["by_type"]:
                    results["by_type"]["knn_distance"] += knn_count
                else:
                    results["by_type"]["knn_distance"] = knn_count
        except Exception as knn_error:
                logger.error(f"Error counting KNN_DISTANCE events: {knn_error}")
        
        # Get first and last events
        first_event = query.order_by(DriftEvent.timestamp).first()
        if first_event:
            # Determine correct drift type for display (handle KNN_DISTANCE workaround)
            drift_type_value = first_event.drift_type.value
            
            # Check if this is a KNN_DISTANCE event stored as CUSTOM
            if first_event.drift_type == DriftType.CUSTOM and first_event.event_metadata:
                try:
                    if isinstance(first_event.event_metadata, dict) and first_event.event_metadata.get("original_drift_type") == "knn_distance":
                        drift_type_value = "knn_distance"
                except Exception:
                    pass  # Use default value if metadata access fails
            
            results["first_event"] = {
                "event_id": first_event.event_id,
                "timestamp": first_event.timestamp,
                "type": drift_type_value
            }
            
        last_event = query.order_by(desc(DriftEvent.timestamp)).first()
        if last_event:
            # Determine correct drift type for display (handle KNN_DISTANCE workaround)
            drift_type_value = last_event.drift_type.value
            
            # Check if this is a KNN_DISTANCE event stored as CUSTOM
            if last_event.drift_type == DriftType.CUSTOM and last_event.event_metadata:
                try:
                    if isinstance(last_event.event_metadata, dict) and last_event.event_metadata.get("original_drift_type") == "knn_distance":
                        drift_type_value = "knn_distance"
                except Exception:
                    pass  # Use default value if metadata access fails
            
            results["last_event"] = {
                "event_id": last_event.event_id,
                "timestamp": last_event.timestamp,
                "type": drift_type_value
            }
            
        return results

    @staticmethod
    def repair_drift_types(db: Session) -> Dict[str, Any]:
        """
        Repair drift types for all events that can be inferred from descriptions.
        This function repairs both UNKNOWN drift types and corrects KNN_DISTANCE issues.
        
        It will:
        1. Find all valid enum values in the database
        2. Update any events with invalid drift types
        3. Store original drift type info in metadata
        
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
            
            # Get all valid drift type values in the database
            valid_drift_types = []
            try:
                drift_type_values = db.execute(
                    text("SELECT unnest(enum_range(NULL::drifttype))::text FROM generate_series(1,1)")
                ).fetchall()
                valid_drift_types = [val[0] for val in drift_type_values]
                logger.info(f"Found valid drift_type enum values: {valid_drift_types}")
            except Exception as enum_error:
                logger.error(f"Error getting valid drift_type enum values: {enum_error}")
                valid_drift_types = ["unknown", "confidence", "distribution"]  # Default fallback values
            
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
                    """)
                ).fetchall()
                
                # Check if knn_distance is a valid type in the database
                has_knn_distance = "knn_distance" in valid_drift_types
                
                for row in mismatched_results:
                    current_type = row[4]
                    # Only add to mismatched if knn_distance is valid and the current type is different
                    if has_knn_distance and current_type != "knn_distance":
                        knn_mismatched.append({
                            "id": row[0],
                            "event_id": row[1],
                            "description": row[2],
                            "detector_name": row[3],
                            "current_type": current_type
                        })
                    
                logger.info(f"Found {len(knn_mismatched)} events with KNN descriptions that could use knn_distance type")
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
                
                # Check if this is likely a KNN_DISTANCE event
                is_knn_event = False
                description = event.get("description", "").lower()
                detector_name = event.get("detector_name", "").lower()
                
                if "neighbor_distance" in description or "knn" in description or "distance" in detector_name:
                    is_knn_event = True
                
                # Determine the logical drift type to store in metadata
                logical_drift_type = "unknown"
                
                if inferred_type != "unknown":
                    logical_drift_type = inferred_type
                elif is_knn_event:
                    logical_drift_type = "knn_distance"
                
                # Only update if we determined a valid logical type
                if logical_drift_type != "unknown":
                    try:
                        old_type = event.get("current_type", "unknown")
                        
                        # Determine the best valid type to use from the database
                        # This should match the logical drift type if possible
                        new_type = "unknown"  # Default fallback
                        
                        # Try to use the logical drift type if it's a valid enum value
                        if logical_drift_type in valid_drift_types:
                            new_type = logical_drift_type
                            logger.info(f"Drift type '{logical_drift_type}' is valid in database")
                        # Otherwise use a safe valid type
                        elif "distribution" in valid_drift_types:
                            new_type = "distribution"
                        elif "confidence" in valid_drift_types:
                            new_type = "confidence"
                        elif "unknown" in valid_drift_types:
                            new_type = "unknown"
                        elif valid_drift_types:  # Use first available type if others not found
                            new_type = valid_drift_types[0]
                        
                        # Store the logical type in metadata with a display name
                        display_name = logical_drift_type.capitalize()
                        if logical_drift_type == "knn_distance":
                            display_name = "KNN Distance"
                        
                        # Update with direct SQL, storing both the type and a display name in metadata
                        db.execute(
                            text("""
                                UPDATE drift_events 
                                SET drift_type = :new_type, 
                                    event_metadata = jsonb_set(
                                        jsonb_set(
                                            COALESCE(event_metadata, '{}'::jsonb), 
                                            '{original_drift_type}', 
                                            :orig_type::jsonb
                                        ),
                                        '{drift_type_display}',
                                        :display_name::jsonb
                                    )
                                WHERE id = :id
                            """),
                            {
                                "id": event["id"],
                                "new_type": new_type,
                                "orig_type": f'"{logical_drift_type}"',
                                "display_name": f'"{display_name}"'
                            }
                        )
                        
                        repaired_count += 1
                        
                        # For KNN events, show the logical mapping, not just database value
                        display_new_type = "knn_distance" if is_knn_event else new_type
                        logger.info(f"Repaired event {event['event_id']}: {old_type} -> {display_new_type} (DB type: {new_type})")
                        
                        repaired_events.append({
                            "event_id": event["event_id"],
                            "old_type": old_type,
                            "new_type": display_new_type,
                            "db_type": new_type,
                            "is_knn_mapped": is_knn_event,
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