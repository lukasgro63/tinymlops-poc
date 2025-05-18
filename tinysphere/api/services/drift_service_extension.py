"""
Extensions to the Drift Service API Implementation
-------------------------------------------------
Contains additional methods for the DriftService class to support
dashboard charts for drift event visualization.
"""

import logging
from datetime import datetime, timedelta, timezone
from typing import Dict, List, Optional, Any, Dict

from sqlalchemy import func, desc, and_
from sqlalchemy.orm import Session
from sqlalchemy.sql import text

from tinysphere.db.models import DriftEvent, DriftSample, DriftValidation, Device, DriftStatus, DriftType

logger = logging.getLogger(__name__)

class DriftServiceExtension:
    """
    These methods can be added to the DriftService class.
    """
    
    @staticmethod
    def get_drift_statistics_by_device(
        db: Session, 
        days: int = 30,
        drift_type: Optional[str] = None,
        limit: int = 10  # Limit the number of devices in the result
    ) -> List[Dict[str, Any]]:
        """
        Get drift event statistics grouped by device.
        
        Args:
            db: Database session
            days: Number of days to include in time series
            drift_type: Optional filter by drift type
            limit: Maximum number of devices to include (sorted by total events)
            
        Returns:
            List of device statistics with status counts
        """
        try:
            # Set start date for filtering
            start_date = datetime.now(timezone.utc) - timedelta(days=days)
            
            # Base query to filter by date range
            base_query = db.query(DriftEvent).filter(DriftEvent.timestamp >= start_date)
            
            # Apply drift type filter - simple approach without complex SQL
            if drift_type:
                try:
                    # Import the DriftService module here to avoid circular imports
                    from tinysphere.api.services.drift_service import DriftService
                    
                    # Simply get all drift events for the period and filter them in Python
                    all_events = db.query(DriftEvent.id, DriftEvent.drift_type).filter(
                        DriftEvent.timestamp >= start_date
                    ).all()
                    
                    # Filter events by drift type string comparison with extra KNN_DISTANCE handling
                    drift_type_lower = drift_type.lower()
                    matching_ids = []
                    
                    # Special handling for KNN_DISTANCE with alternative formats
                    is_knn_search = drift_type_lower in ["knn_distance", "knn", "knndistance", "knn distance"]
                    
                    for event_id, event_drift_type in all_events:
                        # Get value and convert to lowercase for comparison
                        event_type_value = event_drift_type.value.lower() if hasattr(event_drift_type, 'value') else str(event_drift_type).lower()
                        
                        # First check exact match
                        if event_type_value == drift_type_lower:
                            matching_ids.append(event_id)
                        # Special handling for KNN_DISTANCE to catch variations
                        elif is_knn_search and ("knn" in event_type_value or "distance" in event_type_value):
                            matching_ids.append(event_id)
                    
                    if matching_ids:
                        # Filter base query to only include these IDs
                        base_query = base_query.filter(DriftEvent.id.in_(matching_ids))
                        logger.info(f"Found {len(matching_ids)} events with drift_type '{drift_type_lower}'")
                    else:
                        logger.warning(f"No events found with drift_type '{drift_type_lower}'")
                except Exception as e:
                    logger.error(f"Error applying drift type filter (simple approach): {e}")
            
            # Step 1: Get all devices with drift events
            device_counts = {}
            try:
                # First query: Count events by device
                device_query = base_query.with_entities(
                    DriftEvent.device_id,
                    func.count(DriftEvent.id).label('total')
                ).group_by(DriftEvent.device_id)
                
                # Convert to dictionary of device_id -> total_count
                device_totals = {device_id: total for device_id, total in device_query}
                
                # Sort devices by total events and take top ones based on limit
                top_devices = sorted(
                    device_totals.items(), 
                    key=lambda x: x[1], 
                    reverse=True
                )[:limit]
                
                # Create initial structure with device IDs and total counts
                for device_id, total in top_devices:
                    device_counts[device_id] = {
                        'device_id': device_id,
                        'total': total,
                        'pending': 0,
                        'validated': 0,
                        'rejected': 0,
                        'resolved': 0,
                        'ignored': 0
                    }
            except Exception as device_err:
                logger.error(f"Error getting device counts: {device_err}")
            
            # No data found
            if not device_counts:
                return []
            
            # Step 2: For each device, get counts by status
            try:
                # Second query: Count events by device and status
                status_query = base_query.with_entities(
                    DriftEvent.device_id,
                    DriftEvent.status,
                    func.count(DriftEvent.id).label('count')
                ).filter(
                    DriftEvent.device_id.in_(device_counts.keys())
                ).group_by(
                    DriftEvent.device_id,
                    DriftEvent.status
                )
                
                # Update device_counts with status breakdown
                for device_id, status, count in status_query:
                    if device_id in device_counts:
                        status_key = status.value  # Convert enum to string
                        device_counts[device_id][status_key] = count
            except Exception as status_err:
                logger.error(f"Error getting status counts: {status_err}")
            
            # Step 3: Add device names where available
            try:
                # Get device info for the top devices
                devices = db.query(Device).filter(
                    Device.device_id.in_(device_counts.keys())
                ).all()
                
                # Update with device names
                for device in devices:
                    if device.device_id in device_counts:
                        name = device.hostname or device.device_id
                        device_counts[device.device_id]['name'] = name
            except Exception as device_err:
                logger.error(f"Error getting device info: {device_err}")
            
            # Convert to list and sort by total events (descending)
            result = list(device_counts.values())
            result.sort(key=lambda x: x['total'], reverse=True)
            
            return result
            
        except Exception as e:
            logger.error(f"Error in get_drift_statistics_by_device: {e}")
            return []
    
    @staticmethod
    def get_drift_timeline_by_device(
        db: Session, 
        period: str = "week",
        drift_type: Optional[str] = None,
        limit: int = 10  # Limit the number of devices in the result
    ) -> List[Dict[str, Any]]:
        """
        Get drift events timeline grouped by device and date.
        
        Args:
            db: Database session
            period: Time period - "day", "week", or "month"
            drift_type: Optional filter by drift type
            limit: Maximum number of devices to include in timeline
            
        Returns:
            List of daily counts with device breakdown
        """
        try:
            # Determine date range based on period
            days_to_include = 7  # Default for week
            if period == "day":
                days_to_include = 1
            elif period == "week":
                days_to_include = 7
            elif period == "month":
                days_to_include = 30
            
            start_date = datetime.now(timezone.utc) - timedelta(days=days_to_include)
            
            # Base query to filter by date range
            base_query = db.query(DriftEvent).filter(DriftEvent.timestamp >= start_date)
            
            # Apply drift type filter - simple approach without complex SQL
            if drift_type:
                try:
                    # Import the DriftService module here to avoid circular imports
                    from tinysphere.api.services.drift_service import DriftService
                    
                    # Simply get all drift events for the period and filter them in Python
                    all_events = db.query(DriftEvent.id, DriftEvent.drift_type).filter(
                        DriftEvent.timestamp >= start_date
                    ).all()
                    
                    # Filter events by drift type string comparison with extra KNN_DISTANCE handling
                    drift_type_lower = drift_type.lower()
                    matching_ids = []
                    
                    # Special handling for KNN_DISTANCE with alternative formats
                    is_knn_search = drift_type_lower in ["knn_distance", "knn", "knndistance", "knn distance"]
                    
                    for event_id, event_drift_type in all_events:
                        # Get value and convert to lowercase for comparison
                        event_type_value = event_drift_type.value.lower() if hasattr(event_drift_type, 'value') else str(event_drift_type).lower()
                        
                        # First check exact match
                        if event_type_value == drift_type_lower:
                            matching_ids.append(event_id)
                        # Special handling for KNN_DISTANCE to catch variations
                        elif is_knn_search and ("knn" in event_type_value or "distance" in event_type_value):
                            matching_ids.append(event_id)
                    
                    if matching_ids:
                        # Filter base query to only include these IDs
                        base_query = base_query.filter(DriftEvent.id.in_(matching_ids))
                        logger.info(f"Found {len(matching_ids)} events with drift_type '{drift_type_lower}'")
                    else:
                        logger.warning(f"No events found with drift_type '{drift_type_lower}'")
                except Exception as e:
                    logger.error(f"Error applying drift type filter (simple approach): {e}")
            
            # Step 1: Get all devices with drift events and count
            device_counts = {}
            try:
                # First query: Count events by device
                device_query = base_query.with_entities(
                    DriftEvent.device_id,
                    func.count(DriftEvent.id).label('total')
                ).group_by(DriftEvent.device_id)
                
                # Sort by total count and keep top devices
                top_devices = [
                    device_id for device_id, _ in sorted(
                        device_query, 
                        key=lambda x: x[1], 
                        reverse=True
                    )[:limit]
                ]
                
                # No data found
                if not top_devices:
                    return []
                    
                # Keep track of these devices
                device_counts = {device_id: 0 for device_id in top_devices}
            except Exception as device_err:
                logger.error(f"Error getting top devices: {device_err}")
                return []
            
            # Step 2: Get daily counts for each device
            timeline_data = {}
            try:
                # Query for events by day and device
                daily_query = base_query.with_entities(
                    func.date_trunc('day', DriftEvent.timestamp).label('day'),
                    DriftEvent.device_id,
                    func.count(DriftEvent.id).label('count')
                ).filter(
                    DriftEvent.device_id.in_(device_counts.keys())
                ).group_by(
                    func.date_trunc('day', DriftEvent.timestamp),
                    DriftEvent.device_id
                ).order_by(
                    func.date_trunc('day', DriftEvent.timestamp)
                )
                
                # Process results
                for day, device_id, count in daily_query:
                    date_str = day.strftime('%Y-%m-%d')
                    
                    # Create entry for this date if it doesn't exist
                    if date_str not in timeline_data:
                        timeline_data[date_str] = {
                            'date': date_str,
                            **{device_id: 0 for device_id in device_counts.keys()}
                        }
                    
                    # Add count for this device on this day
                    timeline_data[date_str][device_id] = count
            except Exception as timeline_err:
                logger.error(f"Error getting daily device counts: {timeline_err}")
            
            # Step 3: Fill in missing dates with zero values
            for day_offset in range(days_to_include):
                current_date = (datetime.now(timezone.utc) - timedelta(days=day_offset)).strftime('%Y-%m-%d')
                if current_date not in timeline_data:
                    timeline_data[current_date] = {
                        'date': current_date,
                        **{device_id: 0 for device_id in device_counts.keys()}
                    }
            
            # Convert to list and sort by date
            result = list(timeline_data.values())
            result.sort(key=lambda x: x['date'])
            
            return result
            
        except Exception as e:
            logger.error(f"Error in get_drift_timeline_by_device: {e}")
            return []