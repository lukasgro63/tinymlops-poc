# api/routes/drift_dashboard.py
import logging
from datetime import datetime, timedelta
from typing import List, Optional, Dict, Any

from fastapi import APIRouter, Depends, HTTPException, Query
from sqlalchemy.orm import Session

from tinysphere.api.dependencies.db import get_db
from tinysphere.api.services.drift_service_extension import DriftServiceExtension

logger = logging.getLogger(__name__)

router = APIRouter()

@router.get("/statistics/by-device")
def get_drift_statistics_by_device(
    days: int = 30,
    drift_type: Optional[str] = None,
    limit: int = 10,
    db: Session = Depends(get_db)
):
    """
    Get drift event statistics grouped by device.
    
    Args:
        days: Number of days to include
        drift_type: Optional filter by drift type
        limit: Maximum number of devices to include
    
    Returns:
        List of device statistics with status counts
    """
    try:
        result = DriftServiceExtension.get_drift_statistics_by_device(
            db, days=days, drift_type=drift_type, limit=limit
        )
        return result
    except Exception as e:
        logger.error(f"Error in get_drift_statistics_by_device: {e}")
        # Return empty list instead of raising an error
        return []

@router.get("/timeline/by-device")
def get_drift_timeline_by_device(
    period: str = "week",
    drift_type: Optional[str] = None,
    limit: int = 10,
    db: Session = Depends(get_db)
):
    """
    Get drift events timeline grouped by device and date.
    
    Args:
        period: Time period - "day", "week", or "month"
        drift_type: Optional filter by drift type
        limit: Maximum number of devices to include
    
    Returns:
        List of daily counts with device breakdown
    """
    try:
        # Validate period parameter
        if period not in ["day", "week", "month"]:
            period = "week"  # Default to week if invalid
            
        result = DriftServiceExtension.get_drift_timeline_by_device(
            db, period=period, drift_type=drift_type, limit=limit
        )
        return result
    except Exception as e:
        logger.error(f"Error in get_drift_timeline_by_device: {e}")
        # Return empty list instead of raising an error
        return []