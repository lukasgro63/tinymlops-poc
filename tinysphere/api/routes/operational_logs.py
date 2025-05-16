# api/routes/operational_logs.py
from typing import List, Optional, Dict, Any
import logging
import fastapi
from fastapi import APIRouter, Depends, HTTPException, Query

from tinysphere.api.dependencies.db import get_db
from tinysphere.api.services.operational_logs_service import OperationalLogsService

logger = logging.getLogger(__name__)
router = APIRouter()
operational_logs_service = OperationalLogsService()

@router.get("/devices", response_model=List[str])
def list_device_ids():
    """List all device IDs that have operational logs."""
    return operational_logs_service.list_device_ids()

@router.get("/devices/{device_id}/sessions", response_model=List[str])
def list_session_ids(device_id: str):
    """List all session IDs for a specific device."""
    return operational_logs_service.list_session_ids(device_id)

@router.get("/devices/{device_id}/types", response_model=List[str])
def list_log_types(device_id: str):
    """List all log types for a specific device."""
    return operational_logs_service.list_log_types(device_id)

@router.get("/list", response_model=Dict[str, Any])
def list_logs(
    device_id: Optional[str] = None,
    session_id: Optional[str] = None,
    log_type: Optional[str] = None,
    limit: int = Query(100, ge=1, le=1000),
    offset: int = Query(0, ge=0),
    sort_order: Optional[str] = "desc",
    start_date: Optional[str] = None,
    end_date: Optional[str] = None
):
    """
    List operational logs with optional filtering.
    
    Args:
        device_id: Filter by device ID
        session_id: Filter by session ID
        log_type: Filter by log type
        limit: Maximum number of logs to return
        offset: Number of logs to skip
        sort_order: Sort direction ('asc' or 'desc'), defaults to 'desc' (newest first)
        start_date: Filter logs after this date (YYYY-MM-DD format)
        end_date: Filter logs before this date (YYYY-MM-DD format)
    """
    return operational_logs_service.list_logs(
        device_id=device_id,
        session_id=session_id,
        log_type=log_type,
        limit=limit,
        offset=offset,
        sort_order=sort_order,
        start_date=start_date,
        end_date=end_date
    )

@router.get("/url/{log_key:path}", response_model=Dict[str, str])
def get_log_url(log_key: str):
    """
    Generate a URL for accessing a specific log through the API.
    
    Args:
        log_key: Full S3 key of the log (device_id/session_id/filename)
    """
    url = operational_logs_service.get_log_url(log_key)
    if not url:
        raise HTTPException(status_code=404, detail="Log not found or error generating URL")
    
    # Return the API proxy URL
    return {"url": url}

@router.get("/log/{log_key:path}", response_class=fastapi.responses.StreamingResponse)
async def get_log(log_key: str, download: bool = False):
    """
    Stream a log file directly from MinIO through the API.
    
    Args:
        log_key: Full S3 key of the log (device_id/session_id/filename)
        download: Set to true to download the file instead of viewing it
    """
    try:
        # Get the log from MinIO
        response = operational_logs_service.s3_client.get_object(
            Bucket=operational_logs_service.bucket_name,
            Key=log_key
        )
        
        # Get filename from the log key
        filename = log_key.split("/")[-1]
        
        # Determine content type based on file extension
        content_type = "application/json"  # Default
        if log_key.lower().endswith(".jsonl"):
            content_type = "application/jsonlines"
        elif log_key.lower().endswith(".csv"):
            content_type = "text/csv"
        elif log_key.lower().endswith(".txt"):
            content_type = "text/plain"
        
        # Set content disposition based on whether we're downloading or viewing
        content_disposition = f'attachment; filename="{filename}"' if download else f'inline; filename="{filename}"'
        
        # Stream the log back
        return fastapi.responses.StreamingResponse(
            response["Body"],
            media_type=content_type,
            headers={
                "Content-Disposition": content_disposition
            }
        )
    except Exception as e:
        logger.error(f"Error retrieving log {log_key}: {e}")
        raise HTTPException(status_code=404, detail="Log not found")