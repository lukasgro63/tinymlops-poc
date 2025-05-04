"""
Performance monitoring middleware for the TinySphere API.
Measures request processing time and logs slow requests.
"""
import time
import logging
from fastapi import Request

logger = logging.getLogger(__name__)

async def performance_middleware(request: Request, call_next):
    """
    Middleware to measure request processing time and log slow requests.
    
    Args:
        request: The incoming request
        call_next: The next middleware or endpoint handler
        
    Returns:
        The response from downstream handlers
    """
    # Record start time
    start_time = time.time()
    
    # Get the client IP for logging
    client_host = request.client.host if request.client else "unknown"
    request_id = request.headers.get("X-Request-ID", "")
    
    # Process the request
    response = await call_next(request)
    
    # Calculate processing time
    process_time = time.time() - start_time
    process_time_ms = process_time * 1000
    
    # Add processing time to response headers
    response.headers["X-Process-Time"] = f"{process_time_ms:.2f}ms"
    
    # Log request details
    log_message = (
        f"{client_host} - "
        f"{request.method} {request.url.path} "
        f"completed in {process_time_ms:.2f}ms - "
        f"Status: {response.status_code}"
    )
    
    # Warning threshold (1 second)
    if process_time > 1.0:
        logger.warning(f"SLOW REQUEST: {log_message}")
    else:
        logger.debug(log_message)
    
    return response