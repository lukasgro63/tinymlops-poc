"""
Base models and utilities for API models
"""
from datetime import datetime

def format_datetime_with_z(dt):
    """Format a datetime with the Z suffix for UTC time"""
    if dt is None:
        return None
    return dt.strftime('%Y-%m-%dT%H:%M:%S.%f')[:-3] + 'Z'