# api/models/notification.py
from datetime import datetime
from typing import Dict, Optional, Any, List

from pydantic import BaseModel

from tinysphere.db.models import NotificationType


class NotificationBase(BaseModel):
    message: str
    notification_type: NotificationType = NotificationType.INFO
    source: Optional[str] = None
    source_id: Optional[str] = None
    details: Optional[Dict[str, Any]] = None


class NotificationCreate(NotificationBase):
    pass


class NotificationUpdate(BaseModel):
    read: Optional[bool] = None
    read_at: Optional[datetime] = None


class NotificationInDB(NotificationBase):
    id: int
    created_at: datetime
    read: bool
    read_at: Optional[datetime] = None
    
    class Config:
        from_attributes = True


class Notification(NotificationInDB):
    pass


class NotificationResponse(BaseModel):
    items: List[Notification]
    total: int
    unread: int


class NotificationCountResponse(BaseModel):
    total: int
    unread: int


class NotificationActionResponse(BaseModel):
    success: bool
    count: int
    action: str


class NotificationAction(BaseModel):
    action: str  # "mark_read", "mark_all_read", "delete"
    notification_ids: Optional[List[int]] = None