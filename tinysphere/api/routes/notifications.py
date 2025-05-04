# api/routes/notifications.py
from typing import List, Optional

from fastapi import APIRouter, Depends, HTTPException, Query
from sqlalchemy.orm import Session

from tinysphere.api.dependencies.db import get_db
from tinysphere.api.models.notification import (Notification,
                                                NotificationAction,
                                                NotificationCreate,
                                                NotificationResponse,
                                                NotificationUpdate)
from tinysphere.api.services.notification_service import NotificationService

router = APIRouter()


@router.get("/", response_model=NotificationResponse)
def get_notifications(
    skip: int = 0,
    limit: int = 50,
    unread_only: bool = False,
    db: Session = Depends(get_db)
):
    """
    Get all notifications with pagination.
    
    - **skip**: Number of notifications to skip (for pagination)
    - **limit**: Maximum number of notifications to return
    - **unread_only**: If true, return only unread notifications
    """
    notifications = NotificationService.get_notifications(
        db, skip=skip, limit=limit, unread_only=unread_only
    )
    total = NotificationService.get_notifications_count(db)
    unread = NotificationService.get_notifications_count(db, unread_only=True)
    
    return {
        "items": notifications,
        "total": total,
        "unread": unread
    }


@router.get("/count")
def get_notification_counts(db: Session = Depends(get_db)):
    """Get the count of all and unread notifications."""
    total = NotificationService.get_notifications_count(db)
    unread = NotificationService.get_notifications_count(db, unread_only=True)
    
    return {
        "total": total,
        "unread": unread
    }


@router.get("/{notification_id}", response_model=Notification)
def get_notification(notification_id: int, db: Session = Depends(get_db)):
    """Get a single notification by its ID."""
    notification = NotificationService.get_notification_by_id(db, notification_id)
    
    if notification is None:
        raise HTTPException(status_code=404, detail="Notification not found")
    
    return notification


@router.post("/", response_model=Notification)
def create_notification(
    notification: NotificationCreate,
    db: Session = Depends(get_db)
):
    """Create a new notification."""
    return NotificationService.create_notification(db, notification)


@router.patch("/{notification_id}", response_model=Notification)
def update_notification(
    notification_id: int,
    notification_update: NotificationUpdate,
    db: Session = Depends(get_db)
):
    """Update a notification's read status."""
    updated_notification = NotificationService.update_notification(
        db, notification_id, notification_update
    )
    
    if updated_notification is None:
        raise HTTPException(status_code=404, detail="Notification not found")
    
    return updated_notification


@router.delete("/{notification_id}")
def delete_notification(notification_id: int, db: Session = Depends(get_db)):
    """Delete a notification."""
    result = NotificationService.delete_notification(db, notification_id)
    
    if not result:
        raise HTTPException(status_code=404, detail="Notification not found")
    
    return {"success": True, "message": "Notification deleted"}


@router.post("/actions")
def notification_actions(
    action: NotificationAction,
    db: Session = Depends(get_db)
):
    """Perform actions on multiple notifications."""
    if action.action == "mark_read":
        # Mark notifications as read
        updated = NotificationService.mark_notifications_as_read(
            db, action.notification_ids
        )
        return {"success": True, "count": updated, "action": "mark_read"}
    
    elif action.action == "mark_all_read":
        # Mark all notifications as read
        updated = NotificationService.mark_all_as_read(db)
        return {"success": True, "count": updated, "action": "mark_all_read"}
    
    elif action.action == "delete":
        # Delete notifications
        deleted = NotificationService.delete_notifications(
            db, action.notification_ids
        )
        return {"success": True, "count": deleted, "action": "delete"}
    
    else:
        raise HTTPException(
            status_code=400,
            detail=f"Unknown action: {action.action}"
        )