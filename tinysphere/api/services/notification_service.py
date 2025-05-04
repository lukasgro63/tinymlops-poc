# api/services/notification_service.py
import logging
from datetime import datetime, timezone
from typing import Dict, List, Optional, Union

from sqlalchemy import desc, func
from sqlalchemy.orm import Session

from tinysphere.api.models.notification import (NotificationCreate,
                                                NotificationUpdate)
from tinysphere.db.models import Notification, NotificationType

logger = logging.getLogger(__name__)


class NotificationService:
    """Service for managing notifications."""
    
    @staticmethod
    def create_notification(
        db: Session,
        notification: NotificationCreate
    ) -> Notification:
        """Create a new notification."""
        db_notification = Notification(
            message=notification.message,
            notification_type=notification.notification_type,
            source=notification.source,
            source_id=notification.source_id,
            details=notification.details,
            created_at=datetime.now(timezone.utc),
            read=False
        )
        
        db.add(db_notification)
        db.commit()
        db.refresh(db_notification)
        
        logger.info(f"Created notification: {notification.message} ({notification.notification_type})")
        return db_notification
    
    @staticmethod
    def get_notifications(
        db: Session, 
        skip: int = 0, 
        limit: int = 100,
        unread_only: bool = False
    ) -> List[Notification]:
        """Get all notifications with pagination."""
        query = db.query(Notification).order_by(desc(Notification.created_at))
        
        if unread_only:
            query = query.filter(Notification.read == False)
        
        return query.offset(skip).limit(limit).all()
    
    @staticmethod
    def get_notification_by_id(db: Session, notification_id: int) -> Optional[Notification]:
        """Get a notification by its ID."""
        return db.query(Notification).filter(Notification.id == notification_id).first()
    
    @staticmethod
    def get_notifications_count(db: Session, unread_only: bool = False) -> int:
        """Get the count of notifications."""
        query = db.query(func.count(Notification.id))
        
        if unread_only:
            query = query.filter(Notification.read == False)
        
        return query.scalar()
    
    @staticmethod
    def update_notification(
        db: Session, 
        notification_id: int, 
        notification_update: NotificationUpdate
    ) -> Optional[Notification]:
        """Update a notification by its ID."""
        db_notification = NotificationService.get_notification_by_id(db, notification_id)
        
        if not db_notification:
            return None
        
        update_data = notification_update.dict(exclude_unset=True)
        
        # If marking as read and read_at is not set, set it
        if update_data.get("read") is True and "read_at" not in update_data:
            update_data["read_at"] = datetime.now(timezone.utc)
        
        for key, value in update_data.items():
            setattr(db_notification, key, value)
        
        db.commit()
        db.refresh(db_notification)
        return db_notification
    
    @staticmethod
    def delete_notification(db: Session, notification_id: int) -> bool:
        """Delete a notification by its ID."""
        db_notification = NotificationService.get_notification_by_id(db, notification_id)
        
        if not db_notification:
            return False
        
        db.delete(db_notification)
        db.commit()
        return True
    
    @staticmethod
    def mark_notifications_as_read(
        db: Session, 
        notification_ids: List[int]
    ) -> int:
        """Mark multiple notifications as read."""
        now = datetime.now(timezone.utc)
        result = db.query(Notification).filter(
            Notification.id.in_(notification_ids),
            Notification.read == False
        ).update(
            {"read": True, "read_at": now},
            synchronize_session=False
        )
        
        db.commit()
        return result  # Number of updated records
    
    @staticmethod
    def mark_all_as_read(db: Session) -> int:
        """Mark all notifications as read."""
        now = datetime.now(timezone.utc)
        result = db.query(Notification).filter(
            Notification.read == False
        ).update(
            {"read": True, "read_at": now},
            synchronize_session=False
        )
        
        db.commit()
        return result  # Number of updated records
    
    @staticmethod
    def delete_notifications(db: Session, notification_ids: List[int]) -> int:
        """Delete multiple notifications."""
        result = db.query(Notification).filter(
            Notification.id.in_(notification_ids)
        ).delete(synchronize_session=False)
        
        db.commit()
        return result  # Number of deleted records
    
    # Hilfsmethoden für häufige Ereignisse
    
    @staticmethod
    def notify_device_connected(db: Session, device_id: str, hostname: str = None) -> Notification:
        """Create a notification for a device connection event."""
        device_name = hostname or device_id
        return NotificationService.create_notification(
            db,
            NotificationCreate(
                message=f"Device {device_name} connected",
                notification_type=NotificationType.INFO,
                source="device",
                source_id=device_id,
                details={"hostname": hostname}
            )
        )
    
    @staticmethod
    def notify_package_uploaded(db: Session, package_id: str, device_id: str, package_type: str) -> Notification:
        """Create a notification for a package upload event."""
        return NotificationService.create_notification(
            db,
            NotificationCreate(
                message=f"New {package_type} package uploaded from device {device_id}",
                notification_type=NotificationType.INFO,
                source="package",
                source_id=package_id,
                details={"device_id": device_id, "package_type": package_type}
            )
        )
    
    @staticmethod
    def notify_package_processed(db: Session, package_id: str, success: bool, error_msg: str = None) -> Notification:
        """Create a notification for a package processing event."""
        if success:
            return NotificationService.create_notification(
                db,
                NotificationCreate(
                    message=f"Package {package_id} processed successfully",
                    notification_type=NotificationType.SUCCESS,
                    source="package",
                    source_id=package_id
                )
            )
        else:
            return NotificationService.create_notification(
                db,
                NotificationCreate(
                    message=f"Error processing package {package_id}",
                    notification_type=NotificationType.ERROR,
                    source="package",
                    source_id=package_id,
                    details={"error": error_msg}
                )
            )
    
    @staticmethod
    def notify_model_registered(db: Session, model_name: str, version: int) -> Notification:
        """Create a notification for a model registration event."""
        return NotificationService.create_notification(
            db,
            NotificationCreate(
                message=f"New model registered: {model_name} (version {version})",
                notification_type=NotificationType.SUCCESS,
                source="model",
                source_id=model_name,
                details={"version": version}
            )
        )