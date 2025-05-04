import os
import shutil
import time
from datetime import datetime, timezone
from pathlib import Path
from typing import Any, BinaryIO, Dict, List, Optional

# In-Memory-Cache für Verarbeitungsfortschritt
# Format: {package_id: {"status": str, "progress": float, "started_at": float, "steps": List[str]}}
processing_status_cache = {}

from sqlalchemy.orm import Session

from tinysphere.api.models.package import PackageCreate, PackageUpdate
from tinysphere.api.services.device_service import DeviceService
from tinysphere.db.models import Package


class PackageService:
    @staticmethod
    def get_all_packages(db: Session, skip: int = 0, limit: int = 100) -> List[Package]:
        return db.query(Package).offset(skip).limit(limit).all()
    
    @staticmethod
    def get_packages_by_device(db: Session, device_id: str, skip: int = 0, limit: int = 100) -> List[Package]:
        return db.query(Package).filter(Package.device_id == device_id).offset(skip).limit(limit).all()
    
    @staticmethod
    def get_package_by_id(db: Session, package_id: str) -> Optional[Package]:
        return db.query(Package).filter(Package.package_id == package_id).first()
    
    @staticmethod
    def create_package(db: Session, package: PackageCreate) -> Package:
        db_package = Package(
            package_id=package.package_id,
            device_id=package.device_id,
            package_type=package.package_type,
            filename=package.filename,
            file_path=package.file_path,
            file_hash=package.file_hash,
            file_size=package.file_size,
            package_metadata=package.package_metadata,
            uploaded_at=datetime.now(timezone.utc),
            is_processed=False,
            processing_status="pending"
        )
        db.add(db_package)
        db.commit()
        db.refresh(db_package)
        
        DeviceService.update_sync_time(db, package.device_id)
        
        return db_package
    
    @staticmethod
    def update_package(db: Session, package_id: str, package_update: PackageUpdate) -> Optional[Package]:
        db_package = PackageService.get_package_by_id(db, package_id)
        if db_package is None:
            return None
        
        update_data = package_update.dict(exclude_unset=True)
        
        for key, value in update_data.items():
            setattr(db_package, key, value)
        
        db.commit()
        db.refresh(db_package)
        return db_package
    
    @staticmethod
    def save_package_file(file: BinaryIO, package_id: str, device_id: str, filename: str) -> str:
        upload_dir = Path("uploads")
        device_dir = upload_dir / device_id
        os.makedirs(device_dir, exist_ok=True)
        
        file_path = device_dir / filename
        with open(file_path, "wb") as buffer:
            shutil.copyfileobj(file, buffer)
        
        return str(file_path)
    
    @staticmethod
    def process_uploaded_package(db: Session, package_data: Dict[str, Any], file_path: str, file_size: int) -> Package:
        package_id = package_data.get("package_id")
        device_id = package_data.get("device_id")
        package_type = package_data.get("package_type", "unknown")
        file_hash = package_data.get("hash")
        filename = os.path.basename(file_path)
        
        package_create = PackageCreate(
            package_id=package_id,
            device_id=device_id,
            package_type=package_type,
            filename=filename,
            file_path=file_path,
            file_hash=file_hash,
            file_size=file_size,
            package_metadata=package_data
        )
        
        return PackageService.create_package(db, package_create)
    
    @staticmethod
    def get_pending_packages(db: Session, limit: int = 100) -> List[Package]:
        return db.query(Package).filter(Package.is_processed == False).limit(limit).all()
        
    @staticmethod
    def set_processing_progress(package_id: str, progress: float, status: str = None, step: str = None) -> None:
        """
        Setzt den Verarbeitungsfortschritt für ein Paket im In-Memory-Cache.
        
        Args:
            package_id: ID des Pakets
            progress: Fortschritt zwischen 0.0 und 1.0
            status: Optionaler Statustext
            step: Optionale Beschreibung des aktuellen Verarbeitungsschritts
        """
        if package_id not in processing_status_cache:
            processing_status_cache[package_id] = {
                "status": "initializing",
                "progress": 0.0,
                "started_at": time.time(),
                "steps": []
            }
        
        # Cache aktualisieren
        if status:
            processing_status_cache[package_id]["status"] = status
        
        processing_status_cache[package_id]["progress"] = max(0.0, min(1.0, progress))
        
        if step:
            processing_status_cache[package_id]["steps"].append(step)
            
    @staticmethod
    def get_processing_progress(package_id: str) -> Optional[Dict[str, Any]]:
        """
        Gibt den aktuellen Verarbeitungsfortschritt für ein Paket zurück.
        
        Args:
            package_id: ID des Pakets
            
        Returns:
            Dict mit Fortschrittsinformationen oder None, wenn kein Fortschritt gefunden wurde
        """
        if package_id not in processing_status_cache:
            return None
            
        progress_data = processing_status_cache[package_id].copy()
        
        # Laufzeit berechnen
        if "started_at" in progress_data:
            elapsed_time = time.time() - progress_data["started_at"]
            progress_data["elapsed_seconds"] = round(elapsed_time, 2)
            
            # Geschätzte verbleibende Zeit berechnen (ETA)
            progress = progress_data.get("progress", 0)
            if progress > 0:
                eta = (elapsed_time / progress) * (1 - progress)
                progress_data["eta_seconds"] = round(eta, 2)
                
        return progress_data
        
    @staticmethod
    def clear_processing_progress(package_id: str) -> None:
        """
        Entfernt die Fortschrittsinformationen für ein Paket aus dem Cache.
        
        Args:
            package_id: ID des Pakets
        """
        if package_id in processing_status_cache:
            del processing_status_cache[package_id]