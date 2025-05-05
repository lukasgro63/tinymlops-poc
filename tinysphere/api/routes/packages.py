import json
import os
import traceback
import threading
from typing import List

import aiofiles
from fastapi import (APIRouter, Depends, File, Form,
                     HTTPException, UploadFile)
from sqlalchemy.orm import Session

from tinysphere.api.dependencies.db import get_db
from tinysphere.api.models.package import (Package, PackageUpdate,
                                           PackageUploadResponse)
from tinysphere.api.services.mlflow_service import MLflowService
from tinysphere.api.services.notification_service import NotificationService
from tinysphere.api.services.package_service import PackageService

router = APIRouter()
mlflow_service = MLflowService()

import os

import mlflow


@router.get("/mlflow-diagnostics", response_model=None)
def mlflow_diagnostics():
    """Test the connection to MLflow and provide diagnostic information."""
    try:
        # Check environment
        mlflow_url = os.environ.get("MLFLOW_TRACKING_URI", "http://mlflow:5000")
        
        # Try to connect to MLflow
        mlflow.set_tracking_uri(mlflow_url)
        
        # Get list of experiments
        experiments = mlflow.search_experiments()
        experiment_names = [exp.name for exp in experiments]
        
        # Check if we can create an experiment
        test_experiment_name = "diagnostic-test"
        if test_experiment_name not in experiment_names:
            mlflow.create_experiment(test_experiment_name)
        
        # Try to log a test run
        mlflow.set_experiment(test_experiment_name)
        with mlflow.start_run(run_name="connectivity-test") as run:
            mlflow.log_param("test_param", "test_value")
            mlflow.log_metric("test_metric", 1.0)
            run_id = run.info.run_id
        
        return {
            "status": "success",
            "mlflow_url": mlflow_url,
            "experiments": experiment_names,
            "test_run_id": run_id
        }
    except Exception as e:
        import traceback
        return {
            "status": "error",
            "message": str(e),
            "traceback": traceback.format_exc(),
            "mlflow_url": os.environ.get("MLFLOW_TRACKING_URI", "http://mlflow:5000")
        }

@router.get("/", response_model=List[Package])
def get_packages(skip: int = 0, limit: int = 100, db: Session = Depends(get_db)):
    packages = PackageService.get_all_packages(db, skip=skip, limit=limit)
    return packages

@router.get("/{package_id}", response_model=Package)
def get_package(package_id: str, db: Session = Depends(get_db)):
    db_package = PackageService.get_package_by_id(db, package_id=package_id)
    if db_package is None:
        raise HTTPException(status_code=404, detail="Package not found")
    return db_package

@router.get("/{package_id}/status")
def get_package_status(package_id: str, db: Session = Depends(get_db)):
    """Gibt den Verarbeitungsstatus eines Pakets zurück."""
    db_package = PackageService.get_package_by_id(db, package_id=package_id)
    if db_package is None:
        raise HTTPException(status_code=404, detail="Package not found")
    
    # Basisinformationen aus der Datenbank
    status_info = {
        "package_id": db_package.package_id,
        "device_id": db_package.device_id,
        "package_type": db_package.package_type,
        "is_processed": db_package.is_processed,
        "processing_status": db_package.processing_status,
        "uploaded_at": db_package.uploaded_at,
        "processed_at": db_package.processed_at,
        "error": db_package.processing_error,
        "detailed_progress": None
    }
    
    # Detaillierte Fortschrittsinformationen, falls verfügbar
    progress_data = PackageService.get_processing_progress(package_id)
    if progress_data:
        status_info["detailed_progress"] = progress_data
    
    return status_info

@router.get("/{package_id}/progress")
def get_package_progress(package_id: str, db: Session = Depends(get_db)):
    """Gibt nur den Verarbeitungsfortschritt eines Pakets zurück (für Polling)."""
    # Prüfen, ob das Paket existiert
    db_package = PackageService.get_package_by_id(db, package_id=package_id)
    if db_package is None:
        raise HTTPException(status_code=404, detail="Package not found")
    
    # Detaillierte Fortschrittsinformationen abrufen
    progress_data = PackageService.get_processing_progress(package_id)
    
    if not progress_data:
        # Wenn keine Fortschrittsdaten vorhanden sind, aber das Paket verarbeitet wurde
        if db_package.is_processed:
            return {
                "status": db_package.processing_status,
                "progress": 1.0,
                "is_complete": True,
                "message": "Processing completed",
                "error": db_package.processing_error
            }
        # Wenn keine Fortschrittsdaten vorhanden sind und das Paket nicht verarbeitet wurde
        else:
            return {
                "status": db_package.processing_status,
                "progress": 0.0,
                "is_complete": False,
                "message": "Processing not started",
                "error": None
            }
    
    # Fortschrittsdaten zurückgeben
    result = {
        "status": progress_data.get("status", db_package.processing_status),
        "progress": progress_data.get("progress", 0.0),
        "is_complete": progress_data.get("progress", 0.0) >= 1.0,
        "started_at": progress_data.get("started_at"),
        "elapsed_seconds": progress_data.get("elapsed_seconds"),
        "eta_seconds": progress_data.get("eta_seconds"),
        "message": progress_data.get("steps", [])[-1] if progress_data.get("steps") else None,
        "steps": progress_data.get("steps", [])
    }
    
    return result

@router.get("/device/{device_id}", response_model=List[Package])
def get_packages_by_device(device_id: str, skip: int = 0, limit: int = 100, db: Session = Depends(get_db)):
    packages = PackageService.get_packages_by_device(db, device_id=device_id, skip=skip, limit=limit)
    return packages

@router.post("/upload", response_model=PackageUploadResponse)
async def upload_package(
    package: UploadFile = File(...),
    metadata: str = Form(...),
    db: Session = Depends(get_db)
):
    try:
        os.makedirs("uploads", exist_ok=True)
        
        package_data = json.loads(metadata)
        
        package_id = package_data.get("package_id")
        device_id = package_data.get("device_id")
        
        if not package_id or not device_id:
            raise HTTPException(status_code=400, detail="Missing package_id or device_id in metadata")
        
        # Create device directory
        upload_dir = os.path.join("uploads", device_id)
        os.makedirs(upload_dir, exist_ok=True)
        
        # Save the file
        filename = package.filename or f"{package_id}.package"
        file_path = os.path.join(upload_dir, filename)
        
        # Asynchrones Lesen und Schreiben mit aiofiles
        content = await package.read()
        async with aiofiles.open(file_path, "wb") as buffer:
            await buffer.write(content)
        
        file_size = os.path.getsize(file_path)
        
        # Process the package with service
        db_package = PackageService.process_uploaded_package(db, package_data, file_path, file_size)
        
        # Create notification for package upload
        NotificationService.notify_package_uploaded(
            db,
            package_id=db_package.package_id,
            device_id=db_package.device_id,
            package_type=db_package.package_type
        )
        
        # Trigger package processing after upload (synchronously since we can't use BackgroundTasks)
        # Set the package status to "processing"
        PackageService.update_package(
            db, 
            db_package.package_id, 
            PackageUpdate(processing_status="processing")
        )
        
        # Create a new thread to process the package
        import threading
        
        def process_package_thread(package_id):
            # Create a new DB session for the thread
            from tinysphere.api.dependencies.db import SessionLocal
            thread_db = SessionLocal()
            
            try:
                service = MLflowService()
                service.process_package(thread_db, package_id)
            except Exception as e:
                print(f"Background processing error: {str(e)}")
                print(traceback.format_exc())
                # Set error status in case of failure
                PackageService.update_package(
                    thread_db, 
                    package_id, 
                    PackageUpdate(
                        processing_status="error",
                        processing_error=str(e)
                    )
                )
            finally:
                # Close connection
                thread_db.close()
                
        # Start a new thread to process the package
        threading.Thread(
            target=process_package_thread,
            args=(db_package.package_id,),
            daemon=True
        ).start()
        
        return PackageUploadResponse(
            package_id=db_package.package_id,
            status="success",
            message="Package uploaded successfully and processing started",
            uploaded=True  # Required field in the model
        )
    except Exception as e:
        import traceback
        print(f"Upload error: {str(e)}")
        print(traceback.format_exc())
        
        raise HTTPException(status_code=500, detail=f"Upload failed: {str(e)}")
    
@router.post("/{package_id}/process")
async def process_package(
    package_id: str,
    db: Session = Depends(get_db)
):
    try:
        db_package = PackageService.get_package_by_id(db, package_id=package_id)
        if db_package is None:
            raise HTTPException(status_code=404, detail="Package not found")
        
        # Paket-Status auf "processing" setzen
        PackageService.update_package(
            db, 
            package_id, 
            PackageUpdate(processing_status="processing")
        )
        
        # Neue DB-Session für Hintergrundaufgabe erstellen (um Konflikte zu vermeiden)
        from tinysphere.api.dependencies.db import SessionLocal
        db_copy = SessionLocal()
        
        # Verarbeitungsfunktion für Hintergrundtask definieren
        def process_package_background(db, package_id):
            try:
                service = MLflowService()
                result = service.process_package(db, package_id)
                # Status wird bereits in process_package aktualisiert
                return result
            except Exception as e:
                print(f"Background processing error: {str(e)}")
                print(traceback.format_exc())
                # Fehler im Fehlerfall setzen
                PackageService.update_package(
                    db, 
                    package_id, 
                    PackageUpdate(
                        processing_status="error",
                        processing_error=str(e)
                    )
                )
            finally:
                # Verbindung schließen
                db.close()
        
        # Verarbeitung im Hintergrund starten mit Thread
        threading.Thread(
            target=process_package_background,
            args=(db_copy, package_id),
            daemon=True
        ).start()
        
        return {
            "status": "accepted",
            "message": f"Processing package {package_id} in the background",
            "package_id": package_id
        }
    except Exception as e:
        print(f"Error setting up background processing: {str(e)}")
        print(traceback.format_exc())
        raise HTTPException(status_code=500, detail=f"Package processing failed: {str(e)}")

@router.post("/process-pending")
async def process_pending_packages(
    db: Session = Depends(get_db),
    limit: int = 10
):
    try:
        # Hole alle ausstehenden Pakete
        pending_packages = PackageService.get_pending_packages(db, limit=limit)
        
        if not pending_packages:
            return {
                "status": "info",
                "message": "No pending packages found",
                "count": 0
            }
        
        # Neue DB-Session für Hintergrundaufgaben
        from tinysphere.api.dependencies.db import SessionLocal
        db_copy = SessionLocal()
        
        # Für jedes ausstehende Paket eine Hintergrundaufgabe starten
        package_ids = []
        for package in pending_packages:
            # Status auf "queued" setzen
            PackageService.update_package(
                db,
                package.package_id,
                PackageUpdate(processing_status="queued")
            )
            package_ids.append(package.package_id)
        
        # Hintergrundtask zum Verarbeiten aller Pakete
        def process_packages_background(db, package_ids):
            try:
                service = MLflowService()
                for package_id in package_ids:
                    try:
                        service.process_package(db, package_id)
                    except Exception as e:
                        print(f"Error processing package {package_id}: {e}")
                        # Einzelne Paketfehler abfangen, aber weitermachen
                        continue
            finally:
                db.close()
        
        # Starte die Verarbeitung im Hintergrund mit Thread
        threading.Thread(
            target=process_packages_background,
            args=(db_copy, package_ids),
            daemon=True
        ).start()
        
        return {
            "status": "accepted",
            "message": f"Processing {len(pending_packages)} packages in the background",
            "packages": package_ids,
            "count": len(pending_packages)
        }
    except Exception as e:
        print(f"Error setting up background processing: {str(e)}")
        print(traceback.format_exc())
        raise HTTPException(status_code=500, detail=f"Package processing setup failed: {str(e)}")

