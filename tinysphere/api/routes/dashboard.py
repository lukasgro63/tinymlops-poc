from datetime import datetime, timedelta, timezone
from typing import Any, Dict, List

import mlflow
from fastapi import APIRouter, Depends, HTTPException
from sqlalchemy import func
from sqlalchemy.orm import Session

from tinysphere.api.dependencies.db import get_db
from tinysphere.api.services.device_service import DeviceService
from tinysphere.api.services.package_service import PackageService
from tinysphere.db.models import Device, Package

router = APIRouter()

@router.get("/status")
def get_system_status(db: Session = Depends(get_db)):
    """Liefert den Gesamtstatus des Systems."""
    try:
        devices = DeviceService.get_all_devices(db)
        packages = PackageService.get_all_packages(db)
        
        active_devices = len([d for d in devices if d.is_active])
        
        package_types = {}
        for p in packages:
            if p.package_type not in package_types:
                package_types[p.package_type] = 0
            package_types[p.package_type] += 1
        
        processed = len([p for p in packages if p.is_processed])
        pending = len(packages) - processed
        
        return {
            "status": "operational",
            "statistics": {
                "total_devices": len(devices),
                "active_devices": active_devices,
                "total_packages": len(packages),
                "processed_packages": processed,
                "pending_packages": pending,
                "package_types": package_types
            }
        }
    except Exception as e:
        return {
            "status": "error",
            "message": f"Error fetching system status: {str(e)}"
        }

@router.get("/devices/summary")
def get_devices_summary(db: Session = Depends(get_db)):
    """Liefert eine Zusammenfassung aller Geräte."""
    try:
        devices = DeviceService.get_all_devices(db)
        
        device_summaries = []
        for device in devices:
            packages = PackageService.get_packages_by_device(db, device.device_id)
            
            latest_package = None
            if packages:
                latest_package = max([p.uploaded_at for p in packages])
            
            device_summaries.append({
                "device_id": device.device_id,
                "hostname": device.hostname,
                "platform": device.platform,
                "is_active": device.is_active,
                "last_sync_time": device.last_sync_time,
                "package_count": len(packages),
                "latest_package": latest_package
            })
        
        return device_summaries
    except Exception as e:
        raise HTTPException(status_code=500, detail=f"Error fetching device summary: {str(e)}")

@router.get("/models/summary")
def get_models_summary(db: Session = Depends(get_db)):
    """Liefert eine Zusammenfassung aller Modelle in der Registry."""
    try:
        # MLflow Client initialisieren
        mlflow_client = mlflow.tracking.MlflowClient()
        
        # Alle registrierten Modelle abrufen
        registered_models = mlflow_client.search_registered_models()
        
        models_summary = []
        for model in registered_models:
            # Neueste Versionen abrufen
            latest_versions = mlflow_client.get_latest_versions(model.name)
            
            # Aktive Versionen zählen (Production, Staging)
            production_version = next((v for v in latest_versions if v.current_stage == "Production"), None)
            staging_version = next((v for v in latest_versions if v.current_stage == "Staging"), None)
            
            # Modell-Zusammenfassung erstellen
            model_summary = {
                "name": model.name,
                "total_versions": len(latest_versions),
                "latest_version": max(v.version for v in latest_versions) if latest_versions else 0,
                "has_production": production_version is not None,
                "has_staging": staging_version is not None,
                "production_version": production_version.version if production_version else None,
                "staging_version": staging_version.version if staging_version else None,
                "device_id": model.name.split('-')[0] if '-' in model.name else None,
            }
            
            models_summary.append(model_summary)
        
        return models_summary
    except Exception as e:
        raise HTTPException(status_code=500, detail=f"Error fetching models summary: {str(e)}")

@router.get("/models/device/{device_id}")
def get_device_models(device_id: str, db: Session = Depends(get_db)):
    """Liefert alle Modelle für ein bestimmtes Gerät."""
    try:
        # MLflow Client initialisieren
        mlflow_client = mlflow.tracking.MlflowClient()
        
        # Alle registrierten Modelle abrufen
        registered_models = mlflow_client.search_registered_models()
        
        # Nach Modellen filtern, die zum angegebenen Gerät gehören
        device_models = []
        for model in registered_models:
            if model.name.startswith(f"{device_id}-"):
                # Neueste Versionen abrufen
                latest_versions = mlflow_client.get_latest_versions(model.name)
                
                # Details für jede Version sammeln
                versions = []
                for version in latest_versions:
                    # Run abrufen, um Metriken zu bekommen
                    run = mlflow_client.get_run(version.run_id) if version.run_id else None
                    
                    versions.append({
                        "version": version.version,
                        "stage": version.current_stage,
                        "run_id": version.run_id,
                        "metrics": run.data.metrics if run and hasattr(run, "data") else {},
                        "creation_timestamp": version.creation_timestamp,
                        "last_updated_timestamp": version.last_updated_timestamp
                    })
                
                device_models.append({
                    "name": model.name,
                    "versions": versions
                })
        
        return device_models
    except Exception as e:
        raise HTTPException(status_code=500, detail=f"Error fetching device models: {str(e)}")

@router.get("/packages/activity")
def get_package_activity(limit: int = 10, db: Session = Depends(get_db)):
    """Liefert die neuesten Paketaktivitäten."""
    try:
        # Neueste Pakete abrufen, sortiert nach Upload-Zeit
        packages = db.query(Package).order_by(Package.uploaded_at.desc()).limit(limit).all()
        
        activities = []
        for package in packages:
            device = db.query(Device).filter(Device.device_id == package.device_id).first()
            
            from api.models.base import format_datetime_with_z

            # Die datetime-Objekte manuell mit Z-Suffix formatieren
            activities.append({
                "package_id": package.package_id,
                "device_id": package.device_id,
                "device_name": device.hostname if device else "Unknown",
                "package_type": package.package_type,
                "uploaded_at": format_datetime_with_z(package.uploaded_at) if package.uploaded_at else None,
                "processed_at": format_datetime_with_z(package.processed_at) if package.processed_at else None,
                "is_processed": package.is_processed,
                "processing_status": package.processing_status
            })
        
        return activities
    except Exception as e:
        raise HTTPException(status_code=500, detail=f"Error fetching package activity: {str(e)}")

@router.get("/packages/statistics")
def get_package_statistics(db: Session = Depends(get_db)):
    """Liefert Statistiken zu hochgeladenen Paketen."""
    try:
        # Statistiken über Pakettypen abrufen
        package_types = (
            db.query(Package.package_type, func.count(Package.id))
            .group_by(Package.package_type)
            .all()
        )
        
        # Statistiken über Verarbeitungsstatus abrufen
        processing_status = (
            db.query(Package.processing_status, func.count(Package.id))
            .group_by(Package.processing_status)
            .all()
        )
        
        # Statistiken über Geräte abrufen
        device_packages = (
            db.query(Package.device_id, func.count(Package.id))
            .group_by(Package.device_id)
            .all()
        )
        
        # Statistiken über die Zeit abrufen (pro Monat, vereinfacht)
        monthly_uploads = (
            db.query(func.date_trunc('month', Package.uploaded_at), func.count(Package.id))
            .group_by(func.date_trunc('month', Package.uploaded_at))
            .all()
        )
        
        return {
            "package_types": {t[0]: t[1] for t in package_types},
            "processing_status": {s[0]: s[1] for s in processing_status},
            "devices": {d[0]: d[1] for d in device_packages},
            "monthly_uploads": {str(m[0].date()): m[1] for m in monthly_uploads if m[0]}
        }
    except Exception as e:
        raise HTTPException(status_code=500, detail=f"Error fetching package statistics: {str(e)}")
    
@router.get("/packages/timeline")
def get_package_timeline(period: str = "week", db: Session = Depends(get_db)):
    """Returns package upload statistics over time for charts."""
    try:
        # Default date range
        days_to_include = 7
        if period == "day":
            days_to_include = 1
        elif period == "week":
            days_to_include = 7
        elif period == "month":
            days_to_include = 30
            
        start_date = datetime.now(timezone.utc) - timedelta(days=days_to_include)
        
        # SQL query to group by day and package type
        result = db.query(
            func.date_trunc('day', Package.uploaded_at).label('date'),
            Package.package_type,
            func.count(Package.id).label('count')
        ).filter(
            Package.uploaded_at >= start_date
        ).group_by(
            func.date_trunc('day', Package.uploaded_at),
            Package.package_type
        ).order_by(
            func.date_trunc('day', Package.uploaded_at)
        ).all()
        
        # Transform into the format needed by the chart
        timeline_data = {}
        for date, package_type, count in result:
            date_str = date.strftime('%Y-%m-%d')
            if date_str not in timeline_data:
                timeline_data[date_str] = {
                    'date': date_str,
                    'models': 0,
                    'metrics': 0,
                    'data_logs': 0
                }
            
            # Map package types to chart categories
            if package_type == 'model':
                timeline_data[date_str]['models'] += count
            elif package_type == 'metrics':
                timeline_data[date_str]['metrics'] += count
            elif package_type == 'data_log':
                timeline_data[date_str]['data_logs'] += count
        
        # Fill in missing dates with zero values
        for day_offset in range(days_to_include):
            current_date = (datetime.now(timezone.utc) - timedelta(days=day_offset)).strftime('%Y-%m-%d')
            if current_date not in timeline_data:
                timeline_data[current_date] = {
                    'date': current_date,
                    'models': 0,
                    'metrics': 0,
                    'data_logs': 0
                }
        
        # Convert to list and sort by date
        timeline_list = list(timeline_data.values())
        timeline_list.sort(key=lambda x: x['date'])
        
        return timeline_list
        
    except Exception as e:
        raise HTTPException(status_code=500, detail=f"Error fetching package timeline: {str(e)}")

@router.get("/models/performance")
def get_models_performance(metric: str = "accuracy", db: Session = Depends(get_db)):
    """Fetch performance metrics for models from MLflow."""
    try:
        client = mlflow.tracking.MlflowClient()
        
        # Get all registered models
        models = client.search_registered_models()
        
        performance_data = []
        
        for model in models:
            # Get latest versions
            versions = client.get_latest_versions(model.name)
            
            for version in versions:
                # Skip if no run ID
                if not version.run_id:
                    continue
                    
                try:
                    # Get run data
                    run = client.get_run(version.run_id)
                    
                    # Check if the run has the requested metric
                    if metric in run.data.metrics:
                        performance_data.append({
                            "model_name": model.name,
                            "version": version.version,
                            "stage": version.current_stage,
                            "metric_name": metric,
                            "value": run.data.metrics[metric],
                            "timestamp": run.info.start_time
                        })
                except Exception as run_err:
                    continue
                    
        return performance_data
    
    except Exception as e:
        raise HTTPException(status_code=500, detail=f"Error fetching model performance: {str(e)}")