import os
from datetime import datetime, timedelta, timezone
from typing import Any, Dict, List, Optional

import mlflow
import pandas as pd
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
        # Safer approach to querying system data with exception handling
        # for individual components to prevent total failure
        
        # Get device data
        try:
            devices = DeviceService.get_all_devices(db)
            active_devices = len([d for d in devices if d.is_active])
            total_devices = len(devices)
        except Exception as device_err:
            import traceback
            print(f"Error getting device data: {device_err}")
            print(traceback.format_exc())
            active_devices = 0
            total_devices = 0
        
        # Get package data
        try:
            # Get all packages without limit
            packages = PackageService.get_all_packages(db, limit=0)
            
            package_types = {}
            for p in packages:
                if p.package_type not in package_types:
                    package_types[p.package_type] = 0
                package_types[p.package_type] += 1
            
            processed = len([p for p in packages if p.is_processed])
            pending = len(packages) - processed
            total_packages = len(packages)
        except Exception as package_err:
            import traceback
            print(f"Error getting package data: {package_err}")
            print(traceback.format_exc())
            package_types = {}
            processed = 0
            pending = 0
            total_packages = 0
        
        # Get MLflow data
        try:
            import mlflow
            mlflow_uri = os.environ.get("MLFLOW_TRACKING_URI", "http://mlflow:5000")
            # Set connection timeout to avoid hanging if MLflow is down
            os.environ["MLFLOW_HTTP_REQUEST_TIMEOUT"] = "5"  # 5 seconds timeout
            mlflow.set_tracking_uri(mlflow_uri)
            experiments = mlflow.search_experiments()
            mlflow_status = "connected"
            mlflow_experiments = len(experiments)
        except Exception as mlflow_err:
            import traceback
            print(f"Error connecting to MLflow: {mlflow_err}")
            print(traceback.format_exc()) 
            mlflow_status = "disconnected"
            mlflow_experiments = 0
        
        return {
            "status": "operational",
            "statistics": {
                "total_devices": total_devices,
                "active_devices": active_devices,
                "total_packages": total_packages,
                "processed_packages": processed,
                "pending_packages": pending,
                "package_types": package_types,
                "mlflow_status": mlflow_status,
                "mlflow_experiments": mlflow_experiments
            }
        }
    except Exception as e:
        import traceback
        print(f"Critical error in get_system_status: {e}")
        print(traceback.format_exc())
        return {
            "status": "error",
            "message": f"Error fetching system status: {str(e)}"
        }

@router.get("/devices/summary")
def get_devices_summary(db: Session = Depends(get_db)):
    """Liefert eine Zusammenfassung aller Geräte."""
    try:
        try:
            # Try to import format function from base.py
            try:
                from tinysphere.api.models.base import format_datetime_with_z as format_datetime
            except ImportError:
                # Fallback to local implementation
                def format_datetime(dt):
                    if dt is None:
                        return None
                    return dt.strftime('%Y-%m-%dT%H:%M:%S.%f')[:-3] + 'Z'

            devices = DeviceService.get_all_devices(db)

            # Try to get MLflow data for each device
            mlflow_metrics = {}
            try:
                # Connect to MLflow
                mlflow_uri = os.environ.get("MLFLOW_TRACKING_URI", "http://mlflow:5000")
                mlflow.set_tracking_uri(mlflow_uri)
                from mlflow.tracking import MlflowClient
                client = MlflowClient()

                # For each device, try to find MLflow experiments and metrics
                for device in devices:
                    try:
                        # Search for experiments related to this device
                        experiment_name = f"device_{device.device_id}"
                        experiment = mlflow.get_experiment_by_name(experiment_name)

                        if experiment:
                            # Search runs in this experiment, limit to 5 most recent runs
                            runs = mlflow.search_runs(
                                experiment_ids=[experiment.experiment_id],
                                max_results=5,  # Limit to 5 most recent runs
                                order_by=["start_time DESC"]  # Sort by start time descending (newest first)
                            )

                            if not runs.empty:
                                # Extract metrics we care about
                                inference_metrics = []
                                cpu_metrics = []
                                memory_metrics = []

                                # Debug information has been removed now that we understand the DataFrame structure

                                for _, row in runs.iterrows():
                                    # Look for specific metrics with their actual names in MLflow
                                    try:
                                        # Map MLflow metric names to our internal categories
                                        # Get metrics from the actual DataFrame structure
                                        # MLflow typically stores metrics in columns with 'metrics.' prefix

                                        # Inference time metrics - check for metrics in column names
                                        for metric_name in ["latency_mean_ms", "inference_time_ms", "latency_ms", "inference_time"]:
                                            metric_column = f"metrics.{metric_name}"
                                            if metric_column in row:
                                                try:
                                                    value = float(row[metric_column])
                                                    if not pd.isna(value):  # Check for NaN values
                                                        inference_metrics.append(value)
                                                        break
                                                except (ValueError, TypeError) as e:
                                                    print(f"Error converting {metric_name} value: {e}")
                                                    continue

                                        # CPU usage metrics
                                        for metric_name in ["system_cpu_percent_avg", "cpu_usage_percent", "cpu_usage", "cpu_percent"]:
                                            metric_column = f"metrics.{metric_name}"
                                            if metric_column in row:
                                                try:
                                                    value = float(row[metric_column])
                                                    if not pd.isna(value):  # Check for NaN values
                                                        cpu_metrics.append(value)
                                                        break
                                                except (ValueError, TypeError) as e:
                                                    print(f"Error converting {metric_name} value: {e}")
                                                    continue

                                        # Memory usage metrics
                                        for metric_name in ["system_memory_percent_avg", "memory_usage_percent", "memory_usage", "memory_percent"]:
                                            metric_column = f"metrics.{metric_name}"
                                            if metric_column in row:
                                                try:
                                                    value = float(row[metric_column])
                                                    if not pd.isna(value):  # Check for NaN values
                                                        memory_metrics.append(value)
                                                        break
                                                except (ValueError, TypeError) as e:
                                                    print(f"Error converting {metric_name} value: {e}")
                                                    continue

                                        # Also look for drift score to highlight drift issues
                                        for metric_name in ["drift_score", "drift_magnitude", "concept_drift", "distribution_shift"]:
                                            metric_column = f"metrics.{metric_name}"
                                            if metric_column in row:
                                                try:
                                                    if device.device_id not in mlflow_metrics:
                                                        mlflow_metrics[device.device_id] = {}

                                                    if "drift_score" not in mlflow_metrics[device.device_id]:
                                                        mlflow_metrics[device.device_id]["drift_score"] = []

                                                    value = float(row[metric_column])
                                                    if not pd.isna(value):  # Check for NaN values
                                                        mlflow_metrics[device.device_id]["drift_score"].append(value)
                                                    break
                                                except (ValueError, TypeError) as e:
                                                    print(f"Error converting {metric_name} value: {e}")
                                                    continue
                                    except:
                                        # Skip errors in individual metrics
                                        continue

                                # Log available metrics for debugging
                                try:
                                    if device.device_id not in mlflow_metrics:
                                        mlflow_metrics[device.device_id] = {}

                                    if "available_metrics" not in mlflow_metrics[device.device_id]:
                                        mlflow_metrics[device.device_id]["available_metrics"] = set()

                                    # Store unique metric names found in all runs
                                    # Extract metrics from column names that start with "metrics."
                                    for col in row.keys():
                                        if col.startswith('metrics.'):
                                            try:
                                                # Strip the "metrics." prefix to get just the metric name
                                                metric_name = col.replace('metrics.', '')
                                                mlflow_metrics[device.device_id]["available_metrics"].add(metric_name)
                                            except Exception as col_err:
                                                print(f"Error processing column {col}: {col_err}")
                                except Exception as metrics_err:
                                    print(f"Error processing available metrics for device {device.device_id}: {metrics_err}")

                                # Compute averages if we have data
                                device_metrics = {}

                                # Safely calculate metrics statistics with proper error handling
                                try:
                                    if inference_metrics:
                                        device_metrics["inference_time"] = {
                                            "avg": sum(inference_metrics) / len(inference_metrics),
                                            "min": min(inference_metrics),
                                            "max": max(inference_metrics),
                                            "count": len(inference_metrics)
                                        }
                                except Exception as e:
                                    print(f"Error calculating inference time metrics: {e}")

                                try:
                                    if cpu_metrics:
                                        device_metrics["cpu_usage"] = {
                                            "avg": sum(cpu_metrics) / len(cpu_metrics),
                                            "min": min(cpu_metrics),
                                            "max": max(cpu_metrics),
                                            "count": len(cpu_metrics)
                                        }
                                except Exception as e:
                                    print(f"Error calculating CPU usage metrics: {e}")

                                try:
                                    if memory_metrics:
                                        device_metrics["memory_usage"] = {
                                            "avg": sum(memory_metrics) / len(memory_metrics),
                                            "min": min(memory_metrics),
                                            "max": max(memory_metrics),
                                            "count": len(memory_metrics)
                                        }
                                except Exception as e:
                                    print(f"Error calculating memory usage metrics: {e}")

                                if device_metrics:
                                    mlflow_metrics[device.device_id] = {
                                        **mlflow_metrics.get(device.device_id, {}),
                                        **device_metrics
                                    }

                                    # Convert set to list for JSON serialization
                                    if "available_metrics" in mlflow_metrics[device.device_id] and isinstance(mlflow_metrics[device.device_id]["available_metrics"], set):
                                        mlflow_metrics[device.device_id]["available_metrics"] = list(mlflow_metrics[device.device_id]["available_metrics"])
                    except Exception as exp_err:
                        # Skip this device's MLflow data if there's an error
                        print(f"Error getting MLflow data for device {device.device_id}: {exp_err}")
                        continue
            except Exception as mlflow_err:
                print(f"Error connecting to MLflow: {mlflow_err}")
                # Continue without MLflow data

            device_summaries = []
            for device in devices:
                try:
                    # Get all packages for this device without limit
                    packages = PackageService.get_packages_by_device(db, device.device_id, limit=0)

                    latest_package = None
                    if packages:
                        latest_package = max([p.uploaded_at for p in packages])

                    summary = {
                        "device_id": device.device_id,
                        "hostname": device.hostname,
                        "platform": device.platform,
                        "is_active": device.is_active,
                        "last_sync_time": format_datetime(device.last_sync_time),
                        "package_count": len(packages),
                        "latest_package": format_datetime(latest_package)
                    }

                    # Add MLflow metrics if available
                    if device.device_id in mlflow_metrics:
                        summary["mlflow_metrics"] = mlflow_metrics[device.device_id]

                    device_summaries.append(summary)
                except Exception as device_err:
                    # Log error but continue with next device
                    print(f"Error processing device {device.device_id}: {device_err}")
                    continue

            return device_summaries
        except Exception as db_err:
            # Return empty list if database operation fails
            import traceback
            print(f"Database error in get_devices_summary: {db_err}")
            print(traceback.format_exc())
            return []
    except Exception as e:
        import traceback
        print(f"Unexpected error in get_devices_summary: {e}")
        print(traceback.format_exc())
        # Return empty list instead of raising an exception
        return []

@router.get("/models/summary")
def get_models_summary(db: Session = Depends(get_db)):
    """Liefert eine Zusammenfassung aller Modelle in der Registry."""
    try:
        # MLflow Client initialisieren
        try:
            # Set connection timeout to avoid hanging if MLflow is down
            os.environ["MLFLOW_HTTP_REQUEST_TIMEOUT"] = "5"  # 5 seconds timeout
            mlflow_uri = os.environ.get("MLFLOW_TRACKING_URI", "http://mlflow:5000")
            mlflow.set_tracking_uri(mlflow_uri)
            mlflow_client = mlflow.tracking.MlflowClient()
            
            # Alle registrierten Modelle abrufen
            registered_models = mlflow_client.search_registered_models()
            
            models_summary = []
            for model in registered_models:
                try:
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
                except Exception as model_err:
                    # Log error but continue with next model
                    print(f"Error processing model {model.name}: {model_err}")
                    # Add a minimal entry for this model
                    models_summary.append({
                        "name": model.name,
                        "error": str(model_err),
                        "total_versions": 0,
                        "latest_version": 0,
                        "has_production": False,
                        "has_staging": False,
                        "device_id": model.name.split('-')[0] if '-' in model.name else None,
                    })
            
            return models_summary
        
        except Exception as mlflow_err:
            # If MLflow is completely unavailable, return an empty list
            # rather than failing the entire request
            print(f"Error connecting to MLflow: {mlflow_err}")
            import traceback
            traceback.print_exc()
            return []
        
    except Exception as e:
        import traceback
        print(f"Unexpected error in models_summary: {e}")
        print(traceback.format_exc())
        # Return empty list instead of raising an exception
        # This allows the dashboard to continue working with partial data
        return []

@router.get("/models/device/{device_id}")
def get_device_models(device_id: str, db: Session = Depends(get_db)):
    """Liefert alle Modelle für ein bestimmtes Gerät."""
    try:
        try:
            # MLflow Client initialisieren
            # Set connection timeout to avoid hanging if MLflow is down
            os.environ["MLFLOW_HTTP_REQUEST_TIMEOUT"] = "5"  # 5 seconds timeout
            mlflow_uri = os.environ.get("MLFLOW_TRACKING_URI", "http://mlflow:5000")
            mlflow.set_tracking_uri(mlflow_uri)
            mlflow_client = mlflow.tracking.MlflowClient()
            
            # Alle registrierten Modelle abrufen
            registered_models = mlflow_client.search_registered_models()
            
            # Nach Modellen filtern, die zum angegebenen Gerät gehören
            device_models = []
            for model in models:
                try:
                    if model.name.startswith(f"{device_id}-"):
                        # Neueste Versionen abrufen
                        latest_versions = mlflow_client.get_latest_versions(model.name)
                        
                        # Details für jede Version sammeln
                        versions = []
                        for version in latest_versions:
                            try:
                                # Get run metrics directly from the run object
                                metrics_dict = {}
                                if version.run_id:
                                    try:
                                        # Get the run directly
                                        run = mlflow_client.get_run(version.run_id)
                                        
                                        # Extract metrics from the run data object
                                        if hasattr(run, "data") and hasattr(run.data, "metrics"):
                                            print(f"[DEBUG] Found metrics for run {version.run_id}: {list(run.data.metrics.keys())}")
                                            metrics_dict = run.data.metrics
                                    except Exception as run_err:
                                        print(f"[DEBUG] Error getting run data for {version.run_id}: {run_err}")

                                versions.append({
                                    "version": version.version,
                                    "stage": version.current_stage,
                                    "run_id": version.run_id,
                                    "metrics": metrics_dict,
                                    "creation_timestamp": version.creation_timestamp,
                                    "last_updated_timestamp": version.last_updated_timestamp
                                })
                            except Exception as version_err:
                                print(f"Error processing version {version.version} of model {model.name}: {version_err}")
                                continue
                                
                        device_models.append({
                            "name": model.name,
                            "versions": versions
                        })
                except Exception as model_err:
                    print(f"Error processing model {model.name}: {model_err}")
                    continue
            
            return device_models
        except Exception as mlflow_err:
            import traceback
            print(f"MLflow error in get_device_models: {mlflow_err}")
            print(traceback.format_exc())
            return []
    except Exception as e:
        import traceback
        print(f"Unexpected error in get_device_models: {e}")
        print(traceback.format_exc())
        # Return empty list instead of raising an exception
        return []

@router.get("/packages/activity")
def get_package_activity(limit: int = 10, db: Session = Depends(get_db)):
    """Liefert die neuesten Paketaktivitäten."""
    try:
        # Import the datetime formatting function
        try:
            from tinysphere.api.models.base import format_datetime_with_z as format_datetime
        except ImportError:
            # Fallback to local implementation if import fails
            def format_datetime(dt):
                """Format a datetime with the Z suffix for UTC time"""
                if dt is None:
                    return None
                return dt.strftime('%Y-%m-%dT%H:%M:%S.%f')[:-3] + 'Z'
        
        try:
            # Neueste Pakete abrufen, sortiert nach Upload-Zeit
            packages = db.query(Package).order_by(Package.uploaded_at.desc()).limit(limit).all()
            
            activities = []
            for package in packages:
                try:
                    # Safely get device info
                    device_name = "Unknown"
                    try:
                        device = db.query(Device).filter(Device.device_id == package.device_id).first()
                        if device and device.hostname:
                            device_name = device.hostname
                    except Exception as device_err:
                        print(f"Error getting device for package {package.package_id}: {device_err}")
                    
                    # Die datetime-Objekte manuell mit Z-Suffix formatieren
                    activities.append({
                        "package_id": package.package_id,
                        "device_id": package.device_id,
                        "device_name": device_name,
                        "package_type": package.package_type,
                        "uploaded_at": format_datetime(package.uploaded_at),
                        "processed_at": format_datetime(package.processed_at),
                        "is_processed": package.is_processed if package.is_processed is not None else False,
                        "processing_status": package.processing_status or "unknown"
                    })
                except Exception as package_err:
                    print(f"Error processing package {package.package_id}: {package_err}")
                    # Add a minimal entry for this package
                    activities.append({
                        "package_id": package.package_id,
                        "device_id": package.device_id,
                        "device_name": "Unknown",
                        "package_type": "unknown",
                        "uploaded_at": format_datetime(datetime.now(timezone.utc)),
                        "processed_at": None,
                        "is_processed": False,
                        "processing_status": "error",
                        "error": str(package_err)
                    })
            
            return activities
        except Exception as db_err:
            # Return an empty list if any database operations fail
            import traceback
            print(f"Database error in get_package_activity: {db_err}")
            print(traceback.format_exc())
            return []
        
    except Exception as e:
        # Instead of throwing an exception, return an empty list
        import traceback
        print(f"Unexpected error in get_package_activity: {e}")
        print(traceback.format_exc())
        return []

@router.get("/packages/statistics")
def get_package_statistics(db: Session = Depends(get_db)):
    """Liefert Statistiken zu hochgeladenen Paketen."""
    try:
        try:
            # Initialize with default empty values
            package_types_dict = {}
            processing_status_dict = {}
            devices_dict = {}
            monthly_uploads_dict = {}
            
            try:
                # Statistiken über Pakettypen abrufen
                package_types = (
                    db.query(Package.package_type, func.count(Package.id))
                    .group_by(Package.package_type)
                    .all()
                )
                package_types_dict = {t[0] or "unknown": t[1] for t in package_types}
            except Exception as types_err:
                print(f"Error fetching package types: {types_err}")
            
            try:
                # Statistiken über Verarbeitungsstatus abrufen
                processing_status = (
                    db.query(Package.processing_status, func.count(Package.id))
                    .group_by(Package.processing_status)
                    .all()
                )
                processing_status_dict = {s[0] or "unknown": s[1] for s in processing_status}
            except Exception as status_err:
                print(f"Error fetching processing status: {status_err}")
            
            try:
                # Statistiken über Geräte abrufen
                device_packages = (
                    db.query(Package.device_id, func.count(Package.id))
                    .group_by(Package.device_id)
                    .all()
                )
                devices_dict = {d[0] or "unknown": d[1] for d in device_packages}
            except Exception as device_err:
                print(f"Error fetching device packages: {device_err}")
            
            try:
                # Statistiken über die Zeit abrufen (pro Monat, vereinfacht)
                monthly_uploads = (
                    db.query(func.date_trunc('month', Package.uploaded_at), func.count(Package.id))
                    .group_by(func.date_trunc('month', Package.uploaded_at))
                    .all()
                )
                monthly_uploads_dict = {str(m[0].date()) if m[0] else "unknown": m[1] for m in monthly_uploads if m[0]}
            except Exception as monthly_err:
                print(f"Error fetching monthly uploads: {monthly_err}")
            
            return {
                "package_types": package_types_dict,
                "processing_status": processing_status_dict,
                "devices": devices_dict,
                "monthly_uploads": monthly_uploads_dict
            }
            
        except Exception as db_err:
            import traceback
            print(f"Database error in package statistics: {db_err}")
            print(traceback.format_exc())
            # Return empty data structure
            return {
                "package_types": {},
                "processing_status": {},
                "devices": {},
                "monthly_uploads": {}
            }
            
    except Exception as e:
        import traceback
        print(f"Unexpected error in package statistics: {e}")
        print(traceback.format_exc())
        # Return empty data structure instead of throwing an error
        return {
            "package_types": {},
            "processing_status": {},
            "devices": {},
            "monthly_uploads": {}
        }
    
@router.get("/packages/timeline")
def get_package_timeline(period: str = "week", db: Session = Depends(get_db)):
    """Returns package upload statistics over time for charts."""
    try:
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
            
            try:
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
            except Exception as db_err:
                print(f"Database error in package timeline: {db_err}")
                # Return empty result for no data
                result = []
            
            # Transform into the format needed by the chart
            timeline_data = {}
            for date, package_type, count in result:
                try:
                    date_str = date.strftime('%Y-%m-%d')
                    if date_str not in timeline_data:
                        timeline_data[date_str] = {
                            'date': date_str,
                            'models': 0,
                            'metrics': 0,
                            'data_logs': 0,
                            'drift_events': 0,
                            'prediction_images': 0
                        }

                    # Map package types to chart categories
                    if package_type == 'model':
                        timeline_data[date_str]['models'] += count
                    elif package_type == 'metrics':
                        timeline_data[date_str]['metrics'] += count
                    elif package_type == 'data_log':
                        timeline_data[date_str]['data_logs'] += count
                    elif 'drift' in package_type.lower():
                        timeline_data[date_str]['drift_events'] += count
                    elif 'prediction' in package_type.lower() or 'image' in package_type.lower():
                        timeline_data[date_str]['prediction_images'] += count
                except Exception as item_err:
                    print(f"Error processing timeline item: {item_err}")
                    continue
            
            # Fill in missing dates with zero values
            for day_offset in range(days_to_include):
                current_date = (datetime.now(timezone.utc) - timedelta(days=day_offset)).strftime('%Y-%m-%d')
                if current_date not in timeline_data:
                    timeline_data[current_date] = {
                        'date': current_date,
                        'models': 0,
                        'metrics': 0,
                        'data_logs': 0,
                        'drift_events': 0,
                        'prediction_images': 0
                    }
            
            # Convert to list and sort by date
            timeline_list = list(timeline_data.values())
            timeline_list.sort(key=lambda x: x['date'])
            
            return timeline_list
            
        except Exception as e:
            import traceback
            print(f"Error in package timeline: {e}")
            print(traceback.format_exc())
            
            # Return empty dataset on error to allow dashboard to continue working
            return [
                {
                    'date': datetime.now(timezone.utc).strftime('%Y-%m-%d'),
                    'models': 0,
                    'metrics': 0,
                    'data_logs': 0,
                    'drift_events': 0,
                    'prediction_images': 0
                }
            ]
            
    except Exception as e:
        import traceback
        print(f"Unexpected error in package timeline: {e}")
        print(traceback.format_exc())
        # Return an empty array so the dashboard doesn't break
        return []

@router.get("/models/performance")
def get_models_performance(
    metric: str = "accuracy",
    days: Optional[int] = None,
    limit: Optional[int] = None,
    model_name: Optional[str] = None,
    tags: Optional[str] = None,
    include_operational_metrics: bool = True,
    db: Session = Depends(get_db)
):
    """
    Fetch performance metrics for models from MLflow with enhanced filtering options.

    Parameters:
    - metric: Main metric name to retrieve (default: "accuracy")
    - days: Filter runs from last N days
    - limit: Maximum number of runs to return per model
    - model_name: Filter by specific model name
    - tags: Comma-separated list of tags in format "key=value"
    - include_operational_metrics: Include operational metrics like latency, CPU usage, etc.
    """
    try:
        try:
            # Add detailed logging
            print(f"[DEBUG] get_models_performance: Starting with params: metric={metric}, days={days}, limit={limit}, model_name={model_name}, tags={tags}")

            # Set connection timeout to avoid hanging if MLflow is down
            os.environ["MLFLOW_HTTP_REQUEST_TIMEOUT"] = "10"  # 10 seconds timeout
            mlflow_uri = os.environ.get("MLFLOW_TRACKING_URI", "http://mlflow:5000")
            print(f"[DEBUG] Using MLflow URI: {mlflow_uri}")

            mlflow.set_tracking_uri(mlflow_uri)
            client = mlflow.tracking.MlflowClient()
            print(f"[DEBUG] MLflow client created successfully")

            # Calculate date filter if days provided
            date_filter = None
            if days:
                start_time = int((datetime.now(timezone.utc) - timedelta(days=days)).timestamp() * 1000)
                date_filter = start_time

            # Parse tags filter from parameter
            user_tag_filters = {}
            if tags:
                try:
                    tag_pairs = tags.split(",")
                    for pair in tag_pairs:
                        key, value = pair.strip().split("=")
                        user_tag_filters[key.strip()] = value.strip()
                except Exception as tag_error:
                    print(f"Invalid tag format: {tags}. Use 'key1=value1,key2=value2'")

            # Get all registered models or filter by name
            registered_models = []
            if model_name:
                try:
                    print(f"[DEBUG] Trying to get registered model: {model_name}")
                    model = client.get_registered_model(model_name)
                    registered_models = [model]
                    print(f"[DEBUG] Found registered model: {model_name}")
                except Exception as m_err:
                    print(f"[DEBUG] Error getting model {model_name}: {m_err}")
            else:
                print(f"[DEBUG] Searching for all registered models")
                registered_models = client.search_registered_models()
                print(f"[DEBUG] Found {len(registered_models)} registered models")

            # Define operational metrics to include
            operational_metrics = [
                # TinyLCM specific metrics
                "confidence_mean", "confidence_median", "confidence_min", "confidence_max",
                "latency_mean_ms", "latency_median_ms", "latency_min_ms", "latency_max_ms",
                "system_cpu_percent_avg", "system_memory_percent_avg",
                "total_inferences"
            ]

            performance_data = []

            # Find all experiments to search through
            print(f"[DEBUG] Searching for all experiments")
            all_experiments = mlflow.search_experiments()
            print(f"[DEBUG] Found {len(all_experiments)} total experiments")

            # Process each registered model
            for reg_model in registered_models:
                print(f"[DEBUG] Processing registered model: {reg_model.name}")
                model_data = []  # Collect data for this model

                # Build search filter for runs with this model tag
                base_filter_string = f"tags.registered_model_name = '{reg_model.name}'"

                # Add user tags if provided
                tag_filter_parts = [base_filter_string]
                for tag_key, tag_value in user_tag_filters.items():
                    tag_filter_parts.append(f"tags.{tag_key} = '{tag_value}'")

                filter_string = " and ".join(tag_filter_parts)
                print(f"[DEBUG] Using filter string: {filter_string}")

                # Search all experiments for runs related to this model
                for experiment in all_experiments:
                    try:
                        print(f"[DEBUG] Searching in experiment: {experiment.name} (ID: {experiment.experiment_id})")

                        # Search runs with model tag
                        runs = mlflow.search_runs(
                            experiment_ids=[experiment.experiment_id],
                            filter_string=filter_string,
                            max_results=100  # Set a reasonable limit
                        )

                        if runs.empty:
                            print(f"[DEBUG] No matching runs found in experiment {experiment.name}")
                            continue

                        print(f"[DEBUG] Found {len(runs)} matching runs in experiment {experiment.name}")

                        # Process each run's metrics
                        for idx, row in runs.iterrows():
                            run_id = row.get("run_id", "unknown")
                            run_timestamp = row.get("start_time", 0)
                            version_tag = row.get("tags.version", "1")  # Default to version 1 if not specified

                            try:
                                version_num = int(version_tag)
                            except (ValueError, TypeError):
                                version_num = 1

                            stage = row.get("tags.stage", "None")

                            print(f"[DEBUG] Processing run {run_id} with {len([c for c in row.index if c.startswith('metrics.')])} metrics")

                            # Date filtering if required
                            if date_filter and run_timestamp < date_filter:
                                print(f"[DEBUG] Skipping run with timestamp {run_timestamp} (before {date_filter})")
                                continue

                            # Extract all metrics from the run
                            all_metrics = {}
                            for col in row.index:
                                if col.startswith('metrics.'):
                                    try:
                                        metric_name = col.replace('metrics.', '')
                                        metric_value = row[col]
                                        # Try to convert to float and check for NaN
                                        try:
                                            float_value = float(metric_value)
                                            # Skip NaN values which are not JSON serializable
                                            if not pd.isna(float_value):
                                                all_metrics[metric_name] = float_value
                                            else:
                                                print(f"[DEBUG] Skipping NaN value for metric {metric_name}")
                                        except (ValueError, TypeError):
                                            # For non-numeric values, keep as is if they're serializable
                                            if isinstance(metric_value, (str, int, bool)) or metric_value is None:
                                                all_metrics[metric_name] = metric_value
                                    except Exception as metric_err:
                                        print(f"[DEBUG] Error getting metric {col}: {metric_err}")

                            print(f"[DEBUG] Found metrics: {list(all_metrics.keys())}")

                            # Check for the main requested metric
                            if metric in all_metrics:
                                try:
                                    # Value should already be filtered for NaN at this point
                                    value = all_metrics[metric]
                                    if isinstance(value, (int, float)) and not pd.isna(value):
                                        print(f"[DEBUG] Found main metric {metric} = {value}")
                                        model_data.append({
                                            "model_name": reg_model.name,
                                            "version": version_num,
                                            "stage": stage,
                                            "metric_name": metric,
                                            "value": value,
                                            "timestamp": run_timestamp,
                                            "run_id": run_id
                                        })
                                    else:
                                        print(f"[DEBUG] Skipping non-numeric or NaN value for main metric {metric}")
                                except Exception as e:
                                    print(f"Error processing metric {metric} for run {run_id}: {e}")

                            # Include operational metrics if requested
                            if include_operational_metrics:
                                for op_metric in operational_metrics:
                                    if op_metric in all_metrics:
                                        try:
                                            # Value should already be filtered for NaN
                                            value = all_metrics[op_metric]
                                            if isinstance(value, (int, float)) and not pd.isna(value):
                                                print(f"[DEBUG] Found operational metric {op_metric} = {value}")
                                                model_data.append({
                                                    "model_name": reg_model.name,
                                                    "version": version_num,
                                                    "stage": stage,
                                                    "metric_name": op_metric,
                                                    "value": value,
                                                    "timestamp": run_timestamp,
                                                    "run_id": run_id
                                                })
                                            else:
                                                print(f"[DEBUG] Skipping non-numeric or NaN value for operational metric {op_metric}")
                                        except Exception as e:
                                            print(f"Error processing operational metric {op_metric}: {e}")
                    except Exception as exp_err:
                        print(f"[DEBUG] Error searching experiment {experiment.name}: {exp_err}")
                        continue

                # If we have no data from tags search, fall back to direct version search
                if not model_data:
                    print(f"[DEBUG] No metrics found via tag search, trying registered model versions")
                    versions = client.get_latest_versions(reg_model.name)
                    print(f"[DEBUG] Found {len(versions)} versions for model: {reg_model.name}")

                    for version in versions:
                        # Skip if no run ID
                        if not version.run_id:
                            print(f"[DEBUG] Skipping version {version.version} - no run ID")
                            continue

                        try:
                            # Get run data directly
                            run = client.get_run(version.run_id)
                            run_timestamp = run.info.start_time if hasattr(run.info, "start_time") else 0

                            # Extract metrics
                            if hasattr(run, "data") and hasattr(run.data, "metrics"):
                                metrics_dict = run.data.metrics
                                print(f"[DEBUG] Found {len(metrics_dict)} metrics for version {version.version}")

                                # Check for the main requested metric
                                if metric in metrics_dict:
                                    try:
                                        value = metrics_dict[metric]
                                        if isinstance(value, (int, float)) and not pd.isna(value):
                                            model_data.append({
                                                "model_name": reg_model.name,
                                                "version": version.version,
                                                "stage": version.current_stage,
                                                "metric_name": metric,
                                                "value": value,
                                                "timestamp": run_timestamp,
                                                "run_id": version.run_id
                                            })
                                        else:
                                            print(f"[DEBUG] Skipping non-numeric or NaN value for main metric {metric}")
                                    except Exception as e:
                                        print(f"Error processing metric {metric}: {e}")

                                # Include operational metrics if requested
                                if include_operational_metrics:
                                    for op_metric in operational_metrics:
                                        if op_metric in metrics_dict:
                                            try:
                                                value = metrics_dict[op_metric]
                                                if isinstance(value, (int, float)) and not pd.isna(value):
                                                    model_data.append({
                                                        "model_name": reg_model.name,
                                                        "version": version.version,
                                                        "stage": version.current_stage,
                                                        "metric_name": op_metric,
                                                        "value": value,
                                                        "timestamp": run_timestamp,
                                                        "run_id": version.run_id
                                                    })
                                                else:
                                                    print(f"[DEBUG] Skipping non-numeric or NaN value for operational metric {op_metric}")
                                            except Exception as e:
                                                print(f"Error processing operational metric {op_metric}: {e}")
                        except Exception as run_err:
                            print(f"Error getting run data for version {version.version}: {run_err}")
                            continue

                # Sort by version and timestamp
                model_data.sort(key=lambda x: (x["version"], x.get("timestamp", 0)))

                # Apply limit if specified
                if limit and len(model_data) > limit:
                    model_data = model_data[-limit:]  # Take the latest ones

                print(f"[DEBUG] Collected {len(model_data)} data points for model {reg_model.name}")
                performance_data.extend(model_data)

            print(f"[DEBUG] Returning {len(performance_data)} total performance data points")
            return performance_data

        except Exception as mlflow_err:
            # If MLflow is completely unavailable, return an empty list
            # rather than failing the entire request
            print(f"Error connecting to MLflow in performance endpoint: {mlflow_err}")
            import traceback
            traceback.print_exc()
            return []

    except Exception as e:
        import traceback
        print(f"Unexpected error in models_performance: {e}")
        print(traceback.format_exc())
        # Return empty list instead of raising an exception
        # This allows the dashboard to continue working with partial data
        return []