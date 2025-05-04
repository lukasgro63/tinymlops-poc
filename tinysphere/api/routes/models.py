from typing import List, Optional

import mlflow
from fastapi import APIRouter, Depends, HTTPException, Path
from pydantic import BaseModel
from sqlalchemy.orm import Session

from tinysphere.api.dependencies.db import get_db

router = APIRouter()

class ModelStageTransition(BaseModel):
    stage: str  # "None", "Staging", "Production", "Archived"

class ModelVersion(BaseModel):
    name: str
    version: int
    stage: str
    source: Optional[str] = None
    run_id: Optional[str] = None

@router.get("/", response_model=List[str])
def list_models(db: Session = Depends(get_db)):
    """Liste aller registrierten Modelle in MLflow."""
    try:
        mlflow_client = mlflow.tracking.MlflowClient()
        models = mlflow_client.search_registered_models()
        return [model.name for model in models]
    except Exception as e:
        raise HTTPException(status_code=500, detail=f"Error fetching models: {str(e)}")

@router.get("/{model_name}/versions", response_model=List[ModelVersion])
def get_model_versions(model_name: str, db: Session = Depends(get_db)):
    """Liste aller Versionen eines Modells."""
    try:
        mlflow_client = mlflow.tracking.MlflowClient()
        versions = mlflow_client.get_latest_versions(model_name)
        
        result = []
        for version in versions:
            result.append(ModelVersion(
                name=version.name,
                version=version.version,
                stage=version.current_stage,
                source=version.source,
                run_id=version.run_id
            ))
        
        return result
    except Exception as e:
        raise HTTPException(status_code=500, detail=f"Error fetching model versions: {str(e)}")

@router.post("/{model_name}/versions/{version}/transition", response_model=ModelVersion)
def transition_model_version(
    model_name: str,
    version: int,
    transition: ModelStageTransition,
    db: Session = Depends(get_db)
):
    """Ändert den Status einer Modellversion (None, Staging, Production, Archived)."""
    try:
        if transition.stage not in ["None", "Staging", "Production", "Archived"]:
            raise HTTPException(
                status_code=400, 
                detail=f"Invalid stage: {transition.stage}. Must be one of: None, Staging, Production, Archived"
            )
            
        mlflow_client = mlflow.tracking.MlflowClient()
        version_obj = mlflow_client.transition_model_version_stage(
            name=model_name,
            version=version,
            stage=transition.stage
        )
        
        return ModelVersion(
            name=version_obj.name,
            version=version_obj.version,
            stage=version_obj.current_stage,
            source=version_obj.source,
            run_id=version_obj.run_id
        )
    except mlflow.exceptions.MlflowException as e:
        raise HTTPException(status_code=404, detail=f"Model or version not found: {str(e)}")
    except Exception as e:
        raise HTTPException(status_code=500, detail=f"Error transitioning model: {str(e)}")
    
@router.get("/{model_name}/metrics")
def get_model_metrics(
    model_name: str, 
    metric: str = "accuracy",
    db: Session = Depends(get_db)
):
    """Fetch metrics for all versions of a specific model from MLflow."""
    try:
        mlflow_client = mlflow.tracking.MlflowClient()
        
        # Get all versions of the model
        versions = mlflow_client.get_latest_versions(model_name)
        
        metrics_data = []
        
        for version in versions:
            # Skip if no run ID
            if not version.run_id:
                continue
                
            try:
                # Get run data
                run = mlflow_client.get_run(version.run_id)
                
                # Create base version data
                version_data = {
                    "version": version.version,
                    "stage": version.current_stage,
                    "run_id": version.run_id,
                    "created_at": version.creation_timestamp,
                    "metrics": {}
                }
                
                # Add all available metrics
                if hasattr(run, "data") and hasattr(run.data, "metrics"):
                    version_data["metrics"] = run.data.metrics
                
                metrics_data.append(version_data)
            except Exception as run_err:
                continue
                
        return metrics_data
    
    except Exception as e:
        raise HTTPException(status_code=500, detail=f"Error fetching model metrics: {str(e)}")

@router.post("/compare")
def compare_model_versions(
    model_name: str,
    versions: List[int],
    db: Session = Depends(get_db)
):
    """Compare metrics between different versions of a model."""
    try:
        if len(versions) < 1 or len(versions) > 3:
            raise HTTPException(status_code=400, detail="Please provide 1-3 version numbers to compare")
            
        mlflow_client = mlflow.tracking.MlflowClient()
        
        # Get the model versions
        model_versions = mlflow_client.get_latest_versions(model_name)
        
        # Filter to requested versions
        requested_versions = [v for v in model_versions if v.version in versions]
        
        if len(requested_versions) != len(versions):
            raise HTTPException(status_code=404, detail="One or more requested versions not found")
            
        # Get metrics for each version
        comparison_data = []
        metrics_keys = set()
        version_metrics = {}
        
        for version in requested_versions:
            if not version.run_id:
                continue
                
            try:
                # Get run data
                run = mlflow_client.get_run(version.run_id)
                
                # Store metrics for this version
                if hasattr(run, "data") and hasattr(run.data, "metrics"):
                    metrics = run.data.metrics
                    version_metrics[version.version] = metrics
                    
                    # Add all metric keys to our set
                    for key in metrics.keys():
                        metrics_keys.add(key)
            except Exception as run_err:
                continue
        
        # Now build comparison rows for each metric
        for metric_key in sorted(metrics_keys):
            row = {
                "metric": metric_key,
                "values": {}
            }
            
            # Add values for each version
            for version_num in versions:
                if version_num in version_metrics and metric_key in version_metrics[version_num]:
                    row["values"][version_num] = version_metrics[version_num][metric_key]
                else:
                    row["values"][version_num] = None
            
            # Calculate diff between versions if there are exactly 2
            if len(versions) == 2:
                ver1, ver2 = sorted(versions)
                if (ver1 in version_metrics and metric_key in version_metrics[ver1] and
                    ver2 in version_metrics and metric_key in version_metrics[ver2]):
                    
                    val1 = version_metrics[ver1][metric_key]
                    val2 = version_metrics[ver2][metric_key]
                    
                    if isinstance(val1, (int, float)) and isinstance(val2, (int, float)):
                        # For loss metrics, lower is better, so invert the diff
                        if "loss" in metric_key.lower():
                            row["diff"] = val2 - val1  # newer version - older version
                        else:
                            row["diff"] = val1 - val2  # older version - newer version
            
            comparison_data.append(row)
            
        return comparison_data
        
    except HTTPException:
        raise
    except Exception as e:
        raise HTTPException(status_code=500, detail=f"Error comparing model versions: {str(e)}")