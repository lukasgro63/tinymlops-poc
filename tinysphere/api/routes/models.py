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
    creation_timestamp: Optional[int] = None
    last_updated_timestamp: Optional[int] = None
    description: Optional[str] = None
    is_production: bool = False

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
                run_id=version.run_id,
                creation_timestamp=version.creation_timestamp,
                last_updated_timestamp=version.last_updated_timestamp,
                description=version.description,
                is_production=(version.current_stage == "Production")
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
            run_id=version_obj.run_id,
            creation_timestamp=version_obj.creation_timestamp,
            last_updated_timestamp=version_obj.last_updated_timestamp,
            description=version_obj.description,
            is_production=(version_obj.current_stage == "Production")
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
        print(f"[DEBUG] get_model_metrics: model_name={model_name}, metric={metric}")
        mlflow_client = mlflow.tracking.MlflowClient()

        # Get all versions of the model
        print(f"[DEBUG] Getting latest versions for model: {model_name}")
        versions = mlflow_client.get_latest_versions(model_name)
        print(f"[DEBUG] Found {len(versions)} versions for model: {model_name}")

        metrics_data = []

        # First, try searching for metrics via tags
        try:
            print(f"[DEBUG] Searching for runs with registered_model_name={model_name} tag")

            # Find all experiments
            all_experiments = mlflow.search_experiments()
            print(f"[DEBUG] Found {len(all_experiments)} total experiments to search through")

            found_metrics_via_tags = False

            # Search for runs with this model tag
            for experiment in all_experiments:
                try:
                    # Build filter string for registered model name
                    filter_string = f"tags.registered_model_name = '{model_name}'"

                    # Search for runs with this tag
                    runs = mlflow.search_runs(
                        experiment_ids=[experiment.experiment_id],
                        filter_string=filter_string,
                        max_results=100  # Set a reasonable limit
                    )

                    if runs.empty:
                        continue

                    print(f"[DEBUG] Found {len(runs)} runs with model tag in experiment {experiment.name}")

                    # Process each run
                    for idx, row in runs.iterrows():
                        run_id = row.get("run_id", "unknown")
                        run_timestamp = row.get("start_time", 0)
                        version_tag = row.get("tags.version", "1")  # Default to version 1 if not specified

                        try:
                            version_num = int(version_tag)
                        except (ValueError, TypeError):
                            # If we can't parse the version tag, try to match it with a registered version
                            version_match = next((v for v in versions if v.run_id == run_id), None)
                            if version_match:
                                version_num = version_match.version
                            else:
                                version_num = 1

                        # Get stage from tags or default to "None"
                        stage = row.get("tags.stage", "None")

                        # Check the registered versions for matching run_id to get official stage
                        for version in versions:
                            if version.run_id == run_id:
                                stage = version.current_stage
                                break

                        # Extract all metrics from the run
                        run_metrics = {}
                        for col in row.index:
                            if col.startswith('metrics.'):
                                try:
                                    metric_name = col.replace('metrics.', '')
                                    metric_value = row[col]
                                    try:
                                        # Convert to float for consistent handling
                                        value = float(metric_value)
                                        # Skip NaN values which are not JSON serializable
                                        if not pd.isna(value):
                                            run_metrics[metric_name] = value
                                    except (ValueError, TypeError):
                                        # Skip non-numeric metrics
                                        pass
                                except Exception as metric_err:
                                    continue

                        # Create version data
                        if run_metrics:
                            found_metrics_via_tags = True

                            # Get creation timestamp either from the run or from the matching version
                            creation_ts = run_timestamp
                            version_match = next((v for v in versions if v.run_id == run_id), None)
                            if version_match and version_match.creation_timestamp:
                                creation_ts = version_match.creation_timestamp

                            version_data = {
                                "version": version_num,
                                "stage": stage,
                                "run_id": run_id,
                                "created_at": creation_ts,
                                "metrics": run_metrics
                            }

                            metrics_data.append(version_data)
                            print(f"[DEBUG] Added metrics data for version {version_num} from tagged run")
                except Exception as exp_err:
                    print(f"[DEBUG] Error searching experiment {experiment.name}: {exp_err}")
                    continue

            if found_metrics_via_tags:
                print(f"[DEBUG] Successfully found metrics data via tags search")
            else:
                print(f"[DEBUG] No metrics found via tags search, falling back to direct version lookup")
        except Exception as tag_err:
            print(f"[DEBUG] Error during tag-based search: {tag_err}")
            print("[DEBUG] Falling back to direct version lookup")

        # If we didn't find any metrics via tags or encountered an error, fall back to direct version lookup
        if not metrics_data:
            print(f"[DEBUG] Checking registered model versions directly")

            for version in versions:
                # Skip if no run ID
                if not version.run_id:
                    print(f"[DEBUG] Skipping version {version.version} - no run ID")
                    continue

                try:
                    # Get run data
                    print(f"[DEBUG] Getting run data for run_id: {version.run_id}")
                    run = mlflow_client.get_run(version.run_id)
                    print(f"[DEBUG] Successfully retrieved run data for run_id: {version.run_id}")

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
                        metric_keys = list(run.data.metrics.keys())
                        print(f"[DEBUG] Found metrics for run_id {version.run_id}: {metric_keys}")

                        # Create a copy of the metrics dictionary to avoid any reference issues
                        metrics_copy = {}
                        for key, value in run.data.metrics.items():
                            try:
                                # Convert to float for consistent handling
                                float_value = float(value)
                                # Skip NaN values which are not JSON serializable
                                if not pd.isna(float_value):
                                    metrics_copy[key] = float_value
                                else:
                                    print(f"[DEBUG] Skipping NaN value for metric {key}")
                            except (ValueError, TypeError):
                                # Skip any metrics that can't be converted to float
                                print(f"[DEBUG] Skipping non-numeric metric {key}")

                        version_data["metrics"] = metrics_copy

                        # Look specifically for the requested metric
                        if metric in run.data.metrics:
                            print(f"[DEBUG] Specifically found requested metric: {metric}={run.data.metrics[metric]}")
                    else:
                        print(f"[DEBUG] No metrics found for run_id {version.run_id}")

                    metrics_data.append(version_data)
                    print(f"[DEBUG] Added metrics data for version {version.version}")
                except Exception as run_err:
                    print(f"[DEBUG] Error getting run data: {run_err}")
                    continue

        # Sort data by version number for consistency
        metrics_data.sort(key=lambda x: x["version"])

        print(f"[DEBUG] Returning metrics data for {len(metrics_data)} versions")
        return metrics_data

    except Exception as e:
        print(f"[ERROR] Failed to fetch model metrics: {e}")
        import traceback
        traceback.print_exc()
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