import csv
import json
import logging
from pathlib import Path
from typing import Any, Dict, List

import mlflow

from tinysphere.importer.transformers.base import DataTransformer


class MetricsTransformer(DataTransformer):
    def __init__(self):
        # Logger initialisieren
        self.logger = logging.getLogger(f"{__name__}.{self.__class__.__name__}")
    
    def can_transform(self, package_type: str, files: List[Path]) -> bool:
        if package_type.lower() not in ["metrics", "components"]:
            return False
        
        # Erweiterte Suche nach Metrik-Dateien mit Mustern
        metrics_files = []
        for file in files:
            # Standardnamen
            if file.name.lower() in ['metrics.json', 'metrics.csv']:
                metrics_files.append(file)
            # Zeitstempel-Namen (metrics_TIMESTAMP_ID.json)
            elif file.name.lower().startswith('metrics_') and (file.suffix.lower() == '.json' or file.suffix.lower() == '.csv'):
                metrics_files.append(file)
        
        has_metrics_files = len(metrics_files) > 0
        
        if has_metrics_files:
            self.logger.info(f"Found metrics files in package with type '{package_type}': {[f.name for f in metrics_files]}")
        
        return has_metrics_files
    
    def transform(self, package_id: str, device_id: str, files: List[Path], metadata: Dict[str, Any]) -> Dict[str, Any]:
        # Set up MLflow tracking
        mlflow.set_experiment(f"device_{device_id}")
        
        # Find metrics file
        metrics_files = []
        for file in files:
            if file.name.lower() in ['metrics.json', 'metrics.csv']:
                metrics_files.append(file)
            elif file.name.lower().startswith('metrics_') and (file.suffix.lower() == '.json' or file.suffix.lower() == '.csv'):
                metrics_files.append(file)
        
        if not metrics_files:
            return {"status": "error", "message": "No metrics file found in package"}
        
        metrics_file = metrics_files[0]
        self.logger.info(f"Using metrics file: {metrics_file}")
        metrics_data = {}
        
        # Parse metrics file
        if metrics_file.suffix.lower() == '.json':
            with open(metrics_file, 'r') as f:
                metrics_data = json.load(f)
        elif metrics_file.suffix.lower() == '.csv':
            with open(metrics_file, 'r') as f:
                reader = csv.DictReader(f)
                for row in reader:
                    # Convert string numbers to floats where possible
                    for key, value in row.items():
                        try:
                            row[key] = float(value)
                        except:
                            pass
                    metrics_data.update(row)
        
        # Determine model name - check for combined package types
        package_type = metadata.get("package_type", "").lower()
        model_format = None
        
        # Get model information
        model_name = None
        model_version = None
        
        # Extract model format from combined package type, e.g., "model_metrics" -> "model"
        if "_" in package_type:
            package_components = package_type.split("_")
            if "model" in package_components:
                # Look for model files to determine format
                model_extensions = ['.tflite', '.onnx', '.pt', '.pth', '.pkl', '.h5']
                model_files = [f for f in files if any(f.suffix.lower() == ext for ext in model_extensions)]
                
                if model_files:
                    model_format = model_files[0].suffix.lstrip('.')
        
        # Try to find model_info.json which might contain model information
        model_info_files = []
        for file in files:
            if file.name.lower() in ['model_info.json', 'metadata.json']:
                model_info_files.append(file)
            elif file.name.lower().startswith('model_info_') and file.suffix.lower() == '.json':
                model_info_files.append(file)
            
        if model_info_files:
            try:
                with open(model_info_files[0], 'r') as f:
                    model_info = json.load(f)
                    if "model_name" in model_info:
                        model_name = model_info["model_name"]
                    if "model_version" in model_info:
                        model_version = model_info["model_version"]
                    if "format" in model_info and not model_format:
                        model_format = model_info["format"]
            except Exception as e:
                self.logger.warning(f"Could not read model info file: {str(e)}")
        
        # If model format is found but model_name is not set, use device_id to construct model name
        if model_format and not model_name:
            model_name = f"{device_id}-{model_format}-model"
            self.logger.info(f"Using constructed model name: {model_name}")
        
        # Find production version of the model if available
        production_version = None
        if model_name:
            try:
                from mlflow.tracking import MlflowClient
                client = MlflowClient()
                versions = client.get_latest_versions(model_name, stages=["Production"])
                if versions:
                    production_version = versions[0].version
                    self.logger.info(f"Found production version {production_version} for model {model_name}")
            except Exception as e:
                self.logger.warning(f"Error finding production version: {str(e)}")
        
        # Start MLflow run
        with mlflow.start_run(run_name=f"metrics_import_{package_id}"):
            # Log metrics
            flat_metrics = self._flatten_metrics(metrics_data)
            for key, value in flat_metrics.items():
                if isinstance(value, (int, float)):
                    mlflow.log_metric(key, value)
            
            # Log parameters about the metrics collection
            mlflow.log_param("device_id", device_id)
            mlflow.log_param("package_id", package_id)
            mlflow.log_param("metrics_source", metrics_file.name)
            
            # Log model information if available
            if model_name:
                mlflow.log_param("model_name", model_name)
            if model_format:
                mlflow.log_param("model_format", model_format)
            if production_version:
                mlflow.log_param("model_version", production_version)
                
            # Create a standard metrics directory for the artifact
            artifact_path = "metrics"
            # Log the original file as artifact in the metrics directory
            mlflow.log_artifact(str(metrics_file), artifact_path)
            
            # Get run info
            run_id = mlflow.active_run().info.run_id
            
            # Link the run to the model if we have model information
            # This uses direct MLflow Model Registry integration
            if model_name and production_version:
                try:
                    from mlflow.tracking import MlflowClient
                    client = MlflowClient()
                    
                    # Log the run directly to the model version
                    # First, get the model version details
                    model_version = client.get_model_version(model_name, production_version)
                    
                    # Create a link between the run and the model version
                    # This makes the run appear under the model in the UI
                    
                    # 1. Set standard MLflow tags for model linkage
                    client.set_tag(run_id, "mlflow.registeredModelName", model_name)
                    client.set_tag(run_id, "mlflow.registeredModelVersion", production_version)
                    
                    # 2. Create a direct alias for the metrics run to appear in the model version UI
                    try:
                        # Use the metrics directory for model URI
                        mlflow.register_model(
                            f"runs:/{run_id}/{artifact_path}/{metrics_file.name}",
                            model_name,
                            tags={"metrics_run_id": run_id, "original_run_id": run_id}
                        )
                        self.logger.info(f"Directly registered metrics run with model {model_name}")
                    except Exception as reg_error:
                        self.logger.warning(f"Could not register run directly with model: {str(reg_error)}")
                        # Fallback - at least add proper tags
                        client.set_tag(run_id, "registered_model_name", model_name)
                        client.set_tag(run_id, "registered_model_version", production_version)
                    
                    self.logger.info(f"Linked metrics run {run_id} to model {model_name} version {production_version}")
                except Exception as e:
                    self.logger.warning(f"Error linking run to model version: {str(e)}")
        
        return {
            "status": "success",
            "message": "Metrics imported successfully",
            "mlflow_run_id": run_id,
            "metrics_count": len(flat_metrics),
            "model_name": model_name,
            "model_version": production_version
        }
    
    def _flatten_metrics(self, metrics: Dict[str, Any], parent_key: str = '') -> Dict[str, Any]:
        """Flatten nested metrics dictionary"""
        flat_metrics = {}
        
        for key, value in metrics.items():
            new_key = f"{parent_key}_{key}" if parent_key else key
            
            if isinstance(value, dict):
                flat_metrics.update(self._flatten_metrics(value, new_key))
            elif isinstance(value, (int, float)):
                flat_metrics[new_key] = value
        
        return flat_metrics