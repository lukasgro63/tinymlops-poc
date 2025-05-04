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
            
            # Log the original file as artifact
            mlflow.log_artifact(str(metrics_file))
            
            # Get run info
            run_id = mlflow.active_run().info.run_id
        
        return {
            "status": "success",
            "message": "Metrics imported successfully",
            "mlflow_run_id": run_id,
            "metrics_count": len(flat_metrics)
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