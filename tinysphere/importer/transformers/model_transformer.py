import json
import logging
import os
import shutil
from pathlib import Path
from typing import Any, Dict, List, Optional

import mlflow

from tinysphere.importer.transformers.base import DataTransformer


class ModelTransformer(DataTransformer):
    def __init__(self):
        # Logger initialisieren
        self.logger = logging.getLogger(f"{__name__}.{self.__class__.__name__}")
    
    def can_transform(self, package_type: str, files: List[Path]) -> bool:
        # Accept either "model" or "components" package types
        if package_type.lower() not in ["model", "components"]:
            return False
        
        # Check if the package contains model files
        model_extensions = ['.tflite', '.onnx', '.pt', '.pth', '.pkl', '.h5']
        has_model_files = any(any(file.suffix.lower() == ext for ext in model_extensions) for file in files)
        
        # Log what we found for diagnostic purposes
        if has_model_files:
            self.logger.info(f"Found model files in package with type '{package_type}'")
        
        return has_model_files
    
    def transform(self, package_id: str, device_id: str, files: List[Path], metadata: Dict[str, Any]) -> Dict[str, Any]:
        # Set up MLflow tracking
        mlflow.set_experiment(f"device_{device_id}")
        
        # Find model file
        model_extensions = ['.tflite', '.onnx', '.pt', '.pth', '.pkl', '.h5']
        model_files = [f for f in files if any(f.suffix.lower() == ext for ext in model_extensions)]
        
        if not model_files:
            return {"status": "error", "message": "No model file found in package"}
        
        model_file = model_files[0]
        model_format = model_file.suffix.lstrip('.')
        
        # Find metadata file - erweitert, um verschiedene Namenskonventionen zu unterstützen
        metadata_files = []
        for file in files:
            # Standard-Dateinamen
            if file.name.lower() in ['metadata.json', 'model_info.json']:
                metadata_files.append(file)
            # Dateinamen mit Zeitstempeln
            elif file.name.lower().startswith('model_info_') and file.suffix.lower() == '.json':
                metadata_files.append(file)
            elif file.name.lower().startswith('metadata_') and file.suffix.lower() == '.json':
                metadata_files.append(file)
        
        model_metadata = {}
        
        if metadata_files:
            try:
                with open(metadata_files[0], 'r') as f:
                    model_metadata = json.load(f)
            except Exception as e:
                self.logger.warning(f"Could not read metadata file: {str(e)}")
        
        # Start MLflow run
        with mlflow.start_run(run_name=f"model_import_{package_id}"):
            # Log model information
            mlflow.log_param("model_format", model_format)
            mlflow.log_param("device_id", device_id)
            mlflow.log_param("package_id", package_id)
            
            # Log additional parameters from metadata
            if "metrics" in model_metadata:
                for k, v in model_metadata["metrics"].items():
                    if isinstance(v, (int, float)):
                        mlflow.log_metric(k, v)
            
            if "params" in model_metadata:
                for k, v in model_metadata["params"].items():
                    mlflow.log_param(k, v)
            
            # Log model file as artifact
            mlflow.log_artifact(str(model_file))
            
            # Log other artifacts
            for file in files:
                if file != model_file and file.is_file():
                    mlflow.log_artifact(str(file))
            
            # Get run info
            run_id = mlflow.active_run().info.run_id
            
            result = {
                "status": "success",
                "message": "Model imported successfully",
                "mlflow_run_id": run_id,
                "model_format": model_format
            }
            
            # Register model in MLflow Model Registry
            try:
                # Modellname aus device_id und Formatinformationen erstellen
                model_name = f"{device_id}-{model_format}-model"
                
                # Modell-URI für die Registrierung definieren
                model_uri = f"runs:/{run_id}/{model_file.name}"
                
                self.logger.info(f"Registering model in MLflow Model Registry: {model_name} from {model_uri}")
                
                # Modell in der Registry registrieren
                registered_model = mlflow.register_model(
                    model_uri=model_uri,
                    name=model_name
                )
                
                # Erweiterte Ergebnisinformationen hinzufügen
                result.update({
                    "message": "Model imported and registered successfully",
                    "registered_model_name": model_name,
                    "registered_model_version": registered_model.version
                })
                
                self.logger.info(f"Model registered successfully as {model_name} version {registered_model.version}")
                
            except Exception as e:
                self.logger.error(f"Failed to register model in MLflow registry: {str(e)}")
                # Füge Fehlerinformationen zum Ergebnis hinzu, aber setze status nicht auf error
                # Da der Import an sich erfolgreich war
                result.update({
                    "registry_error": str(e),
                    "registry_status": "failed"
                })
        
        return result