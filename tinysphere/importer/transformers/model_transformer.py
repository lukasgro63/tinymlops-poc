import json
import logging
import os
import shutil
import hashlib
from pathlib import Path
from typing import Any, Dict, List, Optional, Tuple

import mlflow
import mlflow.exceptions
from mlflow.tracking import MlflowClient

from tinysphere.importer.transformers.base import DataTransformer


class ModelTransformer(DataTransformer):
    def __init__(self):
        # Logger initialisieren
        self.logger = logging.getLogger(f"{__name__}.{self.__class__.__name__}")
        self.client = MlflowClient()
    
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
            
            # Register model in MLflow Model Registry with duplicate detection
            try:
                # Modellname aus device_id und Formatinformationen erstellen
                model_name = f"{device_id}-{model_format}-model"
                
                # Modell-URI für die Registrierung definieren
                model_uri = f"runs:/{run_id}/{model_file.name}"
                
                # Berechne Hash der Modelldatei zur Duplikaterkennung
                model_hash = self._calculate_file_hash(str(model_file))
                
                # Prüfe, ob dieses Modell bereits existiert (anhand des Hashes)
                existing_version = self._find_existing_model_version(model_name, model_hash)
                
                if existing_version:
                    self.logger.info(f"Model with hash {model_hash} already exists as {model_name} version {existing_version}. Skipping registration.")
                    
                    # Füge existierende Versionsinformationen zum Ergebnis hinzu
                    registered_model = type('obj', (object,), {
                        'version': existing_version
                    })
                else:
                    self.logger.info(f"Registering model in MLflow Model Registry: {model_name} from {model_uri}")
                    
                    # Modell in der Registry registrieren
                    registered_model = mlflow.register_model(
                        model_uri=model_uri,
                        name=model_name
                    )
                    
                    # Speichere Hash als Beschreibung der Version für zukünftige Vergleiche
                    self.client.update_model_version(
                        name=model_name,
                        version=registered_model.version,
                        description=f"File hash: {model_hash}"
                    )
                    
                    # Handle model version stages
                    # First, get all versions of this model
                    try:
                        all_versions = self.client.get_latest_versions(model_name)
                        
                        # Set previous Production versions to Archived
                        for version in all_versions:
                            if version.current_stage == "Production":
                                self.client.transition_model_version_stage(
                                    name=model_name,
                                    version=version.version,
                                    stage="Archived"
                                )
                                self.logger.info(f"Moved previous Production model {model_name} version {version.version} to Archived stage")
                    except Exception as stage_e:
                        self.logger.warning(f"Error handling previous model versions: {str(stage_e)}")
                    
                    # Set new version to Production stage automatically as it came from a device
                    self.client.transition_model_version_stage(
                        name=model_name,
                        version=registered_model.version,
                        stage="Production"
                    )
                    self.logger.info(f"Set model {model_name} version {registered_model.version} to Production stage")
                
                # Erweiterte Ergebnisinformationen hinzufügen
                result.update({
                    "message": "Model imported and registered successfully",
                    "registered_model_name": model_name,
                    "registered_model_version": registered_model.version,
                    "stage": "Production"
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
        
    def _calculate_file_hash(self, file_path: str, algorithm: str = "sha256", buffer_size: int = 65536) -> str:
        """
        Berechnet den Hash einer Datei mit dem angegebenen Algorithmus.
        
        Args:
            file_path: Pfad zur Datei
            algorithm: Hash-Algorithmus ('md5', 'sha1', 'sha256')
            buffer_size: Puffergröße für die Verarbeitung
            
        Returns:
            Hexadezimaler Hash-String
        """
        if algorithm == "md5":
            hash_obj = hashlib.md5()
        elif algorithm == "sha1":
            hash_obj = hashlib.sha1()
        elif algorithm == "sha256":
            hash_obj = hashlib.sha256()
        else:
            raise ValueError(f"Unsupported hash algorithm: {algorithm}")
            
        with open(file_path, "rb") as f:
            while True:
                data = f.read(buffer_size)
                if not data:
                    break
                hash_obj.update(data)
                
        return hash_obj.hexdigest()
        
    def _find_existing_model_version(self, model_name: str, model_hash: str) -> Optional[int]:
        """
        Sucht nach einer existierenden Modellversion mit demselben Hash.
        
        Args:
            model_name: Name des Modells
            model_hash: Hash des Modells
            
        Returns:
            Versionsnummer, wenn eine existierende Version gefunden wurde, sonst None
        """
        try:
            # Prüfe, ob das Modell bereits existiert
            try:
                versions = self.client.get_latest_versions(model_name)
            except mlflow.exceptions.MlflowException:
                # Modell existiert noch nicht
                return None
                
            # Durchsuche alle Versionen nach dem angegebenen Hash
            for version in versions:
                description = version.description or ""
                if f"File hash: {model_hash}" in description:
                    # Hash gefunden - identisches Modell bereits registriert
                    return int(version.version)
                    
            # Kein passendes Modell gefunden
            return None
            
        except Exception as e:
            self.logger.warning(f"Error checking for existing model versions: {e}")
            return None