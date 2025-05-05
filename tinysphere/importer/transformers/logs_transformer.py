import csv
import datetime
import json
import logging
from pathlib import Path
from typing import Any, Dict, List

import mlflow

from tinysphere.importer.transformers.base import DataTransformer


class LogsTransformer(DataTransformer):
    def __init__(self):
        # Logger initialisieren
        self.logger = logging.getLogger(f"{__name__}.{self.__class__.__name__}")
    
    def can_transform(self, package_type: str, files: List[Path]) -> bool:
        # Unterstützte Pakettypen
        if package_type.lower() not in ["logs", "data_log", "components"]:
            return False
        
        # Erweiterte Suche nach Log-Dateien mit verschiedenen Namensmustern
        log_files = []
        log_extensions = ['.csv', '.jsonl', '.log']
        
        for file in files:
            # Nach Dateien mit unterstützten Endungen suchen
            if any(file.suffix.lower() == ext for ext in log_extensions):
                # Für JSONL-Dateien spezielle Regeln anwenden
                if file.suffix.lower() == '.jsonl':
                    # Standard-Logdateien akzeptieren
                    if file.name.lower() == 'inference_log.jsonl':
                        log_files.append(file)
                    # Auch Logdateien mit Zeitstempeln akzeptieren
                    elif file.name.lower().startswith('inference_log_'):
                        log_files.append(file)
                    # Allgemeine Logdateien mit extension akzeptieren
                    elif 'log' in file.name.lower():
                        log_files.append(file)
                # Für andere Log-Dateitypen sind wir weniger restriktiv
                else:
                    log_files.append(file)
        
        has_log_files = len(log_files) > 0
        
        # Für "components"-Pakete: Vermeiden von Konflikten mit anderen Transformern

        
        # Log-Dateien gefunden
        if has_log_files:
            self.logger.info(f"Found log files in package with type '{package_type}': {[f.name for f in log_files]}")
        
        return has_log_files
    
    def transform(self, package_id: str, device_id: str, files: List[Path], metadata: Dict[str, Any]) -> Dict[str, Any]:
        # MLflow-Experiment einrichten
        mlflow.set_experiment(f"device_{device_id}")
        
        # Erweiterte Log-Dateisuche
        log_files = []
        log_extensions = ['.csv', '.jsonl', '.log']
        
        for file in files:
            if any(file.suffix.lower() == ext for ext in log_extensions):
                if file.suffix.lower() == '.jsonl':
                    if file.name.lower() == 'inference_log.jsonl' or file.name.lower().startswith('inference_log_'):
                        log_files.append(file)
                    elif 'log' in file.name.lower():
                        log_files.append(file)
                else:
                    log_files.append(file)
        
        if not log_files:
            return {"status": "error", "message": "No log file found in package"}
        
        # Erste gefundene Log-Datei verwenden
        log_file = log_files[0]
        self.logger.info(f"Using log file: {log_file}")
        
        log_stats = {
            "entry_count": 0,
            "predictions": {},
            "confidence_avg": 0,
            "confidence_values": []
        }
        
        # Log-Datei parsen basierend auf dem Dateityp
        if log_file.suffix.lower() == '.csv':
            log_stats = self._process_csv_log(log_file)
        elif log_file.suffix.lower() == '.jsonl':
            log_stats = self._process_jsonl_log(log_file)
        else:
            # Einfache Textlog-Datei
            log_stats["entry_count"] = sum(1 for _ in open(log_file, 'r'))
            
        # Determine model name - check for combined package types
        package_type = metadata.get("package_type", "").lower()
        model_format = None
        
        # Get model information
        model_name = None
        model_version = None
        
        # Extract model format from combined package type, e.g., "model_logs" -> "model"
        if "_" in package_type:
            package_components = package_type.split("_")
            if "model" in package_components:
                # Look for model files to determine format
                model_extensions = ['.tflite', '.onnx', '.pt', '.pth', '.pkl', '.h5']
                model_files = [f for f in files if any(f.suffix.lower() == ext for ext in model_extensions)]
                
                if model_files:
                    model_format = model_files[0].suffix.lstrip('.')
                    
        # Try to find model_info.json or check log entries for model info
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
        
        # Check log entries for model_id in first few entries
        if log_file.suffix.lower() == '.jsonl' and not model_name:
            try:
                with open(log_file, 'r') as f:
                    for _ in range(10):  # Check first 10 entries
                        line = f.readline()
                        if not line:
                            break
                        try:
                            entry = json.loads(line)
                            if "model_id" in entry:
                                model_name = entry["model_id"]
                                break
                            if "model_name" in entry:
                                model_name = entry["model_name"]
                                break
                        except:
                            pass
            except Exception as e:
                self.logger.warning(f"Error checking log file for model ID: {str(e)}")
        
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
        
        # MLflow-Run starten
        with mlflow.start_run(run_name=f"logs_import_{package_id}"):
            # Parameter loggen
            mlflow.log_param("device_id", device_id)
            mlflow.log_param("package_id", package_id)
            mlflow.log_param("log_source", log_file.name)
            mlflow.log_param("entry_count", log_stats["entry_count"])
            
            # Log model information if available
            if model_name:
                mlflow.log_param("model_name", model_name)
            if model_format:
                mlflow.log_param("model_format", model_format)
            if production_version:
                mlflow.log_param("model_version", production_version)
            
            # Metriken loggen
            if log_stats["confidence_values"]:
                mlflow.log_metric("avg_confidence", log_stats["confidence_avg"])
            
            # Vorhersageverteilung als Parameter loggen
            for pred, count in log_stats["predictions"].items():
                mlflow.log_param(f"prediction_{pred}", count)
            
            # Create a standard logs directory for the artifact
            artifact_path = "logs"
            # Original-Datei als Artefakt in den logs-Ordner loggen
            mlflow.log_artifact(str(log_file), artifact_path)
            
            # Run-ID erhalten
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
                    
                    # 2. Create a direct alias for the logs run to appear in the model version UI
                    try:
                        # Log the file as an artifact with a standard name
                        artifact_path = f"logs/{log_file.name}"
                        
                        # Register the run as an additional version of the same model
                        mlflow.register_model(
                            f"runs:/{run_id}/{artifact_path}",
                            model_name,
                            tags={"logs_run_id": run_id, "original_run_id": run_id}
                        )
                        self.logger.info(f"Directly registered logs run with model {model_name}")
                    except Exception as reg_error:
                        self.logger.warning(f"Could not register run directly with model: {str(reg_error)}")
                        # Fallback - at least add proper tags
                        client.set_tag(run_id, "registered_model_name", model_name)
                        client.set_tag(run_id, "registered_model_version", production_version)
                    
                    self.logger.info(f"Linked logs run {run_id} to model {model_name} version {production_version}")
                except Exception as e:
                    self.logger.warning(f"Error linking run to model version: {str(e)}")
        
        return {
            "status": "success",
            "message": "Logs imported successfully",
            "mlflow_run_id": run_id,
            "entry_count": log_stats["entry_count"],
            "model_name": model_name,
            "model_version": production_version
        }
    
    def _process_csv_log(self, log_file: Path) -> Dict[str, Any]:
        """Process CSV log file"""
        stats = {
            "entry_count": 0,
            "predictions": {},
            "confidence_avg": 0,
            "confidence_values": []
        }
        
        with open(log_file, 'r') as f:
            reader = csv.DictReader(f)
            for row in reader:
                stats["entry_count"] += 1
                
                if "prediction" in row:
                    pred = row["prediction"]
                    if pred not in stats["predictions"]:
                        stats["predictions"][pred] = 0
                    stats["predictions"][pred] += 1
                
                if "confidence" in row:
                    try:
                        conf = float(row["confidence"])
                        stats["confidence_values"].append(conf)
                    except:
                        pass
        
        if stats["confidence_values"]:
            stats["confidence_avg"] = sum(stats["confidence_values"]) / len(stats["confidence_values"])
        
        return stats
    
    def _process_jsonl_log(self, log_file: Path) -> Dict[str, Any]:
        """Process JSONL log file"""
        stats = {
            "entry_count": 0,
            "predictions": {},
            "confidence_avg": 0,
            "confidence_values": []
        }
        
        with open(log_file, 'r') as f:
            for line in f:
                try:
                    entry = json.loads(line.strip())
                    stats["entry_count"] += 1
                    
                    if "prediction" in entry:
                        pred = entry["prediction"]
                        if pred not in stats["predictions"]:
                            stats["predictions"][pred] = 0
                        stats["predictions"][pred] += 1
                    
                    if "confidence" in entry and entry["confidence"] is not None:
                        conf = float(entry["confidence"])
                        stats["confidence_values"].append(conf)
                except:
                    pass
        
        if stats["confidence_values"]:
            stats["confidence_avg"] = sum(stats["confidence_values"]) / len(stats["confidence_values"])
        
        return stats