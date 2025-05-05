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
        client = None
        found_model = False
        
        try:
            from mlflow.tracking import MlflowClient
            client = MlflowClient()
            
            # Direkte Suche mit dem gegebenen Modellnamen
            if model_name:
                # Suche zuerst nach Production-Versionen
                versions = client.get_latest_versions(model_name, stages=["Production"])
                if versions:
                    production_version = versions[0].version
                    self.logger.info(f"Found production version {production_version} for model {model_name}")
                    found_model = True
                else:
                    # Wenn keine Production-Version gefunden wurde, nimm die neueste Version
                    all_versions = client.get_latest_versions(model_name)
                    if all_versions:
                        production_version = all_versions[0].version
                        self.logger.info(f"No production version found, using latest version {production_version} for model {model_name}")
                        found_model = True
            
            # Wenn kein Modell gefunden wurde, versuche alle Modelle für dieses Gerät zu finden
            if not found_model and device_id:
                self.logger.info(f"No model found with name {model_name}, searching for any model from device {device_id}")
                # Alle Modelle suchen, die das device_id-Präfix haben
                possible_model_prefix = f"{device_id}-"
                
                # Alle registrierten Modelle durchsuchen
                registered_models = client.search_registered_models()
                for registered_model in registered_models:
                    if registered_model.name.startswith(possible_model_prefix):
                        model_name = registered_model.name
                        # Neueste Version nehmen
                        versions = client.get_latest_versions(model_name, stages=["Production"])
                        if versions:
                            production_version = versions[0].version
                            self.logger.info(f"Found model {model_name} version {production_version} for device {device_id}")
                            found_model = True
                            break
                        else:
                            all_versions = client.get_latest_versions(model_name)
                            if all_versions:
                                production_version = all_versions[0].version
                                self.logger.info(f"Found model {model_name} version {production_version} for device {device_id}")
                                found_model = True
                                break
                
                if not found_model:
                    self.logger.warning(f"Could not find any model for device {device_id}")
        except Exception as e:
            self.logger.warning(f"Error finding production version: {str(e)}")
        
        # MLflow-Run starten mit korrekten Quelltags
        run_tags = {
            "mlflow.source.name": f"device_{device_id}",
            "mlflow.source.type": "EDGE_DEVICE"
        }
        with mlflow.start_run(run_name=f"logs_import_{package_id}", tags=run_tags):
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
            if model_name and production_version and found_model:
                try:
                    # Wir verwenden den bereits initialisierten Client von oben
                    # Wenn noch nicht erstellt, erzeugen wir ihn hier
                    if not client:
                        from mlflow.tracking import MlflowClient
                        client = MlflowClient()
                    
                    # Log the run directly to the model version
                    # First, get the model version details
                    model_version = client.get_model_version(model_name, production_version)
                    
                    # Create a link between the run and the model version
                    # This makes the run appear under the model in the UI by using MLflow's standard tags
                    
                    # Set the key MLflow standard tags for associating with a registered model
                    client.set_tag(run_id, "mlflow.registeredModelName", model_name)
                    client.set_tag(run_id, "mlflow.registeredModelVersion", production_version)
                    
                    # Bidirektionale Verknüpfung herstellen:
                    try:
                        # Set the artifact path for consistency
                        artifact_path = f"logs/{log_file.name}"
                        
                        # Direkte Verknüpfung mit MLflow
                        # 1. Hol die Modell-URI, mit der wir das Modell in MLflow registerieren können
                        model_artifact_path = "logs"  # Standardpfad für Logs-Artefakte
                        
                        # Wir benötigen einen Artefakt als "Modell" für die Registrierung
                        log_file_name = log_file.name
                        log_uri = f"runs:/{run_id}/{model_artifact_path}/{log_file_name}"
                        
                        self.logger.info(f"Registering logs run {run_id} with model {model_name} version {production_version}")
                        
                        try:
                            # KRITISCH: Hier fügen wir den Run dem Modell als registrierte Version hinzu
                            # Dies ist der Schlüssel, um im MLflow UI in der "Registered Models"-Sektion zu erscheinen
                            # Wir müssen das als Alias-Tag setzen, damit es richtig angezeigt wird
                            registered_alias = f"logs_{device_id}"
                            
                            # Setze Tags, die vom MLflow UI für die Modellverknüpfung verwendet werden
                            client.set_tag(run_id, "mlflow.registeredModelName", model_name)
                            client.set_tag(run_id, "mlflow.registeredModelVersion", production_version)
                            
                            # 2. Registriere die Logs als Alias des Modells
                            try:
                                # Versuche, einen Alias für das Modell zu erstellen
                                client.set_registered_model_alias(
                                    name=model_name,
                                    version=production_version,
                                    alias=registered_alias
                                )
                                self.logger.info(f"Set alias {registered_alias} for model {model_name} version {production_version}")
                            except Exception as alias_error:
                                self.logger.warning(f"Could not set alias: {str(alias_error)}")
                                
                                # Wenn Alias nicht unterstützt wird, registriere als separates Modell
                                try:
                                    alias_model_name = f"{model_name}_logs"
                                    client.create_registered_model(alias_model_name)
                                    mlflow.register_model(
                                        model_uri=log_uri,
                                        name=alias_model_name,
                                        tags={
                                            "original_model": model_name,
                                            "original_version": production_version,
                                            "logs_run_id": run_id,
                                            "device_id": device_id
                                        }
                                    )
                                    self.logger.info(f"Registered logs as model {alias_model_name}")
                                except Exception as model_reg_error:
                                    self.logger.warning(f"Error registering logs as model: {str(model_reg_error)}")
                            
                            # 3. Zusätzliche Informationen für UI-Anzeige
                            additional_tags = {
                                "logs_type": "model_logs",
                                "device_id": device_id,
                                "model_link": f"models:/{model_name}/{production_version}",
                                "linked_entity": "LOGS",
                                "registered_model_name": model_name,
                                "registered_model_version": production_version
                            }
                            
                            for tag_key, tag_value in additional_tags.items():
                                client.set_tag(run_id, tag_key, str(tag_value))
                            
                            # 4. Rückwärtsverknüpfung vom Modell zum Run
                            client.set_model_version_tag(
                                name=model_name,
                                version=production_version,
                                key="logs.runId",
                                value=run_id
                            )
                        except Exception as e:
                            self.logger.error(f"Error registering logs with model: {str(e)}")
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