"""
DriftTransformer for TinySphere
-------------------------------
Transforms drift event packages uploaded by TinyLCM devices.
Integrates with MLflow for tracking and visualizing drift events.
"""

import json
import logging
import os
from datetime import datetime
from pathlib import Path
from typing import Any, Dict, List

import mlflow

from tinysphere.api.models.notification import NotificationCreate
from tinysphere.api.services.drift_service import DriftService
from tinysphere.db.models import DriftStatus, DriftType
from tinysphere.importer.transformers.base import DataTransformer

logger = logging.getLogger(__name__)

class DriftTransformer(DataTransformer):
    """
    Transformer for drift event packages.
    Extracts drift event data and samples from TinyLCM device packages.
    """
    
    def can_transform(self, package_type: str, files: List[Path]) -> bool:
        """
        Determine if this transformer can handle the given package.

        Args:
            package_type: Type of the package
            files: List of file paths in the package

        Returns:
            True if this transformer can handle the package, False otherwise
        """
        logger.info(f"DriftTransformer.can_transform called with package_type: {package_type}")

        # Check for drift event files with more robust detection first, to enable finding
        # drift data in any package, including prediction_images packages that might contain drift data
        drift_files = []
        drift_images = []

        # Log a few files for debugging
        file_count = 0
        for file in files:
            if file_count < 5:  # Only log first few files to avoid log spam
                logger.info(f"DriftTransformer examining file: {file}")

            # Check drift-related JSON or JSONL files
            if file.name.lower().startswith("drift") or "drift" in file.name.lower():
                if file.suffix.lower() in [".json", ".jsonl"]:
                    drift_files.append(file)
                    logger.info(f"DriftTransformer found drift JSON file: {file}")
                # Check for drift-related images
                elif file.suffix.lower() in [".jpg", ".jpeg", ".png"]:
                    drift_images.append(file)
                    logger.info(f"DriftTransformer found drift image file: {file}")
            file_count += 1

        # If we found actual drift files, we should process them regardless of package type
        if drift_files:
            logger.info(f"DriftTransformer found {len(drift_files)} drift JSON/JSONL files in package with type: {package_type}")
            return True

        # Accept packages with drift_event type (with more flexible matching)
        if package_type and "drift" in package_type.lower():
            logger.info(f"DriftTransformer will handle package with type: {package_type}")
            return True

        # If these are drift images in a package with drift in the name, handle them
        if drift_images and package_type and "drift" in package_type.lower():
            logger.info(f"DriftTransformer found {len(drift_images)} drift image files in package with type: {package_type}")
            return True

        # If no other criteria matched, this isn't a drift package
        logger.info(f"DriftTransformer cannot handle package of type: {package_type}")
        return False
    
    def transform(self, package_id: str, device_id: str, files: List[Path], metadata: Dict[str, Any]) -> Dict[str, Any]:
        """
        Transform drift event package content.

        Args:
            package_id: ID of the package
            device_id: ID of the device that sent the package
            files: List of file paths in the package
            metadata: Package metadata

        Returns:
            Dictionary with transformation results
        """
        logger.info(f"DriftTransformer.transform called for package: {package_id} from device: {device_id}")
        logger.info(f"Metadata: {metadata}")

        # Set up MLflow tracking
        mlflow.set_experiment(f"device_{device_id}_drift")

        result = {
            "transformer": "drift",
            "status": "success",  # Use "success" instead of "processed" to match other transformers
            "processed_files": [],
            "drift_events": []
        }

        # Process drift event files
        drift_files = []
        for file in files:
            logger.info(f"Examining file for drift data: {file}")
            if "drift" in str(file).lower() and file.suffix.lower() in [".json", ".jsonl"]:
                drift_files.append(file)
                logger.info(f"Added drift file for processing: {file}")

        if not drift_files:
            logger.warning(f"No drift event files found in package {package_id}")
            # Look for any files with "json" extension that might contain drift data
            json_files = [f for f in files if f.suffix.lower() == '.json']
            if json_files:
                logger.info(f"Found {len(json_files)} JSON files, checking for drift content")
                for json_file in json_files:
                    try:
                        with open(json_file, "r") as f:
                            content = f.read()
                            if "drift" in content.lower() or "detector" in content.lower():
                                logger.info(f"JSON file {json_file} contains drift-related content")
                                drift_files.append(json_file)
                    except Exception as e:
                        logger.error(f"Error checking JSON file {json_file}: {e}")

            if not drift_files:
                # Return with error status for proper handling
                result["status"] = "error"
                result["message"] = f"No drift files found in package {package_id}"
                logger.warning(f"After additional checks, still no drift files found in package {package_id}")
                return result

        # Determine model name - needed for MLflow integration
        model_name = None
        model_version = None

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
            except Exception as e:
                logger.warning(f"Could not read model info file: {str(e)}")

        # If no model info available, construct model name from device_id
        if not model_name:
            model_name = f"{device_id}-model"
            logger.info(f"Using constructed model name: {model_name}")

        # Find production version of the model if available
        production_version = None
        client = None
        found_model = False

        try:
            from mlflow.tracking import MlflowClient
            client = MlflowClient()

            # Search with the given model name
            if model_name:
                # First look for production versions
                versions = client.get_latest_versions(model_name, stages=["Production"])
                if versions:
                    production_version = versions[0].version
                    logger.info(f"Found production version {production_version} for model {model_name}")
                    found_model = True
                else:
                    # If no production version found, use the latest version
                    all_versions = client.get_latest_versions(model_name)
                    if all_versions:
                        production_version = all_versions[0].version
                        logger.info(f"No production version found, using latest version {production_version} for model {model_name}")
                        found_model = True

            # If no model found, try to find any model for this device
            if not found_model and device_id:
                logger.info(f"No model found with name {model_name}, searching for any model from device {device_id}")
                # Search for models with device_id prefix
                possible_model_prefix = f"{device_id}-"

                # Search all registered models
                registered_models = client.search_registered_models()
                for registered_model in registered_models:
                    if registered_model.name.startswith(possible_model_prefix):
                        model_name = registered_model.name
                        # Get latest version
                        versions = client.get_latest_versions(model_name, stages=["Production"])
                        if versions:
                            production_version = versions[0].version
                            logger.info(f"Found model {model_name} version {production_version} for device {device_id}")
                            found_model = True
                            break
                        else:
                            all_versions = client.get_latest_versions(model_name)
                            if all_versions:
                                production_version = all_versions[0].version
                                logger.info(f"Found model {model_name} version {production_version} for device {device_id}")
                                found_model = True
                                break

                if not found_model:
                    logger.warning(f"Could not find any model for device {device_id}")
        except Exception as e:
            logger.warning(f"Error finding production version: {str(e)}")

        # Process each drift file
        mlflow_run_ids = []  # List to store created MLflow run IDs

        for drift_file in drift_files:
            try:
                logger.debug(f"Processing drift file: {drift_file}")

                # Handle different file formats
                if drift_file.suffix.lower() == ".json":
                    with open(drift_file, "r") as f:
                        data = json.load(f)

                    # Process the single event
                    drift_event = {}
                    self._process_drift_event(data, device_id, result, package_id)
                    result["processed_files"].append(str(drift_file))

                    # Get the last processed drift event from the result
                    if result.get("drift_events"):
                        drift_event = result["drift_events"][-1]

                    # Log the drift event to MLflow
                    if drift_event:
                        mlflow_run_id = self._log_drift_to_mlflow(
                            drift_event=drift_event,
                            device_id=device_id,
                            model_name=model_name,
                            model_version=production_version,
                            package_id=package_id,
                            drift_file=drift_file
                        )
                        if mlflow_run_id:
                            mlflow_run_ids.append(mlflow_run_id)

                elif drift_file.suffix.lower() == ".jsonl":
                    # Process line by line for JSONL format
                    events_processed = 0
                    with open(drift_file, "r") as f:
                        for line in f:
                            line = line.strip()
                            if not line:
                                continue

                            try:
                                data = json.loads(line)

                                # Process drift event and update result
                                result_before = len(result.get("drift_events", []))
                                self._process_drift_event(data, device_id, result, package_id)
                                events_processed += 1

                                # Check if a new drift event was added
                                if len(result.get("drift_events", [])) > result_before:
                                    drift_event = result["drift_events"][-1]

                                    # Log the drift event to MLflow
                                    mlflow_run_id = self._log_drift_to_mlflow(
                                        drift_event=drift_event,
                                        device_id=device_id,
                                        model_name=model_name,
                                        model_version=production_version,
                                        package_id=package_id,
                                        drift_file=drift_file
                                    )
                                    if mlflow_run_id:
                                        mlflow_run_ids.append(mlflow_run_id)

                            except json.JSONDecodeError as e:
                                logger.error(f"Error parsing JSON line in {drift_file}: {e}")
                                continue

                    if events_processed > 0:
                        result["processed_files"].append(str(drift_file))
                        logger.info(f"Processed {events_processed} drift events from {drift_file}")

            except Exception as e:
                logger.error(f"Error processing drift file {drift_file}: {e}")
                result["errors"] = result.get("errors", [])
                result["errors"].append(f"Failed to process {drift_file}: {str(e)}")

        # Process sample images if any
        sample_images = []
        for file in files:
            if file.suffix.lower() in [".jpg", ".jpeg", ".png"] and "sample" in file.name.lower():
                sample_images.append(file)

        if sample_images:
            for img_file in sample_images:
                # Find sample ID from filename if possible
                sample_id = None
                file_name = img_file.stem
                if "_" in file_name:
                    parts = file_name.split("_")
                    if len(parts) > 1:
                        # Assume last part is sample_id
                        sample_id = parts[-1]

                # Record processed image
                result["processed_files"].append(str(img_file))
                result["sample_images"] = result.get("sample_images", [])
                result["sample_images"].append({
                    "file": str(img_file),
                    "sample_id": sample_id
                })

        # Check if we processed events successfully and set status accordingly
        processed_events = len(result.get('drift_events', []))
        logger.info(f"Drift transformer processed {processed_events} events from package {package_id}")

        # Add MLflow run IDs to result
        if mlflow_run_ids:
            result["mlflow_run_ids"] = mlflow_run_ids
            logger.info(f"Created {len(mlflow_run_ids)} MLflow runs for drift events")

        # Final result should use same format as other transformers
        if processed_events > 0:
            # We processed at least one event successfully
            return {
                "status": "success",
                "message": f"Successfully processed {processed_events} drift events",
                "drift_event_count": processed_events,
                "processed_files": result["processed_files"],
                "drift_events": result.get("drift_events", []),
                "mlflow_run_ids": mlflow_run_ids
            }
        else:
            # No events processed
            return {
                "status": "error",
                "message": "Failed to process any drift events",
                "processed_files": result["processed_files"],
                "errors": result.get("errors", [])
            }
    
    def _process_drift_event(self, data: Dict[str, Any], device_id: str, result: Dict[str, Any], package_id: str = None) -> None:
        """
        Process a single drift event and update the result.

        Args:
            data: Drift event data
            device_id: ID of the device
            result: Result dictionary to update
            package_id: ID of the package (optional)
        """
        try:
            logger.info(f"Processing drift event data: {data}")

            # Extract drift event information
            detector_name = data.get("detector_name", "unknown")
            reason = data.get("reason", "")
            metrics = data.get("metrics", {})
            sample_data = data.get("sample", {})

            logger.info(f"Extracted detector_name: {detector_name}, reason: {reason}")
            logger.info(f"Metrics: {metrics}")
            if sample_data:
                logger.info(f"Sample data available: {sample_data}")

            # Determine drift score from metrics
            drift_score = None
            if metrics and isinstance(metrics, dict):
                # Try various possible metric names for the drift score
                for key in ["change_magnitude", "drift_score", "magnitude", "score"]:
                    if key in metrics:
                        drift_score = float(metrics[key])
                        logger.info(f"Found drift score in key '{key}': {drift_score}")
                        break

            if drift_score is None:
                # If no specific drift score metric found, check for any numeric values
                for key, value in metrics.items():
                    if isinstance(value, (int, float)) and "confidence" not in key.lower():
                        drift_score = float(value)
                        logger.info(f"Using numeric value from '{key}' as drift score: {drift_score}")
                        break

            if drift_score is None:
                # Fall back to 0.0 if no suitable numeric value found
                drift_score = 0.0
                logger.warning("No suitable drift score found in metrics, using 0.0")

            # Prepare drift event data for the API
            drift_event = {
                "device_id": device_id,
                "drift_type": self._determine_drift_type(detector_name),
                "drift_score": drift_score,
                "detector_name": detector_name,
                "model_id": data.get("model_id"),
                "description": reason,
                "timestamp": self._parse_timestamp(data.get("timestamp")),
                "metadata": {
                    "original_metrics": metrics,
                    "package_id": package_id
                }
            }

            logger.info(f"Created drift event: {drift_event}")

            # Add sample if provided
            if sample_data:
                sample = {
                    "sample_id": sample_data.get("sample_id"),
                    "prediction": sample_data.get("prediction"),
                    "confidence": float(sample_data.get("confidence", 0.0)) if sample_data.get("confidence") is not None else None,
                    "drift_score": drift_score,
                    "timestamp": self._parse_timestamp(sample_data.get("timestamp")),
                    "metadata": {}
                }

                # Set feature and raw data paths if available
                feature_path = sample_data.get("feature_path")
                if feature_path:
                    sample["feature_path"] = feature_path

                raw_data_path = sample_data.get("raw_data_path")
                if raw_data_path:
                    sample["raw_data_path"] = raw_data_path

                logger.info(f"Added sample to drift event: {sample}")
                drift_event["samples"] = [sample]

            # Add to the results list
            result["drift_events"] = result.get("drift_events", [])
            result["drift_events"].append(drift_event)

            logger.info(f"Successfully processed drift event from detector {detector_name}")

            # Create a drift event in the database via the DriftService API
            try:
                from tinysphere.api.services.drift_service import DriftService
                from sqlalchemy.orm import Session
                from tinysphere.api.dependencies.db import SessionLocal

                # Create a new DB session
                db = SessionLocal()
                try:
                    logger.info(f"Creating drift event in database for device {device_id}")
                    DriftService.process_drift_event(db, device_id, drift_event)
                    logger.info(f"Successfully created drift event in database")
                finally:
                    db.close()
            except Exception as db_error:
                logger.error(f"Error saving drift event to database: {db_error}")
                result["warnings"] = result.get("warnings", [])
                result["warnings"].append(f"Failed to save drift event to database: {str(db_error)}")

        except Exception as e:
            import traceback
            logger.error(f"Error processing drift event: {e}")
            logger.error(traceback.format_exc())
            result["errors"] = result.get("errors", [])
            result["errors"].append(f"Failed to process drift event: {str(e)}")
    
    def _determine_drift_type(self, detector_name: str) -> str:
        """
        Determine drift type from detector name.
        
        Args:
            detector_name: Name of the detector
            
        Returns:
            Drift type as string
        """
        # Map common detector types to drift types
        detector_lower = detector_name.lower()
        
        if "confidence" in detector_lower:
            return "CONFIDENCE"
        elif "distribution" in detector_lower:
            return "FEATURE_DISTRIBUTION"
        elif "feature" in detector_lower:
            return "FEATURE"
        elif "concept" in detector_lower:
            return "CONCEPT"
        elif "performance" in detector_lower:
            return "PERFORMANCE"
        elif "knn" in detector_lower or "distance" in detector_lower:
            return "KNN_DISTANCE"
        else:
            return "UNKNOWN"
    
    def _log_drift_to_mlflow(self, drift_event: Dict[str, Any], device_id: str, model_name: str, model_version: str, package_id: str, drift_file: Path) -> str:
        """
        Log a drift event to MLflow as a run.

        Args:
            drift_event: Drift event data to log
            device_id: Device ID
            model_name: Model name (if available)
            model_version: Model version (if available)
            package_id: Package ID
            drift_file: Path to the drift file

        Returns:
            MLflow run ID if successful, None otherwise
        """
        try:
            # Set up the run name based on drift event data
            detector_name = drift_event.get("detector_name", "unknown")
            drift_type = drift_event.get("drift_type", "UNKNOWN")
            timestamp = drift_event.get("timestamp", datetime.now().isoformat())

            # Format timestamp for display
            try:
                if isinstance(timestamp, str):
                    dt = datetime.fromisoformat(timestamp)
                    timestamp_str = dt.strftime("%Y-%m-%d_%H-%M-%S")
                else:
                    timestamp_str = datetime.now().strftime("%Y-%m-%d_%H-%M-%S")
            except Exception:
                timestamp_str = datetime.now().strftime("%Y-%m-%d_%H-%M-%S")

            # Create a descriptive run name
            run_name = f"drift_{drift_type.lower()}_{detector_name}_{timestamp_str}"

            # Set up run tags
            run_tags = {
                "mlflow.source.name": f"device_{device_id}",
                "mlflow.source.type": "EDGE_DEVICE",
                "drift_type": drift_type,
                "detector_name": detector_name,
                "device_id": device_id,
                "package_id": package_id,
                "entity_type": "DRIFT_EVENT"
            }

            # Start MLflow run
            with mlflow.start_run(run_name=run_name, tags=run_tags):
                # Log drift score and other metrics
                drift_score = drift_event.get("drift_score", 0.0)
                mlflow.log_metric("drift_score", drift_score)

                # Log additional metrics from the original metrics
                metrics = drift_event.get("metadata", {}).get("original_metrics", {})
                if isinstance(metrics, dict):
                    for key, value in metrics.items():
                        if isinstance(value, (int, float)):
                            mlflow.log_metric(f"drift_{key}", value)

                # Log parameters
                mlflow.log_param("device_id", device_id)
                mlflow.log_param("package_id", package_id)
                mlflow.log_param("drift_type", drift_type)
                mlflow.log_param("detector_name", detector_name)

                if "description" in drift_event and drift_event["description"]:
                    mlflow.log_param("drift_reason", drift_event["description"])

                # Log model information if available
                if model_name:
                    mlflow.log_param("model_name", model_name)
                if model_version:
                    mlflow.log_param("model_version", model_version)

                # Log the drift file as an artifact
                artifact_path = "drift_data"
                mlflow.log_artifact(str(drift_file), artifact_path)

                # Get run info
                run_id = mlflow.active_run().info.run_id

                # Link the run to the model if we have model information
                if model_name and model_version and run_id:
                    try:
                        # Use the MLflow client for Registry operations
                        from mlflow.tracking import MlflowClient
                        client = MlflowClient()

                        # Set the standard MLflow tags for model association
                        client.set_tag(run_id, "mlflow.registeredModelName", model_name)
                        client.set_tag(run_id, "mlflow.registeredModelVersion", model_version)

                        # Create bidirectional link
                        try:
                            # Create alias for drift events
                            registered_alias = f"drift_{device_id}"

                            # Try to set the registered model alias
                            try:
                                client.set_registered_model_alias(
                                    name=model_name,
                                    version=model_version,
                                    alias=registered_alias
                                )
                                logger.info(f"Set alias {registered_alias} for model {model_name} version {model_version}")
                            except Exception as alias_error:
                                logger.warning(f"Could not set alias: {str(alias_error)}")

                                # If alias is not supported, register as a separate model
                                try:
                                    alias_model_name = f"{model_name}_drift"

                                    # Check if model exists, create if not
                                    try:
                                        client.get_registered_model(alias_model_name)
                                    except:
                                        client.create_registered_model(alias_model_name)

                                    # Create a dummy artifact to register
                                    import tempfile
                                    with tempfile.NamedTemporaryFile(suffix=".json") as tmp:
                                        # Write drift data to temp file
                                        json_str = json.dumps(drift_event, indent=2)
                                        tmp.write(json_str.encode('utf-8'))
                                        tmp.flush()

                                        # Log as artifact
                                        mlflow.log_artifact(tmp.name, "drift_data")

                                    # Get artifact URI for the drift data
                                    artifact_uri = f"runs:/{run_id}/drift_data"

                                    # Register the drift event as a "model"
                                    model_details = mlflow.register_model(
                                        model_uri=artifact_uri,
                                        name=alias_model_name,
                                        tags={
                                            "original_model": model_name,
                                            "original_version": model_version,
                                            "drift_run_id": run_id,
                                            "device_id": device_id,
                                            "drift_type": drift_type
                                        }
                                    )
                                    logger.info(f"Registered drift as model {alias_model_name} version {model_details.version}")
                                except Exception as model_reg_error:
                                    logger.warning(f"Error registering drift as model: {str(model_reg_error)}")

                            # Additional tags for UI display
                            additional_tags = {
                                "drift_type": drift_type,
                                "device_id": device_id,
                                "model_link": f"models:/{model_name}/{model_version}",
                                "linked_entity": "DRIFT",
                                "registered_model_name": model_name,
                                "registered_model_version": model_version,
                                "drift_score": str(drift_score)
                            }

                            for tag_key, tag_value in additional_tags.items():
                                client.set_tag(run_id, tag_key, str(tag_value))

                            # Add backward link from model to run
                            try:
                                client.set_model_version_tag(
                                    name=model_name,
                                    version=model_version,
                                    key="drift.runId",
                                    value=run_id
                                )

                                # Add drift score tag to model version for easy visibility
                                client.set_model_version_tag(
                                    name=model_name,
                                    version=model_version,
                                    key="drift.score",
                                    value=str(drift_score)
                                )

                                # Add drift type tag to model version
                                client.set_model_version_tag(
                                    name=model_name,
                                    version=model_version,
                                    key="drift.type",
                                    value=drift_type
                                )
                            except Exception as tag_error:
                                logger.warning(f"Error setting model version tags: {str(tag_error)}")

                        except Exception as reg_error:
                            logger.warning(f"Could not register run with model directly: {str(reg_error)}")
                            # Fallback - at least add basic tags
                            client.set_tag(run_id, "registered_model_name", model_name)
                            client.set_tag(run_id, "registered_model_version", model_version)

                        logger.info(f"Linked drift run {run_id} to model {model_name} version {model_version}")

                    except Exception as e:
                        logger.warning(f"Error linking run to model version: {str(e)}")

                logger.info(f"Created MLflow run {run_id} for drift event from {detector_name}")
                return run_id

        except Exception as e:
            logger.error(f"Error logging drift event to MLflow: {e}")
            return None

    def _parse_timestamp(self, timestamp) -> str:
        """
        Parse timestamp from various formats to ISO format.

        Args:
            timestamp: Timestamp in various formats (Unix timestamp, ISO string)

        Returns:
            Timestamp in ISO format
        """
        if timestamp is None:
            return datetime.now().isoformat()

        try:
            if isinstance(timestamp, (int, float)):
                dt = datetime.fromtimestamp(timestamp)
                return dt.isoformat()
            elif isinstance(timestamp, str):
                # Try to parse as ISO
                dt = datetime.fromisoformat(timestamp)
                return dt.isoformat()
        except (ValueError, TypeError):
            pass

        # Default to current time
        return datetime.now().isoformat()