"""
DriftTransformer for TinySphere
-------------------------------
Transforms drift event packages uploaded by TinyLCM devices.
"""

import json
import logging
import os
from datetime import datetime
from pathlib import Path
from typing import Any, Dict, List

from tinysphere.api.models.notification import NotificationCreate
from tinysphere.api.services.drift_service import DriftService
from tinysphere.db.models import DriftStatus, DriftType, NotificationType
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

        # Log all files for debugging
        for file in files:
            logger.info(f"DriftTransformer examining file: {file}")

        # Accept packages with drift_event type (with more flexible matching)
        if package_type and "drift" in package_type.lower():
            logger.info(f"DriftTransformer will handle package with type: {package_type}")
            return True

        # Check for drift event files with more robust detection
        drift_files = []
        drift_images = []

        for file in files:
            # Check drift-related JSON or JSONL files
            if file.name.lower().startswith("drift") or "drift" in file.name.lower():
                if file.suffix.lower() in [".json", ".jsonl"]:
                    drift_files.append(file)
                    logger.info(f"DriftTransformer found drift JSON file: {file}")
                # Check for drift-related images
                elif file.suffix.lower() in [".jpg", ".jpeg", ".png"]:
                    drift_images.append(file)
                    logger.info(f"DriftTransformer found drift image file: {file}")

        # Even if no drift JSON files, if we have drift images with a package_type containing 'drift', we can process it
        if drift_files:
            logger.info(f"DriftTransformer found {len(drift_files)} drift JSON/JSONL files in package with type: {package_type}")
            return True
        elif drift_images and package_type and "drift" in package_type.lower():
            logger.info(f"DriftTransformer found {len(drift_images)} drift image files in package with type: {package_type}")
            return True

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

        result = {
            "transformer": "drift",
            "status": "processed",
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
                result["status"] = "no_content"
                logger.warning(f"After additional checks, still no drift files found in package {package_id}")
                return result
        
        # Process each drift file
        for drift_file in drift_files:
            try:
                logger.debug(f"Processing drift file: {drift_file}")
                
                # Handle different file formats
                if drift_file.suffix.lower() == ".json":
                    with open(drift_file, "r") as f:
                        data = json.load(f)
                    
                    # Process the single event
                    self._process_drift_event(data, device_id, result)
                    result["processed_files"].append(str(drift_file))
                    
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
                                self._process_drift_event(data, device_id, result)
                                events_processed += 1
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
        
        logger.info(f"Drift transformer processed {len(result.get('drift_events', []))} events from package {package_id}")
        return result
    
    def _process_drift_event(self, data: Dict[str, Any], device_id: str, result: Dict[str, Any]) -> None:
        """
        Process a single drift event and update the result.

        Args:
            data: Drift event data
            device_id: ID of the device
            result: Result dictionary to update
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
        else:
            return "UNKNOWN"
    
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