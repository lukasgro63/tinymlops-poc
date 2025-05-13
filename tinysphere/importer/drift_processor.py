"""
DriftProcessor for TinySphere
----------------------------
Processes drift event packages uploaded by TinyLCM devices.
"""

import json
import logging
from datetime import datetime
from pathlib import Path
from typing import Any, Dict, List

from sqlalchemy import desc
from sqlalchemy.orm import Session
from tinysphere.api.models.notification import NotificationCreate
from tinysphere.db.models import Device, DriftEvent, DriftSample, DriftType
from tinysphere.api.services.drift_service import DriftService

logger = logging.getLogger(__name__)

class DriftProcessor:
    """Processor for drift event packages."""
    
    def __init__(self):
        """Initialize the drift processor."""
        self.logger = logging.getLogger(f"{__name__}.{self.__class__.__name__}")
    
    def can_process(self, content_path: Path, metadata: Dict[str, Any]) -> bool:
        """
        Determine if this processor can handle the content.
        
        Args:
            content_path: Path to the extracted package content
            metadata: Package metadata
            
        Returns:
            True if this processor can handle the content, False otherwise
        """
        # Check package type
        package_type = metadata.get("package_type", "").lower()
        
        # Accept drift event packages
        if "drift" in package_type:
            self.logger.info(f"DriftProcessor handling package with type: {package_type}")
            return True
        
        # Check for drift event files
        drift_files = list(content_path.glob("**/drift_*.json"))
        drift_files.extend(list(content_path.glob("**/drift*.jsonl")))
        
        if drift_files:
            self.logger.info(f"DriftProcessor found {len(drift_files)} drift files in {package_type} package")
            return True
            
        return False
    
    def process(self, content_path: Path, metadata: Dict[str, Any]) -> Dict[str, Any]:
        """
        Process drift event package content.

        Args:
            content_path: Path to the extracted package content
            metadata: Package metadata

        Returns:
            Dictionary with processing results
        """
        package_id = metadata.get('package_id')
        device_id = metadata.get('device_id')
        self.logger.info(f"DriftProcessor processing drift event package: {package_id} from device: {device_id}")
        self.logger.info(f"Metadata: {metadata}")

        result = {
            "processor": "drift_events",
            "status": "processed",
            "processed_files": []
        }

        # Look for drift event files
        drift_files = list(content_path.glob("**/drift*.json"))
        drift_files.extend(list(content_path.glob("**/drift*.jsonl")))

        # Log found files for debugging
        for file in Path(content_path).glob("**/*"):
            if file.is_file():
                self.logger.info(f"Found file in drift package: {file}")

        if not drift_files:
            self.logger.warning(f"No drift event files found in package {package_id}")

            # Try more generic search for JSON files that might contain drift data
            json_files = list(content_path.glob("**/*.json"))
            if json_files:
                self.logger.info(f"Found {len(json_files)} JSON files, checking for drift content")
                for json_file in json_files:
                    try:
                        with open(json_file, "r") as f:
                            content = f.read()
                            if "drift" in content.lower() or "detector" in content.lower():
                                self.logger.info(f"JSON file {json_file} contains drift-related content")
                                drift_files.append(json_file)
                    except Exception as e:
                        self.logger.error(f"Error checking JSON file {json_file}: {e}")

            if not drift_files:
                result["status"] = "no_content"
                self.logger.warning(f"After additional checks, still no drift files found in package {package_id}")
                return result

        # Process each drift file
        events_found = 0
        created_events = 0
        for drift_file in drift_files:
            try:
                self.logger.info(f"Processing drift file: {drift_file}")

                # Read and parse drift data
                with open(drift_file, "r") as f:
                    if drift_file.suffix.lower() == ".json":
                        drift_data = json.load(f)
                        self.logger.info(f"Loaded drift data from JSON: {drift_data}")
                        events_found += 1

                        # Try to create drift event in database
                        try:
                            self._create_drift_event(drift_data, device_id, package_id)
                            created_events += 1
                        except Exception as e:
                            self.logger.error(f"Error creating drift event from {drift_file}: {e}")
                            import traceback
                            self.logger.error(traceback.format_exc())

                    elif drift_file.suffix.lower() == ".jsonl":
                        # For JSONL, process each line
                        line_count = 0
                        for line in f:
                            line = line.strip()
                            if line:
                                try:
                                    drift_data = json.loads(line)
                                    self.logger.info(f"Loaded drift data from JSONL line: {drift_data}")
                                    events_found += 1

                                    # Try to create drift event in database
                                    self._create_drift_event(drift_data, device_id, package_id)
                                    created_events += 1
                                except Exception as e:
                                    self.logger.error(f"Error processing JSONL line: {e}")
                                line_count += 1

                result["processed_files"].append(str(drift_file))

            except Exception as e:
                self.logger.error(f"Error processing drift file {drift_file}: {e}")
                import traceback
                self.logger.error(traceback.format_exc())
                result["errors"] = result.get("errors", [])
                result["errors"].append(f"Failed to process {drift_file}: {str(e)}")

        # Look for any sample images
        sample_images = list(content_path.glob("**/*.jpg"))
        sample_images.extend(list(content_path.glob("**/*.jpeg")))
        sample_images.extend(list(content_path.glob("**/*.png")))

        for img_file in sample_images:
            self.logger.info(f"Found sample image: {img_file}")
            result["processed_files"].append(str(img_file))

        # Update result with event counts
        result["drift_events_found"] = events_found
        result["drift_events_created"] = created_events
        result["sample_images_found"] = len(sample_images)

        if created_events > 0:
            result["status"] = "success"
            self.logger.info(f"Successfully created {created_events} drift events in database")
        elif events_found > 0:
            result["status"] = "partial"
            self.logger.warning(f"Found {events_found} drift events but failed to create them in database")
        else:
            result["status"] = "no_events"
            self.logger.warning("No drift events found in package")

        return result

    def _create_drift_event(self, data: Dict[str, Any], device_id: str, package_id: str) -> None:
        """
        Create a drift event in the database.

        Args:
            data: Drift event data
            device_id: ID of the device
            package_id: ID of the package
        """
        self.logger.info(f"Creating drift event in database for device {device_id}")

        # Extract drift event information
        detector_name = data.get("detector_name", "unknown")
        reason = data.get("reason", "")
        metrics = data.get("metrics", {})
        sample_data = data.get("sample", {})

        # Determine drift type
        drift_type = "unknown"  # Lowercase to match enum value string
        detector_lower = detector_name.lower() if detector_name else ""
        if "confidence" in detector_lower:
            drift_type = "confidence"  # Lowercase to match enum value string
        elif "distribution" in detector_lower:
            drift_type = "distribution"  # Lowercase to match enum value string
        elif "feature" in detector_lower:
            drift_type = "feature"  # Lowercase to match enum value string
        elif "concept" in detector_lower:
            drift_type = "custom"  # Use custom as the concept is not in enum
        elif "performance" in detector_lower:
            drift_type = "outlier"  # Use outlier as performance is not in enum
        elif "knn" in detector_lower or "distance" in detector_lower:
            drift_type = "knn_distance"  # Lowercase to match enum value string

        # Determine drift score
        drift_score = None
        if metrics and isinstance(metrics, dict):
            # Try various possible metric names for the drift score
            for key in ["change_magnitude", "drift_score", "magnitude", "score"]:
                if key in metrics:
                    drift_score = float(metrics[key])
                    self.logger.info(f"Found drift score in key '{key}': {drift_score}")
                    break

        if drift_score is None:
            # If no specific drift score metric found, check for any numeric values
            for key, value in metrics.items():
                if isinstance(value, (int, float)) and "confidence" not in key.lower():
                    drift_score = float(value)
                    self.logger.info(f"Using numeric value from '{key}' as drift score: {drift_score}")
                    break

        if drift_score is None:
            # Fall back to 0.0 if no suitable numeric value found
            drift_score = 0.0
            self.logger.warning("No suitable drift score found in metrics, using 0.0")

        # Parse timestamp
        timestamp = None
        if "timestamp" in data:
            try:
                if isinstance(data["timestamp"], (int, float)):
                    timestamp = datetime.fromtimestamp(data["timestamp"])
                elif isinstance(data["timestamp"], str):
                    timestamp = datetime.fromisoformat(data["timestamp"])
            except (ValueError, TypeError):
                timestamp = datetime.now()

        if timestamp is None:
            timestamp = datetime.now()

        # Prepare drift event data for the API
        event_data = {
            "device_id": device_id,
            "drift_type": drift_type,
            "drift_score": drift_score,
            "detector_name": detector_name,
            "model_id": data.get("model_id"),
            "description": reason,
            "timestamp": timestamp.isoformat(),
            "metadata": {
                "original_metrics": metrics,
                "package_id": package_id
            }
        }

        # Create drift event in database
        try:
            from tinysphere.api.services.drift_service import DriftService
            from sqlalchemy.orm import Session
            from tinysphere.api.dependencies.db import SessionLocal

            # Create a new DB session
            db = SessionLocal()
            try:
                self.logger.info(f"Creating drift event with data: {event_data}")
                DriftService.process_drift_event(db, device_id, event_data)
                self.logger.info(f"Successfully created drift event in database")

                # Add sample if provided
                if sample_data:
                    try:
                        # Get the newly created event
                        event = db.query(DriftEvent).filter(
                            DriftEvent.device_id == device_id,
                            DriftEvent.detector_name == detector_name
                        ).order_by(desc(DriftEvent.received_at)).first()

                        if event:
                            sample = {
                                "sample_id": sample_data.get("sample_id"),
                                "prediction": sample_data.get("prediction"),
                                "confidence": float(sample_data.get("confidence", 0.0)) if sample_data.get("confidence") is not None else None,
                                "drift_score": drift_score,
                                "timestamp": timestamp.isoformat(),
                            }

                            self.logger.info(f"Adding sample to drift event {event.event_id}: {sample}")
                            DriftService.add_drift_sample(db, event.event_id, sample)
                    except Exception as sample_error:
                        self.logger.error(f"Error adding sample to drift event: {sample_error}")
            finally:
                db.close()
        except Exception as db_error:
            self.logger.error(f"Error saving drift event to database: {db_error}")
            import traceback
            self.logger.error(traceback.format_exc())
            raise