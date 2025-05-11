"""
Package processor for importing uploaded packages into the TinySphere platform.

This module handles extraction, validation, and processing of packages from TinyLCM clients.
"""

import json
import logging
import os
import shutil
import tarfile
import zipfile
from datetime import datetime
from pathlib import Path
from typing import Any, Dict, List, Optional, Protocol, Set, TypeVar, Union

from sqlalchemy.orm import Session

from tinysphere.db.models import Device, DeviceMetric
from tinysphere.importer.drift_processor import DriftProcessor

T = TypeVar('T')


class ProcessorPlugin(Protocol):
    """Protocol defining the interface for package content processors."""

    def can_process(self, content_path: Path, metadata: Dict[str, Any]) -> bool:
        """
        Determine if this processor can handle the given content.

        Args:
            content_path: Path to the extracted package content
            metadata: Package metadata

        Returns:
            True if this processor can handle the content, False otherwise
        """
        ...

    def process(self, content_path: Path, metadata: Dict[str, Any]) -> Dict[str, Any]:
        """
        Process the package content.

        Args:
            content_path: Path to the extracted package content
            metadata: Package metadata

        Returns:
            Dictionary with processing results
        """
        ...


class BaseProcessor:
    """Base class for package content processors."""
    
    def __init__(self):
        self.logger = logging.getLogger(f"{__name__}.{self.__class__.__name__}")


class MetricsProcessor(BaseProcessor):
    """Processor for metrics packages."""
    
    def can_process(self, content_path: Path, metadata: Dict[str, Any]) -> bool:
        """Check if this processor can handle the content."""
        # Get package type and consider combined types
        package_type = metadata.get("package_type", "")
        
        # Accept all packages that indicate they contain metrics
        if any(metric_type in package_type for metric_type in ["metrics", "metric"]):
            self.logger.info(f"Metrics processor handling package with type: {package_type}")
            return True
            
        # For model, components, or other packages: Check if they contain metrics files
        metrics_files = list(content_path.glob("**/metrics*.json"))
        if metrics_files:
            self.logger.info(f"Metrics processor will handle {package_type} package with {len(metrics_files)} metrics files")
            return True
                
        return False
    
    def process(self, content_path: Path, metadata: Dict[str, Any]) -> Dict[str, Any]:
        """Process metrics data."""
        self.logger.info(f"Processing metrics package: {metadata.get('package_id')}")
        result = {
            "processor": "metrics",
            "status": "processed",
            "processed_files": []
        }
        
        # Look for metrics files
        metrics_files = list(content_path.glob("**/metrics*.json"))
        for metrics_file in metrics_files:
            self.logger.debug(f"Found metrics file: {metrics_file}")
            try:
                with open(metrics_file, "r") as f:
                    metrics_data = json.load(f)
                
                result["processed_files"].append(str(metrics_file))
                result["metrics_found"] = True
                result["metrics_summary"] = {
                    "file": str(metrics_file),
                    "keys": list(metrics_data.keys())
                }
            except Exception as e:
                self.logger.error(f"Error processing metrics file {metrics_file}: {e}")
                result["errors"] = result.get("errors", [])
                result["errors"].append(f"Failed to process {metrics_file}: {str(e)}")
        
        return result


class ModelProcessor(BaseProcessor):
    """Processor for model packages."""
    
    def can_process(self, content_path: Path, metadata: Dict[str, Any]) -> bool:
        """Check if this processor can handle the content."""
        package_type = metadata.get("package_type", "")
        
        # Accept all packages that indicate they contain models
        if any(model_type in package_type for model_type in ["model", "models"]):
            # Verify by looking for actual model files
            model_files = []
            for ext in [".tflite", ".onnx", ".pt", ".pkl"]:
                model_files.extend(list(content_path.glob(f"**/*{ext}")))
                
            if model_files:
                self.logger.info(f"Model processor handling package with type: {package_type} containing {len(model_files)} model files")
                return True
            
            # Special case: Check for model_metadata files indicating a model reference without the actual model file
            model_metadata_files = list(content_path.glob("**/model_metadata*.json"))
            if model_metadata_files:
                self.logger.info(f"Model processor handling {package_type} package with model metadata (without model file)")
                return True
        
        # For any package type, check if we can find model files
        model_files = []
        for ext in [".tflite", ".onnx", ".pt", ".pkl"]:
            model_files.extend(list(content_path.glob(f"**/*{ext}")))
            
        if model_files:
            self.logger.info(f"Model processor will handle {package_type} package with {len(model_files)} model files")
            return True
            
        return False
    
    def process(self, content_path: Path, metadata: Dict[str, Any]) -> Dict[str, Any]:
        """Process model data."""
        self.logger.info(f"Processing model package: {metadata.get('package_id')}")
        result = {
            "processor": "models",
            "status": "processed",
            "processed_files": []
        }
        
        # Look for model files and metadata
        model_files = []
        for ext in [".tflite", ".onnx", ".pt", ".pkl"]:
            model_files.extend(list(content_path.glob(f"**/*{ext}")))
        
        model_metadata_files = list(content_path.glob("**/model_info*.json"))
        
        for model_file in model_files:
            self.logger.debug(f"Found model file: {model_file}")
            result["processed_files"].append(str(model_file))
        
        for meta_file in model_metadata_files:
            self.logger.debug(f"Found model metadata file: {meta_file}")
            try:
                with open(meta_file, "r") as f:
                    model_meta = json.load(f)
                
                result["processed_files"].append(str(meta_file))
                result["model_metadata_found"] = True
                result["model_metadata"] = {
                    "file": str(meta_file),
                    "format": model_meta.get("format"),
                    "flavor": model_meta.get("flavor")
                }
            except Exception as e:
                self.logger.error(f"Error processing model metadata file {meta_file}: {e}")
                result["errors"] = result.get("errors", [])
                result["errors"].append(f"Failed to process {meta_file}: {str(e)}")
        
        return result


class DataLogProcessor(BaseProcessor):
    """Processor for data log packages."""
    
    def can_process(self, content_path: Path, metadata: Dict[str, Any]) -> bool:
        """Check if this processor can handle the content."""
        package_type = metadata.get("package_type", "")
        
        # Accept all packages that indicate they contain data logs
        if any(log_type in package_type for log_type in ["data_log", "log"]):
            self.logger.info(f"DataLog processor handling package with type: {package_type}")
            return True
            
        # For any package type, check if it contains log files
        # Look for specific log file patterns first
        inference_logs = list(content_path.glob("**/inference_log*.jsonl"))
        data_logs = list(content_path.glob("**/data_log*.csv"))
        
        if inference_logs or data_logs:
            file_count = len(inference_logs) + len(data_logs)
            self.logger.info(f"DataLog processor found {file_count} specific log files in {package_type} package")
            return True
            
        # General log file search
        csv_files = list(content_path.glob("**/*.csv"))
        jsonl_files = list(content_path.glob("**/*.jsonl"))
        
        # At least one log file found
        if csv_files or jsonl_files:
            file_count = len(csv_files) + len(jsonl_files)
            self.logger.info(f"DataLog processor will handle {package_type} package with {file_count} log files")
            return True
                
        return False
    
    def process(self, content_path: Path, metadata: Dict[str, Any]) -> Dict[str, Any]:
        """Process data log files."""
        self.logger.info(f"Processing data log package: {metadata.get('package_id')}")
        result = {
            "processor": "data_logs",
            "status": "processed",
            "processed_files": []
        }
        
        # Look for CSV and JSON log files
        csv_files = list(content_path.glob("**/*.csv"))
        json_files = list(content_path.glob("**/*.json"))
        jsonl_files = list(content_path.glob("**/*.jsonl"))
        
        for csv_file in csv_files:
            self.logger.debug(f"Found CSV log file: {csv_file}")
            result["processed_files"].append(str(csv_file))
        
        for json_file in json_files:
            self.logger.debug(f"Found JSON log file: {json_file}")
            result["processed_files"].append(str(json_file))
        
        for jsonl_file in jsonl_files:
            self.logger.debug(f"Found JSONL log file: {jsonl_file}")
            result["processed_files"].append(str(jsonl_file))
        
        result["log_files_found"] = len(csv_files) + len(json_files) + len(jsonl_files)
        
        return result


class PackageImporter:
    """Processor for uploaded TinyLCM packages."""
    
    def __init__(self, extract_dir: str = "extracted_packages"):
        """
        Initialize a new package importer.

        Args:
            extract_dir: Directory where packages will be extracted
        """
        self.extract_dir = Path(extract_dir)
        self.processors: List[ProcessorPlugin] = [
            MetricsProcessor(),
            ModelProcessor(),
            DriftProcessor(),
            DataLogProcessor()
        ]
        self.logger = logging.getLogger(__name__)
        os.makedirs(self.extract_dir, exist_ok=True)
    
    def register_processor(self, processor: ProcessorPlugin) -> None:
        """
        Register a new processor for package content.
        
        Args:
            processor: Processor implementing the ProcessorPlugin protocol
        """
        self.processors.append(processor)
    
    def extract_package(self, package_path: str, package_id: str, device_id: str) -> Path:
        """
        Extract a package to the extraction directory.
        
        Args:
            package_path: Path to the package file
            package_id: ID of the package
            device_id: ID of the device that sent the package
            
        Returns:
            Path to the directory containing extracted content
        """
        package_path = Path(package_path)
        if not package_path.exists():
            raise FileNotFoundError(f"Package file not found: {package_path}")
        
        target_dir = self.extract_dir / device_id / package_id
        os.makedirs(target_dir, exist_ok=True)
        
        self.logger.info(f"Extracting package {package_id} to {target_dir}")
        
        if package_path.suffix == '.gz' and package_path.suffixes == ['.tar', '.gz']:
            with tarfile.open(package_path, 'r:gz') as tar:
                tar.extractall(path=target_dir)
        elif package_path.suffix == '.zip':
            with zipfile.ZipFile(package_path, 'r') as zip_ref:
                zip_ref.extractall(target_dir)
        elif package_path.suffix == '.tar':
            with tarfile.open(package_path, 'r') as tar:
                tar.extractall(path=target_dir)
        else:
            raise ValueError(f"Unsupported package format: {package_path.suffix}")
        
        self.logger.info(f"Package extracted successfully: {package_id}")
        return target_dir
    
    def process_package(self, package_path: str, metadata: Dict[str, Any], db: Session = None) -> Dict[str, Any]:
        """
        Process a package and return results.
        
        Args:
            package_path: Path to the package file
            metadata: Package metadata
            
        Returns:
            Dictionary with processing results
        """
        package_id = metadata.get("package_id")
        device_id = metadata.get("device_id")
        package_type = metadata.get("package_type", "unknown")
        
        if not package_id or not device_id:
            raise ValueError("Missing package_id or device_id in metadata")
        
        self.logger.info(f"Processing package: {package_id} (type: {package_type})")
        
        # Extract the package
        extract_dir = self.extract_package(package_path, package_id, device_id)

        # Initialize result dictionary early to use in error handling
        result = {
            "package_id": package_id,
            "device_id": device_id,
            "package_type": package_type,
            "extract_dir": str(extract_dir),
            "processed_at": datetime.now().isoformat(),
            "processors_applied": [],
            "file_count": 0,
            "processed_files": [],
            "errors": []
        }
        
        # Now that package is extracted, process device metrics
        if db is not None:
            try:
                self.extract_device_metrics(extract_dir, metadata, db)
            except Exception as e:
                self.logger.error(f"Error extracting device metrics: {e}")
                result["errors"].append(f"Metrics extraction failed: {str(e)}")
        
        # Count files
        file_count = 0
        for root, _, files in os.walk(extract_dir):
            file_count += len(files)
            for file in files:
                file_path = os.path.join(root, file)
                result["processed_files"].append(file_path)
        
        result["file_count"] = file_count
        
        # Inspect the package contents to refine the package type if needed
        refined_type = self._determine_package_type(extract_dir, package_type)
        if refined_type != package_type:
            self.logger.info(f"Refined package type from '{package_type}' to '{refined_type}' based on content")
            result["original_package_type"] = package_type
            result["package_type"] = refined_type
            metadata["package_type"] = refined_type  # Update metadata for processors
        
        # Apply applicable processors
        processor_results = []
        for processor in self.processors:
            if processor.can_process(extract_dir, metadata):
                try:
                    self.logger.debug(f"Applying processor {processor.__class__.__name__} to package {package_id}")
                    proc_result = processor.process(extract_dir, metadata)
                    processor_results.append(proc_result)
                    result["processors_applied"].append(processor.__class__.__name__)
                except Exception as e:
                    self.logger.error(f"Error in processor {processor.__class__.__name__}: {e}")
                    result["errors"] = result.get("errors", [])
                    result["errors"].append(f"Processor {processor.__class__.__name__} failed: {str(e)}")
        
        result["processor_results"] = processor_results
        
        self.logger.info(f"Package {package_id} processed successfully with {len(processor_results)} processors")
        return result
    
    def _determine_package_type(self, extract_dir: Path, original_type: str) -> str:
        """
        Determines a more accurate package type based on content inspection.

        Args:
            extract_dir: Directory containing extracted package contents
            original_type: Original package type from metadata

        Returns:
            Refined package type
        """
        # Check for model files
        model_files = []
        for ext in [".tflite", ".onnx", ".pt", ".pkl"]:
            model_files.extend(list(extract_dir.glob(f"**/*{ext}")))

        # Check for metrics files
        metrics_files = list(extract_dir.glob("**/metrics*.json"))

        # Check for drift event files
        drift_files = list(extract_dir.glob("**/drift*.json"))
        drift_files.extend(list(extract_dir.glob("**/drift*.jsonl")))

        # Check for log files
        log_files = list(extract_dir.glob("**/*.csv"))
        log_files.extend(list(extract_dir.glob("**/*.jsonl")))

        # Determine content types present
        has_model = len(model_files) > 0
        has_metrics = len(metrics_files) > 0
        has_drift = len(drift_files) > 0
        has_logs = len(log_files) > 0

        # If original type already includes underscores (combined type), keep it
        if "_" in original_type:
            return original_type

        # If the original type contains drift, prioritize it
        if "drift" in original_type.lower():
            return "drift_event"

        # Set the package type based on content
        if has_drift:
            if has_model and has_metrics:
                return "drift_model_metrics"
            elif has_model:
                return "drift_model"
            elif has_metrics:
                return "drift_metrics"
            else:
                return "drift_event"
        elif has_model and has_metrics and has_logs:
            return "model_metrics_logs"
        elif has_model and has_metrics:
            return "model_metrics"
        elif has_model and has_logs:
            return "model_logs"
        elif has_metrics and has_logs:
            return "metrics_logs"
        elif has_model:
            return "model"
        elif has_metrics:
            return "metrics"
        elif has_logs:
            return "data_log"

        # Default fallback to original type
        return original_type
    

    def extract_device_metrics(self, extract_dir: Path, metadata: Dict[str, Any], db: Session) -> None:
        """Extract device metrics from uploaded packages."""
        package_id = metadata.get("package_id")
        device_id = metadata.get("device_id")
        
        if not package_id or not device_id:
            self.logger.warning("Missing package_id or device_id in metadata")
            return
        
        # Find device
        device = db.query(Device).filter(Device.device_id == device_id).first()
        if not device:
            self.logger.warning(f"Device not found for extraction: {device_id}")
            return
        
        # Process metrics based on content, not just package type
        try:
            package_type = metadata.get("package_type", "unknown")
            self.logger.info(f"Extracting metrics from package: {package_id} (type: {package_type})")
            
            # Process metrics files regardless of package type
            metrics_files = list(Path(extract_dir).glob("**/metrics*.json"))
            metrics_processed = 0
            
            for metrics_file in metrics_files:
                try:
                    with open(metrics_file, "r") as f:
                        metrics_data = json.load(f)
                    
                    # Check for all possible metrics in a generic way
                    metrics_to_extract = {
                        "inference_time": ["inference_time", "latency", "latency_ms", "inference_latency"],
                        "cpu_usage": ["cpu_usage", "cpu_percent", "cpu_utilization"],
                        "memory_usage": ["memory_usage", "ram_usage", "memory_percent"],
                        "temperature": ["temperature", "temp", "device_temp"],
                        "battery": ["battery", "battery_level", "battery_percent"],
                        "disk_usage": ["disk_usage", "storage_usage", "disk_percent"],
                        "drift_score": ["drift_score", "drift_magnitude", "drift_level"]
                    }
                    
                    # Helper function for safe float conversion
                    def safe_float(value):
                        try:
                            return float(value)
                        except (ValueError, TypeError):
                            return None
                    
                    # Process each metric type
                    metrics_added = 0
                    for metric_type, possible_keys in metrics_to_extract.items():
                        for key in possible_keys:
                            if key in metrics_data and metrics_data[key] is not None:
                                value = safe_float(metrics_data[key])
                                if value is not None:
                                    db.add(DeviceMetric(
                                        device_id=device_id,
                                        metric_type=metric_type,
                                        value=value,
                                        timestamp=datetime.now()  # Add timestamp
                                    ))
                                    metrics_added += 1
                                    break  # Found a value for this metric type, move to next
                    
                    # Process nested metrics
                    for section_name, section_data in metrics_data.items():
                        if isinstance(section_data, dict):
                            for metric_type, possible_keys in metrics_to_extract.items():
                                for key in possible_keys:
                                    if key in section_data and section_data[key] is not None:
                                        value = safe_float(section_data[key])
                                        if value is not None:
                                            db.add(DeviceMetric(
                                                device_id=device_id,
                                                metric_type=f"{section_name}_{metric_type}",
                                                value=value,
                                                timestamp=datetime.now()
                                            ))
                                            metrics_added += 1
                                            break
                    
                    db.commit()
                    metrics_processed += 1
                    self.logger.info(f"Extracted {metrics_added} metrics from {metrics_file} for device {device_id}")
                    
                except Exception as e:
                    self.logger.error(f"Error processing metrics file {metrics_file}: {e}")
                    db.rollback()
            
            # Process inference logs for latency metrics
            inference_logs = list(Path(extract_dir).glob("**/inference_log*.jsonl"))
            for log_file in inference_logs:
                try:
                    line_count = 0
                    metrics_added = 0
                    
                    with open(log_file, "r") as f:
                        for line in f:
                            try:
                                log_entry = json.loads(line.strip())
                                line_count += 1
                                
                                # Extract latency metrics - check multiple possible keys
                                for key in ["latency_ms", "latency", "inference_time"]:
                                    if key in log_entry and log_entry[key] is not None:
                                        try:
                                            value = float(log_entry[key])
                                            db.add(DeviceMetric(
                                                device_id=device_id,
                                                metric_type="inference_time",
                                                value=value,
                                                timestamp=datetime.now()
                                            ))
                                            metrics_added += 1
                                            break  # Found one valid latency metric, no need to check others
                                        except (ValueError, TypeError):
                                            pass  # Skip if conversion to float fails
                            except json.JSONDecodeError:
                                continue  # Skip invalid JSON lines
                    
                    db.commit()
                    if metrics_added > 0:
                        self.logger.info(f"Extracted {metrics_added} latency metrics from {line_count} entries in {log_file}")
                    
                except Exception as e:
                    self.logger.error(f"Error processing inference log {log_file}: {e}")
                    db.rollback()
                    
            # Return early if we processed any metrics
            if metrics_processed > 0 or len(inference_logs) > 0:
                return
                
            # If no specific metrics files were found, check for any system stats
            system_logs = list(Path(extract_dir).glob("**/system_*.json"))
            system_logs.extend(list(Path(extract_dir).glob("**/stats_*.json")))
            
            for stats_file in system_logs:
                try:
                    with open(stats_file, "r") as f:
                        stats_data = json.load(f)
                        
                    metrics_added = 0
                    
                    # Check for common system stats
                    for stat_type in ["cpu", "memory", "disk", "temperature"]:
                        if stat_type in stats_data:
                            if isinstance(stats_data[stat_type], (int, float)):
                                db.add(DeviceMetric(
                                    device_id=device_id,
                                    metric_type=f"{stat_type}_usage",
                                    value=float(stats_data[stat_type]),
                                    timestamp=datetime.now()
                                ))
                                metrics_added += 1
                            elif isinstance(stats_data[stat_type], dict):
                                for key, value in stats_data[stat_type].items():
                                    if isinstance(value, (int, float)):
                                        db.add(DeviceMetric(
                                            device_id=device_id,
                                            metric_type=f"{stat_type}_{key}",
                                            value=float(value),
                                            timestamp=datetime.now()
                                        ))
                                        metrics_added += 1
                    
                    db.commit()
                    if metrics_added > 0:
                        self.logger.info(f"Extracted {metrics_added} system metrics from {stats_file}")
                        
                except Exception as e:
                    self.logger.error(f"Error processing system stats file {stats_file}: {e}")
                    db.rollback()
        
        except Exception as e:
            self.logger.error(f"Error extracting device metrics: {e}")
            db.rollback()