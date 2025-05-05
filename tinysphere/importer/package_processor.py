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
        # Entweder ein dediziertes Metrics-Paket oder ein Modellpaket mit Metriken
        package_type = metadata.get("package_type")
        
        # Direkt akzeptieren, wenn es ein metrics-Paket ist
        if package_type == "metrics":
            return True
            
        # Für model-Pakete: Prüfen ob Metrikdateien vorhanden sind
        if package_type == "model" or package_type == "components":
            # Suchen nach Metrikdateien im Paket
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
        package_type = metadata.get("package_type")
        
        # Direkt akzeptieren für models-Pakete
        if package_type == "models":
            return True
            
        # Alle model- und components-Pakete akzeptieren
        if package_type == "model" or package_type == "components":
            # Nach Modelldateien suchen
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
        package_type = metadata.get("package_type")
        
        # Direkt akzeptieren für data_log-Pakete
        if package_type == "data_log":
            return True
            
        # Auch model und components Pakete akzeptieren, wenn sie Log-Dateien enthalten
        if package_type in ["model", "components"]:
            # Nach Log-Dateien suchen
            csv_files = list(content_path.glob("**/*.csv"))
            json_files = list(content_path.glob("**/*.json"))
            jsonl_files = list(content_path.glob("**/*.jsonl"))
            
            # Mindestens eine Log-Datei gefunden
            if csv_files or json_files or jsonl_files:
                file_count = len(csv_files) + len(json_files) + len(jsonl_files)
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
        
        # Process metrics based on package type
        try:
            package_type = metadata.get("package_type", "unknown")
            # extract_dir is now passed as an argument, no need to build it
            
            if package_type == "metrics":
                # Extract metrics from a metrics package
                metrics_files = list(Path(extract_dir).glob("**/metrics*.json"))
                for metrics_file in metrics_files:
                    try:
                        with open(metrics_file, "r") as f:
                            metrics_data = json.load(f)
                        
                        # Process inference time
                        if "inference_time" in metrics_data:
                            db.add(DeviceMetric(
                                device_id=device_id,
                                metric_type="inference_time",
                                value=float(metrics_data["inference_time"])
                            ))
                        
                        # Process CPU usage
                        if "cpu_usage" in metrics_data:
                            db.add(DeviceMetric(
                                device_id=device_id,
                                metric_type="cpu_usage",
                                value=float(metrics_data["cpu_usage"])
                            ))
                        
                        # Process memory usage
                        if "memory_usage" in metrics_data:
                            db.add(DeviceMetric(
                                device_id=device_id,
                                metric_type="memory_usage",
                                value=float(metrics_data["memory_usage"])
                            ))
                        
                        db.commit()
                        self.logger.info(f"Extracted metrics from {metrics_file} for device {device_id}")
                        
                    except Exception as e:
                        self.logger.error(f"Error processing metrics file {metrics_file}: {e}")
                        db.rollback()
            
            # Auch aus InferenceMonitor-Logs extrahieren
            if package_type in ["metrics", "inference_logs"]:
                inference_logs = list(Path(extract_dir).glob("**/inference_log*.jsonl"))
                for log_file in inference_logs:
                    try:
                        with open(log_file, "r") as f:
                            for line in f:
                                log_entry = json.loads(line.strip())
                                
                                # Latenz-Metrik extrahieren
                                if "latency_ms" in log_entry and log_entry["latency_ms"] is not None:
                                    db.add(DeviceMetric(
                                        device_id=device_id,
                                        metric_type="inference_time",
                                        value=float(log_entry["latency_ms"])
                                    ))
                                
                        db.commit()
                        self.logger.info(f"Extracted metrics from inference log {log_file} for device {device_id}")
                        
                    except Exception as e:
                        self.logger.error(f"Error processing inference log {log_file}: {e}")
                        db.rollback()
        
        except Exception as e:
            self.logger.error(f"Error extracting device metrics: {e}")
            db.rollback()