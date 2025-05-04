import logging
import os
import traceback
from datetime import datetime
from pathlib import Path
from typing import Any, Dict, List, Optional

from sqlalchemy.orm import Session

from tinysphere.api.models.package import PackageUpdate
from tinysphere.api.services.notification_service import NotificationService
from tinysphere.api.services.package_service import (PackageService,
                                                     processing_status_cache)
from tinysphere.importer.package_processor import PackageImporter
from tinysphere.importer.transformers import (LogsTransformer,
                                              MetricsTransformer,
                                              ModelTransformer)
from tinysphere.importer.transformers.base import DataTransformer

logging.basicConfig(level=logging.INFO)
logger = logging.getLogger(__name__)


class MLflowService:
    def __init__(self):
        self.package_importer = PackageImporter(extract_dir="package_extracts")
        self.transformers = [
            ModelTransformer(),
            MetricsTransformer(),
            LogsTransformer()
        ]
    
    def process_package(self, db: Session, package_id: str) -> Dict[str, Any]:
        try:
            logger.info(f"Processing package {package_id}")
            
            # Fortschritt initialisieren
            PackageService.set_processing_progress(
                package_id, 
                progress=0.0, 
                status="starting", 
                step="Initializing package processing"
            )
            
            package = PackageService.get_package_by_id(db, package_id)
            if not package:
                logger.warning(f"Package {package_id} not found")
                PackageService.set_processing_progress(
                    package_id, 
                    progress=1.0, 
                    status="error", 
                    step="Package not found"
                )
                return {"status": "error", "message": f"Package {package_id} not found"}
            
            if package.is_processed:
                logger.info(f"Package {package_id} already processed")
                PackageService.set_processing_progress(
                    package_id, 
                    progress=1.0, 
                    status="completed", 
                    step="Package already processed"
                )
                return {"status": "info", "message": f"Package {package_id} already processed"}
            
            # Verify file exists
            if not os.path.exists(package.file_path):
                error_msg = f"Package file not found at {package.file_path}"
                logger.error(error_msg)
                self._update_package_status(db, package_id, "error", error_msg)
                PackageService.set_processing_progress(
                    package_id, 
                    progress=1.0, 
                    status="error", 
                    step="Package file not found"
                )
                return {"status": "error", "message": error_msg}

            # Log MLflow environment settings
            mlflow_uri = os.environ.get("MLFLOW_TRACKING_URI", "http://mlflow:5000")
            s3_endpoint = os.environ.get("MLFLOW_S3_ENDPOINT_URL", "http://minio:9000")
            aws_key = os.environ.get("AWS_ACCESS_KEY_ID", "")
            logger.info(f"MLflow URI: {mlflow_uri}, S3 Endpoint: {s3_endpoint}, AWS Key: {aws_key[:4]}...")
            
            try:
                # Test MLflow connection
                import mlflow
                mlflow.set_tracking_uri(mlflow_uri)
                experiments = mlflow.search_experiments()
                logger.info(f"Found {len(experiments)} MLflow experiments")
            except Exception as mlflow_err:
                logger.error(f"MLflow connection error: {str(mlflow_err)}")
                logger.error(traceback.format_exc())
                    
            # Fortschritt aktualisieren
            PackageService.set_processing_progress(
                package_id, 
                progress=0.1, 
                status="extracting", 
                step="Extracting package files"
            )
            
            # Extract the package
            extract_dir = self.package_importer.extract_package(
                package_path=package.file_path,
                package_id=package.package_id,
                device_id=package.device_id
            )
            logger.info(f"Package extracted to {extract_dir}")
            
            # Fortschritt aktualisieren
            PackageService.set_processing_progress(
                package_id, 
                progress=0.3, 
                status="scanning", 
                step="Scanning extracted files"
            )
            
            # Get all files in the extracted directory
            extracted_files = []
            extract_path = Path(extract_dir)
            for file_path in extract_path.glob("**/*"):
                if file_path.is_file():
                    extracted_files.append(file_path)
                    logger.info(f"Found file: {file_path}")
            
            logger.info(f"Found {len(extracted_files)} files in package")
            
            # Fortschritt aktualisieren
            PackageService.set_processing_progress(
                package_id, 
                progress=0.4, 
                status="analyzing", 
                step=f"Analyzing {len(extracted_files)} files"
            )
            
            # Find all appropriate transformers for the package
            package_type = package.package_type
            metadata = package.package_metadata or {}
            
            # Sammle alle passenden Transformer
            matching_transformers = []
            for transformer in self.transformers:
                logger.info(f"Checking transformer {transformer.__class__.__name__}")
                if transformer.can_transform(package_type, extracted_files):
                    logger.info(f"Found matching transformer: {transformer.__class__.__name__}")
                    matching_transformers.append(transformer)
            
            if not matching_transformers:
                error_msg = f"No appropriate transformers found for package type: {package_type}"
                logger.warning(error_msg)
                self._update_package_status(db, package_id, "error", error_msg)
                PackageService.set_processing_progress(
                    package_id, 
                    progress=1.0, 
                    status="error", 
                    step="No matching transformers found"
                )
                return {"status": "error", "message": error_msg}
            
            # Wende alle passenden Transformer an
            results = []
            success_count = 0
            
            # Fortschritt für Transformer-Anwendung berechnen
            transformer_progress_per_step = 0.5 / len(matching_transformers)
            current_progress = 0.4  # Beginnend bei 40%
            
            for i, transformer in enumerate(matching_transformers):
                transformer_name = transformer.__class__.__name__
                logger.info(f"Applying transformer {transformer_name}")
                
                # Fortschritt aktualisieren
                current_progress += transformer_progress_per_step / 2
                PackageService.set_processing_progress(
                    package_id, 
                    progress=current_progress, 
                    status="transforming", 
                    step=f"Applying transformer {i+1}/{len(matching_transformers)}: {transformer_name}"
                )
                
                try:
                    result = transformer.transform(
                        package_id=package.package_id,
                        device_id=package.device_id,
                        files=extracted_files,
                        metadata=metadata
                    )
                    
                    transformer_result = {
                        "transformer": transformer_name,
                        "result": result
                    }
                    
                    if result.get("status") == "success":
                        success_count += 1
                    
                    results.append(transformer_result)
                    logger.info(f"Transformer {transformer_name} result: {result}")
                    
                    # Fortschritt nach erfolgreichem Abschluss aktualisieren
                    current_progress += transformer_progress_per_step / 2
                    PackageService.set_processing_progress(
                        package_id, 
                        progress=current_progress, 
                        status="transforming", 
                        step=f"Completed transformer {transformer_name}"
                    )
                    
                except Exception as e:
                    error_msg = f"Transformer error in {transformer_name}: {str(e)}"
                    logger.error(error_msg)
                    logger.error(traceback.format_exc())
                    
                    results.append({
                        "transformer": transformer_name,
                        "status": "error",
                        "message": error_msg
                    })
                    
                    # Fortschritt trotz Fehler aktualisieren
                    current_progress += transformer_progress_per_step / 2
                    PackageService.set_processing_progress(
                        package_id, 
                        progress=current_progress, 
                        status="partial_error", 
                        step=f"Error in transformer {transformer_name}: {str(e)}"
                    )
            
            # Update package status based on overall results
            if success_count > 0:
                self._update_package_status(db, package_id, "success")
                
                # Fortschritt auf 90% setzen
                PackageService.set_processing_progress(
                    package_id, 
                    progress=0.9, 
                    status="finalizing", 
                    step="Finalizing successful processing"
                )
            else:
                error_msgs = []
                for result in results:
                    if "status" in result and result.get("status") == "error":
                        error_msgs.append(result.get("message", "Unknown error"))
                    elif "result" in result and result["result"].get("status") == "error":
                        error_msgs.append(result["result"].get("message", "Unknown error"))
                
                error_summary = "; ".join(error_msgs) if error_msgs else "Unknown error"
                self._update_package_status(db, package_id, "error", error_summary)
                
                # Fortschritt auf 90% setzen, aber mit Fehlerstatus
                PackageService.set_processing_progress(
                    package_id, 
                    progress=0.9, 
                    status="error", 
                    step=f"Processing failed: {error_summary[:100]}" + ('...' if len(error_summary) > 100 else '')
                )

            # Extract device metrics from package if possible
            try:
                self.package_importer.extract_device_metrics(
                    package_path=extract_dir,
                    metadata=metadata,
                    db=db
                )
            except Exception as e:
                logger.warning(f"Error extracting device metrics: {e}")

            # Fortschritt auf 100% setzen - Verarbeitung abgeschlossen
            final_status = "success" if success_count > 0 else "error"
            PackageService.set_processing_progress(
                package_id, 
                progress=1.0, 
                status=final_status, 
                step="Processing completed"
            )
            
            # Return combined results with summary
            return {
                "status": final_status,
                "message": f"Processed package with {success_count}/{len(matching_transformers)} successful transformers",
                "transformers_applied": len(matching_transformers),
                "successful_transformers": success_count,
                "components_processed": [r["transformer"].replace("Transformer", "").lower() for r in results if r.get("result", {}).get("status") == "success"],
                "results": results
            }

        except Exception as e:
            error_msg = f"Error processing package {package_id}: {str(e)}"
            logger.error(error_msg)
            logger.error(traceback.format_exc())
            self._update_package_status(db, package_id, "error", str(e))
            
            # Fortschritt auf 100% setzen mit Fehlerstatus
            PackageService.set_processing_progress(
                package_id, 
                progress=1.0, 
                status="error", 
                step=f"Unhandled error: {str(e)[:100]}" + ('...' if len(str(e)) > 100 else '')
            )
            
            return {"status": "error", "message": str(e)}
    
    def _update_package_status(self, db: Session, package_id: str, status: str, error_message: str = None):
        try:
            update_data = {
                "processing_status": status
            }
            
            if status == "success":
                update_data["is_processed"] = True
                update_data["processed_at"] = datetime.utcnow()
                
                # Create success notification
                NotificationService.notify_package_processed(
                    db, package_id=package_id, success=True
                )
            elif status == "error" and error_message:
                update_data["processing_error"] = error_message
                
                # Create error notification
                NotificationService.notify_package_processed(
                    db, package_id=package_id, success=False, error_msg=error_message
                )
            
            package_update = PackageUpdate(**update_data)
            PackageService.update_package(db, package_id, package_update)
        except Exception as e:
            logger.error(f"Failed to update package status: {str(e)}")
    
    def process_pending_packages(self, db: Session, limit: int = 10) -> List[Dict[str, Any]]:
        try:
            logger.info(f"Processing up to {limit} pending packages")
            pending_packages = PackageService.get_pending_packages(db, limit=limit)
            results = []
            
            for package in pending_packages:
                try:
                    result = self.process_package(db, package.package_id)
                    results.append({
                        "package_id": package.package_id,
                        "result": result
                    })
                except Exception as e:
                    error_msg = f"Error processing package {package.package_id}: {str(e)}"
                    logger.error(error_msg)
                    logger.error(traceback.format_exc())
                    results.append({
                        "package_id": package.package_id,
                        "result": {"status": "error", "message": error_msg}
                    })
            
            return results
        except Exception as e:
            error_msg = f"Error in process_pending_packages: {str(e)}"
            logger.error(error_msg)
            logger.error(traceback.format_exc())
            return [{"status": "error", "message": error_msg}]