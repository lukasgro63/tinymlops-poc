import json
import os
import shutil
import time
import uuid
from pathlib import Path
from typing import Any, Callable, Dict, List, Optional, Union

from tinylcm.constants import (
    DEFAULT_TRAINING_DIR,
    FILE_FORMAT_JSON,
    STATUS_COMPLETED,
    STATUS_FAILED,
    STATUS_RUNNING,
)
from tinylcm.utils.config import Config, get_config
from tinylcm.utils.file_utils import ensure_dir, load_json, save_json
from tinylcm.utils.logging import setup_logger
from tinylcm.utils.versioning import calculate_file_hash


class TrainingTracker:

    
    def __init__(
        self,
        storage_dir: Optional[Union[str, Path]] = None,
        config: Optional[Config] = None
    ):

        self.config = config or get_config()
        component_config = self.config.get_component_config("training_tracker")
        
        self.logger = setup_logger(f"{__name__}.{self.__class__.__name__}")
        
        self.storage_dir = Path(storage_dir or component_config.get("storage_dir", DEFAULT_TRAINING_DIR))
        self.runs_dir = ensure_dir(self.storage_dir / "runs")
        self.artifacts_dir = ensure_dir(self.storage_dir / "artifacts")
        self.backups_dir = ensure_dir(self.storage_dir / "backups")
        
        self.log_artifacts = component_config.get("log_artifacts", True)
        
        self.active_run_id = None
        self.run_stack = []  # For nested runs
        
        self.logger.info(f"Initialized training tracker with storage at: {self.storage_dir}")
    
    def start_run(
        self,
        run_name: str,
        run_id: Optional[str] = None,
        description: Optional[str] = None,
        tags: Optional[Dict[str, str]] = None,
        nested: bool = False,
        resume: bool = False
    ) -> str:

        if resume:
            existing_runs = self.list_runs(
                filter_func=lambda run: run.get("run_name") == run_name and run.get("status") != STATUS_RUNNING
            )
            if existing_runs:
                existing_runs.sort(key=lambda run: run.get("end_time", 0), reverse=True)
                run_id = existing_runs[0]["run_id"]
                self.logger.info(f"Resuming existing run: {run_id}")
        
        parent_run_id = None
        if nested:
            if not self.active_run_id:
                raise ValueError("Cannot start a nested run with no active parent run")
            parent_run_id = self.active_run_id
            self.run_stack.append(self.active_run_id)
        elif self.active_run_id:
            self.logger.warning(f"Starting a new run without ending the previous run: {self.active_run_id}")
            self.end_run()  # Auto-end previous run
        
        if run_id is None:
            run_id = str(uuid.uuid4())
        
        run_dir = ensure_dir(self.runs_dir / run_id)
        
        metadata = {
            "run_id": run_id,
            "run_name": run_name,
            "description": description or "",
            "tags": tags or {},
            "start_time": time.time(),
            "status": STATUS_RUNNING,
            "parent_run_id": parent_run_id
        }
        
        metadata_path = run_dir / "metadata.json"
        save_json(metadata, metadata_path)
        
        self.active_run_id = run_id
        
        self.logger.info(f"Started run '{run_name}' with ID: {run_id}")
        return run_id
    
    def end_run(
        self,
        run_id: Optional[str] = None,
        status: str = STATUS_COMPLETED
    ) -> bool:

        if run_id is None:
            run_id = self.active_run_id
            if run_id is None:
                raise ValueError("No active run to end")
        
        run_dir = self.runs_dir / run_id
        if not run_dir.exists():
            self.logger.warning(f"Run directory not found: {run_dir}")
            return False
        
        metadata_path = run_dir / "metadata.json"
        if not metadata_path.exists():
            self.logger.warning(f"Run metadata not found: {metadata_path}")
            return False
        
        try:
            metadata = load_json(metadata_path)
            metadata["status"] = status
            metadata["end_time"] = time.time()
            save_json(metadata, metadata_path)
            
            if run_id == self.active_run_id:
                if self.run_stack:
                    self.active_run_id = self.run_stack.pop()
                    self.logger.info(f"Restored parent run: {self.active_run_id}")
                else:
                    self.active_run_id = None
            
            self.logger.info(f"Ended run {run_id} with status: {status}")
            return True
            
        except Exception as e:
            self.logger.error(f"Error ending run {run_id}: {str(e)}")
            return False
    
    def log_param(
        self,
        key: str,
        value: Any,
        run_id: Optional[str] = None
    ) -> bool:

        if not key:
            raise ValueError("Parameter key cannot be empty")
        
        if run_id is None:
            run_id = self.active_run_id
            if run_id is None:
                raise ValueError("No active run for logging parameters")
        
        run_dir = self.runs_dir / run_id
        if not run_dir.exists():
            self.logger.warning(f"Run directory not found: {run_dir}")
            return False
        
        params_path = run_dir / "params.json"
        if params_path.exists():
            try:
                params = load_json(params_path)
            except Exception:
                params = {}
        else:
            params = {}
        
        params[key] = value
        
        try:
            save_json(params, params_path)
            self.logger.debug(f"Logged parameter '{key}' for run {run_id}")
            return True
        except Exception as e:
            self.logger.error(f"Error logging parameter '{key}' for run {run_id}: {str(e)}")
            return False
    
    def log_params(
        self,
        params_dict: Dict[str, Any],
        run_id: Optional[str] = None
    ) -> bool:

        if not params_dict:
            return True
        
        if run_id is None:
            run_id = self.active_run_id
            if run_id is None:
                raise ValueError("No active run for logging parameters")
        
        run_dir = self.runs_dir / run_id
        if not run_dir.exists():
            self.logger.warning(f"Run directory not found: {run_dir}")
            return False
        
        params_path = run_dir / "params.json"
        if params_path.exists():
            try:
                existing_params = load_json(params_path)
            except Exception:
                existing_params = {}
        else:
            existing_params = {}
        
        existing_params.update(params_dict)
        
        try:
            save_json(existing_params, params_path)
            self.logger.debug(f"Logged {len(params_dict)} parameters for run {run_id}")
            return True
        except Exception as e:
            self.logger.error(f"Error logging parameters for run {run_id}: {str(e)}")
            return False
    
    def log_metric(
        self,
        key: str,
        value: float,
        step: Optional[int] = None,
        run_id: Optional[str] = None
    ) -> bool:

        if not key:
            raise ValueError("Metric key cannot be empty")
        
        if run_id is None:
            run_id = self.active_run_id
            if run_id is None:
                raise ValueError("No active run for logging metrics")
        
        run_dir = self.runs_dir / run_id
        if not run_dir.exists():
            self.logger.warning(f"Run directory not found: {run_dir}")
            return False
        
        metrics_path = run_dir / "metrics.json"
        if metrics_path.exists():
            try:
                metrics = load_json(metrics_path)
            except Exception:
                metrics = {}
        else:
            metrics = {}
        
        metric_entry = {
            "value": value,
            "timestamp": time.time()
        }
        if step is not None:
            metric_entry["step"] = step
        
        if key not in metrics:
            metrics[key] = []
        metrics[key].append(metric_entry)
        
        try:
            save_json(metrics, metrics_path)
            self.logger.debug(f"Logged metric '{key}' with value {value} for run {run_id}")
            return True
        except Exception as e:
            self.logger.error(f"Error logging metric '{key}' for run {run_id}: {str(e)}")
            return False
    
    def log_metrics(
        self,
        metrics_dict: Dict[str, float],
        step: Optional[int] = None,
        run_id: Optional[str] = None
    ) -> bool:

        if not metrics_dict:
            return True  # Nothing to log
        
        success = True
        for key, value in metrics_dict.items():
            result = self.log_metric(key, value, step=step, run_id=run_id)
            success = success and result
        
        return success
    
    def log_artifact(
        self,
        local_path: Union[str, Path],
        artifact_path: Optional[str] = None,
        description: Optional[str] = None,
        run_id: Optional[str] = None
    ) -> str:

        if run_id is None:
            run_id = self.active_run_id
            if run_id is None:
                raise ValueError("No active run for logging artifacts")
        
        path_obj = Path(local_path)
        if not path_obj.exists():
            raise FileNotFoundError(f"Artifact file not found: {local_path}")
        
        if artifact_path is None:
            artifact_path = path_obj.name
        
        run_artifacts_dir = ensure_dir(self.artifacts_dir / run_id)
        
        dest_path = run_artifacts_dir / artifact_path
        ensure_dir(dest_path.parent)
        
        try:
            shutil.copy2(path_obj, dest_path)
            
            self._update_artifacts_list(
                run_id=run_id,
                artifact_name=artifact_path,
                artifact_path=str(dest_path),
                artifact_type="file",
                description=description
            )
            
            self.logger.debug(f"Logged artifact '{artifact_path}' for run {run_id}")
            return str(dest_path)
        except Exception as e:
            self.logger.error(f"Error logging artifact '{artifact_path}' for run {run_id}: {str(e)}")
            raise
    
    def log_figure(
        self,
        figure_data: bytes,
        name: str,
        description: Optional[str] = None,
        run_id: Optional[str] = None
    ) -> str:

        if run_id is None:
            run_id = self.active_run_id
            if run_id is None:
                raise ValueError("No active run for logging figures")
        
        run_artifacts_dir = ensure_dir(self.artifacts_dir / run_id / "figures")
        
        dest_path = run_artifacts_dir / name
        ensure_dir(dest_path.parent)
        
        try:
            with open(dest_path, "wb") as f:
                f.write(figure_data)
            
            self._update_artifacts_list(
                run_id=run_id,
                artifact_name=name,
                artifact_path=str(dest_path),
                artifact_type="figure",
                description=description
            )
            
            self.logger.debug(f"Logged figure '{name}' for run {run_id}")
            return str(dest_path)
        except Exception as e:
            self.logger.error(f"Error logging figure '{name}' for run {run_id}: {str(e)}")
            raise
    
    def log_model(
        self,
        model_path: Union[str, Path],
        model_format: str,
        flavor: str = "custom",
        custom_properties: Optional[Dict[str, Any]] = None,
        run_id: Optional[str] = None
    ) -> str:

        path_obj = Path(model_path)
        if not path_obj.exists():
            raise FileNotFoundError(f"Model file not found: {model_path}")
        
        if run_id is None:
            run_id = self.active_run_id
            if run_id is None:
                raise ValueError("No active run for logging models")
        
        run_models_dir = ensure_dir(self.artifacts_dir / run_id / "models")
        model_dir = ensure_dir(run_models_dir / path_obj.stem)
        
        dest_path = model_dir / path_obj.name
        try:
            shutil.copy2(path_obj, dest_path)
            
            model_meta = {
                "format": model_format,
                "flavor": flavor,
                "filename": path_obj.name,
                "timestamp": time.time(),
                "hash": calculate_file_hash(dest_path),
                "custom_properties": custom_properties or {}
            }
            
            meta_path = model_dir / "model_info.json"
            save_json(model_meta, meta_path)
            
            self._update_artifacts_list(
                run_id=run_id,
                artifact_name=path_obj.name,
                artifact_path=str(model_dir),
                artifact_type="model",
                description=f"{flavor} model in {model_format} format"
            )
            
            self.logger.debug(f"Logged model '{path_obj.name}' for run {run_id}")
            return str(model_dir)
        except Exception as e:
            self.logger.error(f"Error logging model '{path_obj.name}' for run {run_id}: {str(e)}")
            raise
    
    def _update_artifacts_list(
        self,
        run_id: str,
        artifact_name: str,
        artifact_path: str,
        artifact_type: str,
        description: Optional[str] = None
    ) -> None:

        run_dir = self.runs_dir / run_id
        
        artifacts_path = run_dir / "artifacts.json"
        if artifacts_path.exists():
            try:
                artifacts = load_json(artifacts_path)
            except Exception:
                artifacts = []
        else:
            artifacts = []
        
        artifact_entry = {
            "name": artifact_name,
            "path": artifact_path,
            "type": artifact_type,
            "timestamp": time.time()
        }
        
        if description:
            artifact_entry["description"] = description
        
        artifacts.append(artifact_entry)
        
        save_json(artifacts, artifacts_path)
    
    def get_run_info(self, run_id: str) -> Dict[str, Any]:

        run_dir = self.runs_dir / run_id
        if not run_dir.exists():
            raise ValueError(f"Run not found: {run_id}")
        
        metadata_path = run_dir / "metadata.json"
        if not metadata_path.exists():
            raise ValueError(f"Run metadata not found: {run_id}")
        
        metadata = load_json(metadata_path)
        
        params_path = run_dir / "params.json"
        if params_path.exists():
            try:
                params = load_json(params_path)
                metadata["params"] = params
            except Exception:
                metadata["params"] = {}
        else:
            metadata["params"] = {}
        
        metrics_path = run_dir / "metrics.json"
        if metrics_path.exists():
            try:
                metrics = load_json(metrics_path)
                metadata["metrics"] = metrics
            except Exception:
                metadata["metrics"] = {}
        else:
            metadata["metrics"] = {}
        
        artifacts_path = run_dir / "artifacts.json"
        if artifacts_path.exists():
            try:
                artifacts = load_json(artifacts_path)
                metadata["artifacts"] = artifacts
            except Exception:
                metadata["artifacts"] = []
        else:
            metadata["artifacts"] = []
        
        return metadata
    
    def list_runs(
        self,
        filter_func: Optional[Callable[[Dict[str, Any]], bool]] = None
    ) -> List[Dict[str, Any]]:

        runs = []
        
        for run_dir in self.runs_dir.iterdir():
            if not run_dir.is_dir():
                continue
            
            metadata_path = run_dir / "metadata.json"
            if not metadata_path.exists():
                continue
            
            try:
                metadata = load_json(metadata_path)
                
                if filter_func is None or filter_func(metadata):
                    runs.append(metadata)
            except Exception as e:
                self.logger.warning(f"Error loading metadata for run {run_dir.name}: {str(e)}")
        
        return runs
    
    def delete_run(self, run_id: str) -> bool:

        run_dir = self.runs_dir / run_id
        if not run_dir.exists():
            return False
        
        try:
            shutil.rmtree(run_dir)
            
            run_artifacts_dir = self.artifacts_dir / run_id
            if run_artifacts_dir.exists():
                shutil.rmtree(run_artifacts_dir)
            
            self.logger.info(f"Deleted run {run_id}")
            return True
        except Exception as e:
            self.logger.error(f"Error deleting run {run_id}: {str(e)}")
            return False
    
    def backup_run(self, run_id: str) -> bool:

        run_dir = self.runs_dir / run_id
        if not run_dir.exists():
            self.logger.warning(f"Run not found for backup: {run_id}")
            return False
        
        backup_dir = ensure_dir(self.backups_dir / run_id)
        
        try:
            for item in run_dir.iterdir():
                if item.is_file():
                    shutil.copy2(item, backup_dir)
            
            run_artifacts_dir = self.artifacts_dir / run_id
            if run_artifacts_dir.exists():
                backup_artifacts_dir = ensure_dir(backup_dir / "artifacts")
                shutil.copytree(run_artifacts_dir, backup_artifacts_dir, dirs_exist_ok=True)
            
            self.logger.info(f"Backed up run {run_id}")
            return True
        except Exception as e:
            self.logger.error(f"Error backing up run {run_id}: {str(e)}")
            return False
    
    def restore_run(self, run_id: str) -> bool:

        backup_dir = self.backups_dir / run_id
        if not backup_dir.exists():
            self.logger.warning(f"Backup not found for run: {run_id}")
            return False
        
        try:
            run_dir = ensure_dir(self.runs_dir / run_id)
            
            for item in backup_dir.iterdir():
                if item.is_file():
                    shutil.copy2(item, run_dir)
                elif item.name == "artifacts":
                    shutil.copytree(item, self.artifacts_dir / run_id, dirs_exist_ok=True)
            
            self.logger.info(f"Restored run {run_id} from backup")
            return True
        except Exception as e:
            self.logger.error(f"Error restoring run {run_id}: {str(e)}")
            return False
    
    def export_to_mlflow_format(
        self,
        run_id: str,
        output_dir: Union[str, Path]
    ) -> bool:

        try:
            run_info = self.get_run_info(run_id)
        except ValueError as e:
            self.logger.error(f"Error getting run info: {str(e)}")
            return False
        
        mlflow_run_dir = ensure_dir(Path(output_dir) / run_id)
        mlflow_params_dir = ensure_dir(mlflow_run_dir / "params")
        mlflow_metrics_dir = ensure_dir(mlflow_run_dir / "metrics")
        mlflow_artifacts_dir = ensure_dir(mlflow_run_dir / "artifacts")
        
        try:
            # Create meta.yaml
            meta = {
                "name": run_info.get("run_name", ""),
                "tags": run_info.get("tags", {}),
                "status": run_info.get("status", ""),
                "start_time": run_info.get("start_time", 0),
                "end_time": run_info.get("end_time", 0),
                "tinylcm.run_id": run_id
            }
            
            with open(mlflow_run_dir / "meta.yaml", "w") as f:
                f.write(self._yaml_format(meta))
            
            params = run_info.get("params", {})
            for param_name, param_value in params.items():
                with open(mlflow_params_dir / param_name, "w") as f:
                    f.write(str(param_value))
            
            metrics = run_info.get("metrics", {})
            for metric_name, metric_values in metrics.items():
                for i, metric_entry in enumerate(metric_values):
                    timestamp = metric_entry.get("timestamp", 0)
                    step = metric_entry.get("step", i)
                    value = metric_entry.get("value", 0)
                    
                    metric_filename = f"{metric_name}-{timestamp:.0f}-{i}"
                    with open(mlflow_metrics_dir / metric_filename, "w") as f:
                        f.write(f"{value} {step} {timestamp:.0f}")
            
            artifacts = run_info.get("artifacts", [])
            for artifact in artifacts:
                src_path = artifact.get("path")
                if not src_path:
                    continue
                
                name = artifact.get("name", "")
                dest_path = mlflow_artifacts_dir / name
                
                src_path_obj = Path(src_path)
                if src_path_obj.is_file():
                    ensure_dir(dest_path.parent)
                    shutil.copy2(src_path_obj, dest_path)
                elif src_path_obj.is_dir():
                    shutil.copytree(src_path_obj, dest_path, dirs_exist_ok=True)
            
            self.logger.info(f"Exported run {run_id} to MLflow-compatible format at {output_dir}")
            return True
            
        except Exception as e:
            self.logger.error(f"Error exporting run {run_id} to MLflow-compatible format: {str(e)}")
            return False
    
    def _yaml_format(self, data: Dict[str, Any]) -> str:

        lines = []
        
        for key, value in data.items():
            if isinstance(value, dict):
                lines.append(f"{key}:")
                for sub_key, sub_value in value.items():
                    lines.append(f"  {sub_key}: {sub_value}")
            else:
                lines.append(f"{key}: {value}")
        
        return "\n".join(lines)
    
    def close(self) -> None:

        if self.active_run_id:
            self.logger.info(f"Auto-ending active run during close: {self.active_run_id}")
            self.end_run(status=STATUS_COMPLETED)  # Assume completed
        
        while self.run_stack:
            run_id = self.run_stack.pop()
            self.logger.warning(f"Auto-ending stacked run during close: {run_id}")
            self.end_run(run_id=run_id, status=STATUS_COMPLETED)
        
        if self.active_run_id:
            self.logger.info(f"Auto-ending remaining active run during close: {self.active_run_id}")
            self.end_run(status=STATUS_COMPLETED)
        
        self.active_run_id = None
        
        self.logger.info("Closed training tracker")