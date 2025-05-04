"""Tests for the TrainingTracker class."""

import json
import os
import time
from pathlib import Path

import pytest

from tinylcm.core.training_tracker import TrainingTracker
from tinylcm.constants import STATUS_COMPLETED, STATUS_FAILED, STATUS_RUNNING
from tinylcm.utils.file_utils import load_json


@pytest.fixture
def tracker(tmp_path):
    """Create a TrainingTracker instance for testing."""
    tracker = TrainingTracker(storage_dir=tmp_path)
    yield tracker
    tracker.close()


def test_tracker_init(tmp_path):
    """Test that the TrainingTracker initializes correctly."""
    tracker = TrainingTracker(storage_dir=tmp_path)
    
    assert tracker.storage_dir == tmp_path
    assert tracker.runs_dir == tmp_path / "runs"
    assert tracker.artifacts_dir == tmp_path / "artifacts"
    assert tracker.backups_dir == tmp_path / "backups"
    assert tracker.active_run_id is None
    assert len(tracker.run_stack) == 0


def test_start_run(tracker):
    """Test that start_run creates a run correctly."""
    run_id = tracker.start_run("test_run", description="Test description", tags={"key": "value"})
    
    assert run_id is not None
    assert tracker.active_run_id == run_id
    
    # Check that run directory exists
    run_dir = tracker.runs_dir / run_id
    assert run_dir.exists()
    
    # Check that metadata was written correctly
    metadata_path = run_dir / "metadata.json"
    assert metadata_path.exists()
    
    metadata = load_json(metadata_path)
    assert metadata["run_id"] == run_id
    assert metadata["run_name"] == "test_run"
    assert metadata["description"] == "Test description"
    assert metadata["tags"] == {"key": "value"}
    assert metadata["status"] == STATUS_RUNNING
    assert "start_time" in metadata


def test_start_run_with_custom_id(tracker):
    """Test that start_run works with a custom run ID."""
    custom_id = "custom_run_id"
    run_id = tracker.start_run("test_run", run_id=custom_id)
    
    assert run_id == custom_id
    assert tracker.active_run_id == custom_id


def test_nested_runs(tracker):
    """Test nested runs functionality."""
    parent_id = tracker.start_run("parent_run")
    
    # Start a nested run
    child_id = tracker.start_run("child_run", nested=True)
    
    assert tracker.active_run_id == child_id
    assert parent_id in tracker.run_stack
    
    # End the child run
    tracker.end_run()
    
    # Parent should now be active again
    assert tracker.active_run_id == parent_id
    assert len(tracker.run_stack) == 0


def test_nested_run_requires_active_parent(tracker):
    """Test that starting a nested run requires an active parent run."""
    with pytest.raises(ValueError, match="Cannot start a nested run with no active parent run"):
        tracker.start_run("child_run", nested=True)


def test_end_run(tracker):
    """Test that end_run works correctly."""
    run_id = tracker.start_run("test_run")
    
    assert tracker.end_run() is True
    assert tracker.active_run_id is None
    
    # Check that metadata was updated
    metadata_path = tracker.runs_dir / run_id / "metadata.json"
    metadata = load_json(metadata_path)
    
    assert metadata["status"] == STATUS_COMPLETED
    assert "end_time" in metadata


def test_end_run_with_status(tracker):
    """Test that end_run works with custom status."""
    run_id = tracker.start_run("test_run")
    
    assert tracker.end_run(status=STATUS_FAILED) is True
    
    # Check that metadata was updated with correct status
    metadata_path = tracker.runs_dir / run_id / "metadata.json"
    metadata = load_json(metadata_path)
    
    assert metadata["status"] == STATUS_FAILED


def test_log_param(tracker):
    """Test that log_param works correctly."""
    run_id = tracker.start_run("test_run")
    
    # Log a parameter
    assert tracker.log_param("test_param", 123) is True
    
    # Check that the parameter was saved
    params_path = tracker.runs_dir / run_id / "params.json"
    assert params_path.exists()
    
    params = load_json(params_path)
    assert "test_param" in params
    assert params["test_param"] == 123


def test_log_params(tracker):
    """Test that log_params works correctly."""
    run_id = tracker.start_run("test_run")
    
    # Log multiple parameters
    params_dict = {
        "param1": "value1",
        "param2": 2,
        "param3": [1, 2, 3],
        "param4": {"key": "value"}
    }
    
    assert tracker.log_params(params_dict) is True
    
    # Check that parameters were saved
    params_path = tracker.runs_dir / run_id / "params.json"
    assert params_path.exists()
    
    params = load_json(params_path)
    assert params == params_dict


def test_log_metric(tracker):
    """Test that log_metric works correctly."""
    run_id = tracker.start_run("test_run")
    
    # Log a metric
    assert tracker.log_metric("accuracy", 0.95, step=1) is True
    
    # Check that metric was saved
    metrics_path = tracker.runs_dir / run_id / "metrics.json"
    assert metrics_path.exists()
    
    metrics = load_json(metrics_path)
    assert "accuracy" in metrics
    assert len(metrics["accuracy"]) == 1
    assert metrics["accuracy"][0]["value"] == 0.95
    assert metrics["accuracy"][0]["step"] == 1
    assert "timestamp" in metrics["accuracy"][0]


def test_log_metrics(tracker):
    """Test that log_metrics works correctly."""
    run_id = tracker.start_run("test_run")
    
    # Log multiple metrics
    metrics_dict = {
        "accuracy": 0.95,
        "loss": 0.1,
        "f1_score": 0.97
    }
    
    assert tracker.log_metrics(metrics_dict, step=5) is True
    
    # Check that metrics were saved
    metrics_path = tracker.runs_dir / run_id / "metrics.json"
    assert metrics_path.exists()
    
    metrics = load_json(metrics_path)
    for key, value in metrics_dict.items():
        assert key in metrics
        assert len(metrics[key]) == 1
        assert metrics[key][0]["value"] == value
        assert metrics[key][0]["step"] == 5


def test_log_artifact(tracker, tmp_path):
    """Test that log_artifact works correctly."""
    run_id = tracker.start_run("test_run")
    
    # Create a test artifact
    artifact_content = "Test artifact content"
    artifact_path = tmp_path / "test_artifact.txt"
    with open(artifact_path, "w") as f:
        f.write(artifact_content)
    
    # Log the artifact
    result = tracker.log_artifact(artifact_path, description="Test artifact")
    
    # Check that artifact was saved
    assert Path(result).exists()
    with open(result, "r") as f:
        saved_content = f.read()
    assert saved_content == artifact_content
    
    # Check that artifact was added to the list
    artifacts_path = tracker.runs_dir / run_id / "artifacts.json"
    artifacts = load_json(artifacts_path)
    
    assert len(artifacts) == 1
    assert artifacts[0]["name"] == "test_artifact.txt"
    assert artifacts[0]["type"] == "file"
    assert artifacts[0]["description"] == "Test artifact"


def test_log_figure(tracker):
    """Test that log_figure works correctly."""
    run_id = tracker.start_run("test_run")
    
    # Create figure data
    figure_data = b"PNG\r\n\x1a\n\x00\x00\x00\rIHDR\x00\x00\x00\x01\x00\x00\x00\x01\x08\x06\x00\x00\x00\x1f\x15\xc4\x89\x00\x00\x00\nIDATx\x9cc\x00\x01\x00\x00}\x00\x07\x1f\x1d\xd9.\x00\x00\x00\x00IEND\xaeB`\x82"
    
    # Log the figure
    result = tracker.log_figure(figure_data, "test_figure.png", description="Test figure")
    
    # Check that figure was saved
    assert Path(result).exists()
    with open(result, "rb") as f:
        saved_content = f.read()
    assert saved_content == figure_data
    
    # Check that figure was added to the list
    artifacts_path = tracker.runs_dir / run_id / "artifacts.json"
    artifacts = load_json(artifacts_path)
    
    assert len(artifacts) == 1
    assert artifacts[0]["name"] == "test_figure.png"
    assert artifacts[0]["type"] == "figure"
    assert artifacts[0]["description"] == "Test figure"


def test_log_model(tracker, tmp_path):
    """Test that log_model works correctly."""
    run_id = tracker.start_run("test_run")
    
    # Create a test model file
    model_content = json.dumps({"weights": [1, 2, 3], "biases": [0.1, 0.2]})
    model_path = tmp_path / "test_model.json"
    with open(model_path, "w") as f:
        f.write(model_content)
    
    # Log the model
    model_dir = tracker.log_model(
        model_path=model_path,
        model_format="json",
        flavor="test",
        custom_properties={"key": "value"}
    )
    
    # Check that model was saved
    model_file = Path(model_dir) / "test_model.json"
    assert model_file.exists()
    with open(model_file, "r") as f:
        saved_content = f.read()
    assert saved_content == model_content
    
    # Check that model metadata was created
    model_meta_path = Path(model_dir) / "model_info.json"
    assert model_meta_path.exists()
    
    model_meta = load_json(model_meta_path)
    assert model_meta["format"] == "json"
    assert model_meta["flavor"] == "test"
    assert model_meta["filename"] == "test_model.json"
    assert model_meta["custom_properties"] == {"key": "value"}
    
    # Check that model was added to artifacts list
    artifacts_path = tracker.runs_dir / run_id / "artifacts.json"
    artifacts = load_json(artifacts_path)
    
    assert len(artifacts) == 1
    assert artifacts[0]["type"] == "model"


def test_get_run_info(tracker):
    """Test that get_run_info works correctly."""
    # Create a run with parameters, metrics, and artifacts
    run_id = tracker.start_run("test_run", description="Test description")
    tracker.log_param("param1", "value1")
    tracker.log_metric("metric1", 0.95)
    
    # Create and log an artifact
    artifact_content = "Test artifact"
    artifact_path = tracker.runs_dir / "test_artifact.txt"
    with open(artifact_path, "w") as f:
        f.write(artifact_content)
    tracker.log_artifact(artifact_path)
    
    # Get run info
    run_info = tracker.get_run_info(run_id)
    
    # Check basic metadata
    assert run_info["run_id"] == run_id
    assert run_info["run_name"] == "test_run"
    assert run_info["description"] == "Test description"
    assert run_info["status"] == STATUS_RUNNING
    
    # Check parameters
    assert "params" in run_info
    assert run_info["params"]["param1"] == "value1"
    
    # Check metrics
    assert "metrics" in run_info
    assert "metric1" in run_info["metrics"]
    assert len(run_info["metrics"]["metric1"]) == 1
    assert run_info["metrics"]["metric1"][0]["value"] == 0.95
    
    # Check artifacts
    assert "artifacts" in run_info
    assert len(run_info["artifacts"]) == 1
    assert run_info["artifacts"][0]["name"] == "test_artifact.txt"


def test_list_runs(tracker):
    """Test that list_runs works correctly."""
    # Create multiple runs
    run1_id = tracker.start_run("run1", tags={"tag1": "value1"})
    tracker.end_run()
    
    run2_id = tracker.start_run("run2", tags={"tag2": "value2"})
    tracker.end_run()
    
    run3_id = tracker.start_run("run3", tags={"tag1": "value1", "tag3": "value3"})
    tracker.end_run()
    
    # List all runs
    runs = tracker.list_runs()
    assert len(runs) == 3
    
    # Check that runs are returned in the right format
    assert runs[0]["run_id"] in [run1_id, run2_id, run3_id]
    assert runs[1]["run_id"] in [run1_id, run2_id, run3_id]
    assert runs[2]["run_id"] in [run1_id, run2_id, run3_id]
    
    # Test filtering
    filtered_runs = tracker.list_runs(
        filter_func=lambda run: "tag1" in run.get("tags", {})
    )
    assert len(filtered_runs) == 2
    assert filtered_runs[0]["run_id"] in [run1_id, run3_id]
    assert filtered_runs[1]["run_id"] in [run1_id, run3_id]


def test_delete_run(tracker):
    """Test that delete_run works correctly."""
    # Create a run
    run_id = tracker.start_run("test_run")
    tracker.end_run()
    
    # Verify run exists
    assert (tracker.runs_dir / run_id).exists()
    
    # Delete run
    result = tracker.delete_run(run_id)
    assert result is True
    
    # Verify run directory is gone
    assert not (tracker.runs_dir / run_id).exists()


def test_backup_and_restore_run(tracker):
    """Test that backup_run and restore_run work correctly."""
    # Create a run with parameters
    run_id = tracker.start_run("test_run")
    tracker.log_param("param1", "value1")
    tracker.end_run()
    
    # Backup the run
    result = tracker.backup_run(run_id)
    assert result is True
    
    # Verify backup exists
    assert (tracker.backups_dir / run_id).exists()
    
    # Delete the original run
    tracker.delete_run(run_id)
    assert not (tracker.runs_dir / run_id).exists()
    
    # Restore the run
    result = tracker.restore_run(run_id)
    assert result is True
    
    # Verify run was restored
    assert (tracker.runs_dir / run_id).exists()
    
    # Check that parameters were restored
    run_info = tracker.get_run_info(run_id)
    assert run_info["params"]["param1"] == "value1"


def test_export_to_mlflow_format(tracker, tmp_path):
    """Test that export_to_mlflow_format works correctly."""
    # Create a run with parameters, metrics, and artifacts
    run_id = tracker.start_run("test_run", tags={"tag1": "value1"})
    
    # Log parameters
    tracker.log_params({
        "param1": "value1",
        "param2": 42
    })
    
    # Log metrics
    tracker.log_metric("accuracy", 0.95, step=1)
    tracker.log_metric("accuracy", 0.97, step=2)
    tracker.log_metric("loss", 0.1, step=1)
    
    # Create and log an artifact
    artifact_content = "Test artifact"
    artifact_path = tracker.runs_dir / "test_artifact.txt"
    with open(artifact_path, "w") as f:
        f.write(artifact_content)
    tracker.log_artifact(artifact_path)
    
    # End the run
    tracker.end_run()
    
    # Export to MLflow format
    output_dir = tmp_path / "mlflow"
    result = tracker.export_to_mlflow_format(run_id, output_dir)
    assert result is True
    
    # Check that MLflow directory structure was created
    mlflow_run_dir = output_dir / run_id
    assert mlflow_run_dir.exists()
    
    # Check meta.yaml
    meta_yaml_path = mlflow_run_dir / "meta.yaml"
    assert meta_yaml_path.exists()
    with open(meta_yaml_path, "r") as f:
        meta_yaml = f.read()
    assert "name: test_run" in meta_yaml
    assert "status: COMPLETED" in meta_yaml
    
    # Check parameters
    params_dir = mlflow_run_dir / "params"
    assert params_dir.exists()
    with open(params_dir / "param1", "r") as f:
        assert f.read() == "value1"
    with open(params_dir / "param2", "r") as f:
        assert f.read() == "42"
    
    # Check metrics
    metrics_dir = mlflow_run_dir / "metrics"
    assert metrics_dir.exists()
    assert len(list(metrics_dir.glob("accuracy-*"))) == 2
    assert len(list(metrics_dir.glob("loss-*"))) == 1
    
    # Check artifacts
    artifacts_dir = mlflow_run_dir / "artifacts"
    assert artifacts_dir.exists()
    assert (artifacts_dir / "test_artifact.txt").exists()
    with open(artifacts_dir / "test_artifact.txt", "r") as f:
        assert f.read() == artifact_content


def test_close(tracker):
    """Test that close properly ends active runs."""
    # Start a run
    run_id = tracker.start_run("test_run")
    
    # Close tracker
    tracker.close()
    
    # Check that run was ended
    assert tracker.active_run_id is None
    
    # Check metadata
    metadata_path = tracker.runs_dir / run_id / "metadata.json"
    metadata = load_json(metadata_path)
    assert metadata["status"] == STATUS_COMPLETED
    assert "end_time" in metadata


def test_auto_end_with_stack(tracker):
    """Test that close properly ends all runs in the stack."""
    # Start nested runs
    parent_id = tracker.start_run("parent_run")
    child_id = tracker.start_run("child_run", nested=True)
    
    # Close tracker
    tracker.close()
    
    # Check that both runs were ended
    assert tracker.active_run_id is None
    assert len(tracker.run_stack) == 0
    
    # Check parent metadata
    parent_metadata_path = tracker.runs_dir / parent_id / "metadata.json"
    parent_metadata = load_json(parent_metadata_path)
    assert parent_metadata["status"] == STATUS_COMPLETED
    
    # Check child metadata
    child_metadata_path = tracker.runs_dir / child_id / "metadata.json"
    child_metadata = load_json(child_metadata_path)
    assert child_metadata["status"] == STATUS_COMPLETED