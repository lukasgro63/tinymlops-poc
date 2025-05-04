import json
import os
import shutil
import tempfile
import time
from unittest.mock import MagicMock

import pytest

from tinylcm.client.sync_interface import SyncInterface, SyncPackage
from tinylcm.utils.errors import SyncError


class TestSyncInterface:
    def setup_method(self):
        self.temp_dir = tempfile.mkdtemp()
        self.sync_dir = os.path.join(self.temp_dir, "sync")
        self.sync_interface = SyncInterface(sync_dir=self.sync_dir)
        self.test_data_dir = os.path.join(self.temp_dir, "test_data")
        os.makedirs(self.test_data_dir, exist_ok=True)
        self.model_file = os.path.join(self.test_data_dir, "model.json")
        self.model_data = {"weights": [1.0, 2.0, 3.0], "layers": [5, 3, 1]}
        with open(self.model_file, "w", encoding="utf-8") as f:
            json.dump(self.model_data, f)
        self.metrics_file = os.path.join(self.test_data_dir, "metrics.json")
        self.metrics_data = {
            "accuracy": 0.95,
            "latency_ms": {"mean": 10.2, "max": 15.6},
            "predictions": {"class_a": 120, "class_b": 80}
        }
        with open(self.metrics_file, "w", encoding="utf-8") as f:
            json.dump(self.metrics_data, f)
        self.log_file = os.path.join(self.test_data_dir, "log.jsonl")
        self.log_entries = [
            {"timestamp": time.time(), "level": "INFO", "message": "Test log 1"},
            {"timestamp": time.time() + 1, "level": "WARNING", "message": "Test log 2"}
        ]
        with open(self.log_file, "w", encoding="utf-8") as f:
            for entry in self.log_entries:
                f.write(json.dumps(entry) + "\n")

    def teardown_method(self):
        shutil.rmtree(self.temp_dir)

    def test_init_creates_directories(self):
        assert os.path.exists(self.sync_dir)
        assert os.path.exists(os.path.join(self.sync_dir, "packages"))
        assert os.path.exists(os.path.join(self.sync_dir, "history"))

    def test_create_sync_package(self):
        package_id = self.sync_interface.create_package(
            device_id="test_device_1",
            package_type="models",
            description="Test model package"
        )
        assert isinstance(package_id, str)
        self.sync_interface.add_file_to_package(
            package_id=package_id,
            file_path=self.model_file,
            file_type="model",
            metadata={"model_name": "test_model", "version": "1.0"}
        )
        package_path = self.sync_interface.finalize_package(package_id)
        assert os.path.exists(package_path)
        with open(package_path, "rb") as f:
            package_data = f.read()
        assert len(package_data) > 0
        metadata_path = os.path.join(self.sync_dir, "packages", f"{package_id}.meta.json")
        assert os.path.exists(metadata_path)
        with open(metadata_path, "r", encoding="utf-8") as f:
            metadata = json.load(f)
        assert metadata["device_id"] == "test_device_1"
        assert metadata["package_type"] == "models"
        assert "creation_time" in metadata
        assert "files" in metadata
        assert len(metadata["files"]) == 1
        assert metadata["files"][0]["file_type"] == "model"

    def test_add_multiple_files_to_package(self):
        package_id = self.sync_interface.create_package(
            device_id="test_device_1",
            package_type="mixed"
        )
        self.sync_interface.add_file_to_package(
            package_id=package_id,
            file_path=self.model_file,
            file_type="model"
        )
        self.sync_interface.add_file_to_package(
            package_id=package_id,
            file_path=self.metrics_file,
            file_type="metrics"
        )
        self.sync_interface.add_file_to_package(
            package_id=package_id,
            file_path=self.log_file,
            file_type="logs"
        )
        package_path = self.sync_interface.finalize_package(package_id)
        metadata_path = os.path.join(self.sync_dir, "packages", f"{package_id}.meta.json")
        with open(metadata_path, "r", encoding="utf-8") as f:
            metadata = json.load(f)
        assert len(metadata["files"]) == 3
        file_types = [f["file_type"] for f in metadata["files"]]
        assert "model" in file_types
        assert "metrics" in file_types
        assert "logs" in file_types

    def test_add_directory_to_package(self):
        package_id = self.sync_interface.create_package(
            device_id="test_device_1",
            package_type="data"
        )
        self.sync_interface.add_directory_to_package(
            package_id=package_id,
            directory_path=self.test_data_dir,
            recursive=True,
            file_type="data"
        )
        package_path = self.sync_interface.finalize_package(package_id)
        metadata_path = os.path.join(self.sync_dir, "packages", f"{package_id}.meta.json")
        with open(metadata_path, "r", encoding="utf-8") as f:
            metadata = json.load(f)
        assert len(metadata["files"]) == 3

    def test_create_package_from_component(self):
        mock_model_manager = MagicMock()
        mock_model_manager.get_active_model_metadata.return_value = {
            "model_id": "test_model_123",
            "filename": "model.json",
            "version": "1.0"
        }
        mock_model_manager.load_model.return_value = self.model_file
        mock_monitor = MagicMock()
        mock_monitor.export_metrics.return_value = self.metrics_file
        package_id = self.sync_interface.create_package_from_components(
            device_id="test_device_1",
            model_manager=mock_model_manager,
            inference_monitor=mock_monitor
        )
        package_path = self.sync_interface.finalize_package(package_id)
        metadata_path = os.path.join(self.sync_dir, "packages", f"{package_id}.meta.json")
        with open(metadata_path, "r", encoding="utf-8") as f:
            metadata = json.load(f)
        assert len(metadata["files"]) == 2
        mock_model_manager.get_active_model_metadata.assert_called_once()
        mock_monitor.export_metrics.assert_called_once()

    def test_compression_options(self):
        package_id = self.sync_interface.create_package(
            device_id="test_device_1",
            package_type="models",
            compression="gzip"
        )
        self.sync_interface.add_file_to_package(
            package_id=package_id,
            file_path=self.model_file,
            file_type="model"
        )
        gzip_package_path = self.sync_interface.finalize_package(package_id)
        gzip_size = os.path.getsize(gzip_package_path)
        package_id = self.sync_interface.create_package(
            device_id="test_device_1",
            package_type="models",
            compression="zip"
        )
        self.sync_interface.add_file_to_package(
            package_id=package_id,
            file_path=self.model_file,
            file_type="model"
        )
        zip_package_path = self.sync_interface.finalize_package(package_id)
        zip_size = os.path.getsize(zip_package_path)
        assert gzip_size > 0
        assert zip_size > 0

    def test_list_packages(self):
        for i in range(3):
            package_id = self.sync_interface.create_package(
                device_id=f"device_{i}",
                package_type="data"
            )
            self.sync_interface.add_file_to_package(
                package_id=package_id,
                file_path=self.model_file,
                file_type="model"
            )
            self.sync_interface.finalize_package(package_id)
        packages = self.sync_interface.list_packages()
        assert len(packages) == 3
        device1_packages = self.sync_interface.list_packages(
            filter_func=lambda pkg: pkg["device_id"] == "device_1"
        )
        assert len(device1_packages) == 1
        assert device1_packages[0]["device_id"] == "device_1"

    def test_mark_as_synced(self):
        package_id = self.sync_interface.create_package(
            device_id="test_device",
            package_type="logs"
        )
        self.sync_interface.add_file_to_package(
            package_id=package_id,
            file_path=self.log_file,
            file_type="logs"
        )
        self.sync_interface.finalize_package(package_id)
        self.sync_interface.mark_as_synced(
            package_id=package_id,
            sync_time=time.time(),
            server_id="test_server_1",
            status="success"
        )
        packages = self.sync_interface.list_packages(include_synced=True)
        synced_package = next((p for p in packages if p["package_id"] == package_id), None)
        assert synced_package is not None
        assert synced_package["sync_status"] == "success"
        assert "sync_time" in synced_package

    def test_error_handling(self):
        with pytest.raises(SyncError):
            self.sync_interface.add_file_to_package(
                package_id="non_existent_package",
                file_path=self.model_file,
                file_type="model"
            )
        package_id = self.sync_interface.create_package(
            device_id="test_device",
            package_type="models"
        )
        with pytest.raises(FileNotFoundError):
            self.sync_interface.add_file_to_package(
                package_id=package_id,
                file_path="/path/to/nonexistent/file.txt",
                file_type="unknown"
            )
        self.sync_interface.add_file_to_package(
            package_id=package_id,
            file_path=self.model_file,
            file_type="model"
        )
        self.sync_interface.finalize_package(package_id)
        with pytest.raises(SyncError):
            self.sync_interface.finalize_package(package_id)