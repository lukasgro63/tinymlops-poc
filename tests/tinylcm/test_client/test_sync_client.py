import json
import os
import shutil
import tempfile
import time
import uuid
from pathlib import Path
from unittest.mock import MagicMock, patch

import pytest
import requests

from tinylcm.client.sync_client import SyncClient
from tinylcm.client.sync_interface import SyncInterface
from tinylcm.utils.errors import ConnectionError, SyncError


class TestSyncClient:
    def setup_method(self):
        self.temp_dir = tempfile.mkdtemp()
        self.sync_dir = os.path.join(self.temp_dir, "sync")
        self.sync_interface = SyncInterface(sync_dir=self.sync_dir)
        self.server_url = "https://example.com/api"
        self.api_key = "test_api_key_12345"
        self.device_id = "test_device_123"
        with patch('tinylcm.client.sync_client.ConnectionManager') as mock_cm:
            self.mock_connection_manager = mock_cm.return_value
            self.client = SyncClient(
                server_url=self.server_url,
                api_key=self.api_key,
                device_id=self.device_id,
                sync_interface=self.sync_interface,
                auto_register=False
            )
        self.test_data_dir = os.path.join(self.temp_dir, "test_data")
        os.makedirs(self.test_data_dir, exist_ok=True)
        self.model_file = os.path.join(self.test_data_dir, "model.json")
        with open(self.model_file, "w", encoding="utf-8") as f:
            json.dump({"weights": [1.0, 2.0], "layers": [3, 1]}, f)

    def teardown_method(self):
        shutil.rmtree(self.temp_dir)

    def test_initialization(self):
        assert self.client.server_url == self.server_url
        assert self.client.api_key == self.api_key
        assert self.client.device_id == self.device_id
        assert self.client.sync_interface == self.sync_interface
        assert "Authorization" in self.client.headers
        assert "X-Device-ID" in self.client.headers
        assert self.client.headers["X-Device-ID"] == self.device_id

    def test_validate_server_url(self):
        assert SyncClient.validate_server_url("http://example.com") is True
        assert SyncClient.validate_server_url("https://example.com/api") is True
        assert SyncClient.validate_server_url("") is False
        assert SyncClient.validate_server_url("ftp://example.com") is False

    def test_get_device_info(self):
        device_info = self.client._get_device_info()
        assert "device_id" in device_info
        assert device_info["device_id"] == self.device_id
        assert "hostname" in device_info
        assert "platform" in device_info
        assert "python_version" in device_info
        assert "tinylcm_version" in device_info

    def test_register_device_success(self):
        mock_response = MagicMock()
        mock_response.status_code = 200
        mock_response.json.return_value = {"status": "success"}
        self.mock_connection_manager.execute_request.return_value = mock_response
        result = self.client.register_device()
        assert result is True
        self.mock_connection_manager.execute_request.assert_called_once()
        call_args = self.mock_connection_manager.execute_request.call_args[1]
        assert call_args["method"] == "POST"
        assert call_args["endpoint"] == "devices/register"
        assert "json" in call_args
        assert call_args["json"]["device_id"] == self.device_id

    def test_register_device_failure(self):
        mock_response = MagicMock()
        mock_response.status_code = 401
        mock_response.text = "Unauthorized"
        self.mock_connection_manager.execute_request.return_value = mock_response
        with pytest.raises(ConnectionError):
            self.client.register_device()

    def test_register_device_network_error(self):
        self.mock_connection_manager.execute_request.side_effect = requests.RequestException("Network error")
        with pytest.raises(ConnectionError):
            self.client.register_device()

    def test_send_package_success(self):
        package_id = self.sync_interface.create_package(
            device_id=self.device_id,
            package_type="test"
        )
        self.sync_interface.add_file_to_package(
            package_id=package_id,
            file_path=self.model_file,
            file_type="model"
        )
        self.sync_interface.finalize_package(package_id)
        mock_response = MagicMock()
        mock_response.status_code = 200
        mock_response.json.return_value = {"server_id": "test_server_1"}
        self.mock_connection_manager.execute_request.return_value = mock_response
        result = self.client.send_package(package_id)
        assert result is True
        self.mock_connection_manager.execute_request.assert_called_once()
        call_args = self.mock_connection_manager.execute_request.call_args[1]
        assert call_args["method"] == "POST"
        assert call_args["endpoint"] == "packages/upload"
        assert "files" in call_args
        assert "data" in call_args
        packages = self.sync_interface.list_packages(include_synced=True)
        synced_package = next((p for p in packages if p["package_id"] == package_id), None)
        assert synced_package is not None
        assert synced_package["sync_status"] == "success"

    def test_send_package_missing(self):
        with pytest.raises(SyncError):
            self.client.send_package("nonexistent_package_id")

    def test_send_package_server_error(self):
        package_id = self.sync_interface.create_package(
            device_id=self.device_id,
            package_type="test"
        )
        self.sync_interface.add_file_to_package(
            package_id=package_id,
            file_path=self.model_file,
            file_type="model"
        )
        self.sync_interface.finalize_package(package_id)
        mock_response = MagicMock()
        mock_response.status_code = 500
        mock_response.text = "Server error"
        self.mock_connection_manager.execute_request.return_value = mock_response
        with pytest.raises(ConnectionError):
            self.client.send_package(package_id)
        packages = self.sync_interface.list_packages(include_synced=True)
        synced_package = next((p for p in packages if p["package_id"] == package_id), None)
        assert synced_package is not None
        assert synced_package["sync_status"] == "error"

    def test_sync_all_pending_packages(self):
        packages = []
        for i in range(3):
            package_id = self.sync_interface.create_package(
                device_id=self.device_id,
                package_type=f"test_{i}"
            )
            self.sync_interface.add_file_to_package(
                package_id=package_id,
                file_path=self.model_file,
                file_type="model"
            )
            self.sync_interface.finalize_package(package_id)
            packages.append(package_id)
        mock_response = MagicMock()
        mock_response.status_code = 200
        mock_response.json.return_value = {"server_id": "test_server_1"}
        self.mock_connection_manager.execute_request.return_value = mock_response
        original_send_package = self.client.send_package
        self.client.send_package = MagicMock(return_value=True)
        results = self.client.sync_all_pending_packages()
        assert len(results) == 3
        assert all(r["success"] for r in results)
        assert self.client.send_package.call_count == 3
        self.client.send_package = original_send_package

    def test_get_sync_status(self):
        package_id = self.sync_interface.create_package(
            device_id=self.device_id,
            package_type="test"
        )
        self.sync_interface.add_file_to_package(
            package_id=package_id,
            file_path=self.model_file,
            file_type="model"
        )
        self.sync_interface.finalize_package(package_id)
        self.sync_interface.mark_as_synced(
            package_id=package_id,
            sync_time=time.time(),
            server_id="test_server",
            status="success"
        )
        status = self.client.get_sync_status()
        assert "total_packages" in status
        assert status["total_packages"] >= 1
        assert "synced_packages" in status
        assert status["synced_packages"] >= 1
        assert "package_types" in status
        assert "test" in status["package_types"]
        assert "connection_status" in status