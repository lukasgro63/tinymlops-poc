"""Tests for configuration utility functions."""

import logging
import os
import shutil
import tempfile

import pytest

from tinylcm.utils.config import Config, get_config, load_config, set_global_config


class TestConfig:
    """Test configuration management."""

    def setup_method(self):
        """Set up temporary directory for tests."""
        self.temp_dir = tempfile.mkdtemp()

    def teardown_method(self):
        """Clean up temporary directory."""
        shutil.rmtree(self.temp_dir)

    def test_config_defaults(self):
        """Test that Config initializes with default values."""
        config = Config()

        # Check a few default values
        assert config.get("storage", "base_dir") == "tinylcm_data"
        assert config.get("model_manager", "storage_dir") == "tinylcm_data/models"
        assert "log_artifacts" in config.get("training_tracker")

    def test_config_get(self):
        """Test Config.get method."""
        config = Config()

        # Get with default
        value = config.get("nonexistent", "key", default="default_value")
        assert value == "default_value"

        # Get entire section
        storage_config = config.get("storage")
        assert isinstance(storage_config, dict)
        assert "base_dir" in storage_config

    def test_config_set(self):
        """Test Config.set method."""
        config = Config()

        # Set a value in existing section
        config.set("storage", "max_storage_bytes", 500000)
        assert config.get("storage", "max_storage_bytes") == 500000

        # Set a value in new section
        config.set("custom_section", "custom_key", "custom_value")
        assert config.get("custom_section", "custom_key") == "custom_value"

    def test_load_from_file(self):
        """Test loading config from file."""
        config_path = os.path.join(self.temp_dir, "config.json")
        custom_config = {
            "storage": {
                "base_dir": "custom_data_dir"
            },
            "custom_section": {
                "key1": "value1",
                "key2": 42
            }
        }

        # Create config file
        with open(config_path, "w") as f:
            import json
            json.dump(custom_config, f)

        # Load config
        config = Config()
        config.load_from_file(config_path)

        # Check values
        assert config.get("storage", "base_dir") == "custom_data_dir"
        assert config.get("custom_section", "key1") == "value1"
        assert config.get("custom_section", "key2") == 42

        # Non-overridden defaults should still be available
        assert "max_storage_bytes" in config.get("storage")

    def test_save_to_file(self):
        """Test saving config to file."""
        config = Config()
        config.set("custom_section", "custom_key", "custom_value")

        # Save config
        config_path = os.path.join(self.temp_dir, "saved_config.json")
        config.save_to_file(config_path)

        # Load and verify
        new_config = Config()
        new_config.load_from_file(config_path)

        assert new_config.get("custom_section", "custom_key") == "custom_value"

    def test_get_component_config(self):
        """Test getting component-specific config."""
        config = Config()

        model_manager_config = config.get_component_config("model_manager")
        assert isinstance(model_manager_config, dict)
        assert "storage_dir" in model_manager_config

    def test_global_config_functions(self):
        """Test global config functions."""
        # Get default global config
        default_global = get_config()
        assert default_global is not None

        # Load new config
        config_path = os.path.join(self.temp_dir, "global_config.json")
        with open(config_path, "w") as f:
            import json
            json.dump({"custom": {"key": "value"}}, f)

        new_config = load_config(config_path)
        assert new_config.get("custom", "key") == "value"

        # Set as global
        set_global_config(new_config)

        # Check that global config was updated
        current_global = get_config()
        assert current_global.get("custom", "key") == "value"
