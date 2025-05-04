"""Tests for file utility functions."""

import json
import os
import shutil
import tempfile
from pathlib import Path

import pytest

from tinylcm.utils.file_utils import (
    ensure_dir,
    get_file_size,
    list_files,
    load_json,
    save_json,
)


class TestFileUtils:
    """Test file utility functions."""

    def __init__(self):
        """Initialize test class."""
        self.temp_dir = tempfile.mkdtemp()

    def setup_method(self):
        """Set up temporary directory for tests."""
        # Initialization moved to __init__
        pass

    def teardown_method(self):
        """Clean up temporary directory."""
        shutil.rmtree(self.temp_dir)

    def test_ensure_dir_creates_directory(self):
        """Test that ensure_dir creates a directory if it doesn't exist."""
        test_dir = os.path.join(self.temp_dir, "test_dir")
        assert not os.path.exists(test_dir)

        result = ensure_dir(test_dir)

        assert os.path.exists(test_dir)
        assert os.path.isdir(test_dir)
        assert isinstance(result, Path)
        assert str(result) == test_dir

    def test_ensure_dir_with_nested_paths(self):
        """Test that ensure_dir creates nested directories."""
        nested_dir = os.path.join(self.temp_dir, "parent/child/grandchild")
        assert not os.path.exists(nested_dir)

        result = ensure_dir(nested_dir)

        assert os.path.exists(nested_dir)
        assert os.path.isdir(nested_dir)
        assert isinstance(result, Path)

    def test_ensure_dir_accepts_path_object(self):
        """Test that ensure_dir accepts Path objects."""
        path_obj = Path(self.temp_dir) / "path_obj_dir"
        assert not path_obj.exists()

        result = ensure_dir(path_obj)

        assert path_obj.exists()
        assert path_obj.is_dir()
        assert result == path_obj

    def test_ensure_dir_with_existing_directory(self):
        """Test that ensure_dir handles existing directories gracefully."""
        existing_dir = os.path.join(self.temp_dir, "existing")
        os.makedirs(existing_dir)

        result = ensure_dir(existing_dir)

        assert os.path.exists(existing_dir)
        assert str(result) == existing_dir

    def test_save_json_creates_file(self):
        """Test that save_json creates a file with correct content."""
        data = {"key": "value", "nested": {"inner": 42}}
        json_path = os.path.join(self.temp_dir, "test.json")

        save_json(data, json_path)

        assert os.path.exists(json_path)
        with open(json_path, "r", encoding='utf-8') as f:
            loaded_data = json.load(f)
        assert loaded_data == data

    def test_save_json_creates_parent_directories(self):
        """Test that save_json creates parent directories if needed."""
        data = {"key": "value"}
        nested_path = os.path.join(self.temp_dir, "nested/path/test.json")

        save_json(data, nested_path)

        assert os.path.exists(nested_path)

    def test_save_json_with_path_object(self):
        """Test that save_json accepts Path objects."""
        data = {"key": "value"}
        json_path = Path(self.temp_dir) / "test_path_obj.json"

        save_json(data, json_path)

        assert json_path.exists()

    def test_load_json_loads_data_correctly(self):
        """Test that load_json loads data correctly."""
        original_data = {"key": "value", "list": [1, 2, 3]}
        json_path = os.path.join(self.temp_dir, "to_load.json")

        with open(json_path, "w", encoding='utf-8') as f:
            json.dump(original_data, f)

        loaded_data = load_json(json_path)

        assert loaded_data == original_data

    def test_load_json_with_path_object(self):
        """Test that load_json accepts Path objects."""
        original_data = {"key": "value"}
        json_path = Path(self.temp_dir) / "path_to_load.json"

        with open(json_path, "w", encoding='utf-8') as f:
            json.dump(original_data, f)

        loaded_data = load_json(json_path)

        assert loaded_data == original_data

    def test_load_json_raises_on_missing_file(self):
        """Test that load_json raises FileNotFoundError for missing files."""
        nonexistent_path = os.path.join(self.temp_dir, "nonexistent.json")

        with pytest.raises(FileNotFoundError):
            load_json(nonexistent_path)

    def test_get_file_size(self):
        """Test that get_file_size returns the correct size."""
        content = b"Hello, world!" * 10
        file_path = os.path.join(self.temp_dir, "test_size.txt")

        with open(file_path, "wb") as f:
            f.write(content)

        size = get_file_size(file_path)

        assert size == len(content)

    def test_get_file_size_with_path_object(self):
        """Test that get_file_size accepts Path objects."""
        content = b"Hello, world!"
        file_path = Path(self.temp_dir) / "test_size_path.txt"

        with open(file_path, "wb") as f:
            f.write(content)

        size = get_file_size(file_path)

        assert size == len(content)

    def test_get_file_size_raises_on_missing_file(self):
        """Test that get_file_size raises FileNotFoundError for missing files."""
        nonexistent_path = os.path.join(self.temp_dir, "nonexistent.txt")

        with pytest.raises(FileNotFoundError):
            get_file_size(nonexistent_path)

    def test_list_files_returns_matching_files(self):
        """Test that list_files returns files matching a pattern."""
        # Create test files
        files = ["test1.txt", "test2.txt", "other.csv", "nested/test3.txt"]
        for f in files:
            path = os.path.join(self.temp_dir, f)
            ensure_dir(os.path.dirname(path))
            with open(path, "w", encoding='utf-8') as file:
                file.write("content")

        # List txt files
        txt_files = list_files(self.temp_dir, "*.txt", recursive=False)

        assert len(txt_files) == 2
        assert all(f.name.endswith(".txt") for f in txt_files)

        # Test recursive
        all_txt_files = list_files(self.temp_dir, "*.txt", recursive=True)
        assert len(all_txt_files) == 3
