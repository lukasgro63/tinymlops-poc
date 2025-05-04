"""Tests for versioning utility functions."""

import os
import shutil
import tempfile
import time

import pytest

from tinylcm.utils.versioning import (
    calculate_content_hash,
    calculate_file_hash,
    compare_versions,
    create_version_info,
    generate_incremental_version,
    generate_timestamp_version,
    get_version_diff,
)


class TestVersioning:
    """Test versioning utility functions."""

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

    def test_generate_timestamp_version(self):
        """Test timestamp-based version generation."""
        version = generate_timestamp_version()

        # Version should be in format v_YYYYMMDD_HHMMSS
        assert version.startswith("v_")
        assert len(version) >= 16  # v_ + 8 digits for date + _ + 6 digits for time

        # Generate another version and make sure it's different
        time.sleep(1)  # Wait to ensure different timestamp
        second_version = generate_timestamp_version()
        assert second_version != version

    def test_generate_incremental_version(self):
        """Test incremental version generation."""
        # Create directories with existing versions
        for i in range(1, 4):
            version_dir = os.path.join(self.temp_dir, f"v_{i:03d}")
            os.makedirs(version_dir)

        # Generate new version
        version = generate_incremental_version(self.temp_dir)

        assert version == "v_004"  # Should increment to 4

        # Test with different prefix
        custom_prefix = "model_"
        for i in range(1, 3):
            version_dir = os.path.join(self.temp_dir, f"{custom_prefix}{i}")
            os.makedirs(version_dir)

        custom_version = generate_incremental_version(self.temp_dir, prefix=custom_prefix)
        assert custom_version == f"{custom_prefix}3"

    def test_generate_incremental_version_empty_dir(self):
        """Test incremental version with empty directory."""
        empty_dir = os.path.join(self.temp_dir, "empty")
        os.makedirs(empty_dir)

        version = generate_incremental_version(empty_dir)

        assert version == "v_001"  # Should start from 1

    def test_calculate_file_hash(self):
        """Test file hash calculation."""
        # Create a test file
        file_path = os.path.join(self.temp_dir, "test_file.txt")
        content = b"Test content for hashing"

        with open(file_path, "wb") as f:
            f.write(content)

        # Calculate hash
        file_hash = calculate_file_hash(file_path)

        # Should return a hex string
        assert isinstance(file_hash, str)
        assert len(file_hash) == 32  # MD5 hash is 32 characters

        # Same content should have same hash
        file_path2 = os.path.join(self.temp_dir, "test_file2.txt")
        with open(file_path2, "wb") as f:
            f.write(content)

        file_hash2 = calculate_file_hash(file_path2)
        assert file_hash == file_hash2

        # Different content should have different hash
        file_path3 = os.path.join(self.temp_dir, "test_file3.txt")
        with open(file_path3, "wb") as f:
            f.write(b"Different content")

        file_hash3 = calculate_file_hash(file_path3)
        assert file_hash != file_hash3

    def test_calculate_file_hash_with_algorithm(self):
        """Test file hash calculation with different algorithms."""
        file_path = os.path.join(self.temp_dir, "test_file.txt")
        with open(file_path, "wb") as f:
            f.write(b"Test content")

        md5_hash = calculate_file_hash(file_path, algorithm="md5")
        sha1_hash = calculate_file_hash(file_path, algorithm="sha1")
        sha256_hash = calculate_file_hash(file_path, algorithm="sha256")

        assert len(md5_hash) == 32
        assert len(sha1_hash) == 40
        assert len(sha256_hash) == 64

        assert md5_hash != sha1_hash
        assert md5_hash != sha256_hash

    def test_calculate_file_hash_raises_on_missing_file(self):
        """Test file hash calculation with missing file."""
        nonexistent_path = os.path.join(self.temp_dir, "nonexistent.txt")

        with pytest.raises(FileNotFoundError):
            calculate_file_hash(nonexistent_path)

    def test_calculate_content_hash(self):
        """Test content hash calculation."""
        content = "Test content for hashing"

        # Calculate hash
        content_hash = calculate_content_hash(content)

        # Should return a hex string
        assert isinstance(content_hash, str)
        assert len(content_hash) == 32  # MD5 hash is 32 characters

        # Same content should have same hash
        content_hash2 = calculate_content_hash(content)
        assert content_hash == content_hash2

        # Different content should have different hash
        content_hash3 = calculate_content_hash("Different content")
        assert content_hash != content_hash3

    def test_calculate_content_hash_with_bytes(self):
        """Test content hash with bytes input."""
        content_bytes = b"Test content as bytes"
        content_str = "Test content as bytes"

        bytes_hash = calculate_content_hash(content_bytes)
        str_hash = calculate_content_hash(content_str)

        assert bytes_hash == str_hash

    def test_create_version_info_with_file(self):
        """Test version info creation with file."""
        # Create a test file
        file_path = os.path.join(self.temp_dir, "model.json")
        content = '{"model": "test", "version": 1}'

        with open(file_path, "w", encoding='utf-8') as f:
            f.write(content)

        # Create version info
        metadata = {"accuracy": 0.95, "created_by": "test"}
        version_info = create_version_info(source_file=file_path, metadata=metadata)

        # Check fields
        assert "version_id" in version_info
        assert "timestamp" in version_info
        assert "filename" in version_info
        assert version_info["filename"] == "model.json"
        assert "file_size_bytes" in version_info
        assert version_info["file_size_bytes"] == len(content)
        assert "file_hash" in version_info
        assert "metadata" in version_info
        assert version_info["metadata"] == metadata

    def test_create_version_info_with_content(self):
        """Test version info creation with content."""
        content = "Model content for versioning"

        # Create version info
        version_info = create_version_info(content=content)

        # Check fields
        assert "version_id" in version_info
        assert "timestamp" in version_info
        assert "content_hash" in version_info
        assert "content_size_bytes" in version_info
        assert version_info["content_size_bytes"] == len(content)

    def test_create_version_info_requires_file_or_content(self):
        """Test version info creation requires either file or content."""
        with pytest.raises(ValueError):
            create_version_info()

    def test_compare_versions(self):
        """Test version comparison."""
        # Create two version infos with same content
        content = "Same content"
        version1 = create_version_info(content=content)
        version2 = create_version_info(content=content)

        # Should be considered same
        assert compare_versions(version1, version2) is True

        # Create version with different content
        version3 = create_version_info(content="Different content")

        # Should be considered different
        assert compare_versions(version1, version3) is False

    def test_get_version_diff(self):
        """Test version difference calculation."""
        # Create two versions
        metadata1 = {"accuracy": 0.8, "classes": ["cat", "dog"]}
        metadata2 = {"accuracy": 0.9, "classes": ["cat", "dog", "bird"]}

        version1 = create_version_info(
            content="Version 1 content",
            metadata=metadata1
        )

        # Wait a bit to ensure different timestamp
        time.sleep(1)

        version2 = create_version_info(
            content="Version 2 content",
            metadata=metadata2
        )

        # Get diff
        diff = get_version_diff(version1, version2)

        # Check diff fields
        assert "is_same_content" in diff
        assert diff["is_same_content"] is False

        assert "time_difference_seconds" in diff
        assert diff["time_difference_seconds"] > 0

        assert "metadata_changes" in diff
        assert "accuracy" in diff["metadata_changes"]
        assert diff["metadata_changes"]["accuracy"]["from"] == 0.8
        assert diff["metadata_changes"]["accuracy"]["to"] == 0.9
