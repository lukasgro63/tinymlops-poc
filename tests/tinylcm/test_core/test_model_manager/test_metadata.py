"""Tests for model metadata providers."""

import os
import json
from pathlib import Path

import pytest

from tinylcm.core.model_manager import JSONFileMetadataProvider, ModelMetadataProvider
from tinylcm.utils.errors import ModelNotFoundError, StorageError


class TestJSONFileMetadataProvider:
    
    def test_is_metadata_provider(self):
        """Test that JSONFileMetadataProvider implements ModelMetadataProvider."""
        provider = JSONFileMetadataProvider()
        assert isinstance(provider, ModelMetadataProvider)
    
    def test_save_metadata_creates_file(self, temp_dir, sample_metadata):
        provider = JSONFileMetadataProvider()
        
        # Hier den richtigen model_id verwenden - nicht aus sample_metadata
        model_id = "valid_model"
        
        # Metadaten mit der korrekten model_id erstellen
        valid_metadata = sample_metadata.copy()
        valid_metadata["model_id"] = model_id
        
        # Speichern
        provider.save_metadata(
            model_id=model_id,
            metadata=valid_metadata,
            metadata_dir=temp_dir
        )
        
        # Prüfen mit dem neuen model_id
        metadata_file = Path(temp_dir) / f"{model_id}.json"
        assert metadata_file.exists()
        
    def test_save_metadata_writes_correct_content(self, temp_dir, sample_metadata):
        """Test that save_metadata writes the correct content."""
        provider = JSONFileMetadataProvider()
        model_id = sample_metadata["model_id"]
        
        provider.save_metadata(
            model_id=model_id,
            metadata=sample_metadata,
            metadata_dir=temp_dir
        )
        
        metadata_file = Path(temp_dir) / f"{model_id}.json"
        with open(metadata_file, "r", encoding="utf-8") as f:
            loaded_data = json.load(f)
        
        assert loaded_data == sample_metadata
    
    def test_load_metadata_returns_correct_data(self, temp_dir, sample_metadata):
        """Test that load_metadata returns the correct metadata."""
        provider = JSONFileMetadataProvider()
        model_id = sample_metadata["model_id"]
        
        # First save the metadata
        provider.save_metadata(
            model_id=model_id,
            metadata=sample_metadata,
            metadata_dir=temp_dir
        )
        
        # Then load it
        loaded_metadata = provider.load_metadata(
            model_id=model_id,
            metadata_dir=temp_dir
        )
        
        assert loaded_metadata == sample_metadata
    
    def test_load_metadata_raises_when_not_found(self, temp_dir):
        """Test that load_metadata raises ModelNotFoundError when metadata not found."""
        provider = JSONFileMetadataProvider()
        model_id = "nonexistent_model"
        
        with pytest.raises(ModelNotFoundError):
            provider.load_metadata(
                model_id=model_id,
                metadata_dir=temp_dir
            )
    
    def test_load_metadata_raises_on_invalid_json(self, temp_dir):
        """Test that load_metadata raises StorageError on invalid JSON."""
        provider = JSONFileMetadataProvider()
        model_id = "invalid_json_model"
        
        # Create invalid JSON file
        metadata_file = Path(temp_dir) / f"{model_id}.json"
        with open(metadata_file, "w", encoding="utf-8") as f:
            f.write("{invalid json")
        
        with pytest.raises(StorageError):
            provider.load_metadata(
                model_id=model_id,
                metadata_dir=temp_dir
            )
    
    def test_list_metadata_returns_empty_list_for_empty_dir(self, temp_dir):
        """Test that list_metadata returns an empty list for an empty directory."""
        provider = JSONFileMetadataProvider()
        
        metadata_list = provider.list_metadata(
            metadata_dir=temp_dir
        )
        
        assert isinstance(metadata_list, list)
        assert len(metadata_list) == 0
    
    def test_list_metadata_returns_all_metadatas(self, temp_dir, sample_metadata):
        """Test that list_metadata returns all metadata files."""
        provider = JSONFileMetadataProvider()
        
        # Create multiple metadata files
        metadata1 = dict(sample_metadata)
        metadata1["model_id"] = "model1"
        
        metadata2 = dict(sample_metadata)
        metadata2["model_id"] = "model2"
        
        metadata3 = dict(sample_metadata)
        metadata3["model_id"] = "model3"
        
        # Save all metadata
        for metadata in [metadata1, metadata2, metadata3]:
            provider.save_metadata(
                model_id=metadata["model_id"],
                metadata=metadata,
                metadata_dir=temp_dir
            )
        
        # List all metadata
        metadata_list = provider.list_metadata(
            metadata_dir=temp_dir
        )
        
        assert isinstance(metadata_list, list)
        assert len(metadata_list) == 3
        
        # Check if all metadata is included
        model_ids = [m["model_id"] for m in metadata_list]
        assert "model1" in model_ids
        assert "model2" in model_ids
        assert "model3" in model_ids
    
    def test_list_metadata_with_filter(self, temp_dir, sample_metadata):
        """Test that list_metadata respects the filter function."""
        provider = JSONFileMetadataProvider()
        
        # Create multiple metadata files with different tags
        metadata1 = dict(sample_metadata)
        metadata1["model_id"] = "model1"
        metadata1["tags"] = ["tag1", "tag2"]
        
        metadata2 = dict(sample_metadata)
        metadata2["model_id"] = "model2"
        metadata2["tags"] = ["tag1"]
        
        metadata3 = dict(sample_metadata)
        metadata3["model_id"] = "model3"
        metadata3["tags"] = ["tag2", "tag3"]
        
        # Save all metadata
        for metadata in [metadata1, metadata2, metadata3]:
            provider.save_metadata(
                model_id=metadata["model_id"],
                metadata=metadata,
                metadata_dir=temp_dir
            )
        
        # List metadata with tag1
        tag1_filter = lambda m: "tag1" in m.get("tags", [])
        tag1_list = provider.list_metadata(
            metadata_dir=temp_dir,
            filter_func=tag1_filter
        )
        
        assert len(tag1_list) == 2
        model_ids = [m["model_id"] for m in tag1_list]
        assert "model1" in model_ids
        assert "model2" in model_ids
        assert "model3" not in model_ids
        
        # List metadata with tag3
        tag3_filter = lambda m: "tag3" in m.get("tags", [])
        tag3_list = provider.list_metadata(
            metadata_dir=temp_dir,
            filter_func=tag3_filter
        )
        
        assert len(tag3_list) == 1
        assert tag3_list[0]["model_id"] == "model3"
    
    def test_list_metadata_handles_invalid_files(self, temp_dir, sample_metadata):
        provider = JSONFileMetadataProvider()
        
        # Korrekte model_id definieren
        model_id = "valid_model"
        
        # Metadaten mit der korrekten model_id erstellen
        valid_metadata = sample_metadata.copy()
        valid_metadata["model_id"] = model_id
        
        # Speichern mit der korrekten model_id
        provider.save_metadata(
            model_id=model_id,
            metadata=valid_metadata,
            metadata_dir=temp_dir
        )
            
        # Create invalid JSON file
        invalid_file = Path(temp_dir) / "invalid.json"
        with open(invalid_file, "w", encoding="utf-8") as f:
            f.write("{invalid json")
        
        # List should include only valid metadata and ignore errors
        metadata_list = provider.list_metadata(
            metadata_dir=temp_dir
        )
        
        assert len(metadata_list) == 1
        assert metadata_list[0]["model_id"] == "valid_model"
    
    def test_delete_metadata_removes_file(self, temp_dir, sample_metadata):
        """Test that delete_metadata removes the metadata file."""
        provider = JSONFileMetadataProvider()
        model_id = sample_metadata["model_id"]
        
        # First save the metadata
        provider.save_metadata(
            model_id=model_id,
            metadata=sample_metadata,
            metadata_dir=temp_dir
        )
        
        # Check file exists
        metadata_file = Path(temp_dir) / f"{model_id}.json"
        assert metadata_file.exists()
        
        # Delete the metadata
        result = provider.delete_metadata(
            model_id=model_id,
            metadata_dir=temp_dir
        )
        
        # Check result
        assert result is True
        assert not metadata_file.exists()
    
    def test_delete_metadata_returns_false_when_not_found(self, temp_dir):
        """Test that delete_metadata returns False when metadata not found."""
        provider = JSONFileMetadataProvider()
        model_id = "nonexistent_model"
        
        result = provider.delete_metadata(
            model_id=model_id,
            metadata_dir=temp_dir
        )
        
        assert result is False
    
    def test_delete_metadata_raises_on_permission_error(self, temp_dir, sample_metadata, monkeypatch):
        """Test that delete_metadata raises StorageError on permission error."""
        provider = JSONFileMetadataProvider()
        model_id = sample_metadata["model_id"]
        
        # Save metadata
        provider.save_metadata(
            model_id=model_id,
            metadata=sample_metadata,
            metadata_dir=temp_dir
        )
        
        # Mock Path.unlink to raise PermissionError
        original_unlink = Path.unlink
        
        def mock_unlink(self, *args, **kwargs):
            if self.name == f"{model_id}.json":
                raise PermissionError("Permission denied")
            return original_unlink(self, *args, **kwargs)
        
        monkeypatch.setattr(Path, "unlink", mock_unlink)
        
        # Try to delete
        with pytest.raises(StorageError):
            provider.delete_metadata(
                model_id=model_id,
                metadata_dir=temp_dir
            )