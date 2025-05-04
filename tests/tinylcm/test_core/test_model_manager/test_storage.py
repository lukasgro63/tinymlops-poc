"""Tests for model storage strategies."""

import os
import shutil
from pathlib import Path

import pytest

from tinylcm.core.model_manager import FileSystemModelStorage, ModelStorageStrategy
from tinylcm.utils.errors import ModelNotFoundError, StorageError


class TestFileSystemModelStorage:
    
    def test_is_model_storage_strategy(self):
        """Test that FileSystemModelStorage implements ModelStorageStrategy."""
        storage = FileSystemModelStorage()
        assert isinstance(storage, ModelStorageStrategy)
    
    def test_save_model_creates_directory(self, temp_dir, mock_model_file):
        """Test that save_model creates the model directory."""
        storage = FileSystemModelStorage()
        model_id = "test_model_123"
        
        storage.save_model(
            model_path=mock_model_file,
            model_id=model_id,
            target_dir=temp_dir
        )
        
        model_dir = Path(temp_dir) / model_id
        assert model_dir.exists()
        assert model_dir.is_dir()
    
    def test_save_model_copies_file(self, temp_dir, mock_model_file):
        """Test that save_model copies the model file."""
        storage = FileSystemModelStorage()
        model_id = "test_model_123"
        
        storage.save_model(
            model_path=mock_model_file,
            model_id=model_id,
            target_dir=temp_dir
        )
        
        copied_file = Path(temp_dir) / model_id / Path(mock_model_file).name
        assert copied_file.exists()
        assert copied_file.is_file()
    
    def test_save_model_returns_metadata(self, temp_dir, mock_model_file):
        """Test that save_model returns correct metadata."""
        storage = FileSystemModelStorage()
        model_id = "test_model_123"
        
        metadata = storage.save_model(
            model_path=mock_model_file,
            model_id=model_id,
            target_dir=temp_dir
        )
        
        assert isinstance(metadata, dict)
        assert "filename" in metadata
        assert "path" in metadata
        assert "md5_hash" in metadata
        assert metadata["filename"] == Path(mock_model_file).name
    
    def test_save_model_fails_with_invalid_path(self, temp_dir):
        """Test that save_model raises StorageError with invalid model path."""
        storage = FileSystemModelStorage()
        model_id = "test_model_123"
        
        with pytest.raises(StorageError):
            storage.save_model(
                model_path="/nonexistent/path/model.json",
                model_id=model_id,
                target_dir=temp_dir
            )
    
    def test_load_model_returns_path(self, temp_dir, mock_model_file):
        """Test that load_model returns the correct model file path."""
        storage = FileSystemModelStorage()
        model_id = "test_model_123"
        
        # First save the model
        storage.save_model(
            model_path=mock_model_file,
            model_id=model_id,
            target_dir=temp_dir
        )
        
        # Now load it
        model_path = storage.load_model(
            model_id=model_id,
            models_dir=temp_dir
        )
        
        # Check result
        assert isinstance(model_path, str)
        assert Path(model_path).exists()
        assert Path(model_path).is_file()
        assert Path(model_path).name == Path(mock_model_file).name
    
    def test_load_model_raises_when_not_found(self, temp_dir):
        """Test that load_model raises ModelNotFoundError when model not found."""
        storage = FileSystemModelStorage()
        model_id = "nonexistent_model"
        
        with pytest.raises(ModelNotFoundError):
            storage.load_model(
                model_id=model_id,
                models_dir=temp_dir
            )
    
    def test_load_model_raises_when_directory_empty(self, temp_dir):
        """Test that load_model raises ModelNotFoundError when directory is empty."""
        storage = FileSystemModelStorage()
        model_id = "empty_model"
        
        # Create empty model directory
        model_dir = Path(temp_dir) / model_id
        model_dir.mkdir(parents=True, exist_ok=True)
        
        with pytest.raises(ModelNotFoundError):
            storage.load_model(
                model_id=model_id,
                models_dir=temp_dir
            )
    
    def test_delete_model_removes_directory(self, temp_dir, mock_model_file):
        """Test that delete_model removes the model directory."""
        storage = FileSystemModelStorage()
        model_id = "test_model_to_delete"
        
        # First save the model
        storage.save_model(
            model_path=mock_model_file,
            model_id=model_id,
            target_dir=temp_dir
        )
        
        # Check directory exists
        model_dir = Path(temp_dir) / model_id
        assert model_dir.exists()
        
        # Delete the model
        result = storage.delete_model(
            model_id=model_id,
            models_dir=temp_dir
        )
        
        # Check result
        assert result is True
        assert not model_dir.exists()
    
    def test_delete_model_returns_false_when_not_found(self, temp_dir):
        """Test that delete_model returns False when model not found."""
        storage = FileSystemModelStorage()
        model_id = "nonexistent_model"
        
        result = storage.delete_model(
            model_id=model_id,
            models_dir=temp_dir
        )
        
        assert result is False
    
    def test_delete_model_raises_on_permission_error(self, temp_dir, mock_model_file, monkeypatch):
        """Test that delete_model raises StorageError on permission error."""
        storage = FileSystemModelStorage()
        model_id = "test_model_permission"
        
        # First save the model
        storage.save_model(
            model_path=mock_model_file,
            model_id=model_id,
            target_dir=temp_dir
        )
        
        # Mock shutil.rmtree to raise PermissionError
        def mock_rmtree(*args, **kwargs):
            raise PermissionError("Permission denied")
        
        monkeypatch.setattr(shutil, "rmtree", mock_rmtree)
        
        # Try to delete
        with pytest.raises(StorageError):
            storage.delete_model(
                model_id=model_id,
                models_dir=temp_dir
            )
    
    def test_multiple_models(self, temp_dir, mock_model_file, mock_model_binary):
        """Test handling multiple models with the storage strategy."""
        storage = FileSystemModelStorage()
        
        # Save two different models
        model_id1 = "model_json"
        model_id2 = "model_binary"
        
        # Save the models
        storage.save_model(
            model_path=mock_model_file,
            model_id=model_id1,
            target_dir=temp_dir
        )
        
        storage.save_model(
            model_path=mock_model_binary,
            model_id=model_id2,
            target_dir=temp_dir
        )
        
        # Load and verify both models
        path1 = storage.load_model(model_id1, temp_dir)
        path2 = storage.load_model(model_id2, temp_dir)
        
        assert Path(path1).name == Path(mock_model_file).name
        assert Path(path2).name == Path(mock_model_binary).name
        
        # Delete one model and verify the other still exists
        storage.delete_model(model_id1, temp_dir)
        
        with pytest.raises(ModelNotFoundError):
            storage.load_model(model_id1, temp_dir)
            
        path2_again = storage.load_model(model_id2, temp_dir)
        assert path2_again == path2