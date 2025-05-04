"""Tests for the ModelManager main class."""

import os
import shutil
from pathlib import Path
import uuid
from unittest.mock import MagicMock, patch

import pytest

from tinylcm.core.model_manager import (
    ModelManager, 
    FileSystemModelStorage, 
    JSONFileMetadataProvider,
    ModelNotFoundError, 
    ModelIntegrityError,
    StorageError
)
from tinylcm.constants import DEFAULT_ACTIVE_MODEL_LINK, DEFAULT_MODELS_DIR


class TestModelManager:
    
    def test_init_creates_directories(self, temp_dir):
        """Test that init creates required directories."""
        manager = ModelManager(storage_dir=temp_dir)
        
        models_dir = Path(temp_dir) / "models"
        metadata_dir = Path(temp_dir) / "metadata"
        
        assert models_dir.exists()
        assert models_dir.is_dir()
        assert metadata_dir.exists()
        assert metadata_dir.is_dir()
    
    def test_init_with_custom_strategies(self, temp_dir):
        """Test that init accepts custom strategy objects."""
        mock_storage = MagicMock(spec=FileSystemModelStorage)
        mock_metadata = MagicMock(spec=JSONFileMetadataProvider)
        
        manager = ModelManager(
            storage_dir=temp_dir,
            storage_strategy=mock_storage,
            metadata_provider=mock_metadata
        )
        
        assert manager.storage_strategy is mock_storage
        assert manager.metadata_provider is mock_metadata
    
    def test_save_model_generates_id(self, temp_dir, mock_model_file):
        """Test that save_model generates a UUID if not provided."""
        manager = ModelManager(storage_dir=temp_dir)
        
        with patch('uuid.uuid4', return_value=uuid.UUID('12345678-1234-5678-1234-567812345678')):
            model_id = manager.save_model(
                model_path=mock_model_file,
                model_format="json",
                description="Test model"
            )
        
        assert model_id == "12345678-1234-5678-1234-567812345678"
    
    def test_save_model_uses_strategies(self, temp_dir, mock_model_file):
        """Test that save_model uses the storage and metadata strategies."""
        mock_storage = MagicMock(spec=FileSystemModelStorage)
        mock_storage.save_model.return_value = {
            "filename": "test_model.json",
            "md5_hash": "mock_hash"
        }
        
        mock_metadata = MagicMock(spec=JSONFileMetadataProvider)
        
        manager = ModelManager(
            storage_dir=temp_dir,
            storage_strategy=mock_storage,
            metadata_provider=mock_metadata
        )
        
        model_id = manager.save_model(
            model_path=mock_model_file,
            model_format="json",
            description="Test model"
        )
        
        # Check storage strategy was called
        mock_storage.save_model.assert_called_once()
        args = mock_storage.save_model.call_args[1]
        assert args["model_path"] == mock_model_file
        assert args["model_id"] == model_id
        assert args["target_dir"] == manager.models_dir
        
        # Check metadata provider was called
        mock_metadata.save_metadata.assert_called_once()
        args = mock_metadata.save_metadata.call_args[1]
        assert args["model_id"] == model_id
        assert args["metadata_dir"] == manager.metadata_dir
        assert "model_format" in args["metadata"]
        assert args["metadata"]["model_format"] == "json"
        assert args["metadata"]["description"] == "Test model"
    
    def test_save_model_with_set_active(self, temp_dir, mock_model_file):
        """Test that save_model sets active model when requested."""
        manager = ModelManager(storage_dir=temp_dir)
        mock_set_active = MagicMock()
        manager.set_active_model = mock_set_active
        
        model_id = manager.save_model(
            model_path=mock_model_file,
            model_format="json",
            set_active=True
        )
        
        mock_set_active.assert_called_once_with(model_id)
    
    def test_load_model_with_id(self, temp_dir, mock_model_file):
        """Test that load_model loads a model by ID."""
        manager = ModelManager(storage_dir=temp_dir)
        
        # First save a model
        model_id = manager.save_model(
            model_path=mock_model_file,
            model_format="json"
        )
        
        # Then load it by ID
        model_path = manager.load_model(model_id)
        
        assert isinstance(model_path, str)
        assert Path(model_path).exists()
        assert Path(model_path).is_file()
    
    def test_load_model_active(self, temp_dir, mock_model_file):
        """Test that load_model loads the active model when no ID is provided."""
        manager = ModelManager(storage_dir=temp_dir)
        
        # Save a model and set it as active
        model_id = manager.save_model(
            model_path=mock_model_file,
            model_format="json",
            set_active=True
        )
        
        # Load the active model
        model_path = manager.load_model()
        
        assert isinstance(model_path, str)
        assert Path(model_path).exists()
        assert Path(model_path).is_file()
    
    def test_load_model_raises_when_no_active(self, temp_dir):
        """Test that load_model raises ModelNotFoundError when no active model and no ID."""
        manager = ModelManager(storage_dir=temp_dir)
        
        with pytest.raises(ModelNotFoundError, match="No active model set"):
            manager.load_model()
    
    def test_get_model_metadata(self, temp_dir, mock_model_file):
        """Test that get_model_metadata returns correct metadata."""
        manager = ModelManager(storage_dir=temp_dir)
        
        # Save a model with specific metadata
        model_id = manager.save_model(
            model_path=mock_model_file,
            model_format="json",
            description="Test description",
            tags=["test", "model"],
            metrics={"accuracy": 0.95}
        )
        
        # Get metadata
        metadata = manager.get_model_metadata(model_id)
        
        assert metadata["model_id"] == model_id
        assert metadata["model_format"] == "json"
        assert metadata["description"] == "Test description"
        assert metadata["tags"] == ["test", "model"]
        assert metadata["metrics"] == {"accuracy": 0.95}
    
    def test_list_models_unfiltered(self, temp_dir, mock_model_file):
        """Test that list_models returns all models when unfiltered."""
        manager = ModelManager(storage_dir=temp_dir)
        
        # Save multiple models
        model_id1 = manager.save_model(
            model_path=mock_model_file,
            model_format="json",
            tags=["test"]
        )
        
        model_id2 = manager.save_model(
            model_path=mock_model_file,
            model_format="pickle",
            tags=["production"]
        )
        
        # List all models
        models = manager.list_models()
        
        assert len(models) == 2
        model_ids = [m["model_id"] for m in models]
        assert model_id1 in model_ids
        assert model_id2 in model_ids
    
    def test_list_models_filtered_by_tag(self, temp_dir, mock_model_file):
        """Test that list_models filters by tag correctly."""
        manager = ModelManager(storage_dir=temp_dir)
        
        # Save multiple models with different tags
        model_id1 = manager.save_model(
            model_path=mock_model_file,
            model_format="json",
            tags=["test", "dev"]
        )
        
        model_id2 = manager.save_model(
            model_path=mock_model_file,
            model_format="json",
            tags=["prod"]
        )
        
        # Filter by tag
        test_models = manager.list_models(tag="test")
        prod_models = manager.list_models(tag="prod")
        
        assert len(test_models) == 1
        assert test_models[0]["model_id"] == model_id1
        
        assert len(prod_models) == 1
        assert prod_models[0]["model_id"] == model_id2
    
    def test_list_models_filtered_by_format(self, temp_dir, mock_model_file):
        """Test that list_models filters by format correctly."""
        manager = ModelManager(storage_dir=temp_dir)
        
        # Save multiple models with different formats
        model_id1 = manager.save_model(
            model_path=mock_model_file,
            model_format="json"
        )
        
        model_id2 = manager.save_model(
            model_path=mock_model_file,
            model_format="pickle"
        )
        
        # Filter by format
        json_models = manager.list_models(model_format="json")
        pickle_models = manager.list_models(model_format="pickle")
        
        assert len(json_models) == 1
        assert json_models[0]["model_id"] == model_id1
        
        assert len(pickle_models) == 1
        assert pickle_models[0]["model_id"] == model_id2
    
    def test_set_active_model_symlink(self, temp_dir, mock_model_file):
        """Test that set_active_model creates a symbolic link to the model."""
        manager = ModelManager(storage_dir=temp_dir)
        
        # Save a model
        model_id = manager.save_model(
            model_path=mock_model_file,
            model_format="json"
        )
        
        # Set as active
        manager.set_active_model(model_id)
        
        # Check symbolic link
        active_link = Path(temp_dir) / DEFAULT_ACTIVE_MODEL_LINK
        
        # Test passes whether using symlink or fallback text file approach
        if active_link.is_symlink():
            assert active_link.exists()
            target = os.readlink(active_link)
            if os.path.isabs(target):
                assert Path(target).name == model_id
            else:
                path_parts = Path(target).parts
                assert model_id in path_parts
        else:
            # Fallback text file approach
            assert active_link.is_file()
            with open(active_link, "r", encoding="utf-8") as f:
                stored_id = f.read().strip()
            assert stored_id == model_id
    
    def test_set_active_model_updates_metadata(self, temp_dir, mock_model_file):
        """Test that set_active_model updates the is_active flag in metadata."""
        manager = ModelManager(storage_dir=temp_dir)
        
        # Save multiple models
        model_id1 = manager.save_model(
            model_path=mock_model_file,
            model_format="json"
        )
        
        model_id2 = manager.save_model(
            model_path=mock_model_file,
            model_format="json"
        )
        
        # Set model1 as active
        manager.set_active_model(model_id1)
        
        # Check metadata
        metadata1 = manager.get_model_metadata(model_id1)
        metadata2 = manager.get_model_metadata(model_id2)
        
        assert metadata1["is_active"] is True
        assert metadata2["is_active"] is False
        
        # Now set model2 as active
        manager.set_active_model(model_id2)
        
        # Check metadata again
        metadata1 = manager.get_model_metadata(model_id1)
        metadata2 = manager.get_model_metadata(model_id2)
        
        assert metadata1["is_active"] is False
        assert metadata2["is_active"] is True
    
    def test_set_active_model_raises_when_not_found(self, temp_dir):
        """Test that set_active_model raises ModelNotFoundError for nonexistent model."""
        manager = ModelManager(storage_dir=temp_dir)
        
        with pytest.raises(ModelNotFoundError):
            manager.set_active_model("nonexistent_model")
    
    def test_delete_model(self, temp_dir, mock_model_file):
        """Test that delete_model removes the model and its metadata."""
        manager = ModelManager(storage_dir=temp_dir)
        
        # Save a model
        model_id = manager.save_model(
            model_path=mock_model_file,
            model_format="json"
        )
        
        # Delete the model
        result = manager.delete_model(model_id)
        
        # Check result
        assert result is True
        
        # Check model directory and metadata are gone
        model_dir = Path(manager.models_dir) / model_id
        metadata_file = Path(manager.metadata_dir) / f"{model_id}.json"
        
        assert not model_dir.exists()
        assert not metadata_file.exists()
    
    def test_delete_active_model_with_force(self, temp_dir, mock_model_file):
        """Test that delete_model can delete active model with force=True."""
        manager = ModelManager(storage_dir=temp_dir)
        
        # Save a model and set it as active
        model_id = manager.save_model(
            model_path=mock_model_file,
            model_format="json",
            set_active=True
        )
        
        # Delete the model with force
        result = manager.delete_model(model_id, force=True)
        
        # Check result
        assert result is True
        
        # Check model directory and metadata are gone
        model_dir = Path(manager.models_dir) / model_id
        metadata_file = Path(manager.metadata_dir) / f"{model_id}.json"
        active_link = Path(manager.storage_dir) / DEFAULT_ACTIVE_MODEL_LINK
        
        assert not model_dir.exists()
        assert not metadata_file.exists()
        # The active link may still exist but should be broken/invalid
    
    def test_delete_active_model_without_force(self, temp_dir, mock_model_file):
        """Test that delete_model raises ValueError for active model without force."""
        manager = ModelManager(storage_dir=temp_dir)
        
        # Save a model and set it as active
        model_id = manager.save_model(
            model_path=mock_model_file,
            model_format="json",
            set_active=True
        )
        
        # Try to delete without force
        with pytest.raises(ValueError, match="Cannot delete active model"):
            manager.delete_model(model_id, force=False)
        
        # Model should still exist
        model_dir = Path(manager.models_dir) / model_id
        metadata_file = Path(manager.metadata_dir) / f"{model_id}.json"
        
        assert model_dir.exists()
        assert metadata_file.exists()
    
    def test_add_tag(self, temp_dir, mock_model_file):
        """Test that add_tag adds a tag to the model metadata."""
        manager = ModelManager(storage_dir=temp_dir)
        
        # Save a model
        model_id = manager.save_model(
            model_path=mock_model_file,
            model_format="json",
            tags=["initial"]
        )
        
        # Add a tag
        result = manager.add_tag(model_id, "new_tag")
        
        # Check result
        assert result is True
        
        # Check metadata
        metadata = manager.get_model_metadata(model_id)
        assert "new_tag" in metadata["tags"]
        assert "initial" in metadata["tags"]
        assert len(metadata["tags"]) == 2
    
    def test_add_tag_idempotent(self, temp_dir, mock_model_file):
        """Test that add_tag is idempotent (adding same tag twice works)."""
        manager = ModelManager(storage_dir=temp_dir)
        
        # Save a model
        model_id = manager.save_model(
            model_path=mock_model_file,
            model_format="json",
            tags=["tag1"]
        )
        
        # Add same tag again
        result = manager.add_tag(model_id, "tag1")
        
        # Check result
        assert result is True
        
        # Check metadata (tag should still be there exactly once)
        metadata = manager.get_model_metadata(model_id)
        assert metadata["tags"] == ["tag1"]
    
    def test_remove_tag(self, temp_dir, mock_model_file):
        """Test that remove_tag removes a tag from the model metadata."""
        manager = ModelManager(storage_dir=temp_dir)
        
        # Save a model with tags
        model_id = manager.save_model(
            model_path=mock_model_file,
            model_format="json",
            tags=["tag1", "tag2", "tag3"]
        )
        
        # Remove a tag
        result = manager.remove_tag(model_id, "tag2")
        
        # Check result
        assert result is True
        
        # Check metadata
        metadata = manager.get_model_metadata(model_id)
        assert "tag1" in metadata["tags"]
        assert "tag2" not in metadata["tags"]
        assert "tag3" in metadata["tags"]
        assert len(metadata["tags"]) == 2
    
    def test_remove_nonexistent_tag(self, temp_dir, mock_model_file):
        """Test that remove_tag handles removing nonexistent tag gracefully."""
        manager = ModelManager(storage_dir=temp_dir)
        
        # Save a model with tags
        model_id = manager.save_model(
            model_path=mock_model_file,
            model_format="json",
            tags=["tag1"]
        )
        
        # Remove a nonexistent tag
        result = manager.remove_tag(model_id, "nonexistent")
        
        # Check result (should still be True as operation completed)
        assert result is True
        
        # Check metadata (should be unchanged)
        metadata = manager.get_model_metadata(model_id)
        assert metadata["tags"] == ["tag1"]
    
    def test_update_metrics(self, temp_dir, mock_model_file):
        """Test that update_metrics updates model metrics."""
        manager = ModelManager(storage_dir=temp_dir)
        
        # Save a model with initial metrics
        model_id = manager.save_model(
            model_path=mock_model_file,
            model_format="json",
            metrics={"accuracy": 0.9}
        )
        
        # Update metrics
        result = manager.update_metrics(
            model_id=model_id,
            metrics={"precision": 0.85, "recall": 0.88}
        )
        
        # Check result
        assert result is True
        
        # Check metadata
        metadata = manager.get_model_metadata(model_id)
        assert metadata["metrics"]["accuracy"] == 0.9  # Original metric preserved
        assert metadata["metrics"]["precision"] == 0.85  # New metric added
        assert metadata["metrics"]["recall"] == 0.88  # New metric added
    
    def test_verify_model_integrity_valid(self, temp_dir, mock_model_file):
        """Test that verify_model_integrity returns True for valid model."""
        manager = ModelManager(storage_dir=temp_dir)
        
        # Save a model
        model_id = manager.save_model(
            model_path=mock_model_file,
            model_format="json"
        )
        
        # Verify integrity
        result = manager.verify_model_integrity(model_id)
        
        assert result is True
    
    def test_verify_model_integrity_tampered(self, temp_dir, mock_model_file):
        """Test that verify_model_integrity returns False for tampered model."""
        manager = ModelManager(storage_dir=temp_dir)
        
        # Save a model
        model_id = manager.save_model(
            model_path=mock_model_file,
            model_format="json"
        )
        
        # Tamper with the model file
        model_files = list(Path(manager.models_dir / model_id).glob("*"))
        with open(model_files[0], "a", encoding="utf-8") as f:
            f.write("\ntampered")
        
        # Verify integrity
        result = manager.verify_model_integrity(model_id)
        
        assert result is False
    
    def test_verify_model_integrity_missing_hash(self, temp_dir, mock_model_file):
        """Test that verify_model_integrity returns False for missing hash."""
        manager = ModelManager(storage_dir=temp_dir)
        
        # Save a model
        model_id = manager.save_model(
            model_path=mock_model_file,
            model_format="json"
        )
        
        # Remove hash from metadata
        metadata = manager.get_model_metadata(model_id)
        del metadata["md5_hash"]
        
        # Save modified metadata
        manager.metadata_provider.save_metadata(
            model_id=model_id,
            metadata=metadata,
            metadata_dir=manager.metadata_dir
        )
        
        # Verify integrity
        result = manager.verify_model_integrity(model_id)
        
        assert result is False
    
    def test_verify_model_integrity_missing_model(self, temp_dir):
        """Test that verify_model_integrity raises for missing model."""
        manager = ModelManager(storage_dir=temp_dir)
        
        with pytest.raises(ModelNotFoundError):
            manager.verify_model_integrity("nonexistent_model")
    
    def test_from_config(self, temp_dir):
        """Test ModelManager initialization from config."""
        # This would normally use a Config mock, but for simplicity we'll just patch get_config
        from tinylcm.utils.config import Config
        
        config_mock = MagicMock(spec=Config)
        config_mock.get.return_value = temp_dir
        
        with patch('tinylcm.core.model_manager.manager.get_config', return_value=config_mock):
            manager = ModelManager()
            
            assert manager.storage_dir == Path(temp_dir)