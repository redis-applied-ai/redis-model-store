import json
import os
from typing import Generator

import pytest
from redis import Redis

from model_store.store import ModelStore, ModelStoreError, ModelVersion


@pytest.fixture
def redis_client():
    """Provide a Redis client for testing."""
    client = Redis.from_url(os.environ["REDIS_URL"])
    yield client
    # Clean up after tests
    client.flushdb()


@pytest.fixture
def store(redis_client: Redis):
    """Provide a ModelStore instance for testing."""
    return ModelStore(redis_client, shard_size=1024)


@pytest.fixture
def sample_model():
    """Provide a simple model for testing."""
    return {"weights": [1.0, 2.0, 3.0], "config": {"layers": 3, "activation": "relu"}}


@pytest.fixture
def large_model():
    """Provide a model large enough to create multiple shards."""
    return {
        "weights": [1.0456] * 1000,  # Large array
        "config": {"layers": 10, "activation": "relu"},
    }


@pytest.fixture
def populated_store(store: ModelStore, sample_model: dict):
    """Provide a store pre-populated with test models."""
    versions = {
        "model-a": ["v1.0"],
        "model-b": ["v1.0", "v2.0"],
        "model-c": ["v1.0", "v1.1", "v2.0"],
    }

    for name, model_versions in versions.items():
        for version in model_versions:
            store.save_model(
                sample_model,
                name=name,
                version=version,
                description=f"Test model {name} {version}",
            )

    yield store


class TestModelVersion:
    """Test ModelVersion creation and serialization."""

    def test_create_minimal(self):
        """Should create ModelVersion with only required fields."""
        version = ModelVersion(name="test-model")

        assert version.name == "test-model"
        assert version.description == ""
        assert version.version  # auto-generated
        assert version.created_at > 0
        assert version.shard_keys == []

    def test_create_complete(self):
        """Should create ModelVersion with all fields specified."""
        version = ModelVersion(
            name="test-model",
            description="Test model",
            version="v1.0",
            created_at=1234567890.0,
            shard_keys=["shard:1", "shard:2"],
        )

        assert version.name == "test-model"
        assert version.description == "Test model"
        assert version.version == "v1.0"
        assert version.created_at == 1234567890.0
        assert version.shard_keys == ["shard:1", "shard:2"]

    def test_from_dict_valid(self):
        """Should create ModelVersion from valid query result."""
        data = {
            "name": "test-model",
            "description": "Test model",
            "version": "v1.0",
            "created_at": 1234567890.0,
            "$.shard_keys": json.dumps(["shard:1", "shard:2"]),
        }

        version = ModelVersion.from_dict(data)
        assert version.name == "test-model"
        assert version.shard_keys == ["shard:1", "shard:2"]

    @pytest.mark.parametrize(
        "invalid_data",
        [
            {},  # Empty dict
            {"name": "test"},  # Missing required fields
            {  # Invalid shard keys format
                "name": "test",
                "version": "v1",
                "created_at": 123,
                "$.shard_keys": "invalid",
            },
        ],
    )
    def test_from_dict_invalid(self, invalid_data):
        """Should raise error for invalid data formats."""
        with pytest.raises(ModelStoreError, match="Invalid model version data"):
            ModelVersion.from_dict(invalid_data)


class TestModelStore:
    """Test ModelStore operations."""

    def test_init_invalid_client(self):
        """Should reject invalid Redis client."""
        with pytest.raises(TypeError, match="Must provide a valid Redis client"):
            ModelStore(None)

    def test_save_and_load_basic(self, store: ModelStore, sample_model: dict):
        """Should save and load model with basic metadata."""
        version = store.save_model(
            sample_model, name="test-model", description="Test model"
        )
        assert version  # version string returned

        loaded = store.load_model("test-model")
        assert loaded == sample_model

    def test_save_and_load_large(self, store: ModelStore, large_model: dict):
        """Should handle models requiring multiple shards."""
        version = store.save_model(large_model, name="large-model")
        loaded = store.load_model("large-model")
        assert loaded == large_model

    def test_save_duplicate_version(self, store: ModelStore, sample_model: dict):
        """Should prevent duplicate version creation."""
        store.save_model(sample_model, name="test", version="v1.0")

        with pytest.raises(ModelStoreError, match="Version exists"):
            store.save_model(sample_model, name="test", version="v1.0")

    @pytest.mark.parametrize(
        "name,version",
        [
            ("nonexistent", None),  # No such model
            ("test-model", "v999"),  # No such version
        ],
    )
    def test_load_nonexistent(self, store: ModelStore, name: str, version: str):
        """Should handle loading nonexistent models/versions."""
        with pytest.raises(ModelStoreError):
            store.load_model(name, version)

    def test_version_management(self, store: ModelStore, sample_model: dict):
        """Should manage multiple versions correctly."""
        # Create versions
        v1 = store.save_model(sample_model, name="test", version="v1.0")
        v2 = store.save_model(sample_model, name="test", version="v2.0")

        # Get specific version
        version = store.get_version("test", "v1.0")
        assert version.version == "v1.0"

        # Get latest
        latest = store.get_latest_version("test")
        assert latest.version == "v2.0"

        # Get all versions
        versions = store.get_all_versions("test")
        assert len(versions) == 2
        assert [v.version for v in versions] == ["v2.0", "v1.0"]

    def test_list_models(self, populated_store: ModelStore):
        """Should list available models correctly."""
        models = populated_store.list_models()
        assert models == ["model-a", "model-b", "model-c"]

        # After deletion
        populated_store.clear("model-b")
        models = populated_store.list_models()
        assert models == ["model-a", "model-c"]

    def test_delete_version(self, populated_store: ModelStore):
        """Should delete specific version and maintain others."""
        # Delete middle version
        deleted = populated_store.delete_version("model-c", "v1.1")
        assert deleted > 0

        versions = populated_store.get_all_versions("model-c")
        assert [v.version for v in versions] == ["v2.0", "v1.0"]

    def test_clear_specific(self, populated_store: ModelStore):
        """Should clear specific model and maintain others."""
        populated_store.clear("model-b")

        # model-b should be gone
        with pytest.raises(ModelStoreError):
            populated_store.get_version("model-b", "v1.0")

        # others should remain
        assert populated_store.list_models() == ["model-a", "model-c"]

    def test_clear_all(self, populated_store: ModelStore):
        """Should clear entire store."""
        deleted = populated_store.clear()
        assert deleted > 0

        assert populated_store.list_models() == []
