import pytest

from model_store.serialize import PickleSerializer
from model_store.shard_manager import ModelShardManager, SerializationError


@pytest.fixture
def small_shard_manager():
    return ModelShardManager(shard_size=10)  # Very small shard size for testing


@pytest.fixture
def shard_manager():
    return ModelShardManager(shard_size=1024)  # More realistic shard size


class LargeObject:
    def __init__(self, data):
        self.data = data

    def __eq__(self, other):
        return isinstance(other, LargeObject) and self.data == other.data


def test_shard_manager_init():
    # Test valid initialization
    manager = ModelShardManager(shard_size=1024)
    assert manager.shard_size == 1024
    assert isinstance(manager.serializer, PickleSerializer)

    # Test invalid shard size
    with pytest.raises(ValueError):
        ModelShardManager(shard_size=0)

    with pytest.raises(ValueError):
        ModelShardManager(shard_size=-1)


def test_shard_key_format():
    manager = ModelShardManager(shard_size=1024)
    key = manager.shard_key("model1", "v1", 0)
    assert key == "shard:model1:v1:0"


def test_small_object_single_shard(shard_manager):
    # Object small enough to fit in one shard
    obj = {"key": "value"}

    shards = list(shard_manager.to_shards(obj))
    assert len(shards) == 1

    reconstructed = shard_manager.from_shards(shards)
    assert reconstructed == obj


def test_large_object_multiple_shards(small_shard_manager):
    # Create object that will require multiple shards
    obj = LargeObject("x" * 25)  # Will be split into multiple shards

    shards = list(small_shard_manager.to_shards(obj))
    assert len(shards) > 1

    reconstructed = small_shard_manager.from_shards(shards)
    assert isinstance(reconstructed, LargeObject)
    assert reconstructed == obj
    assert reconstructed.data == obj.data


def test_shard_size_boundaries(small_shard_manager):
    # Test objects of different sizes around the shard boundary
    test_sizes = [9, 10, 11, 19, 20, 21, 100]

    for size in test_sizes:
        obj = LargeObject("x" * size)
        shards = list(small_shard_manager.to_shards(obj))
        reconstructed = small_shard_manager.from_shards(shards)
        assert reconstructed == obj


def test_invalid_shards():
    manager = ModelShardManager(shard_size=10)

    # Test with corrupted data
    with pytest.raises(SerializationError):
        manager.from_shards([b"invalid data"])

    # Test with empty shards
    with pytest.raises(SerializationError):
        manager.from_shards([])


def test_complex_object_serialization(shard_manager):
    # Test with a more complex nested structure
    obj = {
        "list": [1, 2, 3],
        "dict": {"nested": "value"},
        "tuple": (4, 5, 6),
        "object": LargeObject("test"),
    }

    shards = list(shard_manager.to_shards(obj))
    reconstructed = shard_manager.from_shards(shards)

    assert reconstructed == obj
    assert isinstance(reconstructed["object"], LargeObject)
    assert reconstructed["object"].data == "test"
