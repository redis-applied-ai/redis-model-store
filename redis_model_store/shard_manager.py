from typing import Any, Iterator, List

from redis_model_store.serialize import PickleSerializer, SerializationError, Serializer


class ModelShardManager:
    """
    Manages serialization and sharding of model data.

    Handles breaking large objects into manageable chunks and provides
    serialization/deserialization. This provides an abstraction layer between
    raw objects and their serialized/sharded representation.

    The manager supports:
    - Configurable shard sizes
    - Pluggable serialization formats
    - Efficient streaming of shards
    """

    def __init__(self, shard_size: int, serializer: Serializer = PickleSerializer()):
        """
        Initialize the shard manager.

        Args:
            shard_size: Maximum size in bytes for each shard
            serializer: Serializer implementation to use (default: pickle)

        Raises:
            ValueError: If shard_size is not positive
        """
        if shard_size <= 0:
            raise ValueError("Shard size must be positive")

        self.shard_size = shard_size
        self.serializer = serializer

    @staticmethod
    def _shardify(data: bytes, shard_size: int) -> Iterator[bytes]:
        """Split serialized data into fixed-size shards."""
        total_size = len(data)
        for start in range(0, total_size, shard_size):
            yield data[start : start + shard_size]

    @staticmethod
    def shard_key(model_name: str, model_version: str, idx: int) -> str:
        """
        Generate a storage key for a model shard.

        Args:
            model_name: Name of the model
            model_version: Version identifier
            idx: Shard index number

        Returns:
            Formatted storage key for the shard
        """
        return f"shard:{model_name}:{model_version}:{idx}"

    def to_shards(self, obj: Any) -> Iterator[bytes]:
        """
        Convert object into shards ready for storage.

        The object is first serialized then split into fixed-size chunks.
        Shards are yielded one at a time to minimize memory usage.

        Args:
            obj: The object to shard

        Returns:
            Iterator yielding binary shards

        Raises:
            SerializationError: If the object cannot be serialized
        """
        try:
            serialized = self.serializer.dumps(obj)
            return self._shardify(serialized, self.shard_size)
        except Exception as e:
            raise SerializationError(f"Failed to serialize object: {str(e)}") from e

    def from_shards(self, shards: List[bytes]) -> Any:
        """
        Reconstruct object from shards.

        Args:
            shards: List of binary shards in order

        Returns:
            The reconstructed object

        Raises:
            SerializationError: If the shards cannot be deserialized
        """
        try:
            serialized = b"".join(shards)
            return self.serializer.loads(serialized)
        except Exception as e:
            raise SerializationError(f"Failed to deserialize object: {str(e)}") from e
