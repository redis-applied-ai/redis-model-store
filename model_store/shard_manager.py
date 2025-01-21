from typing import Any, Iterator, List

from model_store.serialize import PickleSerializer, SerializationError, Serializer


class ModelShardManager:
    """
    Manages sharding and serialization of model data.

    This class handles breaking large model objects into manageable shards and serializing
    them for storage. It provides an abstraction layer between raw model objects and their
    serialized/sharded representation ready for storage.

    Key features:
    - Configurable shard size to optimize for different storage backends
    - Pluggable serialization via the Serializer protocol
    - Maintains data integrity across sharding/reassembly

    Example:
        >>> manager = ModelShardManager()
        >>> shards = manager.to_shards(large_model)  # Split model into shards
        >>> reconstructed = manager.from_shards(shards)  # Reassemble model
    """

    def __init__(self, shard_size: int, serializer: Serializer = PickleSerializer()):
        """
        Initialize the shard manager.

        Args:
            shard_size (int): Maximum size in bytes for each shard.
            serializer (Serializer): Serializer implementation to use. Defaults to pickle.
        """
        if shard_size <= 0:
            raise ValueError("Shard size must be positive")

        self.shard_size = shard_size
        self.serializer = serializer

    def _shardify(self, data: bytes) -> Iterator[bytes]:
        """
        Split serialized data into fixed-size shards.

        Args:
            data (bytes): The full serialized model data.

        Yields:
            bytes: Successive shards of the data, each up to shard_size in length.
        """
        total_size = len(data)
        for start in range(0, total_size, self.shard_size):
            yield data[start : start + self.shard_size]

    @staticmethod
    def shard_key(model_name: str, model_version: str, idx: int) -> str:
        """
        Generate a storage key for a model shard.

        Args:
            model_name (str): Name of the model.
            model_version (str): Version identifier of the model.
            idx (int): Shard index.

        Returns:
            str: Formatted storage key for the model shard.
        """
        return f"shard:{model_name}:{model_version}:{idx}"

    def to_shards(self, model: Any) -> List[bytes]:
        """
        Convert model into smaller chunks (shards) ready for storage.

        Args:
            model (Any): The model object to shard.

        Returns:
            List[bytes]: List of binary shards derived from the model.
            TODO -- returns a generator here

        Raises:
            SerializationError: If model serialization fails.
        """
        try:
            serialized_data = self.serializer.dumps(model)
            return self._shardify(serialized_data)
        except Exception as e:
            raise SerializationError(f"Failed to serialize model: {str(e)}") from e

    def from_shards(self, shards: List[bytes]) -> Any:
        """
        Reconstruct a model from its shards.

        Args:
            shards (List[bytes]): List of binary shards to reassemble.

        Returns:
            Any: The reconstructed model object.

        Raises:
            SerializationError: If model deserialization fails.
        """
        try:
            serialized_data = b"".join(shards)
            return self.serializer.loads(serialized_data)
        except Exception as e:
            raise SerializationError(f"Failed to deserialize model: {str(e)}") from e
