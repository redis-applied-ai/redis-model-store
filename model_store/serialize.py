import pickle
from typing import Any, Protocol


class SerializationError(Exception):
    """Raised when model serialization or deserialization fails."""

    pass


class Serializer(Protocol):
    """Protocol defining the interface for model serialization."""

    def dumps(self, obj: Any) -> bytes:
        """Serialize object to bytes."""
        pass

    def loads(self, data: bytes) -> Any:
        """Deserialize bytes to object."""
        pass


class PickleSerializer(Serializer):
    """Default serializer implementation using pickle."""

    def dumps(self, obj: Any) -> bytes:
        return pickle.dumps(obj, protocol=pickle.HIGHEST_PROTOCOL)

    def loads(self, data: bytes) -> Any:
        return pickle.loads(data)
