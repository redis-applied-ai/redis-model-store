import pytest

from model_store.serialize import PickleSerializer


@pytest.fixture
def serializer():
    return PickleSerializer()


class SampleClass:
    def __init__(self, x, y):
        self.x = x
        self.y = y

    def __eq__(self, other):
        return (
            isinstance(other, SampleClass) and self.x == other.x and self.y == other.y
        )


def test_pickle_serializer_simple_types(serializer):
    # Test with different types
    test_cases = [
        42,
        "hello world",
        [1, 2, 3],
        {"a": 1, "b": 2},
        (1, "two", 3.0),
        True,
        None,
    ]

    for obj in test_cases:
        serialized = serializer.dumps(obj)
        deserialized = serializer.loads(serialized)
        assert deserialized == obj


def test_pickle_serializer_complex_object(serializer):
    obj = SampleClass(42, "test")

    serialized = serializer.dumps(obj)
    deserialized = serializer.loads(serialized)

    assert deserialized == obj
    assert deserialized.x == 42
    assert deserialized.y == "test"


def test_pickle_serializer_invalid_data(serializer):
    with pytest.raises(Exception):
        serializer.loads(b"invalid data")
