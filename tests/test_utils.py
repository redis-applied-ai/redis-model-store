import logging
import uuid
from datetime import datetime, timezone

from model_store.utils import current_timestamp, new_model_version, setup_logger


def test_setup_logger():
    logger = setup_logger("test_logger")

    assert isinstance(logger, logging.Logger)
    assert logger.name == "test_logger"
    assert logger.level == logging.INFO
    assert len(logger.handlers) == 1
    assert isinstance(logger.handlers[0], logging.StreamHandler)

    # Test that calling setup_logger again doesn't add another handler
    logger = setup_logger("test_logger")
    assert len(logger.handlers) == 1


def test_current_timestamp():
    timestamp = current_timestamp()
    now = datetime.now(timezone.utc).timestamp()

    assert isinstance(timestamp, float)
    # Check if timestamp is recent (within 1 second)
    assert abs(timestamp - now) < 1


def test_new_model_version():
    version = new_model_version()

    assert isinstance(version, str)
    # Verify it's a valid UUID
    assert uuid.UUID(version)

    # Test uniqueness
    another_version = new_model_version()
    assert version != another_version


def test_pipeline_batch_size_constant():
    from model_store.utils import PIPELINE_BATCH_SIZE

    assert isinstance(PIPELINE_BATCH_SIZE, int)
    assert PIPELINE_BATCH_SIZE > 0
