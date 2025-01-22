import logging
from datetime import datetime, timezone
from uuid import uuid4

#: How many commands to queue in a Redis pipeline before executing.
PIPELINE_BATCH_SIZE = 64


def setup_logger(name):
    """Configure the model store logger with appropriate formatting."""
    logger = logging.getLogger(name)

    # Only add handler if logger doesn't already have handlers
    if not logger.handlers:
        # Create console handler
        handler = logging.StreamHandler()

        # Create formatter
        formatter = logging.Formatter(
            "%(asctime)s.%(msecs)03d - %(name)s - %(levelname)s - %(message)s",
            datefmt="%Y-%m-%d %H:%M:%S",
        )

        # Add formatter to handler
        handler.setFormatter(formatter)

        # Add handler to logger
        logger.addHandler(handler)

        # Set default level to INFO
        logger.setLevel(logging.INFO)

    return logger


def current_timestamp() -> float:
    """
    Return the current UTC UNIX timestamp as a float.

    Returns:
        float: The current UTC time in seconds since the epoch.
    """
    return datetime.now(timezone.utc).timestamp()


def new_model_version() -> str:
    """
    Create a new UUID model version.

    Returns:
        str: A new UUID string to use as a model version identifier.
    """
    return str(uuid4())
