from typing import Any, List, Optional

from redis import Redis

from model_store.registry import ModelRegistry, ModelVersion
from model_store.shard_manager import ModelShardManager, SerializationError
from model_store.utils import PIPELINE_BATCH_SIZE, current_timestamp, setup_logger

logger = setup_logger(__name__)


class ModelStoreError(Exception):
    """Raised when model store I/O operations fail."""
    pass


class ModelStore:
    """
    High-level interface for storing and retrieving AI/ML models from Redis.

    The ModelStore provides a simple API for saving and loading models,
    handling versioning, metadata, and efficient storage internally. It uses the
    ModelRegistry for version tracking and metadata management, and the ModelShardManager
    for serialization and sharding.

    Example:
        >>> store = ModelStore(redis_client)
        >>> store.save_model(model, name="my_model", description="My trained model")
        >>> loaded_model = store.load_model("my_model")
    """

    def __init__(self, redis_client: Redis, shard_size: int = 1024 * 1024):
        """
        Initialize the ModelStore.

        Args:
            redis_client (Redis): An initialized Redis client instance.
            shard_size (int, optional): Maximum size in bytes for each model shard.
                Defaults to 1MB.
        """
        if not isinstance(redis_client, Redis):
            raise TypeError("Must provide a valid Redis client instance.")

        self.redis_client = redis_client
        self.model_registry = ModelRegistry(redis_client)
        self.shard_manager = ModelShardManager(shard_size=shard_size)

    def _to_redis(self, model: Any, name: str, version: str) -> List[str]:
        """
        Serialize, shard and store a model in Redis.

        Args:
            model (Any): The model object to store.
            name (str): Name for the model.
            version (str): Version identifier for the model.

        Returns:
            List[str]: List of Redis keys where the model shards were stored.

        Raises:
            ModelStoreError: If serialization or storage fails.
        """
        start_time = current_timestamp()
        logger.info("Starting model serialization and storage")

        try:
            shard_keys: List[str] = []

            # Store shards in Redis
            with self.redis_client.pipeline(transaction=False) as pipe:
                # Serialize and shard the model
                logger.debug("Serializing and sharding model")
                for i, shard in enumerate(self.shard_manager.to_shards(model)):
                    skey = self.shard_manager.shard_key(name, version, i)
                    shard_keys.append(skey)
                    # Store under shard keys in Redis
                    pipe.set(skey, shard)
                    if i % PIPELINE_BATCH_SIZE == 0:
                        logger.debug("Executing pipeline batch")
                        pipe.execute()
                pipe.execute()

        except SerializationError as e:
            raise ModelStoreError(f"Failed to serialize model: {str(e)}") from e
        except Exception as e:
            # Clean up any shards that were stored before the error
            logger.error("Error during storage, cleaning up shards")
            self.redis_client.delete(*shard_keys)
            raise ModelStoreError(f"Failed to store model shards: {str(e)}") from e

        duration = current_timestamp() - start_time
        logger.info(f"Stored model in {len(shard_keys)} shards ({duration:.4f}s)")
        return shard_keys

    def save_model(self, model: Any, name: str, **kwargs) -> str:
        """
        Store a model in Redis with versioning and metadata.

        Saves the model by:
        1. Creating a model version record in the registry.
        2. Serializing and storing the model in chunks.
        3. Updating the registry with chunk locations.

        Args:
            model (Any): The model object to store.
            name (str): Name for the model.
            **kwargs: Additional model metadata fields (version,
                description, shard_keys).

        Returns:
            str: The created model version.

        Raises:
            ModelStoreError: If model storage or registration fails.
        """
        total_start = current_timestamp()
        model_version: Optional[ModelVersion] = None
        logger.info(f"Saving '{name}' model")

        try:
            # Create model version record
            st = current_timestamp()
            logger.debug("Creating model version record")
            model_version = self.model_registry.add_version(name=name, **kwargs)
            logger.info(f"Added model version record ({current_timestamp()-st:.4f}s)")

            # Store model chunks and get their keys
            logger.debug("Starting model storage")
            shard_keys = self._to_redis(model, name, model_version.version)

            # Update model version with shard locations
            st = current_timestamp()
            logger.debug("Updating model version with shard locations")
            self.model_registry.set_model_shards(
                name, model_version.version, shard_keys
            )
            logger.info(f"Set shards ({current_timestamp()-st:.4f}s)")

            total_duration = current_timestamp() - total_start
            logger.info(f"Total save operation completed in {total_duration:.4f}s")
            return model_version.version

        except Exception as e:
            # Clean up the version record if storage fails
            if model_version:
                logger.error("Error during save, cleaning up version record")
                self.model_registry.delete_version(name, model_version.version)
            if not isinstance(e, ModelStoreError):
                raise ModelStoreError(f"Failed to save model: {str(e)}") from e
            raise

    def _from_redis(self, shard_keys: List[str]) -> Any:
        """
        Load and reconstruct a model from its shards in Redis.

        Args:
            shard_keys (List[str]): List of Redis keys containing the model shards.

        Returns:
            Any: The reconstructed model object.

        Raises:
            ModelStoreError: If shard retrieval or deserialization fails
        """
        start_time = current_timestamp()
        shards: List[bytes] = []
        logger.info("Starting model reconstruction from shards")

        try:
            # Retrieve shards from Redis
            logger.debug("Retrieving shards from Redis")
            with self.redis_client.pipeline(transaction=False) as pipe:
                for i, skey in enumerate(shard_keys):
                    pipe.get(skey)
                    if i % PIPELINE_BATCH_SIZE == 0:
                        shards.extend(pipe.execute())
                shards.extend(pipe.execute())

            # Deserialize the model
            logger.debug("Deserializing model from shards")
            model = self.shard_manager.from_shards(shards)

            duration = current_timestamp() - start_time
            logger.info(f"Loaded model from {len(shard_keys)} shards ({duration:.4f}s)")
            return model

        except Exception as e:
            raise ModelStoreError(f"Failed to load model: {str(e)}") from e

    def load_model(self, name: str, version: Optional[str] = None) -> Any:
        """
        Load a model from Redis by name and optional version.

        Args:
            name (str): Unique identifier for the model.
            version (Optional[str]): Specific version to load. If None,
                loads the latest version.

        Returns:
            Any: The reconstructed model object.

        Raises:
            ModelStoreError: If model loading fails
        """
        total_start = current_timestamp()
        logger.info(f"Loading '{name}' model")

        try:
            st = current_timestamp()
            if not version:
                logger.debug("Retrieving latest model version")
                model_version = self.model_registry.get_latest_version(name)
            else:
                logger.debug(f"Retrieving specified model version: {version}")
                model_version = self.model_registry.get_version(name, version)

            logger.info(f"Retrieved model version metadata ({current_timestamp()-st:.4f}s)")
            model = self._from_redis(model_version.shard_keys)

            total_duration = current_timestamp() - total_start
            logger.info(f"Load operation completed ({total_duration:.4f}s)")
            return model

        except Exception as e:
            if not isinstance(e, ModelStoreError):
                raise ModelStoreError(f"Failed to load model {name}: {str(e)}") from e
            raise
