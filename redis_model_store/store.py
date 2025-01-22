import json
from typing import Any, List, Optional, Set

from pydantic import BaseModel, Field
from redis import Redis
from redisvl.index import SearchIndex
from redisvl.query import FilterQuery
from redisvl.query.filter import FilterExpression, Tag

from redis_model_store.shard_manager import ModelShardManager
from redis_model_store.utils import (
    PIPELINE_BATCH_SIZE,
    current_timestamp,
    new_model_version,
    setup_logger,
)

logger = setup_logger(__name__)


class ModelStoreError(Exception):
    """Raised when model store operations fail.

    This is the base exception for all model store operations including:
    - Model saving/loading
    - Version management
    - Store initialization
    """

    pass


class ModelVersion(BaseModel):
    """
    Metadata for a specific model version.

    Contains all metadata associated with a stored model version including
    its name, version identifier, description, creation time, and storage
    locations.
    """

    name: str = Field(..., description="Unique identifier for the model")
    description: str = Field(
        default="", description="Optional description of this model version"
    )
    version: str = Field(
        default_factory=new_model_version,
        description="Version identifier (e.g. semantic version or UUID)",
    )
    created_at: float = Field(
        default_factory=current_timestamp,
        description="Unix timestamp when this version was created",
    )
    shard_keys: List[str] = Field(
        default=[], description="Redis keys containing the serialized model data"
    )

    @classmethod
    def from_dict(cls, result: dict) -> "ModelVersion":
        """Create a ModelVersion instance from a query result dict.

        Args:
            result: Dictionary containing version metadata fields

        Returns:
            New ModelVersion instance

        Raises:
            ModelStoreError: If required fields are missing or malformed
        """
        try:
            if "$.shard_keys" in result:
                result["shard_keys"] = result.pop("$.shard_keys")
            if isinstance(result["shard_keys"], str):
                result["shard_keys"] = json.loads(result["shard_keys"])
            return cls(
                name=result["name"],
                description=result["description"],
                version=result["version"],
                created_at=result["created_at"],
                shard_keys=result["shard_keys"],
            )
        except (KeyError, json.JSONDecodeError) as e:
            raise ModelStoreError(f"Invalid model version data: {str(e)}") from e


class ModelStore:
    """
    High-level interface for storing and retrieving ML models with versioning.

    The ModelStore provides a simple API for saving and loading models while handling:
    - Automatic model versioning
    - Metadata tracking (creation time, descriptions)
    - Efficient storage via sharding
    - Version querying and retrieval

    Models are stored in Redis using a combination of:
    - JSON storage for version metadata
    - Bytes storage for model shards
    - Search indices for efficient querying

    All operations are atomic and handle cleanup on failure.

    Example:
    >>> store = ModelStore(redis_client)
    >>> version = store.save_model(
    ...     model,
    ...     name="bert-qa",
    ...     description="BERT model fine-tuned for QA"
    ... )
    >>> model = store.load_model("bert-qa")  # loads latest version
    >>> model = store.load_model("bert-qa", version=version)  # loads specific version
    """

    def __init__(self, redis_client: Redis, shard_size: int = 1024 * 1024):
        """
        Initialize the model store.

        Args:
            redis_client: Initialized Redis client instance
            shard_size: Maximum size in bytes for each model shard (default: 1MB)

        Raises:
            TypeError: If redis_client is not a valid Redis instance
            ModelStoreError: If store initialization fails
        """
        if not isinstance(redis_client, Redis):
            raise TypeError("Must provide a valid Redis client instance")

        try:
            self.redis_client = redis_client
            self.shard_manager = ModelShardManager(shard_size=shard_size)

            # Initialize model version index
            self._query_return_fields = [
                "name",
                "description",
                "version",
                "created_at",
                "$.shard_keys",
            ]
            self.store_idx = SearchIndex.from_dict(
                {
                    "index": {
                        "name": "model_store",
                        "prefix": "model_version",
                        "storage_type": "json",
                        "key_separator": ":",
                    },
                    "fields": [
                        {"name": "name", "type": "tag"},
                        {"name": "description", "type": "text"},
                        {"name": "version", "type": "tag"},
                        {"name": "created_at", "type": "numeric"},
                    ],
                }
            )
            self.store_idx.set_client(redis_client)
            self.store_idx.create(overwrite=False, drop=False)

        except Exception as e:
            raise ModelStoreError(f"Failed to initialize model store: {str(e)}") from e

    @staticmethod
    def _version_key(name: str, version: str) -> str:
        """Generate Redis key for a model version."""
        return f"model_version:{name}:{version}"

    def _store_shards(self, model: Any, name: str, version: str) -> List[str]:
        """Store model shards in Redis."""
        start_time = current_timestamp()
        logger.info("Starting model serialization and storage")
        shard_keys: List[str] = []

        try:
            with self.redis_client.pipeline(transaction=False) as pipe:
                # Generate shards and pipeline into Redis
                for i, shard in enumerate(self.shard_manager.to_shards(model)):
                    shard_key = self.shard_manager.shard_key(name, version, i)
                    shard_keys.append(shard_key)
                    pipe.set(shard_key, shard)
                    # Flush the pipeline batch
                    if i % PIPELINE_BATCH_SIZE == 0:
                        pipe.execute()
                pipe.execute()

            duration = current_timestamp() - start_time
            logger.info(f"Stored model in {len(shard_keys)} shards ({duration:.4f}s)")
            return shard_keys

        except Exception as e:
            # Clean up any stored shards
            if shard_keys:
                self.redis_client.delete(*shard_keys)
            raise ModelStoreError(f"Failed to store model shards: {str(e)}") from e

    def _load_shards(self, shard_keys: List[str]) -> Any:
        """Load and reconstruct model from shards."""
        start_time = current_timestamp()
        logger.info("Starting model reconstruction from shards")

        try:
            shards: List[bytes] = []
            # Pipeline load shards from Redis
            with self.redis_client.pipeline(transaction=False) as pipe:
                for i, shard_key in enumerate(shard_keys):
                    pipe.get(shard_key)
                    # Flush pipeline batch
                    if i % PIPELINE_BATCH_SIZE == 0:
                        shards.extend(pipe.execute())
                shards.extend(pipe.execute())
            # Deserialize model from shards
            model = self.shard_manager.from_shards(shards)
            duration = current_timestamp() - start_time
            logger.info(f"Loaded model from {len(shard_keys)} shards ({duration:.4f}s)")
            return model

        except Exception as e:
            raise ModelStoreError(f"Failed to load model shards: {str(e)}") from e

    def save_model(self, model: Any, name: str, **kwargs) -> str:
        """
        Store a model with versioning and metadata.

        Handles serialization, sharding, and version tracking for the model.
        If the operation fails, any partially stored data is cleaned up.

        Args:
            model: The model object to store
            name: Unique identifier for the model
            **kwargs: Additional metadata fields including:
                - version: Specific version identifier (optional)
                - description: Human readable description (optional)

        Returns:
            str: The version identifier for the stored model

        Raises:
            ModelStoreError: If the model cannot be saved or version exists
        """
        total_start = current_timestamp()
        model_version: Optional[ModelVersion] = None
        logger.info(f"Saving '{name}' model")

        try:
            # Create version record
            model_version = ModelVersion(name=name, **kwargs)
            model_version_key = self._version_key(name, model_version.version)

            if self.redis_client.exists(model_version_key):
                raise ModelStoreError(
                    f"Version exists: Model {name} version {model_version.version}"
                )

            # Store model chunks and update version
            shard_keys = self._store_shards(model, name, model_version.version)
            model_version.shard_keys = shard_keys
            self.store_idx.load(
                data=[model_version.model_dump()], keys=[model_version_key]
            )

            total_duration = current_timestamp() - total_start
            logger.info(f"Save operation completed in {total_duration:.4f}s")
            return model_version.version

        except Exception as e:
            # Clean up version record if it was created
            if model_version:
                self._delete_version(name, model_version.version)
            if not isinstance(e, ModelStoreError):
                raise ModelStoreError(f"Failed to save model: {str(e)}") from e
            raise

    def load_model(self, name: str, version: Optional[str] = None) -> Any:
        """
        Load a model by name and optional version.

        Args:
            name: Unique identifier for the model
            version: Specific version to load. If None, loads the latest version

        Returns:
            The reconstructed model object

        Raises:
            ModelStoreError: If the model/version is not found or cannot be loaded
        """
        total_start = current_timestamp()
        logger.info(f"Loading '{name}' model")

        try:
            # Get model version metadata
            if version:
                model_version = self.get_version(name, version)
            else:
                model_version = self.get_latest_version(name)

            # Load model data
            model = self._load_shards(model_version.shard_keys)
            total_duration = current_timestamp() - total_start
            logger.info(f"Load operation completed in {total_duration:.4f}s")
            return model

        except ModelStoreError:
            raise
        except Exception as e:
            raise ModelStoreError(f"Failed to load model: {str(e)}") from e

    def get_version(self, name: str, version: str) -> ModelVersion:
        """
        Get metadata for a specific model version.

        Args:
            name: Model identifier
            version: Version identifier

        Returns:
            ModelVersion containing the version metadata

        Raises:
            ModelStoreError: If the version is not found
        """
        model_version_key = self._version_key(name, version)
        model_version_dict = self.redis_client.json().get(model_version_key)
        if not model_version_dict:
            raise ModelStoreError(f"Version not found: Model {name} version {version}")
        return ModelVersion.from_dict(model_version_dict)

    def get_latest_version(self, name: str) -> ModelVersion:
        """
        Get the most recent version of a model.

        Args:
            name: Model identifier

        Returns:
            ModelVersion for the most recently created version

        Raises:
            ModelStoreError: If no versions exist for the model
        """
        query = FilterQuery(
            filter_expression=Tag("name") == name,
            num_results=1,
            return_fields=self._query_return_fields,
        ).sort_by("created_at", asc=False)

        results = self.store_idx.query(query)
        if not results:
            raise ModelStoreError(f"No versions found for model: {name}")

        return ModelVersion.from_dict(results[0])

    def get_all_versions(self, name: str) -> List[ModelVersion]:
        """
        Get all versions of a model sorted by creation time.

        Args:
            name: Model identifier

        Returns:
            List of ModelVersion objects, sorted newest to oldest

        Raises:
            ModelStoreError: If no versions exist or metadata is invalid
        """
        query = FilterQuery(
            filter_expression=Tag("name") == name,
            return_fields=self._query_return_fields,
        ).sort_by("created_at", asc=False)

        versions: List[ModelVersion] = []
        for results in self.store_idx.paginate(query, page_size=50):
            if results:
                versions.extend(ModelVersion.from_dict(result) for result in results)

        if not versions:
            raise ModelStoreError(f"No versions found for model: {name}")

        return versions

    def _delete_version(self, name: str, version: str) -> int:
        """Delete a model version and its shards."""
        try:
            model_version = self.get_version(name, version)
        except ModelStoreError:
            # Ignore case where model version doesn't exist
            pass

        keys = model_version.shard_keys + [self._version_key(name, version)]
        self.store_idx.drop_keys(keys)
        return len(keys)

    def delete_version(self, name: str, version: str) -> int:
        """
        Delete a specific model version.

        Removes both the version metadata and all associated model shards.

        Args:
            name: Model identifier
            version: Version identifier

        Returns:
            Number of Redis keys deleted

        """
        return self._delete_version(name, version)

    def clear(self, name: Optional[str] = None) -> int:
        """
        Clear model versions from the store.

        Args:
            name: If provided, only clear versions of this model.
                If None, clear all models and versions.

        Returns:
            Number of Redis keys deleted

        Raises:
            ModelStoreError: If clearing operation fails
        """
        try:
            # Get versions to delete based on name param
            if name:
                versions = self.get_all_versions(name)
            else:
                # Get all versions across all models
                models = self.list_models()
                versions = []
                for model_name in models:
                    versions.extend(self.get_all_versions(model_name))

            # Delete each version and count total keys removed
            total_deleted = 0
            for version in versions:
                total_deleted += self._delete_version(version.name, version.version)
            return total_deleted

        except ModelStoreError:
            raise
        except Exception as e:
            raise ModelStoreError(f"Failed to clear store: {str(e)}") from e

    def list_models(self) -> List[str]:
        """
        Get a list of all model names in the store.

        Returns:
            List of unique model names, sorted alphabetically

        Raises:
            ModelStoreError: If query fails
        """
        try:
            query = FilterQuery(
                return_fields=["name"],
            ).sort_by("name", asc=True)

            # Use set to get unique model names
            model_names: Set[str] = set()
            for results in self.store_idx.paginate(query, page_size=50):
                if results:
                    model_names.update(result["name"] for result in results)

            return sorted(model_names)

        except Exception as e:
            raise ModelStoreError(f"Failed to list models: {str(e)}") from e
