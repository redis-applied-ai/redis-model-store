"""
model_registry.py

Implements a ModelRegistry class that tracks model versioning and metadata
using RedisVL. Also includes a Pydantic model for model version metadata.
"""

import json
from typing import List, Optional

from pydantic import BaseModel, Field
from redis import Redis
from redisvl.index import SearchIndex
from redisvl.query import FilterQuery
from redisvl.query.filter import FilterExpression, Tag

from model_store.utils import current_timestamp, new_model_version


class ModelVersion(BaseModel):
    """
    Metadata for a specific model version stored in Redis.
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


class ModelRegistry:
    """
    Manages versioning and metadata for machine learning models stored in Redis.

    Uses RedisVL for efficient querying and indexing of model metadata. Each model
    version is stored as a JSON document with searchable fields for name, version,
    description and creation time.

    The registry maintains an index of model versions and their metadata, allowing
    for efficient querying and retrieval of model versions by name, version, or
    creation time.
    """

    def __init__(self, redis_client: Redis) -> None:
        """
        Initialize the model registry.

        Args:
            redis_client (Redis): Initialized Redis client for
               storage and querying.
        """
        self.registry_idx = SearchIndex.from_dict(
            {
                "index": {
                    "name": "model_registry",
                    "prefix": "model",
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
        self.registry_idx.set_client(redis_client)
        # Create index only if it doesn't exist
        self.registry_idx.create(overwrite=False, drop=False)

    @staticmethod
    def model_version_key(name: str, version: str) -> str:
        """
        Generate Redis key for storing a specific model version.

        Args:
            name (str): Model name
            version (str): Model version identifier

        Returns:
            str: Formatted Redis key in the format "model:{name}:{version}"
        """
        return f"model:{name}:{version}"

    def add_version(self, name: str, **kwargs) -> ModelVersion:
        """
        Add a new model version to the registry.

        Args:
            name (str): Model name.
            **kwargs: Additional model metadata fields (version,
                description, shard_keys).

        Returns:
            ModelVersion: Model version metadata object.

        Raises:
            ValidationError: If required fields are missing or invalid.
        """
        model_version = ModelVersion(name=name, **kwargs)
        key = self.model_version_key(model_version.name, model_version.version)
        if not self.registry_idx.client.exists(key):
            self.registry_idx.load(data=[model_version.model_dump()], keys=[key])
            return model_version
        raise ValueError(f"Model {name} version {model_version.version} already exists")

    def set_model_shards(self, name: str, version: str, shard_keys: List[str]):
        key = self.model_version_key(name, version)
        if self.registry_idx.client.exists(key):
            self.registry_idx.client.json().set(key, "$.shard_keys", shard_keys)
        else:
            raise ValueError(
                f"Failed to set shard keys. Model {name} and version {version} does not exist"
            )

    def _query_version(
        self, model_filter: FilterExpression, latest: bool = False
    ) -> ModelVersion:
        """
        Query model versions using a filter expression.

        Args:
            model_filter (FilterExpression): RediSearch filter expression to match models.
            latest (bool): If True, sort by creation time to get most recent version.
                Defaults to False.

        Returns:
            ModelVersion: Matching ModelVersion instance.

        Raises:
            ValueError: If no matching model version is found.
        """
        query = FilterQuery(
            filter_expression=model_filter,
            num_results=1,
            return_fields=[
                "name",
                "description",
                "version",
                "created_at",
                "$.shard_keys",
            ],
        )

        if latest:
            query.sort_by("created_at", asc=False)

        results = self.registry_idx.query(query)

        if not results:
            raise ValueError(f"No model version found matching filter: {model_filter}")

        return ModelVersion(
            name=results[0]["name"],
            description=results[0]["description"],
            version=results[0]["version"],
            created_at=results[0]["created_at"],
            shard_keys=json.loads(results[0]["$.shard_keys"]),
        )

    def get_version(self, name: str, version: str) -> ModelVersion:
        """
        Retrieve metadata for a specific model version.

        Args:
            name (str): Model identifier.
            version (str): Version identifier.

        Returns:
            ModelVersion: ModelVersion instance containing metadata.

        Raises:
            ValueError: If the requested model version is not found
        """
        model_filter = (Tag("name") == name) & (Tag("version") == version)
        return self._query_version(model_filter)

    def get_latest_version(self, name: str) -> ModelVersion:
        """
        Get the most recently created version of a model.

        Args:
            name (str): Model identifier.

        Returns:
            ModelVersion: ModelVersion instance for the latest version.

        Raises:
            ValueError: If no versions exist for the model.
        """
        model_filter = Tag("name") == name
        return self._query_version(model_filter, latest=True)

    def get_all_versions(self, name: str) -> List[ModelVersion]:
        """
        Get all versions of a model sorted by creation time (newest first).

        Args:
            name (str): Model identifier.

        Returns:
            List[ModelVersion]: List of ModelVersion instances for
                all versions of the model, sorted by creation time with
                newest first.

        Raises:
            ValueError: If no versions exist for the model.
        """
        query = FilterQuery(
            filter_expression=Tag("name") == name,
            return_fields=[
                "name",
                "description",
                "version",
                "created_at",
                "$.shard_keys",
            ],
        )
        query.sort_by("created_at", asc=False)

        model_versions: List[ModelVersion] = []

        for results in self.registry_idx.paginate(query, page_size=50):
            if results:
                model_versions.extend(
                    [
                        ModelVersion(
                            name=result["name"],
                            description=result["description"],
                            version=result["version"],
                            created_at=result["created_at"],
                            shard_keys=json.loads(result["$.shard_keys"]),
                        )
                        for result in results
                    ]
                )

        if not model_versions:
            raise ValueError(f"No versions found for model: {name}")

        return model_versions

    def _delete_version_and_shards(self, model_version: ModelVersion) -> int:
        """
        Helper method to delete a model version and its associated shards.

        Args:
            model_version (ModelVersion): The model version to delete.

        Returns:
            int: Number of keys deleted.
        """
        version_key = self.model_version_key(model_version.name, model_version.version)
        keys_to_delete = model_version.shard_keys + [version_key]
        self.registry_idx.drop_keys(keys_to_delete)
        return len(keys_to_delete)

    def delete_version(self, name: str, version: str) -> int:
        """
        Delete a specific model version from the registry.

        Args:
            name (str): Model identifier.
            version (str): Version identifier to delete.

        Returns:
            int: Number of keys deleted.

        Raises:
            ValueError: If the specified version does not exist.
        """
        # This will raise ValueError if version doesn't exist
        model_version = self.get_version(name, version)
        return self._delete_version_and_shards(model_version)

    def clear(self, name: Optional[str] = None) -> int:
        """
        Clear model versions from the registry.

        Args:
            name (Optional[str]): If provided, only clear versions of this model.
                If None, clear all model versions.

        Returns:
            int: Number of keys deleted.
        """
        if name:
            model_versions = self.get_all_versions(name)
            deleted_count = sum(
                self._delete_version_and_shards(version) for version in model_versions
            )
            return deleted_count
        else:
            return self.registry_idx.clear()
