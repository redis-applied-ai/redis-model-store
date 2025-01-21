# Redis Model Store

[![License: MIT](https://img.shields.io/badge/License-MIT-yellow.svg)](https://opensource.org/licenses/MIT)
![Language](https://img.shields.io/github/languages/top/redis-applied-ai/redis-model-store)
[![Code style: black](https://img.shields.io/badge/code%20style-black-000000.svg)](https://github.com/psf/black)
![GitHub last commit](https://img.shields.io/github/last-commit/redis-applied-ai/redis-model-store)
[![pypi](https://badge.fury.io/py/redisvl.svg)](https://pypi.org/project/redis-model-store/)

`redis-model-store` is a simple Python library designed to handle versioning and serialization of AI/ML models into Redis. It provides a streamlined way to manage your machine learning models in Redis.

## Features

- **Pluggable Serialization**: Serialize/deserialize any Python object (Numpy arrays, Scikit-Learn, PyTorch, TensorFlow models, etc.).
- **Sharding for Large Models**: Splits large serialized payloads into manageable chunks to optimize Redis storage.
- **Version Management**: Automatically manages model versions in Redis, allowing you to store and retrieve specific versions.


## Installation
```bash
pip install redis-model-store
```

## Usage

See the fully detailed [example notebook](docs/redis_model_store.ipynb) for more assistance in getting started.

### Init the ModelStore
```python
from model_store import ModelStore
from redis import Redis

# Initialize the Redis client
redis_client = Redis.from_url("redis://localhost:6379")

# Initialize the ModelStore with (optional) shard size
model_store = ModelStore(redis_client, shard_size=1012 * 100)
```

### Store a model 
You can store any serializable Python object.
```python
from sklearn.ensemble import RandomForestClassifier
from sklearn.datasets import load_iris
from sklearn.model_selection import train_test_split

# Load sample data
iris = load_iris()
X_train, X_test, y_train, y_test = train_test_split(
    iris.data, iris.target, test_size=0.2, random_state=42
)

# Train a simple RandomForest model
model = RandomForestClassifier()
model.fit(X_train, y_train)

# Save the model to Redis
version = model_store.save_model(model, name="random_forest", description="Random forest classifier model")
```

### Load models
```python
# Grab the latest model
model = model_store.load_model(name="random_forest")

# Grab a specific model version
model = model_store.load_model(name="random_forest", version=version)
```

## 