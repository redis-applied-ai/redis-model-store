# 🧠 Redis Model Store

[![License: MIT](https://img.shields.io/badge/License-MIT-yellow.svg)](https://opensource.org/licenses/MIT)
![Language](https://img.shields.io/github/languages/top/redis-applied-ai/redis-model-store)
[![Code style: black](https://img.shields.io/badge/code%20style-black-000000.svg)](https://github.com/psf/black)
![GitHub last commit](https://img.shields.io/github/last-commit/redis-applied-ai/redis-model-store)
[![pypi](https://badge.fury.io/py/redisvl.svg)](https://pypi.org/project/redis-model-store/)

Store, version, and manage your ML models in Redis with ease. `redis-model-store` provides a simple yet powerful interface for handling machine learning model artifacts in Redis.

## ✨ Features

- **🔄 Automatic Versioning**: Track and manage multiple versions of your models
- **📦 Smart Storage**: Large models are automatically sharded for optimal storage
- **🔌 Pluggable Serialization**: Works with any Python object (NumPy, PyTorch, TensorFlow, etc.)
- **🏃‍♂️ High Performance**: Efficient storage and retrieval using Redis pipelining
- **🛡️ Safe Operations**: Atomic operations with automatic cleanup on failures

## 🚀 Quick Start

### Installation

```bash
# Using pip
pip install redis-model-store

# Or using poetry
poetry add redis-model-store
```

### Basic Usage

Here's a simple example using scikit-learn:

```python
from redis import Redis
from redis_model_store import ModelStore
from sklearn.ensemble import RandomForestClassifier

# Connect to Redis and initialize store
redis = Redis(host="localhost", port=6379)
store = ModelStore(redis)

# Train your model
model = RandomForestClassifier()
model.fit(X_train, y_train)

# Save model with version tracking
version = store.save_model(
    model,
    name="my-classifier",
    description="Random forest trained on dataset v1"
)

# List available models
models = store.list_models()
print(f"Available models: {models}")

# Load latest version
model = store.load_model("my-classifier")

# Load specific version
model = store.load_model("my-classifier", version=version)

# View all versions
versions = store.get_all_versions("my-classifier")
for v in versions:
    print(f"Version: {v.version}, Created: {v.created_at}")
```

## 🛠️ Contributing

We welcome contributions! Here's how to get started:

### Development Setup

1. Clone the repository:
```bash
git clone https://github.com/redis-applied-ai/redis-model-store.git
cd redis-model-store
```

2. Install poetry if you haven't:
```bash
curl -sSL https://install.python-poetry.org | python3 -
```

3. Install dependencies:
```bash
poetry install --all-extras
```

### Linting and Tests

```bash
poetry run format
poetry run check-mypy
poetry run test
poetry run test-verbose
```

### Making Changes

1. Create a new branch:
```bash
git checkout -b feat/your-feature-name
```

2. Make your changes and ensure:
   - All tests pass (covering new functionality)
   - Code is formatted 
   - Type hints are valid
   - Examples/docs added as notebooks to the `docs/` directory.

3. Push changes and open a PR


## 📚 Documentation

For more usage examples check out tbhis [Example Notebook](docs/redis_model_store.ipynb).
