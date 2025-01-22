import os
import pytest

from testcontainers.compose import DockerCompose


@pytest.fixture(scope="session", autouse=True)
def redis_container():
    # Set the default Redis version if not already set
    os.environ.setdefault("REDIS_VERSION", "edge")

    compose = DockerCompose("tests", compose_file_name="docker-compose.yaml", pull=True)
    compose.start()

    redis_host, redis_port = compose.get_service_host_and_port("redis", 6379)
    redis_url = f"redis://{redis_host}:{redis_port}"
    os.environ["REDIS_URL"] = redis_url

    yield compose

    compose.stop()
