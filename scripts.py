import subprocess


def format():
    subprocess.run(["isort", "./redis_model_store", "./tests", "--profile", "black"], check=True)
    subprocess.run(["black", "./redis_model_store", "./tests"], check=True)


def check_format():
    subprocess.run(["black", "--check", "./redis_model_store"], check=True)


def sort_imports():
    subprocess.run(["isort", "./redis_model_store", "./tests/", "--profile", "black"], check=True)


def check_sort_imports():
    subprocess.run(
        ["isort", "./redis_model_store", "--check-only", "--profile", "black"], check=True
    )


def check_lint():
    subprocess.run(["pylint", "--rcfile=.pylintrc", "./redis_model_store"], check=True)


def check_mypy():
    subprocess.run(["python", "-m", "mypy", "./redis_model_store"], check=True)


def test():
    subprocess.run(["python", "-m", "pytest", "--log-level=CRITICAL"], check=True)


def test_verbose():
    subprocess.run(
        ["python", "-m", "pytest", "-vv", "-s", "--log-level=CRITICAL"], check=True
    )


def test_notebooks():
    subprocess.run(["cd", "docs/", "&&", "poetry run treon", "-v"], check=True)
