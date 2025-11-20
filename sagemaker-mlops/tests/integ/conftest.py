"""Shared pytest fixtures for integration tests."""
import pytest
import os


@pytest.fixture(scope="session")
def test_data_dir():
    """Return the path to the test data directory."""
    return os.path.join(os.path.dirname(__file__), "data")


@pytest.fixture(scope="session")
def test_code_dir():
    """Return the path to the test code directory."""
    return os.path.join(os.path.dirname(__file__), "code")
