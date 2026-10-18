"""Pytest fixtures for API testing."""

from collections.abc import Generator
from pathlib import Path
import sys

import pytest
from fastapi import FastAPI
from fastapi.testclient import TestClient

# Ensure apps/api directory is on sys.path for direct imports
API_ROOT = Path(__file__).resolve().parent.parent
if str(API_ROOT) not in sys.path:
    sys.path.insert(0, str(API_ROOT))

from app.core.config import Settings, get_settings
from app.main import create_app
from app.storage import reset_storage_state


@pytest.fixture(autouse=True)
def isolate_env() -> Generator[None, None, None]:
    """Ensure tests run in isolation and clear get_settings and storage caches before and after each test."""
    get_settings.cache_clear()
    reset_storage_state()
    yield
    get_settings.cache_clear()
    reset_storage_state()


@pytest.fixture
def app() -> FastAPI:
    """Provide a fresh FastAPI application instance isolated from external .env files."""
    app_instance = create_app()
    clean_settings = Settings(_env_file=None)
    app_instance.dependency_overrides[get_settings] = lambda: clean_settings
    return app_instance


@pytest.fixture
def client(app: FastAPI) -> Generator[TestClient, None, None]:
    """Provide a TestClient instance for issuing test requests."""
    with TestClient(app) as test_client:
        yield test_client

