"""Tests for the GET /health endpoint."""

import pytest
from fastapi import status
from fastapi.testclient import TestClient

from app.core.config import Settings, get_settings


def test_health_check_returns_200_and_valid_payload(client: TestClient) -> None:
    """Verify that GET /health returns HTTP 200 and expected schema values."""
    response = client.get("/health")

    assert response.status_code == status.HTTP_200_OK
    data = response.json()
    assert data["status"] == "ok"
    assert data["app"] == "EvidenceDesk"
    assert data["environment"] == "development"
    assert data["version"] == "0.1.0"


def test_health_check_respects_custom_settings(client: TestClient, app) -> None:
    """Verify that health check values reflect injected settings."""
    custom_settings = Settings(
        APP_NAME="CustomEvidenceDesk",
        APP_ENV="staging",
        VERSION="1.2.3",
    )
    app.dependency_overrides[get_settings] = lambda: custom_settings

    response = client.get("/health")

    assert response.status_code == status.HTTP_200_OK
    data = response.json()
    assert data["status"] == "ok"
    assert data["app"] == "CustomEvidenceDesk"
    assert data["environment"] == "staging"
    assert data["version"] == "1.2.3"


def test_health_check_content_type(client: TestClient) -> None:
    """Verify that GET /health response has application/json content-type."""
    response = client.get("/health")
    assert response.status_code == status.HTTP_200_OK
    assert "application/json" in response.headers.get("content-type", "")


def test_health_check_post_not_allowed(client: TestClient) -> None:
    """Verify that POST /health returns 405 Method Not Allowed."""
    response = client.post("/health")
    assert response.status_code == status.HTTP_405_METHOD_NOT_ALLOWED


def test_default_settings_instantiation() -> None:
    """Verify that Settings instantiates with expected defaults without environment variables."""
    settings = Settings()
    assert settings.APP_NAME == "EvidenceDesk"
    assert settings.APP_ENV == "development"
    assert settings.API_PORT == 8000
    assert settings.VERSION == "0.1.0"

