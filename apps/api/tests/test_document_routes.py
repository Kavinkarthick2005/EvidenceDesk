"""Integration tests for Document API routes."""

from collections.abc import Generator
import hashlib
import io
from pathlib import Path
import unittest.mock as mock
import uuid

import pytest
from fastapi import status
from fastapi.testclient import TestClient
import sqlalchemy as sa
from sqlalchemy.orm import Session

from app.api.deps import get_document_service
from app.core.config import Settings, get_settings
from app.db.base import Base
from app.db.session import get_db
from app.main import create_app
from app.models.document import DocumentStatus
from app.services.document import DocumentService
from app.services.exceptions import (
    DocumentPersistenceError,
    DocumentStorageError,
)
from app.storage.local import LocalStorageService


@pytest.fixture
def sqlite_engine() -> Generator[sa.Engine, None, None]:
    """Provide an in-memory SQLite engine with the Document schema created."""
    engine = sa.create_engine(
        "sqlite:///:memory:",
        connect_args={"check_same_thread": False},
        poolclass=sa.pool.StaticPool,
    )
    Base.metadata.create_all(engine)
    yield engine
    engine.dispose()


@pytest.fixture
def storage_service(tmp_path: Path) -> LocalStorageService:
    """Provide an isolated local storage service inside tmp_path."""
    return LocalStorageService(root_dir=tmp_path / "route_storage", max_size_mb=2)


@pytest.fixture
def test_settings(tmp_path: Path) -> Settings:
    """Provide isolated application settings with 2MB upload limit."""
    return Settings(
        MAX_UPLOAD_SIZE_MB=2,
        DOCUMENT_STORAGE_PATH=str(tmp_path / "route_storage"),
        _env_file=None,
    )


@pytest.fixture
def client(
    sqlite_engine: sa.Engine,
    storage_service: LocalStorageService,
    test_settings: Settings,
) -> Generator[TestClient, None, None]:
    """Provide a TestClient with dependency overrides for database, storage, and settings."""
    app_instance = create_app()

    def override_get_db() -> Generator[Session, None, None]:
        with Session(sqlite_engine) as session:
            yield session

    def override_get_settings() -> Settings:
        return test_settings

    def override_get_document_service() -> DocumentService:
        # Note: creates DocumentService with request-scoped session
        session = Session(sqlite_engine)
        return DocumentService(
            db=session,
            storage=storage_service,
            settings=test_settings,
        )

    app_instance.dependency_overrides[get_db] = override_get_db
    app_instance.dependency_overrides[get_settings] = override_get_settings
    app_instance.dependency_overrides[get_document_service] = override_get_document_service

    with TestClient(app_instance) as test_client:
        yield test_client


def test_upload_document_success(client: TestClient) -> None:
    """Verify successful multipart upload returns 201 with document metadata."""
    payload = b"%PDF-1.4\nSample PDF test content\n%%EOF"
    files = {"file": ("report.pdf", io.BytesIO(payload), "application/pdf")}

    response = client.post("/documents", files=files)

    assert response.status_code == status.HTTP_201_CREATED
    data = response.json()

    assert "id" in data
    assert uuid.UUID(data["id"])
    assert data["original_filename"] == "report.pdf"
    assert data["content_type"] == "application/pdf"
    assert data["size_bytes"] == len(payload)
    assert data["sha256"] == hashlib.sha256(payload).hexdigest()
    assert data["status"] == DocumentStatus.UPLOADED
    assert "created_at" in data
    assert "updated_at" in data
    # Ensure internal storage details are never exposed
    assert "storage_key" not in data


def test_get_document_metadata_success(client: TestClient) -> None:
    """Verify retrieving document metadata by UUID returns 200."""
    payload = b"# Notes\nMarkdown document content"
    files = {"file": ("notes.md", io.BytesIO(payload), "text/markdown")}

    upload_res = client.post("/documents", files=files)
    assert upload_res.status_code == status.HTTP_201_CREATED
    doc_id = upload_res.json()["id"]

    get_res = client.get(f"/documents/{doc_id}")
    assert get_res.status_code == status.HTTP_200_OK
    data = get_res.json()

    assert data["id"] == doc_id
    assert data["original_filename"] == "notes.md"
    assert data["content_type"] == "text/markdown"
    assert data["size_bytes"] == len(payload)
    assert data["sha256"] == hashlib.sha256(payload).hexdigest()
    assert "storage_key" not in data


def test_delete_document_success(client: TestClient) -> None:
    """Verify deleting a document by UUID returns 200 and removes the document."""
    payload = b"Plain text content for deletion"
    files = {"file": ("doc.txt", io.BytesIO(payload), "text/plain")}

    upload_res = client.post("/documents", files=files)
    assert upload_res.status_code == status.HTTP_201_CREATED
    doc_id = upload_res.json()["id"]

    # Delete the document
    del_res = client.delete(f"/documents/{doc_id}")
    assert del_res.status_code == status.HTTP_200_OK
    del_data = del_res.json()
    assert del_data["id"] == doc_id
    assert del_data["status"] == "deleted"

    # Subsequent GET returns 404
    get_res = client.get(f"/documents/{doc_id}")
    assert get_res.status_code == status.HTTP_404_NOT_FOUND

    # Subsequent DELETE returns 404
    del_again_res = client.delete(f"/documents/{doc_id}")
    assert del_again_res.status_code == status.HTTP_404_NOT_FOUND


def test_get_document_invalid_uuid_returns_422(client: TestClient) -> None:
    """Verify malformed UUID path parameter returns 422 Unprocessable Content."""
    response = client.get("/documents/not-a-uuid")
    assert response.status_code == status.HTTP_422_UNPROCESSABLE_CONTENT


def test_delete_document_invalid_uuid_returns_422(client: TestClient) -> None:
    """Verify malformed UUID on DELETE returns 422 Unprocessable Content."""
    response = client.delete("/documents/not-a-uuid")
    assert response.status_code == status.HTTP_422_UNPROCESSABLE_CONTENT


def test_get_document_nonexistent_uuid_returns_404(client: TestClient) -> None:
    """Verify non-existent UUID returns 404 Not Found with safe detail."""
    random_id = uuid.uuid4()
    response = client.get(f"/documents/{random_id}")
    assert response.status_code == status.HTTP_404_NOT_FOUND
    assert f"Document with ID '{random_id}' not found" in response.json()["detail"]


def test_delete_document_nonexistent_uuid_returns_404(client: TestClient) -> None:
    """Verify deleting a non-existent UUID returns 404 Not Found."""
    random_id = uuid.uuid4()
    response = client.delete(f"/documents/{random_id}")
    assert response.status_code == status.HTTP_404_NOT_FOUND
    assert f"Document with ID '{random_id}' not found" in response.json()["detail"]


def test_upload_empty_file_returns_400(client: TestClient) -> None:
    """Verify 0-byte file upload returns 400 Bad Request."""
    files = {"file": ("empty.txt", io.BytesIO(b""), "text/plain")}
    response = client.post("/documents", files=files)
    assert response.status_code == status.HTTP_400_BAD_REQUEST
    assert "cannot be empty" in response.json()["detail"]


def test_upload_empty_filename_returns_400(client: TestClient) -> None:
    """Verify upload with whitespace-only filename returns 400 Bad Request."""
    files = {"file": ("   ", io.BytesIO(b"content"), "text/plain")}
    response = client.post("/documents", files=files)
    assert response.status_code == status.HTTP_400_BAD_REQUEST
    assert "cannot be empty" in response.json()["detail"]


def test_upload_unsupported_mime_type_returns_415(client: TestClient) -> None:
    """Verify upload with unsupported MIME type returns 415 Unsupported Media Type."""
    files = {"file": ("image.png", io.BytesIO(b"\x89PNG\r\n\x1a\n"), "image/png")}
    response = client.post("/documents", files=files)
    assert response.status_code == status.HTTP_415_UNSUPPORTED_MEDIA_TYPE
    assert "Unsupported format" in response.json()["detail"]


def test_upload_unsupported_mime_disguised_with_pdf_extension_returns_415(
    client: TestClient,
) -> None:
    """Verify upload with image/png disguised as .pdf is rejected with 415."""
    files = {"file": ("fake.pdf", io.BytesIO(b"\x89PNG\r\n\x1a\n"), "image/png")}
    response = client.post("/documents", files=files)
    assert response.status_code == status.HTTP_415_UNSUPPORTED_MEDIA_TYPE
    assert "Unsupported format" in response.json()["detail"]


def test_upload_unsupported_extension_returns_415(client: TestClient) -> None:
    """Verify upload with executable extension returns 415 even if MIME is application/pdf."""
    files = {"file": ("trojan.exe", io.BytesIO(b"%PDF-1.4 binary"), "application/pdf")}
    response = client.post("/documents", files=files)
    assert response.status_code == status.HTTP_415_UNSUPPORTED_MEDIA_TYPE
    assert "Unsupported format" in response.json()["detail"]


def test_upload_exact_size_limit_allowed(client: TestClient) -> None:
    """Verify upload at exactly 2MB limit (2 * 1024 * 1024 bytes) succeeds with 201."""
    exact_payload = b"x" * (2 * 1024 * 1024)
    files = {"file": ("exact.txt", io.BytesIO(exact_payload), "text/plain")}

    response = client.post("/documents", files=files)
    assert response.status_code == status.HTTP_201_CREATED
    assert response.json()["size_bytes"] == len(exact_payload)


def test_upload_oversized_file_returns_413(client: TestClient) -> None:
    """Verify upload exceeding 2MB limit by 1 byte returns 413 Payload Too Large."""
    oversized_payload = b"x" * (2 * 1024 * 1024 + 1)
    files = {"file": ("large.txt", io.BytesIO(oversized_payload), "text/plain")}

    response = client.post("/documents", files=files)
    assert response.status_code == status.HTTP_413_CONTENT_TOO_LARGE
    assert "exceeds configured maximum" in response.json()["detail"]


def test_upload_persistence_error_returns_500_safe_message(
    client: TestClient,
    monkeypatch: pytest.MonkeyPatch,
) -> None:
    """Verify database persistence failure returns 500 without leaking SQL details or traces."""
    # Force DocumentService.create_document to raise DocumentPersistenceError
    def failing_create(*args: object, **kwargs: object) -> None:
        raise DocumentPersistenceError(
            "SQLAlchemy connection terminated: SELECT * FROM documents WHERE secret='leak'"
        )

    monkeypatch.setattr(DocumentService, "create_document", failing_create)

    files = {"file": ("doc.txt", io.BytesIO(b"content"), "text/plain")}
    response = client.post("/documents", files=files)

    assert response.status_code == status.HTTP_500_INTERNAL_SERVER_ERROR
    detail = response.json()["detail"]
    assert detail == "Database persistence failure."
    # Ensure sensitive SQL details are NOT in the response
    assert "SELECT" not in detail
    assert "secret" not in detail


def test_upload_storage_error_returns_500_safe_message(
    client: TestClient,
    monkeypatch: pytest.MonkeyPatch,
) -> None:
    """Verify storage failure returns 500 without leaking host filesystem paths."""
    def failing_create(*args: object, **kwargs: object) -> None:
        raise DocumentStorageError("Permission denied: /var/data/storage/root/private/file.txt")

    monkeypatch.setattr(DocumentService, "create_document", failing_create)

    files = {"file": ("doc.txt", io.BytesIO(b"content"), "text/plain")}
    response = client.post("/documents", files=files)

    assert response.status_code == status.HTTP_500_INTERNAL_SERVER_ERROR
    detail = response.json()["detail"]
    assert detail == "Document storage operation failed."
    # Ensure host filesystem path is NOT in the response
    assert "/var/data" not in detail


def test_delete_storage_error_returns_500_safe_message(
    client: TestClient,
    monkeypatch: pytest.MonkeyPatch,
) -> None:
    """Verify storage failure during delete returns 500 without leaking internal details."""
    doc_id = uuid.uuid4()

    def failing_delete(self: object, document_id: uuid.UUID) -> None:
        raise DocumentStorageError("Disk I/O error unlinking /var/data/storage/root/file.txt")

    monkeypatch.setattr(DocumentService, "delete_document", failing_delete)

    response = client.delete(f"/documents/{doc_id}")

    assert response.status_code == status.HTTP_500_INTERNAL_SERVER_ERROR
    detail = response.json()["detail"]
    assert detail == "Document storage operation failed."
    assert "/var/data" not in detail
