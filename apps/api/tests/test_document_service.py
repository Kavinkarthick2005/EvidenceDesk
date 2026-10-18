"""Tests for EvidenceDesk DocumentService coordination, validation, and rollback."""

import hashlib
from pathlib import Path
import unittest.mock as mock
import uuid

import pytest
import sqlalchemy as sa
from sqlalchemy.exc import OperationalError, SQLAlchemyError
from sqlalchemy.orm import Session

from app.core.config import Settings
from app.db.base import Base
from app.models.document import Document, DocumentStatus
from app.services.document import DocumentService
from app.services.exceptions import (
    DocumentNotFoundError,
    DocumentPersistenceError,
    DocumentSizeLimitError,
    DocumentStorageError,
    DocumentValidationError,
    EmptyDocumentError,
    UnsupportedDocumentTypeError,
)
from app.storage.exceptions import StorageIOError
from app.storage.local import LocalStorageService


@pytest.fixture
def sqlite_engine() -> sa.Engine:
    """Provide an in-memory SQLite engine with the Document table schema created."""
    engine = sa.create_engine("sqlite:///:memory:")
    Base.metadata.create_all(engine)
    return engine


@pytest.fixture
def db_session(sqlite_engine: sa.Engine) -> Session:
    """Provide a scoped database session for service testing."""
    with Session(sqlite_engine) as session:
        yield session


@pytest.fixture
def storage_service(tmp_path: Path) -> LocalStorageService:
    """Provide an isolated local storage service inside tmp_path."""
    return LocalStorageService(root_dir=tmp_path / "service_storage", max_size_mb=2)


@pytest.fixture
def document_service(db_session: Session, storage_service: LocalStorageService) -> DocumentService:
    """Provide a DocumentService wired with test database session and temporary storage."""
    settings = Settings(MAX_UPLOAD_SIZE_MB=2, _env_file=None)
    return DocumentService(db=db_session, storage=storage_service, settings=settings)


def test_create_and_read_document_success(document_service: DocumentService) -> None:
    """Verify document creation, hash calculation, byte length, and content retrieval."""
    payload = b"%PDF-1.4\nTest PDF content with header and structure\n%%EOF"
    doc = document_service.create_document(
        content=payload,
        original_filename="research_report.pdf",
        content_type="application/pdf",
    )

    assert isinstance(doc.id, uuid.UUID)
    assert doc.original_filename == "research_report.pdf"
    assert doc.content_type == "application/pdf"
    assert doc.size_bytes == len(payload)
    assert doc.sha256 == hashlib.sha256(payload).hexdigest()
    assert doc.status == DocumentStatus.UPLOADED
    assert doc.storage_key.endswith(".pdf")

    # Read back through service
    retrieved_bytes = document_service.read_document_content(doc.id)
    assert retrieved_bytes == payload

    # Read record through get_document
    retrieved_doc = document_service.get_document(doc.id)
    assert retrieved_doc.id == doc.id
    assert retrieved_doc.sha256 == doc.sha256


def test_delete_document_success(
    document_service: DocumentService,
    storage_service: LocalStorageService,
) -> None:
    """Verify document deletion removes the database record and unlinks storage file."""
    payload = b"# Markdown Notes\nDocument to be deleted."
    doc = document_service.create_document(
        content=payload,
        original_filename="notes.md",
        content_type="text/markdown",
    )
    storage_key = doc.storage_key
    assert storage_service.exists(storage_key) is True

    document_service.delete_document(doc.id)

    # Database record must no longer exist
    with pytest.raises(DocumentNotFoundError):
        document_service.get_document(doc.id)

    # Storage file must no longer exist
    assert storage_service.exists(storage_key) is False

    # Second delete raises DocumentNotFoundError
    with pytest.raises(DocumentNotFoundError):
        document_service.delete_document(doc.id)


def test_duplicate_sha256_allowed(document_service: DocumentService) -> None:
    """Verify multiple documents with identical content hashes can coexist with distinct keys."""
    identical_payload = b"Exact identical text content in two uploads"
    doc1 = document_service.create_document(
        content=identical_payload,
        original_filename="copy1.txt",
        content_type="text/plain",
    )
    doc2 = document_service.create_document(
        content=identical_payload,
        original_filename="copy2.txt",
        content_type="text/plain",
    )

    assert doc1.id != doc2.id
    assert doc1.storage_key != doc2.storage_key
    assert doc1.sha256 == doc2.sha256
    assert doc1.sha256 == hashlib.sha256(identical_payload).hexdigest()


def test_empty_content_rejected(document_service: DocumentService) -> None:
    """Verify 0-byte uploads are rejected with EmptyDocumentError."""
    with pytest.raises(EmptyDocumentError, match="cannot be empty"):
        document_service.create_document(
            content=b"",
            original_filename="empty.txt",
            content_type="text/plain",
        )


def test_empty_or_whitespace_filename_rejected(document_service: DocumentService) -> None:
    """Verify empty or whitespace-only filenames are rejected."""
    with pytest.raises(EmptyDocumentError, match="cannot be empty"):
        document_service.create_document(
            content=b"some content",
            original_filename="",
            content_type="text/plain",
        )

    with pytest.raises(EmptyDocumentError, match="cannot be empty"):
        document_service.create_document(
            content=b"some content",
            original_filename="   \t",
            content_type="text/plain",
        )


def test_null_byte_in_filename_rejected(document_service: DocumentService) -> None:
    """Verify filenames with null bytes are rejected as malformed metadata."""
    with pytest.raises(DocumentValidationError, match="null characters"):
        document_service.create_document(
            content=b"some content",
            original_filename="bad\0file.pdf",
            content_type="application/pdf",
        )


def test_oversized_content_rejected(
    db_session: Session,
    storage_service: LocalStorageService,
) -> None:
    """Verify content exceeding MAX_UPLOAD_SIZE_MB is rejected before or during storage."""
    settings = Settings(MAX_UPLOAD_SIZE_MB=1, _env_file=None)
    doc_service = DocumentService(db=db_session, storage=storage_service, settings=settings)

    oversized = b"A" * (1024 * 1024 + 10)  # 1MB + 10 bytes
    with pytest.raises(DocumentSizeLimitError, match="exceeds configured maximum"):
        doc_service.create_document(
            content=oversized,
            original_filename="large.txt",
            content_type="text/plain",
        )


def test_unsupported_formats_rejected(document_service: DocumentService) -> None:
    """Verify non-PDF, non-Markdown, and non-plain-text formats are rejected."""
    with pytest.raises(UnsupportedDocumentTypeError, match="Unsupported format"):
        document_service.create_document(
            content=b"#!/bin/bash\necho hi",
            original_filename="script.sh",
            content_type="application/x-sh",
        )

    with pytest.raises(UnsupportedDocumentTypeError, match="Unsupported format"):
        document_service.create_document(
            content=b"\x89PNG\r\n\x1a\n",
            original_filename="graphic.png",
            content_type="image/png",
        )


def test_database_failure_rolls_back_and_cleans_up_storage(
    document_service: DocumentService,
    storage_service: LocalStorageService,
    db_session: Session,
    monkeypatch: pytest.MonkeyPatch,
) -> None:
    """Verify that a database flush failure (confirmed non-commit) rolls back and deletes the created storage file."""
    payload = b"%PDF-1.4 Uncommitted Document"

    # Simulate database flush failure (e.g. table lock or constraint failure)
    def failing_flush(*args: object, **kwargs: object) -> None:
        raise OperationalError("INSERT INTO documents", {}, Exception("Database locked"))

    monkeypatch.setattr(db_session, "flush", failing_flush)

    with pytest.raises(DocumentPersistenceError, match="Database persistence failed during flush"):
        document_service.create_document(
            content=payload,
            original_filename="will_fail.pdf",
            content_type="application/pdf",
        )

    # Verify no documents exist in database
    monkeypatch.undo()
    assert db_session.query(Document).count() == 0

    # Verify no files remain in storage directory (safe cleanup occurred)
    assert list(storage_service.storage_root.glob("*.pdf")) == []
    assert list(storage_service.storage_root.glob(".tmp_*")) == []


def test_cleanup_failure_does_not_mask_primary_database_error(
    document_service: DocumentService,
    storage_service: LocalStorageService,
    db_session: Session,
    monkeypatch: pytest.MonkeyPatch,
) -> None:
    """Verify that if storage cleanup fails during DB rollback, the primary DB error is not masked."""
    # Simulate DB commit failure
    primary_db_error = SQLAlchemyError("Database disk full")

    def failing_commit() -> None:
        raise primary_db_error

    monkeypatch.setattr(db_session, "commit", failing_commit)

    # Also simulate storage delete failure during cleanup
    def failing_delete(key: str) -> None:
        raise StorageIOError("Storage volume disconnected")

    monkeypatch.setattr(storage_service, "delete", failing_delete)

    with pytest.raises(DocumentPersistenceError) as exc_info:
        document_service.create_document(
            content=b"Content for dual failure",
            original_filename="dual_fail.txt",
            content_type="text/plain",
        )

    # Primary error must be preserved as __cause__
    assert exc_info.value.__cause__ is primary_db_error
    # The message must reference both the database error and the storage cleanup failure
    err_msg = str(exc_info.value)
    assert "Database persistence failed" in err_msg
    assert "Storage volume disconnected" in err_msg


def test_unrelated_file_never_deleted_on_failure(
    document_service: DocumentService,
    storage_service: LocalStorageService,
    db_session: Session,
    monkeypatch: pytest.MonkeyPatch,
) -> None:
    """Verify that error handling deletes only the file created by the current request, not existing files."""
    # 1. Pre-create an unrelated file directly in storage
    unrelated_payload = b"Unrelated existing document"
    unrelated_key = storage_service.save(unrelated_payload, content_type="text/plain")
    assert storage_service.exists(unrelated_key) is True

    # 2. Trigger DB commit failure during new document upload
    def failing_commit() -> None:
        raise SQLAlchemyError("Connection reset")

    monkeypatch.setattr(db_session, "commit", failing_commit)

    with pytest.raises(DocumentPersistenceError):
        document_service.create_document(
            content=b"New payload",
            original_filename="new.txt",
            content_type="text/plain",
        )

    # 3. Unrelated file must remain untouched
    assert storage_service.exists(unrelated_key) is True
    assert storage_service.read(unrelated_key) == unrelated_payload


def test_missing_storage_file_for_existing_db_record(
    document_service: DocumentService,
    storage_service: LocalStorageService,
) -> None:
    """Verify read_document_content raises DocumentNotFoundError when DB record exists but file is missing."""
    doc = document_service.create_document(
        content=b"Valid content",
        original_filename="valid.txt",
        content_type="text/plain",
    )

    # Manually delete the storage file from disk
    storage_service.delete(doc.storage_key)
    assert storage_service.exists(doc.storage_key) is False

    with pytest.raises(DocumentNotFoundError, match="Underlying storage file"):
        document_service.read_document_content(doc.id)


def test_delete_is_idempotent_if_storage_file_already_missing(
    document_service: DocumentService,
    storage_service: LocalStorageService,
    db_session: Session,
) -> None:
    """Verify deleting a document succeeds cleanly even if its underlying storage file is already gone."""
    doc = document_service.create_document(
        content=b"Content to be deleted",
        original_filename="doc.txt",
        content_type="text/plain",
    )
    doc_id = doc.id
    storage_key = doc.storage_key

    # Manually delete underlying storage file beforehand
    storage_service.delete(storage_key)

    # delete_document must remove DB record without failing on missing file
    document_service.delete_document(doc_id)

    assert db_session.get(Document, doc_id) is None


def test_get_document_invalid_uuid_raises_not_found(document_service: DocumentService) -> None:
    """Verify supplying an invalid or non-existent UUID raises DocumentNotFoundError."""
    with pytest.raises(DocumentNotFoundError, match="Invalid document UUID"):
        document_service.get_document("not-a-valid-uuid")  # type: ignore[arg-type]

    with pytest.raises(DocumentNotFoundError, match="not found"):
        document_service.get_document(uuid.uuid4())


def test_filename_exceeding_255_chars_rejected(
    document_service: DocumentService,
    storage_service: LocalStorageService,
) -> None:
    """Verify that filenames exceeding 255 characters are rejected without touching storage."""
    long_filename = ("a" * 252) + ".txt"  # 256 characters total
    assert len(long_filename) == 256

    with pytest.raises(DocumentValidationError, match="exceeds maximum length of 255 characters"):
        document_service.create_document(
            content=b"Short content",
            original_filename=long_filename,
            content_type="text/plain",
        )

    # Ensure no file was written to storage
    assert list(storage_service.storage_root.glob("*.txt")) == []


def test_content_type_exceeding_100_chars_rejected(
    document_service: DocumentService,
    storage_service: LocalStorageService,
) -> None:
    """Verify that content-type strings exceeding 100 characters are rejected."""
    long_content_type = "text/plain;" + ("x" * 90)  # > 100 characters
    assert len(long_content_type) > 100

    with pytest.raises(DocumentValidationError, match="exceeds maximum length of 100 characters"):
        document_service.create_document(
            content=b"Short content",
            original_filename="test.txt",
            content_type=long_content_type,
        )

    # Ensure no file was written to storage
    assert list(storage_service.storage_root.glob("*.txt")) == []


def test_exact_size_limit_boundary_behavior(
    db_session: Session,
    storage_service: LocalStorageService,
) -> None:
    """Verify exact size limit boundary: exactly max_size_bytes is accepted, +1 byte is rejected."""
    settings = Settings(MAX_UPLOAD_SIZE_MB=1, _env_file=None)
    doc_service = DocumentService(db=db_session, storage=storage_service, settings=settings)
    exact_limit = 1024 * 1024  # 1 MB exactly

    # 1. Exactly at limit: must succeed
    exact_content = b"x" * exact_limit
    doc = doc_service.create_document(
        content=exact_content,
        original_filename="exact_limit.txt",
        content_type="text/plain",
    )
    assert doc.size_bytes == exact_limit

    # 2. 1 byte over limit: must fail
    oversized_content = b"x" * (exact_limit + 1)
    with pytest.raises(DocumentSizeLimitError, match="exceeds configured maximum"):
        doc_service.create_document(
            content=oversized_content,
            original_filename="over_limit.txt",
            content_type="text/plain",
        )


def test_delete_document_storage_failure_raises_document_storage_error(
    document_service: DocumentService,
    storage_service: LocalStorageService,
    db_session: Session,
    monkeypatch: pytest.MonkeyPatch,
) -> None:
    """Verify that when storage deletion fails after DB commit, DocumentStorageError is raised and retry fails with 404."""
    doc = document_service.create_document(
        content=b"File to trigger storage error on delete",
        original_filename="delete_fail.txt",
        content_type="text/plain",
    )
    doc_id = doc.id
    storage_key = doc.storage_key
    assert storage_service.exists(storage_key) is True

    # Simulate storage delete failure (e.g. read-only filesystem or lock)
    def failing_storage_delete(key: str) -> None:
        raise StorageIOError("Filesystem volume is read-only")

    monkeypatch.setattr(storage_service, "delete", failing_storage_delete)

    # Deleting document must raise DocumentStorageError
    with pytest.raises(DocumentStorageError, match="failed to delete file from storage"):
        document_service.delete_document(doc_id)

    # Database record was already deleted and committed
    assert db_session.get(Document, doc_id) is None

    # Underlying file still exists on disk (orphan file)
    # Restore unpatched delete method to inspect disk
    monkeypatch.undo()
    assert storage_service.exists(storage_key) is True

    # A subsequent retry by the caller raises DocumentNotFoundError because DB record is gone
    with pytest.raises(DocumentNotFoundError):
        document_service.delete_document(doc_id)


def test_ambiguous_commit_operational_error_preserves_storage_file(
    document_service: DocumentService,
    storage_service: LocalStorageService,
    db_session: Session,
    monkeypatch: pytest.MonkeyPatch,
) -> None:
    """Verify that if commit raises an ambiguous OperationalError, the storage file is preserved to prevent data loss."""
    payload = b"%PDF-1.4 Important Document Content"

    # Simulate connection loss during COMMIT
    conn_error = OperationalError("COMMIT", {}, Exception("Connection lost to server"))

    def failing_commit() -> None:
        raise conn_error

    monkeypatch.setattr(db_session, "commit", failing_commit)

    with pytest.raises(DocumentPersistenceError, match="ambiguous connection error") as exc_info:
        document_service.create_document(
            content=payload,
            original_filename="contract.pdf",
            content_type="application/pdf",
        )

    # Primary connection error must be preserved as cause
    assert exc_info.value.__cause__ is conn_error

    # The storage file MUST be preserved to prevent catastrophic data loss in case the row committed
    saved_files = list(storage_service.storage_root.glob("*.pdf"))
    assert len(saved_files) == 1
    assert saved_files[0].read_bytes() == payload


def test_unsupported_mime_rejected_even_if_filename_has_allowed_extension(
    document_service: DocumentService,
    storage_service: LocalStorageService,
) -> None:
    """Verify that an explicitly unsupported MIME type is rejected even if the filename ends in .pdf."""
    with pytest.raises(UnsupportedDocumentTypeError, match="Unsupported format: content-type 'image/png'"):
        document_service.create_document(
            content=b"\x89PNG\r\n\x1a\nFake PNG with PDF name",
            original_filename="malicious.pdf",
            content_type="image/png",
        )

    # Ensure no file was written to storage
    assert list(storage_service.storage_root.glob("*")) == []


def test_unsupported_extension_rejected_even_if_content_type_allowed(
    document_service: DocumentService,
    storage_service: LocalStorageService,
) -> None:
    """Verify that an unsupported file extension (.exe) is rejected even if content_type is application/pdf."""
    with pytest.raises(UnsupportedDocumentTypeError, match="Unsupported format: file extension '.exe'"):
        document_service.create_document(
            content=b"%PDF-1.4 executable disguise",
            original_filename="trojan.exe",
            content_type="application/pdf",
        )

    # Ensure no file was written to storage
    assert list(storage_service.storage_root.glob("*")) == []


def test_mime_parameters_and_case_normalization(document_service: DocumentService) -> None:
    """Verify that MIME parameters and upper-casing are correctly normalized."""
    payload = b"%PDF-1.4 Normalized PDF"
    doc = document_service.create_document(
        content=payload,
        original_filename="report.pdf",
        content_type="Application/PDF; charset=binary",
    )
    assert doc.content_type == "application/pdf"


def test_empty_mime_infers_from_extension(document_service: DocumentService) -> None:
    """Verify that empty/whitespace MIME type infers correctly from allowed filename extension."""
    doc = document_service.create_document(
        content=b"# Header\nMarkdown content",
        original_filename="notes.md",
        content_type="",
    )
    assert doc.content_type == "text/markdown"


def test_empty_mime_with_unsupported_extension_rejected(
    document_service: DocumentService,
    storage_service: LocalStorageService,
) -> None:
    """Verify that empty MIME with unsupported extension is rejected."""
    with pytest.raises(UnsupportedDocumentTypeError, match="Missing content type and unsupported"):
        document_service.create_document(
            content=b"#!/bin/bash",
            original_filename="script.sh",
            content_type="   ",
        )

    assert list(storage_service.storage_root.glob("*")) == []
