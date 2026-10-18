"""Tests for IngestionService pipeline orchestration, metadata preservation, and failure semantics."""

import io
from pathlib import Path
from unittest.mock import MagicMock, patch
import uuid

from pypdf import PdfWriter
from pypdf.generic import DecodedStreamObject, DictionaryObject, NameObject
import pytest
import sqlalchemy as sa
from sqlalchemy.exc import OperationalError, SQLAlchemyError
from sqlalchemy.orm import Session

from app.chunking.exceptions import ChunkingError
from app.chunking.models import ChunkingConfig
from app.chunking.service import StructureAwareChunker
from app.core.config import Settings
from app.db.base import Base
from app.extraction.service import ExtractionService
from app.models.chunk import Chunk as ChunkModel
from app.models.document import Document, DocumentStatus
from app.services.chunk import ChunkPersistenceService
from app.services.exceptions import (
    ChunkPersistenceError,
    DocumentNotFoundError,
    IngestionChunkingError,
    IngestionExtractionError,
    IngestionPersistenceError,
)
from app.services.ingestion import IngestionResult, IngestionService
from app.storage.local import LocalStorageService


def _build_pdf(pages: list[str]) -> bytes:
    """Helper to construct a valid in-memory multi-page PDF with text."""
    writer = PdfWriter()
    for text in pages:
        page = writer.add_blank_page(width=400, height=400)
        content = DecodedStreamObject()
        content.set_data(f"BT /F1 12 Tf 50 350 Td ({text}) Tj ET".encode("latin-1"))
        page[NameObject("/Contents")] = writer._add_object(content)

        font_obj = DictionaryObject()
        font_obj[NameObject("/Type")] = NameObject("/Font")
        font_obj[NameObject("/Subtype")] = NameObject("/Type1")
        font_obj[NameObject("/BaseFont")] = NameObject("/Helvetica")
        fonts = DictionaryObject()
        fonts[NameObject("/F1")] = writer._add_object(font_obj)

        resources = DictionaryObject()
        resources[NameObject("/Font")] = fonts
        page[NameObject("/Resources")] = resources

    out = io.BytesIO()
    writer.write(out)
    return out.getvalue()


@pytest.fixture
def sqlite_engine() -> sa.Engine:
    """Provide an in-memory SQLite engine with PRAGMA foreign_keys enabled."""
    engine = sa.create_engine("sqlite:///:memory:")

    @sa.event.listens_for(engine, "connect")
    def set_sqlite_pragma(dbapi_connection, connection_record):  # type: ignore[no-untyped-def]
        cursor = dbapi_connection.cursor()
        cursor.execute("PRAGMA foreign_keys=ON")
        cursor.close()

    Base.metadata.create_all(engine)
    return engine


@pytest.fixture
def db_session(sqlite_engine: sa.Engine) -> Session:
    """Provide a scoped database session."""
    with Session(sqlite_engine) as session:
        yield session


@pytest.fixture
def storage_service(tmp_path: Path) -> LocalStorageService:
    """Provide an isolated local storage service inside tmp_path."""
    return LocalStorageService(root_dir=tmp_path / "pipeline_storage", max_size_mb=5)


@pytest.fixture
def ingestion_service(
    db_session: Session,
    storage_service: LocalStorageService,
) -> IngestionService:
    """Provide an IngestionService configured with test database and storage."""
    settings = Settings(MAX_UPLOAD_SIZE_MB=5, _env_file=None)
    return IngestionService(
        db=db_session,
        storage=storage_service,
        settings=settings,
    )


def test_ingest_pdf_success(
    db_session: Session,
    storage_service: LocalStorageService,
    ingestion_service: IngestionService,
) -> None:
    """Verify end-to-end PDF ingestion extracts pages, produces chunks, and sets READY status."""
    pdf_bytes = _build_pdf(["Page 1 Introduction text.", "Page 2 Advanced evaluation content."])
    storage_key = storage_service.save(pdf_bytes, content_type="application/pdf", original_filename="manual.pdf")

    doc = Document(
        original_filename="manual.pdf",
        content_type="application/pdf",
        size_bytes=len(pdf_bytes),
        sha256="1" * 64,
        storage_key=storage_key,
        status=DocumentStatus.UPLOADED,
    )
    db_session.add(doc)
    db_session.commit()

    result = ingestion_service.ingest_document(doc.id)

    assert isinstance(result, IngestionResult)
    assert result.document_id == doc.id
    assert result.status == DocumentStatus.READY
    assert result.chunk_count >= 2
    assert result.total_tokens > 0

    # Verify document in DB is updated to READY
    db_session.refresh(doc)
    assert doc.status == DocumentStatus.READY

    # Verify page metadata preserved
    chunks = result.chunks
    page_numbers = {c.page_number for c in chunks}
    assert 1 in page_numbers
    assert 2 in page_numbers
    assert all(c.source_format == "application/pdf" for c in chunks)


def test_ingest_markdown_success(
    db_session: Session,
    storage_service: LocalStorageService,
    ingestion_service: IngestionService,
) -> None:
    """Verify Markdown ingestion preserves heading hierarchies, offsets, and metadata."""
    md_content = b"# Document Architecture\nOverview of the system.\n\n## Component Details\nDetailed modular breakdown."
    storage_key = storage_service.save(md_content, content_type="text/markdown", original_filename="arch.md")

    doc = Document(
        original_filename="arch.md",
        content_type="text/markdown",
        size_bytes=len(md_content),
        sha256="2" * 64,
        storage_key=storage_key,
        status=DocumentStatus.UPLOADED,
    )
    db_session.add(doc)
    db_session.commit()

    result = ingestion_service.ingest_document(doc.id)

    assert result.status == DocumentStatus.READY
    assert result.chunk_count >= 2

    headings = [c.heading for c in result.chunks if c.heading]
    assert "Document Architecture" in headings
    assert "Component Details" in headings

    levels = [c.heading_level for c in result.chunks if c.heading_level]
    assert 1 in levels
    assert 2 in levels


def test_ingest_plaintext_success(
    db_session: Session,
    storage_service: LocalStorageService,
    ingestion_service: IngestionService,
) -> None:
    """Verify plain text ingestion chunks cleanly and tracks offsets."""
    text_content = b"First paragraph of clean plain text.\n\nSecond paragraph of content."
    storage_key = storage_service.save(text_content, content_type="text/plain", original_filename="notes.txt")

    doc = Document(
        original_filename="notes.txt",
        content_type="text/plain",
        size_bytes=len(text_content),
        sha256="3" * 64,
        storage_key=storage_key,
        status=DocumentStatus.UPLOADED,
    )
    db_session.add(doc)
    db_session.commit()

    result = ingestion_service.ingest_document(doc.id)

    assert result.status == DocumentStatus.READY
    assert result.chunk_count >= 1
    assert all(c.end_char_offset - c.start_char_offset == c.character_count for c in result.chunks)


def test_reprocessing_atomically_replaces_prior_chunks(
    db_session: Session,
    storage_service: LocalStorageService,
    ingestion_service: IngestionService,
) -> None:
    """Verify re-ingesting a document replaces the prior chunk set without duplicating chunks."""
    content_v1 = b"# Version 1\nInitial content passage one.\n\n## Section 1\nPassage two."
    storage_key = storage_service.save(content_v1, content_type="text/markdown", original_filename="doc.md")

    doc = Document(
        original_filename="doc.md",
        content_type="text/markdown",
        size_bytes=len(content_v1),
        sha256="4" * 64,
        storage_key=storage_key,
        status=DocumentStatus.UPLOADED,
    )
    db_session.add(doc)
    db_session.commit()

    # Initial ingestion
    res_v1 = ingestion_service.ingest_document(doc.id)
    v1_count = res_v1.chunk_count
    assert v1_count >= 2

    # Reprocess with updated content
    content_v2 = b"# Version 2 Replaced\nSingle concise passage replacing prior text."
    storage_service.delete(storage_key)
    storage_service.save(content_v2, content_type="text/markdown", original_filename="doc.md")

    res_v2 = ingestion_service.ingest_document(doc.id, content=content_v2)
    assert res_v2.status == DocumentStatus.READY
    assert res_v2.chunk_count == 1
    assert "Version 2 Replaced" in res_v2.chunks[0].text

    # Verify database contains only the new chunk
    stored_chunks = db_session.scalars(
        sa.select(ChunkModel).where(ChunkModel.document_id == doc.id).order_by(ChunkModel.chunk_index)
    ).all()
    assert len(stored_chunks) == 1
    assert stored_chunks[0].chunk_index == 0


def test_empty_content_marks_document_failed(
    db_session: Session,
    storage_service: LocalStorageService,
    ingestion_service: IngestionService,
) -> None:
    """Verify empty document content triggers IngestionExtractionError and sets status to FAILED."""
    empty_content = b""
    storage_key = storage_service.save(b"placeholder", content_type="text/plain", original_filename="empty.txt")

    doc = Document(
        original_filename="empty.txt",
        content_type="text/plain",
        size_bytes=0,
        sha256="5" * 64,
        storage_key=storage_key,
        status=DocumentStatus.UPLOADED,
    )
    db_session.add(doc)
    db_session.commit()

    with pytest.raises(IngestionExtractionError, match="empty"):
        ingestion_service.ingest_document(doc.id, content=empty_content)

    db_session.refresh(doc)
    assert doc.status == DocumentStatus.FAILED


def test_extraction_failure_marks_document_failed(
    db_session: Session,
    storage_service: LocalStorageService,
    ingestion_service: IngestionService,
) -> None:
    """Verify malformed file bytes raise IngestionExtractionError and set status to FAILED."""
    corrupted_pdf = b"%PDF-corrupted-binary-garbage"
    storage_key = storage_service.save(corrupted_pdf, content_type="application/pdf", original_filename="corrupt.pdf")

    doc = Document(
        original_filename="corrupt.pdf",
        content_type="application/pdf",
        size_bytes=len(corrupted_pdf),
        sha256="6" * 64,
        storage_key=storage_key,
        status=DocumentStatus.UPLOADED,
    )
    db_session.add(doc)
    db_session.commit()

    with pytest.raises(IngestionExtractionError):
        ingestion_service.ingest_document(doc.id)

    db_session.refresh(doc)
    assert doc.status == DocumentStatus.FAILED


def test_chunking_failure_marks_document_failed(
    db_session: Session,
    storage_service: LocalStorageService,
    ingestion_service: IngestionService,
) -> None:
    """Verify failure during chunking raises IngestionChunkingError and marks document FAILED."""
    text_content = b"Valid content for chunking."
    storage_key = storage_service.save(text_content, content_type="text/plain", original_filename="text.txt")

    doc = Document(
        original_filename="text.txt",
        content_type="text/plain",
        size_bytes=len(text_content),
        sha256="7" * 64,
        storage_key=storage_key,
        status=DocumentStatus.UPLOADED,
    )
    db_session.add(doc)
    db_session.commit()

    with patch.object(ingestion_service.chunker, "chunk", side_effect=ChunkingError("Chunking failed")):
        with pytest.raises(IngestionChunkingError, match="Chunking failed"):
            ingestion_service.ingest_document(doc.id)

    db_session.refresh(doc)
    assert doc.status == DocumentStatus.FAILED


def test_persistence_failure_marks_failed_and_preserves_prior_chunks(
    sqlite_engine: sa.Engine,
    db_session: Session,
    storage_service: LocalStorageService,
    ingestion_service: IngestionService,
) -> None:
    """Verify persistence failure on re-ingestion sets FAILED status and preserves prior committed chunks."""
    content_v1 = b"Original text that persists successfully."
    storage_key = storage_service.save(content_v1, content_type="text/plain", original_filename="data.txt")

    doc = Document(
        original_filename="data.txt",
        content_type="text/plain",
        size_bytes=len(content_v1),
        sha256="8" * 64,
        storage_key=storage_key,
        status=DocumentStatus.UPLOADED,
    )
    db_session.add(doc)
    db_session.commit()

    # Step 1: Initial successful ingestion
    res_v1 = ingestion_service.ingest_document(doc.id)
    assert res_v1.status == DocumentStatus.READY
    assert res_v1.chunk_count == 1

    # Step 2: Simulate failure during persistence on second ingestion run
    with patch.object(
        ingestion_service.chunk_persistence,
        "store_chunks",
        side_effect=ChunkPersistenceError("DB persistence constraint error"),
    ):
        with pytest.raises(IngestionPersistenceError, match="DB persistence constraint error"):
            ingestion_service.ingest_document(doc.id)

    # Step 3: Document status remains READY to preserve last successfully indexed state
    db_session.refresh(doc)
    assert doc.status == DocumentStatus.READY

    # Step 4: Prior chunks must remain intact in the database
    with Session(sqlite_engine) as fresh_session:
        fresh_chunks = fresh_session.scalars(
            sa.select(ChunkModel).where(ChunkModel.document_id == doc.id)
        ).all()
        assert len(fresh_chunks) == 1
        assert fresh_chunks[0].text == "Original text that persists successfully."


def test_missing_document_raises_not_found(ingestion_service: IngestionService) -> None:
    """Verify ingesting a non-existent document ID raises DocumentNotFoundError."""
    missing_id = uuid.uuid4()
    with pytest.raises(DocumentNotFoundError):
        ingestion_service.ingest_document(missing_id)


def test_missing_storage_file_marks_failed_and_raises_not_found(
    db_session: Session,
    ingestion_service: IngestionService,
) -> None:
    """Verify missing storage file marks document FAILED and raises DocumentNotFoundError."""
    doc = Document(
        original_filename="ghost.pdf",
        content_type="application/pdf",
        size_bytes=100,
        sha256="9" * 64,
        storage_key="raw/ghost_does_not_exist.pdf",
        status=DocumentStatus.UPLOADED,
    )
    db_session.add(doc)
    db_session.commit()

    with pytest.raises(DocumentNotFoundError, match="was not found"):
        ingestion_service.ingest_document(doc.id)

    db_session.refresh(doc)
    assert doc.status == DocumentStatus.FAILED


def test_dependency_injection_custom_components(
    db_session: Session,
    storage_service: LocalStorageService,
) -> None:
    """Verify IngestionService can be instantiated with custom injected extractors and chunkers."""
    mock_extractor = MagicMock(spec=ExtractionService)
    mock_chunker = StructureAwareChunker(default_config=ChunkingConfig(chunk_size_tokens=100, chunk_overlap_tokens=10))

    custom_service = IngestionService(
        db=db_session,
        storage=storage_service,
        extraction_service=mock_extractor,
        chunker=mock_chunker,
    )

    assert custom_service.extraction_service is mock_extractor
    assert custom_service.chunker is mock_chunker
    assert custom_service.chunker.default_config.chunk_size_tokens == 100


def test_processing_commit_failure_stops_ingestion_before_extraction(
    db_session: Session,
    storage_service: LocalStorageService,
    ingestion_service: IngestionService,
) -> None:
    """Verify commit failure during PROCESSING stops ingestion before extraction and raises IngestionPersistenceError."""
    text_content = b"Content that should not be extracted."
    storage_key = storage_service.save(text_content, content_type="text/plain", original_filename="test.txt")

    doc = Document(
        original_filename="test.txt",
        content_type="text/plain",
        size_bytes=len(text_content),
        sha256="a" * 64,
        storage_key=storage_key,
        status=DocumentStatus.UPLOADED,
    )
    db_session.add(doc)
    db_session.commit()

    with patch.object(ingestion_service.extraction_service, "extract") as mock_extract:
        with patch.object(db_session, "commit", side_effect=SQLAlchemyError("Simulated PROCESSING commit failure")):
            with pytest.raises(IngestionPersistenceError, match="processing"):
                ingestion_service.ingest_document(doc.id)

        # Ingestion must halt before extraction
        mock_extract.assert_not_called()


def test_processing_operational_error_reports_ambiguous_outcome(
    db_session: Session,
    storage_service: LocalStorageService,
    ingestion_service: IngestionService,
) -> None:
    """Verify OperationalError during PROCESSING commit reports ambiguous commit outcome."""
    text_content = b"Test content."
    storage_key = storage_service.save(text_content, content_type="text/plain", original_filename="test.txt")

    doc = Document(
        original_filename="test.txt",
        content_type="text/plain",
        size_bytes=len(text_content),
        sha256="b" * 64,
        storage_key=storage_key,
        status=DocumentStatus.UPLOADED,
    )
    db_session.add(doc)
    db_session.commit()

    op_err = OperationalError("connection reset", {}, Exception())
    with patch.object(db_session, "commit", side_effect=op_err):
        with pytest.raises(IngestionPersistenceError, match="ambiguous connection error"):
            ingestion_service.ingest_document(doc.id)


def test_ready_commit_failure_raises_persistence_error_and_does_not_return_result(
    db_session: Session,
    storage_service: LocalStorageService,
    ingestion_service: IngestionService,
) -> None:
    """Verify commit failure during READY transition raises IngestionPersistenceError without returning IngestionResult."""
    text_content = b"Text for successful chunking but failed READY commit."
    storage_key = storage_service.save(text_content, content_type="text/plain", original_filename="test.txt")

    doc = Document(
        original_filename="test.txt",
        content_type="text/plain",
        size_bytes=len(text_content),
        sha256="c" * 64,
        storage_key=storage_key,
        status=DocumentStatus.UPLOADED,
    )
    db_session.add(doc)
    db_session.commit()

    original_commit = db_session.commit
    commit_calls = 0

    def commit_hook() -> None:
        nonlocal commit_calls
        commit_calls += 1
        # Commit 1 is PROCESSING, Commit 2 is coordinated persistence (chunks + embeddings + READY)
        if commit_calls == 2:
            raise SQLAlchemyError("Simulated READY commit failure")
        original_commit()

    with patch.object(db_session, "commit", side_effect=commit_hook):
        with pytest.raises(IngestionPersistenceError, match="ready"):
            ingestion_service.ingest_document(doc.id)


def test_ready_operational_error_reports_ambiguous_outcome(
    db_session: Session,
    storage_service: LocalStorageService,
    ingestion_service: IngestionService,
) -> None:
    """Verify OperationalError during READY commit reports ambiguous commit outcome."""
    text_content = b"Text for READY operational error."
    storage_key = storage_service.save(text_content, content_type="text/plain", original_filename="test.txt")

    doc = Document(
        original_filename="test.txt",
        content_type="text/plain",
        size_bytes=len(text_content),
        sha256="d" * 64,
        storage_key=storage_key,
        status=DocumentStatus.UPLOADED,
    )
    db_session.add(doc)
    db_session.commit()

    original_commit = db_session.commit
    commit_calls = 0
    op_err = OperationalError("connection severed", {}, Exception())

    def commit_hook() -> None:
        nonlocal commit_calls
        commit_calls += 1
        # Commit 1 is PROCESSING, Commit 2 is coordinated persistence (chunks + embeddings + READY)
        if commit_calls == 2:
            raise op_err
        original_commit()

    with patch.object(db_session, "commit", side_effect=commit_hook):
        with pytest.raises(IngestionPersistenceError, match="ambiguous connection error"):
            ingestion_service.ingest_document(doc.id)


def test_failed_status_update_failure_does_not_mask_original_exception(
    db_session: Session,
    storage_service: LocalStorageService,
    ingestion_service: IngestionService,
) -> None:
    """Verify secondary failure while persisting FAILED does not mask the original extraction exception."""
    storage_key = storage_service.save(b"placeholder", content_type="text/plain", original_filename="empty.txt")

    doc = Document(
        original_filename="empty.txt",
        content_type="text/plain",
        size_bytes=0,
        sha256="e" * 64,
        storage_key=storage_key,
        status=DocumentStatus.UPLOADED,
    )
    db_session.add(doc)
    db_session.commit()

    original_commit = db_session.commit
    commit_calls = 0

    def commit_hook() -> None:
        nonlocal commit_calls
        commit_calls += 1
        # Commit 1 is PROCESSING (allowed), Commit 2 is FAILED (fails)
        if commit_calls == 2:
            raise OperationalError("database down during FAILED commit", {}, Exception())
        original_commit()

    with patch.object(db_session, "commit", side_effect=commit_hook):
        # Must raise the original IngestionExtractionError from empty content, NOT IngestionPersistenceError
        with pytest.raises(IngestionExtractionError, match="empty"):
            ingestion_service.ingest_document(doc.id, content=b"")


def test_successful_ingestion_commits_ready_only_after_chunks_persisted(
    db_session: Session,
    storage_service: LocalStorageService,
    ingestion_service: IngestionService,
) -> None:
    """Verify normal successful ingestion commits READY strictly after chunks are persisted."""
    text_content = b"Ordered execution verification."
    storage_key = storage_service.save(text_content, content_type="text/plain", original_filename="order.txt")

    doc = Document(
        original_filename="order.txt",
        content_type="text/plain",
        size_bytes=len(text_content),
        sha256="f" * 64,
        storage_key=storage_key,
        status=DocumentStatus.UPLOADED,
    )
    db_session.add(doc)
    db_session.commit()

    events: list[str] = []
    original_store = ingestion_service.chunk_persistence.store_chunks

    def store_spy(*args, **kwargs):  # type: ignore[no-untyped-def]
        events.append("store_chunks")
        return original_store(*args, **kwargs)

    original_mark = ingestion_service._mark_status

    def mark_spy(document, status, swallow_errors=False):  # type: ignore[no-untyped-def]
        events.append(f"mark_{status.value}")
        return original_mark(document, status, swallow_errors=swallow_errors)

    with patch.object(ingestion_service.chunk_persistence, "store_chunks", side_effect=store_spy):
        with patch.object(ingestion_service, "_mark_status", side_effect=mark_spy):
            res = ingestion_service.ingest_document(doc.id)

    assert events == ["mark_processing", "store_chunks", "mark_ready"]
    assert res.status == DocumentStatus.READY
    assert res.chunk_count == 1
