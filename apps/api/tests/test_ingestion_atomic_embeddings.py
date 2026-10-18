"""Unit tests for atomic coordinated ingestion and vector persistence.

Tests first-time ingestion vs reprocessing failure semantics, rollback guarantees,
and single-transaction atomicity across chunks, embeddings, and READY status.
Runs in-memory using SQLite with zero model downloads.
"""

from unittest.mock import patch
import uuid

import pytest
import sqlalchemy as sa
from sqlalchemy.exc import OperationalError, SQLAlchemyError
from sqlalchemy.orm import Session

from app.chunking.service import StructureAwareChunker
from app.core.config import Settings
from app.db.base import Base
from app.embeddings import DeterministicMockEmbeddingProvider
from app.embeddings.exceptions import EmbeddingInferenceError
from app.extraction.service import ExtractionService
from app.models.chunk import Chunk as ChunkModel
from app.models.document import Document, DocumentStatus
from app.models.embedding import ChunkEmbedding
from app.services.chunk import ChunkPersistenceService
from app.services.embedding import EmbeddingService
from app.services.exceptions import (
    ChunkPersistenceError,
    IngestionExtractionError,
    IngestionPersistenceError,
)
from app.services.ingestion import IngestionService
from app.storage.local import LocalStorageService


@pytest.fixture
def sqlite_engine() -> sa.Engine:
    """Provide an in-memory SQLite engine with PRAGMA foreign_keys enabled and tables created."""
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
def storage_service(tmp_path: pytest.TempPathFactory) -> LocalStorageService:
    """Provide LocalStorageService backed by a temporary directory."""
    return LocalStorageService(root_dir=tmp_path / "documents", max_size_mb=10)


@pytest.fixture
def mock_embedding_provider() -> DeterministicMockEmbeddingProvider:
    """Provide a deterministic mock embedding provider producing 384-dim vectors."""
    return DeterministicMockEmbeddingProvider(model_id="test-bge-384", dimension=384)


@pytest.fixture
def ingestion_service(
    db_session: Session,
    storage_service: LocalStorageService,
    mock_embedding_provider: DeterministicMockEmbeddingProvider,
) -> IngestionService:
    """Provide IngestionService configured with database and mock embedding provider."""
    embedding_service = EmbeddingService(db=db_session, provider=mock_embedding_provider)
    return IngestionService(
        db=db_session,
        storage=storage_service,
        extraction_service=ExtractionService(),
        chunker=StructureAwareChunker(),
        chunk_persistence=ChunkPersistenceService(db_session),
        embedding_service=embedding_service,
    )


def test_first_time_ingestion_persists_chunks_and_embeddings_atomically(
    db_session: Session,
    storage_service: LocalStorageService,
    ingestion_service: IngestionService,
) -> None:
    """Verify first-time ingestion commits chunks and embeddings together in READY status."""
    content = b"Paragraph one with enough content for chunking.\n\nParagraph two with additional words."
    storage_key = storage_service.save(content, content_type="text/plain", original_filename="doc1.txt")

    doc = Document(
        original_filename="doc1.txt",
        content_type="text/plain",
        size_bytes=len(content),
        sha256="1" * 64,
        storage_key=storage_key,
        status=DocumentStatus.UPLOADED,
    )
    db_session.add(doc)
    db_session.commit()

    result = ingestion_service.ingest_document(doc.id)

    assert result.status == DocumentStatus.READY
    assert result.chunk_count > 0
    assert len(result.chunks) == result.chunk_count
    assert len(result.embeddings) == result.chunk_count

    # Verify records in database
    db_chunks = db_session.scalars(sa.select(ChunkModel).where(ChunkModel.document_id == doc.id)).all()
    db_embeddings = db_session.scalars(sa.select(ChunkEmbedding).where(ChunkEmbedding.document_id == doc.id)).all()

    assert len(db_chunks) == result.chunk_count
    assert len(db_embeddings) == result.chunk_count
    for emb in db_embeddings:
        assert emb.model_id == "test-bge-384"
        assert emb.dimension == 384
        assert len(emb.embedding) == 384


def test_first_time_ingestion_embedding_failure_leaves_zero_records(
    db_session: Session,
    storage_service: LocalStorageService,
    ingestion_service: IngestionService,
    mock_embedding_provider: DeterministicMockEmbeddingProvider,
) -> None:
    """Verify first-time ingestion failure during embedding leaves zero chunks and zero embeddings, document FAILED."""
    content = b"Valid content for extraction and chunking."
    storage_key = storage_service.save(content, content_type="text/plain", original_filename="doc2.txt")

    doc = Document(
        original_filename="doc2.txt",
        content_type="text/plain",
        size_bytes=len(content),
        sha256="2" * 64,
        storage_key=storage_key,
        status=DocumentStatus.UPLOADED,
    )
    db_session.add(doc)
    db_session.commit()

    # Simulate in-memory embedding inference failure
    mock_embedding_provider.should_fail = True

    with pytest.raises(IngestionPersistenceError, match="Embedding generation failed"):
        ingestion_service.ingest_document(doc.id)

    db_session.refresh(doc)
    assert doc.status == DocumentStatus.FAILED

    # Zero chunks and zero embeddings must exist in database
    chunks = db_session.scalars(sa.select(ChunkModel).where(ChunkModel.document_id == doc.id)).all()
    embeddings = db_session.scalars(sa.select(ChunkEmbedding).where(ChunkEmbedding.document_id == doc.id)).all()
    assert len(chunks) == 0
    assert len(embeddings) == 0


def test_first_time_ingestion_flush_failure_leaves_zero_records(
    db_session: Session,
    storage_service: LocalStorageService,
    ingestion_service: IngestionService,
) -> None:
    """Verify pre-commit flush failure during first-time persistence leaves zero chunks/embeddings, document FAILED."""
    content = b"Content that will fail on database flush."
    storage_key = storage_service.save(content, content_type="text/plain", original_filename="doc3.txt")

    doc = Document(
        original_filename="doc3.txt",
        content_type="text/plain",
        size_bytes=len(content),
        sha256="3" * 64,
        storage_key=storage_key,
        status=DocumentStatus.UPLOADED,
    )
    db_session.add(doc)
    db_session.commit()

    # Simulate flush error in embedding service
    with patch.object(
        ingestion_service.embedding_service,
        "store_embeddings",
        side_effect=ChunkPersistenceError("Simulated DB flush error on embeddings"),
    ):
        with pytest.raises(IngestionPersistenceError, match="Simulated DB flush error"):
            ingestion_service.ingest_document(doc.id)

    db_session.refresh(doc)
    assert doc.status == DocumentStatus.FAILED

    chunks = db_session.scalars(sa.select(ChunkModel).where(ChunkModel.document_id == doc.id)).all()
    embeddings = db_session.scalars(sa.select(ChunkEmbedding).where(ChunkEmbedding.document_id == doc.id)).all()
    assert len(chunks) == 0
    assert len(embeddings) == 0


def test_reprocessing_success_atomically_replaces_chunks_and_embeddings(
    db_session: Session,
    storage_service: LocalStorageService,
    ingestion_service: IngestionService,
) -> None:
    """Verify successful reprocessing replaces old chunks and embeddings, document remains READY."""
    content_v1 = b"Initial text version one."
    storage_key = storage_service.save(content_v1, content_type="text/plain", original_filename="doc4.txt")

    doc = Document(
        original_filename="doc4.txt",
        content_type="text/plain",
        size_bytes=len(content_v1),
        sha256="4" * 64,
        storage_key=storage_key,
        status=DocumentStatus.UPLOADED,
    )
    db_session.add(doc)
    db_session.commit()

    # Initial ingestion
    res_v1 = ingestion_service.ingest_document(doc.id)
    assert res_v1.status == DocumentStatus.READY
    old_chunks = db_session.scalars(sa.select(ChunkModel).where(ChunkModel.document_id == doc.id)).all()
    old_chunk_ids = {c.id for c in old_chunks}

    # Second ingestion with new content
    content_v2 = b"Updated text version two with distinct wording."
    res_v2 = ingestion_service.ingest_document(doc.id, content=content_v2)

    assert res_v2.status == DocumentStatus.READY
    new_chunks = db_session.scalars(sa.select(ChunkModel).where(ChunkModel.document_id == doc.id)).all()
    new_embeddings = db_session.scalars(sa.select(ChunkEmbedding).where(ChunkEmbedding.document_id == doc.id)).all()

    assert len(new_chunks) == len(res_v2.chunks)
    assert len(new_embeddings) == len(res_v2.embeddings)

    # Old chunk IDs were completely replaced
    new_chunk_ids = {c.id for c in new_chunks}
    assert old_chunk_ids.isdisjoint(new_chunk_ids)


def test_reprocessing_embedding_failure_preserves_prior_ready_data(
    db_session: Session,
    storage_service: LocalStorageService,
    ingestion_service: IngestionService,
    mock_embedding_provider: DeterministicMockEmbeddingProvider,
) -> None:
    """Verify reprocessing failure during embedding preserves prior committed chunks/embeddings, document status remains READY."""
    content_v1 = b"Original reliable content for indexing."
    storage_key = storage_service.save(content_v1, content_type="text/plain", original_filename="doc5.txt")

    doc = Document(
        original_filename="doc5.txt",
        content_type="text/plain",
        size_bytes=len(content_v1),
        sha256="5" * 64,
        storage_key=storage_key,
        status=DocumentStatus.UPLOADED,
    )
    db_session.add(doc)
    db_session.commit()

    # Successful initial ingestion
    res_v1 = ingestion_service.ingest_document(doc.id)
    assert res_v1.status == DocumentStatus.READY

    initial_chunks = db_session.scalars(sa.select(ChunkModel).where(ChunkModel.document_id == doc.id)).all()
    initial_embeddings = db_session.scalars(sa.select(ChunkEmbedding).where(ChunkEmbedding.document_id == doc.id)).all()
    assert len(initial_chunks) > 0
    assert len(initial_embeddings) > 0
    initial_chunk_text = initial_chunks[0].text

    # Attempt second ingestion, but trigger embedding failure
    mock_embedding_provider.should_fail = True

    with pytest.raises(IngestionPersistenceError, match="Embedding generation failed"):
        ingestion_service.ingest_document(doc.id, content=b"New content that will fail during embedding")

    # Document status MUST remain READY
    db_session.refresh(doc)
    assert doc.status == DocumentStatus.READY

    # Prior chunks and embeddings MUST remain intact
    current_chunks = db_session.scalars(sa.select(ChunkModel).where(ChunkModel.document_id == doc.id)).all()
    current_embeddings = db_session.scalars(sa.select(ChunkEmbedding).where(ChunkEmbedding.document_id == doc.id)).all()

    assert len(current_chunks) == len(initial_chunks)
    assert len(current_embeddings) == len(initial_embeddings)
    assert current_chunks[0].text == initial_chunk_text


def test_reprocessing_flush_failure_preserves_prior_ready_data(
    db_session: Session,
    storage_service: LocalStorageService,
    ingestion_service: IngestionService,
) -> None:
    """Verify reprocessing failure during persistence flush preserves prior chunks and embeddings, document status remains READY."""
    content_v1 = b"Original content to preserve."
    storage_key = storage_service.save(content_v1, content_type="text/plain", original_filename="doc6.txt")

    doc = Document(
        original_filename="doc6.txt",
        content_type="text/plain",
        size_bytes=len(content_v1),
        sha256="6" * 64,
        storage_key=storage_key,
        status=DocumentStatus.UPLOADED,
    )
    db_session.add(doc)
    db_session.commit()

    # Initial ingestion
    ingestion_service.ingest_document(doc.id)
    initial_chunks = db_session.scalars(sa.select(ChunkModel).where(ChunkModel.document_id == doc.id)).all()
    initial_embeddings = db_session.scalars(sa.select(ChunkEmbedding).where(ChunkEmbedding.document_id == doc.id)).all()

    # Attempt second ingestion with DB flush error on embedding store
    with patch.object(
        ingestion_service.embedding_service,
        "store_embeddings",
        side_effect=ChunkPersistenceError("Simulated flush crash during re-embedding"),
    ):
        with pytest.raises(IngestionPersistenceError, match="Simulated flush crash"):
            ingestion_service.ingest_document(doc.id, content=b"New text that crashes during persistence")

    # Status must remain READY
    db_session.refresh(doc)
    assert doc.status == DocumentStatus.READY

    # Chunks and embeddings must be intact
    current_chunks = db_session.scalars(sa.select(ChunkModel).where(ChunkModel.document_id == doc.id)).all()
    current_embeddings = db_session.scalars(sa.select(ChunkEmbedding).where(ChunkEmbedding.document_id == doc.id)).all()

    assert len(current_chunks) == len(initial_chunks)
    assert len(current_embeddings) == len(initial_embeddings)


def test_secondary_status_failure_does_not_mask_primary_error(
    db_session: Session,
    storage_service: LocalStorageService,
    ingestion_service: IngestionService,
) -> None:
    """Verify secondary failure while recording status logs the secondary error and raises the primary exception."""
    content = b"Content causing extraction failure."
    storage_key = storage_service.save(content, content_type="text/plain", original_filename="doc7.txt")

    doc = Document(
        original_filename="doc7.txt",
        content_type="text/plain",
        size_bytes=0,
        sha256="7" * 64,
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
        # Commit 1 is PROCESSING, Commit 2 is FAILED status write
        if commit_calls == 2:
            raise OperationalError("database dropped during FAILED status write", {}, Exception())
        original_commit()

    with patch.object(db_session, "commit", side_effect=commit_hook):
        # Must raise the original IngestionExtractionError from empty content, not masking it with secondary commit error
        with pytest.raises(IngestionExtractionError, match="empty"):
            ingestion_service.ingest_document(doc.id, content=b"")
