"""Tests for ChunkPersistenceService coordination, validation, atomic replacement, and rollback."""

from unittest.mock import patch
import uuid

import pytest
import sqlalchemy as sa
from sqlalchemy.exc import OperationalError, SQLAlchemyError
from sqlalchemy.orm import Session

from app.chunking.models import Chunk as DomainChunk, ChunkingConfig, ChunkingResult
from app.db.base import Base
from app.models.chunk import Chunk as ChunkModel
from app.models.document import Document, DocumentStatus
from app.services.chunk import ChunkPersistenceService
from app.services.exceptions import (
    ChunkPersistenceError,
    ChunkValidationError,
    DocumentNotFoundError,
    EmptyChunkSetError,
)


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
    """Provide a scoped database session for service testing."""
    with Session(sqlite_engine) as session:
        yield session


@pytest.fixture
def sample_document(db_session: Session) -> Document:
    """Create and persist a Document for chunk testing."""
    doc = Document(
        original_filename="manual.md",
        content_type="text/markdown",
        size_bytes=2048,
        sha256="b" * 64,
        storage_key="raw/manual.md",
        status=DocumentStatus.READY,
    )
    db_session.add(doc)
    db_session.commit()
    return doc


def test_store_and_get_chunks_with_domain_models(
    db_session: Session,
    sample_document: Document,
) -> None:
    """Verify persisting chunks from domain Chunk models and querying in index order."""
    service = ChunkPersistenceService(db_session)

    domain_chunks = [
        DomainChunk(
            chunk_index=0,
            text="# Title\nIntroduction paragraph.",
            token_count=6,
            character_count=31,
            start_char_offset=0,
            end_char_offset=31,
            document_id=sample_document.id,
            source_format="text/markdown",
            heading="Title",
            heading_level=1,
            metadata={"section": "intro"},
        ),
        DomainChunk(
            chunk_index=1,
            text="Second section with details.",
            token_count=5,
            character_count=28,
            start_char_offset=32,
            end_char_offset=60,
            document_id=sample_document.id,
            source_format="text/markdown",
            heading="Title",
            heading_level=1,
            metadata={"section": "body"},
        ),
    ]

    stored = service.store_chunks(sample_document.id, domain_chunks)
    assert len(stored) == 2
    assert stored[0].chunk_index == 0
    assert stored[1].chunk_index == 1

    # Verify query returns matching models in ascending order
    retrieved = service.get_chunks(sample_document.id)
    assert len(retrieved) == 2
    assert retrieved[0].chunk_index == 0
    assert retrieved[0].text == "# Title\nIntroduction paragraph."
    assert retrieved[0].token_count == 6
    assert retrieved[0].character_count == 31
    assert retrieved[0].start_char_offset == 0
    assert retrieved[0].end_char_offset == 31
    assert retrieved[0].heading == "Title"
    assert retrieved[0].heading_level == 1
    assert retrieved[0].metadata_ == {"section": "intro"}

    assert retrieved[1].chunk_index == 1
    assert retrieved[1].text == "Second section with details."

    # Verify count method
    assert service.get_chunk_count(sample_document.id) == 2


def test_store_chunks_with_chunking_result(
    db_session: Session,
    sample_document: Document,
) -> None:
    """Verify storing chunks directly from a ChunkingResult container."""
    service = ChunkPersistenceService(db_session)

    result = ChunkingResult(
        document_id=sample_document.id,
        config=ChunkingConfig(chunk_size_tokens=500, chunk_overlap_tokens=75),
        chunks=[
            DomainChunk(
                chunk_index=0,
                text="Chunk zero text.",
                token_count=4,
                character_count=16,
                start_char_offset=0,
                end_char_offset=16,
                document_id=sample_document.id,
                source_format="text/plain",
            )
        ],
    )

    stored = service.store_chunks(sample_document.id, result)
    assert len(stored) == 1
    assert stored[0].text == "Chunk zero text."


def test_atomic_replacement(
    db_session: Session,
    sample_document: Document,
) -> None:
    """Verify storing with replace=True replaces all previous chunks cleanly."""
    service = ChunkPersistenceService(db_session)

    # Initial set of 3 chunks
    initial_chunks = [
        DomainChunk(
            chunk_index=i,
            text=f"Initial text {i}",
            token_count=3,
            character_count=len(f"Initial text {i}"),
            start_char_offset=i * 20,
            end_char_offset=(i * 20) + len(f"Initial text {i}"),
        )
        for i in range(3)
    ]
    service.store_chunks(sample_document.id, initial_chunks)
    assert service.get_chunk_count(sample_document.id) == 3

    # Replace with 1 new chunk
    replacement_chunks = [
        DomainChunk(
            chunk_index=0,
            text="Brand new replaced single chunk.",
            token_count=6,
            character_count=32,
            start_char_offset=0,
            end_char_offset=32,
        )
    ]
    service.store_chunks(sample_document.id, replacement_chunks, replace=True)

    retrieved = service.get_chunks(sample_document.id)
    assert len(retrieved) == 1
    assert retrieved[0].text == "Brand new replaced single chunk."
    assert service.get_chunk_count(sample_document.id) == 1


def test_rollback_on_persistence_failure_preserves_prior_chunks(
    sqlite_engine: sa.Engine,
    db_session: Session,
    sample_document: Document,
) -> None:
    """Verify that if chunk persistence fails mid-transaction, rollback preserves prior chunks."""
    service = ChunkPersistenceService(db_session)
    doc_id = sample_document.id

    # 1. Establish initial committed chunk set
    initial_chunks = [
        DomainChunk(
            chunk_index=0,
            text="Original intact chunk.",
            token_count=4,
            character_count=22,
            start_char_offset=0,
            end_char_offset=22,
        )
    ]
    service.store_chunks(doc_id, initial_chunks)
    assert service.get_chunk_count(doc_id) == 1

    # 2. Attempt replacement that causes a database error during flush
    with patch.object(db_session, "flush", side_effect=SQLAlchemyError("DB disk full")):
        new_chunks = [
            DomainChunk(
                chunk_index=0,
                text="Attempted replacement chunk.",
                token_count=4,
                character_count=28,
                start_char_offset=0,
                end_char_offset=28,
            )
        ]
        with pytest.raises(ChunkPersistenceError, match="DB disk full"):
            service.store_chunks(doc_id, new_chunks, replace=True)

    # 3. Verify prior chunk set was NOT deleted or corrupted using a FRESH independent session
    with Session(sqlite_engine) as fresh_session:
        fresh_service = ChunkPersistenceService(fresh_session)
        surviving = fresh_service.get_chunks(doc_id)
        assert len(surviving) == 1
        assert surviving[0].text == "Original intact chunk."


def test_empty_chunk_set_rejected_by_default(
    db_session: Session,
    sample_document: Document,
) -> None:
    """Verify empty chunk input raises EmptyChunkSetError and does not delete existing chunks."""
    service = ChunkPersistenceService(db_session)

    initial_chunk = DomainChunk(
        chunk_index=0,
        text="Keep me safe.",
        token_count=3,
        character_count=13,
        start_char_offset=0,
        end_char_offset=13,
    )
    service.store_chunks(sample_document.id, [initial_chunk])

    # Empty chunk input rejected
    with pytest.raises(EmptyChunkSetError, match="cannot be empty"):
        service.store_chunks(sample_document.id, [])

    # Previous chunk intact
    assert service.get_chunk_count(sample_document.id) == 1


def test_empty_chunk_set_allowed_explicitly(
    db_session: Session,
    sample_document: Document,
) -> None:
    """Verify empty chunk input with allow_empty=True clears existing chunks."""
    service = ChunkPersistenceService(db_session)

    initial_chunk = DomainChunk(
        chunk_index=0,
        text="Will be cleared.",
        token_count=3,
        character_count=16,
        start_char_offset=0,
        end_char_offset=16,
    )
    service.store_chunks(sample_document.id, [initial_chunk])
    assert service.get_chunk_count(sample_document.id) == 1

    # Explicit empty set clears chunks
    result = service.store_chunks(sample_document.id, [], replace=True, allow_empty=True)
    assert result == []
    assert service.get_chunk_count(sample_document.id) == 0


def test_validation_empty_text_rejected(
    db_session: Session,
    sample_document: Document,
) -> None:
    """Verify chunks with empty text are rejected."""
    service = ChunkPersistenceService(db_session)
    bad_chunk = {
        "chunk_index": 0,
        "text": "",
        "token_count": 0,
        "character_count": 0,
        "start_char_offset": 0,
        "end_char_offset": 0,
    }
    with pytest.raises(ChunkValidationError, match="empty or non-string text"):
        service.store_chunks(sample_document.id, [bad_chunk])


def test_validation_character_count_mismatch_rejected(
    db_session: Session,
    sample_document: Document,
) -> None:
    """Verify chunk with character_count != len(text) is rejected."""
    service = ChunkPersistenceService(db_session)
    bad_chunk = {
        "chunk_index": 0,
        "text": "Hello world",
        "token_count": 2,
        "character_count": 99,  # actual length is 11
        "start_char_offset": 0,
        "end_char_offset": 11,
    }
    with pytest.raises(ChunkValidationError, match="inconsistent with text length"):
        service.store_chunks(sample_document.id, [bad_chunk])


def test_validation_invalid_offsets_rejected(
    db_session: Session,
    sample_document: Document,
) -> None:
    """Verify chunk with end_char_offset < start_char_offset is rejected."""
    service = ChunkPersistenceService(db_session)
    bad_chunk = {
        "chunk_index": 0,
        "text": "Hello world",
        "token_count": 2,
        "character_count": 11,
        "start_char_offset": 20,
        "end_char_offset": 10,
    }
    with pytest.raises(ChunkValidationError, match="invalid end_char_offset"):
        service.store_chunks(sample_document.id, [bad_chunk])


def test_validation_non_consecutive_index_rejected(
    db_session: Session,
    sample_document: Document,
) -> None:
    """Verify chunks that do not form consecutive 0..N-1 indices are rejected."""
    service = ChunkPersistenceService(db_session)
    chunks = [
        {
            "chunk_index": 0,
            "text": "First chunk",
            "token_count": 2,
            "character_count": 11,
            "start_char_offset": 0,
            "end_char_offset": 11,
        },
        {
            "chunk_index": 2,  # Skipped index 1
            "text": "Third chunk",
            "token_count": 2,
            "character_count": 11,
            "start_char_offset": 12,
            "end_char_offset": 23,
        },
    ]
    with pytest.raises(ChunkValidationError, match="consecutive zero-based index sequence"):
        service.store_chunks(sample_document.id, chunks)


def test_validation_document_id_mismatch_rejected(
    db_session: Session,
    sample_document: Document,
) -> None:
    """Verify chunk with mismatched document_id is rejected."""
    service = ChunkPersistenceService(db_session)
    alien_doc_id = uuid.uuid4()
    bad_chunk = DomainChunk(
        chunk_index=0,
        text="Alien doc chunk",
        token_count=3,
        character_count=15,
        start_char_offset=0,
        end_char_offset=15,
        document_id=alien_doc_id,
    )
    with pytest.raises(ChunkValidationError, match="does not match target document_id"):
        service.store_chunks(sample_document.id, [bad_chunk])


def test_document_not_found_errors(db_session: Session) -> None:
    """Verify operations on non-existent document ID raise DocumentNotFoundError."""
    service = ChunkPersistenceService(db_session)
    missing_id = uuid.uuid4()

    valid_chunk = DomainChunk(
        chunk_index=0,
        text="Some text",
        token_count=2,
        character_count=9,
        start_char_offset=0,
        end_char_offset=9,
    )

    with pytest.raises(DocumentNotFoundError):
        service.store_chunks(missing_id, [valid_chunk])

    with pytest.raises(DocumentNotFoundError):
        service.get_chunks(missing_id)

    with pytest.raises(DocumentNotFoundError):
        service.get_chunk_count(missing_id)

    with pytest.raises(DocumentNotFoundError):
        service.delete_chunks(missing_id)


def test_delete_chunks(
    db_session: Session,
    sample_document: Document,
) -> None:
    """Verify delete_chunks removes all chunks for the document and returns deleted count."""
    service = ChunkPersistenceService(db_session)

    chunks = [
        DomainChunk(
            chunk_index=i,
            text=f"Chunk {i}",
            token_count=2,
            character_count=len(f"Chunk {i}"),
            start_char_offset=i * 10,
            end_char_offset=(i * 10) + len(f"Chunk {i}"),
        )
        for i in range(3)
    ]
    service.store_chunks(sample_document.id, chunks)
    assert service.get_chunk_count(sample_document.id) == 3

    deleted_count = service.delete_chunks(sample_document.id)
    assert deleted_count == 3
    assert service.get_chunk_count(sample_document.id) == 0
    assert service.get_chunks(sample_document.id) == []
    # Assert parent document remains intact
    assert db_session.get(Document, sample_document.id) is not None


def test_validation_offset_span_mismatch_rejected(
    db_session: Session,
    sample_document: Document,
) -> None:
    """Verify chunk with end_char_offset - start_char_offset != character_count is rejected."""
    service = ChunkPersistenceService(db_session)
    bad_chunk = {
        "chunk_index": 0,
        "text": "Hello world",
        "token_count": 2,
        "character_count": 11,
        "start_char_offset": 0,
        "end_char_offset": 25,  # 25 - 0 = 25 != 11
    }
    with pytest.raises(ChunkValidationError, match="offset span .* inconsistent with character_count"):
        service.store_chunks(sample_document.id, [bad_chunk])


def test_validation_heading_level_bounds(
    db_session: Session,
    sample_document: Document,
) -> None:
    """Verify heading_level is validated to 1..6 inclusive, rejecting 0 and 7."""
    service = ChunkPersistenceService(db_session)

    # 1. Reject heading_level = 0
    bad_h0 = {
        "chunk_index": 0,
        "text": "Heading 0",
        "token_count": 2,
        "character_count": 9,
        "start_char_offset": 0,
        "end_char_offset": 9,
        "heading_level": 0,
    }
    with pytest.raises(ChunkValidationError, match="invalid heading_level"):
        service.store_chunks(sample_document.id, [bad_h0])

    # 2. Reject heading_level = 7
    bad_h7 = {
        "chunk_index": 0,
        "text": "Heading 7",
        "token_count": 2,
        "character_count": 9,
        "start_char_offset": 0,
        "end_char_offset": 9,
        "heading_level": 7,
    }
    with pytest.raises(ChunkValidationError, match="invalid heading_level"):
        service.store_chunks(sample_document.id, [bad_h7])

    # 3. Accept valid boundaries 1 and 6
    valid_chunks = [
        DomainChunk(
            chunk_index=0,
            text="H1 title",
            token_count=2,
            character_count=8,
            start_char_offset=0,
            end_char_offset=8,
            heading_level=1,
        ),
        DomainChunk(
            chunk_index=1,
            text="H6 title",
            token_count=2,
            character_count=8,
            start_char_offset=8,
            end_char_offset=16,
            heading_level=6,
        ),
    ]
    stored = service.store_chunks(sample_document.id, valid_chunks)
    assert len(stored) == 2
    assert stored[0].heading_level == 1
    assert stored[1].heading_level == 6


def test_validation_page_number_consistency(
    db_session: Session,
    sample_document: Document,
) -> None:
    """Verify page_number must appear in page_numbers list when both are provided."""
    service = ChunkPersistenceService(db_session)

    # 1. Inconsistent: page_number 5 not in [1, 2]
    inconsistent_chunk = {
        "chunk_index": 0,
        "text": "Mismatched page.",
        "token_count": 3,
        "character_count": 16,
        "start_char_offset": 0,
        "end_char_offset": 16,
        "page_number": 5,
        "page_numbers": [1, 2],
    }
    with pytest.raises(ChunkValidationError, match="page_number 5 not present in page_numbers list"):
        service.store_chunks(sample_document.id, [inconsistent_chunk])

    # 2. Consistent: page_number 2 in [1, 2, 3]
    consistent_chunk = DomainChunk(
        chunk_index=0,
        text="Matching page.",
        token_count=3,
        character_count=14,
        start_char_offset=0,
        end_char_offset=14,
        page_number=2,
        page_numbers=[1, 2, 3],
    )
    stored = service.store_chunks(sample_document.id, [consistent_chunk])
    assert len(stored) == 1
    assert stored[0].page_number == 2
    assert stored[0].page_numbers == [1, 2, 3]

    # 3. Nullable page_number with empty list accepted
    no_page_chunk = DomainChunk(
        chunk_index=0,
        text="Text with no page.",
        token_count=4,
        character_count=18,
        start_char_offset=0,
        end_char_offset=18,
        page_number=None,
        page_numbers=[],
    )
    stored2 = service.store_chunks(sample_document.id, [no_page_chunk])
    assert len(stored2) == 1
    assert stored2[0].page_number is None
    assert stored2[0].page_numbers == []


def test_ambiguous_commit_operational_error_raises_chunk_persistence_error(
    db_session: Session,
    sample_document: Document,
) -> None:
    """Verify OperationalError on commit is treated as an ambiguous connection error and raises ChunkPersistenceError."""
    service = ChunkPersistenceService(db_session)
    chunk = DomainChunk(
        chunk_index=0,
        text="Ambiguous commit test.",
        token_count=4,
        character_count=22,
        start_char_offset=0,
        end_char_offset=22,
    )
    with patch.object(
        db_session,
        "commit",
        side_effect=OperationalError("Connection reset by peer", None, None),
    ):
        with pytest.raises(ChunkPersistenceError, match="ambiguous connection error"):
            service.store_chunks(sample_document.id, [chunk])


def test_post_commit_refresh_failure_does_not_fail_persistence(
    sqlite_engine: sa.Engine,
    db_session: Session,
    sample_document: Document,
) -> None:
    """Verify that a failure during post-commit refresh does not fail the operation or undo persistence."""
    service = ChunkPersistenceService(db_session)
    doc_id = sample_document.id
    chunk = DomainChunk(
        chunk_index=0,
        text="Committed despite refresh issue.",
        token_count=5,
        character_count=32,
        start_char_offset=0,
        end_char_offset=32,
    )
    with patch.object(
        db_session,
        "refresh",
        side_effect=SQLAlchemyError("Post-commit refresh failure"),
    ):
        result = service.store_chunks(doc_id, [chunk])
        assert len(result) == 1
        assert result[0].text == "Committed despite refresh issue."

    # Verify via fresh session that chunk was indeed committed successfully
    with Session(sqlite_engine) as fresh_session:
        fresh_service = ChunkPersistenceService(fresh_session)
        persisted = fresh_service.get_chunks(doc_id)
        assert len(persisted) == 1
        assert persisted[0].text == "Committed despite refresh issue."
