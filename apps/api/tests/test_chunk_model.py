"""Tests for Chunk persistence model, constraints, cascade deletion, and migration schema.

These tests run in-memory and do not require a live PostgreSQL container.
"""

from datetime import datetime
import importlib.util
from pathlib import Path
import uuid

from alembic.migration import MigrationContext
from alembic.operations import Operations
import pytest
import sqlalchemy as sa
from sqlalchemy.exc import IntegrityError
from sqlalchemy.orm import Session

from app.db.base import Base
from app.models.chunk import Chunk
from app.models.document import Document, DocumentStatus


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
    """Provide a scoped database session for chunk model testing."""
    with Session(sqlite_engine) as session:
        yield session


def _create_sample_document(db_session: Session) -> Document:
    """Helper creating a persisted parent Document record."""
    doc = Document(
        original_filename="sample.pdf",
        content_type="application/pdf",
        size_bytes=1024,
        sha256="a" * 64,
        storage_key="raw/sample.pdf",
        status=DocumentStatus.READY,
    )
    db_session.add(doc)
    db_session.commit()
    return doc


def test_chunk_metadata_registered() -> None:
    """Verify Chunk model is registered in Base.metadata with expected columns and constraints."""
    assert "chunks" in Base.metadata.tables
    table = Base.metadata.tables["chunks"]

    column_names = {col.name for col in table.columns}
    expected_columns = {
        "id",
        "document_id",
        "chunk_index",
        "text",
        "token_count",
        "character_count",
        "start_char_offset",
        "end_char_offset",
        "source_format",
        "page_number",
        "page_numbers",
        "heading",
        "heading_level",
        "metadata",
        "created_at",
        "updated_at",
    }
    assert expected_columns.issubset(column_names)

    # Check primary key
    assert table.columns["id"].primary_key is True

    # Check nullability
    assert table.columns["document_id"].nullable is False
    assert table.columns["chunk_index"].nullable is False
    assert table.columns["text"].nullable is False
    assert table.columns["token_count"].nullable is False
    assert table.columns["character_count"].nullable is False
    assert table.columns["start_char_offset"].nullable is False
    assert table.columns["end_char_offset"].nullable is False
    assert table.columns["source_format"].nullable is False
    assert table.columns["page_number"].nullable is True
    assert table.columns["page_numbers"].nullable is False
    assert table.columns["heading"].nullable is True
    assert table.columns["heading_level"].nullable is True
    assert table.columns["metadata"].nullable is False
    assert table.columns["created_at"].nullable is False
    assert table.columns["updated_at"].nullable is False

    # Check foreign keys
    foreign_keys = list(table.foreign_keys)
    assert len(foreign_keys) == 1
    fk = foreign_keys[0]
    assert fk.column.table.name == "documents"
    assert fk.column.name == "id"
    assert fk.ondelete == "CASCADE"

    # Check indexes
    index_columns = {col.name for idx in table.indexes for col in idx.columns}
    assert "document_id" in index_columns
    assert "created_at" in index_columns

    # Check unique constraint on (document_id, chunk_index)
    unique_constraints = [
        c for c in table.constraints if isinstance(c, sa.UniqueConstraint)
    ]
    unique_col_sets = [{col.name for col in c.columns} for c in unique_constraints]
    assert {"document_id", "chunk_index"} in unique_col_sets

    # Check check constraints
    check_names = {c.name for c in table.constraints if isinstance(c, sa.CheckConstraint)}
    assert "check_chunk_index_non_negative" in check_names
    assert "check_chunk_token_count_non_negative" in check_names
    assert "check_chunk_character_count_non_negative" in check_names
    assert "check_chunk_start_char_offset_non_negative" in check_names
    assert "check_chunk_end_char_offset_valid" in check_names
    assert "check_chunk_offset_span_equals_character_count" in check_names
    assert "check_chunk_page_number_positive" in check_names
    assert "check_chunk_heading_level_valid" in check_names


def test_create_and_retrieve_chunk(db_session: Session) -> None:
    """Verify inserting and querying a valid Chunk with default and custom values."""
    doc = _create_sample_document(db_session)

    chunk = Chunk(
        document_id=doc.id,
        chunk_index=0,
        text="Sample chunk content.",
        token_count=4,
        character_count=21,
        start_char_offset=0,
        end_char_offset=21,
        source_format="application/pdf",
        page_number=1,
        page_numbers=[1],
        heading="Introduction",
        heading_level=1,
        metadata_={"parser": "pdf"},
    )
    db_session.add(chunk)
    db_session.commit()

    retrieved = db_session.get(Chunk, chunk.id)
    assert retrieved is not None
    assert isinstance(retrieved.id, uuid.UUID)
    assert retrieved.document_id == doc.id
    assert retrieved.chunk_index == 0
    assert retrieved.text == "Sample chunk content."
    assert retrieved.token_count == 4
    assert retrieved.character_count == 21
    assert retrieved.start_char_offset == 0
    assert retrieved.end_char_offset == 21
    assert retrieved.source_format == "application/pdf"
    assert retrieved.page_number == 1
    assert retrieved.page_numbers == [1]
    assert retrieved.heading == "Introduction"
    assert retrieved.heading_level == 1
    assert retrieved.metadata_ == {"parser": "pdf"}
    assert isinstance(retrieved.created_at, datetime)
    assert isinstance(retrieved.updated_at, datetime)

    # Check relationship navigation
    assert retrieved.document.id == doc.id
    assert len(doc.chunks) == 1
    assert doc.chunks[0].id == chunk.id


def test_chunk_unique_constraint(db_session: Session) -> None:
    """Verify unique constraint on (document_id, chunk_index) rejects duplicate indices for same document."""
    doc = _create_sample_document(db_session)

    chunk1 = Chunk(
        document_id=doc.id,
        chunk_index=0,
        text="Chunk 0",
        token_count=2,
        character_count=7,
        start_char_offset=0,
        end_char_offset=7,
    )
    chunk2 = Chunk(
        document_id=doc.id,
        chunk_index=0,  # Duplicate index
        text="Another Chunk 0",
        token_count=3,
        character_count=15,
        start_char_offset=8,
        end_char_offset=23,
    )
    db_session.add(chunk1)
    db_session.commit()

    db_session.add(chunk2)
    with pytest.raises(IntegrityError):
        db_session.commit()
    db_session.rollback()


def test_chunk_cascade_deletion(db_session: Session) -> None:
    """Verify deleting a Document cascades and deletes its associated Chunks."""
    doc = _create_sample_document(db_session)

    chunk1 = Chunk(
        document_id=doc.id,
        chunk_index=0,
        text="First passage",
        token_count=2,
        character_count=13,
        start_char_offset=0,
        end_char_offset=13,
    )
    chunk2 = Chunk(
        document_id=doc.id,
        chunk_index=1,
        text="Second passage",
        token_count=2,
        character_count=14,
        start_char_offset=14,
        end_char_offset=28,
    )
    db_session.add_all([chunk1, chunk2])
    db_session.commit()

    chunk1_id = chunk1.id
    chunk2_id = chunk2.id

    # Delete the document
    db_session.delete(doc)
    db_session.commit()

    # Both chunks should have been cascaded and deleted
    assert db_session.get(Chunk, chunk1_id) is None
    assert db_session.get(Chunk, chunk2_id) is None


def test_chunk_check_constraints_rejected(sqlite_engine: sa.Engine) -> None:
    """Verify check constraints reject negative or invalid numeric fields."""
    table = Base.metadata.tables["chunks"]

    # 1. Negative chunk_index
    with sqlite_engine.begin() as conn:
        with pytest.raises(IntegrityError):
            conn.execute(
                table.insert().values(
                    id=uuid.uuid4(),
                    document_id=uuid.uuid4(),
                    chunk_index=-1,
                    text="Invalid",
                    token_count=1,
                    character_count=7,
                    start_char_offset=0,
                    end_char_offset=7,
                )
            )

    # 2. Negative token_count
    with sqlite_engine.begin() as conn:
        with pytest.raises(IntegrityError):
            conn.execute(
                table.insert().values(
                    id=uuid.uuid4(),
                    document_id=uuid.uuid4(),
                    chunk_index=0,
                    text="Invalid",
                    token_count=-5,
                    character_count=7,
                    start_char_offset=0,
                    end_char_offset=7,
                )
            )

    # 3. Negative character_count
    with sqlite_engine.begin() as conn:
        with pytest.raises(IntegrityError):
            conn.execute(
                table.insert().values(
                    id=uuid.uuid4(),
                    document_id=uuid.uuid4(),
                    chunk_index=0,
                    text="Invalid",
                    token_count=1,
                    character_count=-1,
                    start_char_offset=0,
                    end_char_offset=7,
                )
            )

    # 4. Invalid offsets (end < start)
    with sqlite_engine.begin() as conn:
        with pytest.raises(IntegrityError):
            conn.execute(
                table.insert().values(
                    id=uuid.uuid4(),
                    document_id=uuid.uuid4(),
                    chunk_index=0,
                    text="Invalid",
                    token_count=1,
                    character_count=7,
                    start_char_offset=10,
                    end_char_offset=5,
                )
            )

    # 5. Zero page_number (must be >= 1)
    with sqlite_engine.begin() as conn:
        with pytest.raises(IntegrityError):
            conn.execute(
                table.insert().values(
                    id=uuid.uuid4(),
                    document_id=uuid.uuid4(),
                    chunk_index=0,
                    text="Invalid",
                    token_count=1,
                    character_count=7,
                    start_char_offset=0,
                    end_char_offset=7,
                    page_number=0,
                )
            )

    # 6. Zero heading_level (must be >= 1)
    with sqlite_engine.begin() as conn:
        with pytest.raises(IntegrityError):
            conn.execute(
                table.insert().values(
                    id=uuid.uuid4(),
                    document_id=uuid.uuid4(),
                    chunk_index=0,
                    text="Invalid",
                    token_count=1,
                    character_count=7,
                    start_char_offset=0,
                    end_char_offset=7,
                    heading_level=0,
                )
            )

    # 7. Heading level exceeding maximum of 6 (must be <= 6)
    with sqlite_engine.begin() as conn:
        with pytest.raises(IntegrityError):
            conn.execute(
                table.insert().values(
                    id=uuid.uuid4(),
                    document_id=uuid.uuid4(),
                    chunk_index=0,
                    text="Invalid",
                    token_count=1,
                    character_count=7,
                    start_char_offset=0,
                    end_char_offset=7,
                    heading_level=7,
                )
            )

    # 8. Offset span inconsistent with character_count (end - start != char_count)
    with sqlite_engine.begin() as conn:
        with pytest.raises(IntegrityError):
            conn.execute(
                table.insert().values(
                    id=uuid.uuid4(),
                    document_id=uuid.uuid4(),
                    chunk_index=0,
                    text="Invalid",
                    token_count=1,
                    character_count=7,
                    start_char_offset=0,
                    end_char_offset=15,  # span is 15, char_count is 7
                )
            )


def test_chunk_heading_level_boundaries(db_session: Session) -> None:
    """Verify heading_level supports valid boundaries 1 and 6, as well as NULL."""
    doc = _create_sample_document(db_session)

    # Boundary: 1 (H1)
    chunk_h1 = Chunk(
        document_id=doc.id,
        chunk_index=0,
        text="H1 chunk",
        token_count=2,
        character_count=8,
        start_char_offset=0,
        end_char_offset=8,
        heading="Main Title",
        heading_level=1,
    )
    # Boundary: 6 (H6)
    chunk_h6 = Chunk(
        document_id=doc.id,
        chunk_index=1,
        text="H6 chunk",
        token_count=2,
        character_count=8,
        start_char_offset=8,
        end_char_offset=16,
        heading="Sub Sub Section",
        heading_level=6,
    )
    # Nullable heading level
    chunk_no_heading = Chunk(
        document_id=doc.id,
        chunk_index=2,
        text="No head.",
        token_count=2,
        character_count=8,
        start_char_offset=16,
        end_char_offset=24,
        heading=None,
        heading_level=None,
    )
    db_session.add_all([chunk_h1, chunk_h6, chunk_no_heading])
    db_session.commit()

    retrieved_h1 = db_session.get(Chunk, chunk_h1.id)
    retrieved_h6 = db_session.get(Chunk, chunk_h6.id)
    retrieved_none = db_session.get(Chunk, chunk_no_heading.id)

    assert retrieved_h1 is not None and retrieved_h1.heading_level == 1
    assert retrieved_h6 is not None and retrieved_h6.heading_level == 6
    assert retrieved_none is not None and retrieved_none.heading_level is None


def test_direct_database_sql_cascade_deletion(sqlite_engine: sa.Engine) -> None:
    """Verify database-level ON DELETE CASCADE removes chunks when parent document is deleted via raw SQL."""
    doc_id = uuid.uuid4()
    chunk_id1 = uuid.uuid4()
    chunk_id2 = uuid.uuid4()

    doc_table = Base.metadata.tables["documents"]
    chunk_table = Base.metadata.tables["chunks"]

    with sqlite_engine.begin() as conn:
        conn.execute(
            doc_table.insert().values(
                id=doc_id,
                original_filename="raw_test.pdf",
                content_type="application/pdf",
                size_bytes=500,
                sha256="d" * 64,
                storage_key="raw/raw_test.pdf",
                status="ready",
            )
        )
        conn.execute(
            chunk_table.insert().values(
                [
                    {
                        "id": chunk_id1,
                        "document_id": doc_id,
                        "chunk_index": 0,
                        "text": "Part 1",
                        "token_count": 2,
                        "character_count": 6,
                        "start_char_offset": 0,
                        "end_char_offset": 6,
                        "page_numbers": "[]",
                        "metadata": "{}",
                    },
                    {
                        "id": chunk_id2,
                        "document_id": doc_id,
                        "chunk_index": 1,
                        "text": "Part 2",
                        "token_count": 2,
                        "character_count": 6,
                        "start_char_offset": 6,
                        "end_char_offset": 12,
                        "page_numbers": "[]",
                        "metadata": "{}",
                    },
                ]
            )
        )

        # Verify chunks exist before deletion
        pre_chunks = conn.execute(
            sa.select(chunk_table).where(chunk_table.c.document_id == doc_id)
        ).fetchall()
        assert len(pre_chunks) == 2

        # Execute direct SQL DELETE on documents table bypassing ORM
        conn.execute(sa.delete(doc_table).where(doc_table.c.id == doc_id))

        # Assert foreign key cascade deleted all child chunks
        post_chunks = conn.execute(
            sa.select(chunk_table).where(chunk_table.c.document_id == doc_id)
        ).fetchall()
        assert len(post_chunks) == 0


def test_chunk_unicode_and_nested_metadata(db_session: Session) -> None:
    """Verify persisting and querying Unicode text, Unicode headings, and deeply nested JSON metadata."""
    doc = _create_sample_document(db_session)

    unicode_text = "第1章：人工知能の発展と倫理 — Café & Naïve résumé."
    unicode_heading = "第1章: 概要 (Overview)"
    nested_metadata = {
        "source": "日本語テキスト",
        "nested": {
            "hierarchy": {"depth": 3, "tags": ["nlp", "evaluation", "精度"]},
            "layout": {"bounding_boxes": [[10.5, 20.2], [30.8, 40.1]]},
        },
    }

    chunk = Chunk(
        document_id=doc.id,
        chunk_index=0,
        text=unicode_text,
        token_count=18,
        character_count=len(unicode_text),
        start_char_offset=0,
        end_char_offset=len(unicode_text),
        source_format="text/markdown",
        heading=unicode_heading,
        heading_level=2,
        page_numbers=[1, 2, 3],
        metadata_=nested_metadata,
    )
    db_session.add(chunk)
    db_session.commit()

    retrieved = db_session.get(Chunk, chunk.id)
    assert retrieved is not None
    assert retrieved.text == unicode_text
    assert retrieved.heading == unicode_heading
    assert retrieved.page_numbers == [1, 2, 3]
    assert retrieved.metadata_ == nested_metadata
    assert retrieved.metadata_["nested"]["hierarchy"]["tags"][2] == "精度"



def test_chunk_repr() -> None:
    """Verify Chunk string representation contains diagnostic attributes."""
    chunk_id = uuid.uuid4()
    doc_id = uuid.uuid4()
    chunk = Chunk(
        id=chunk_id,
        document_id=doc_id,
        chunk_index=2,
        text="Representation test.",
        token_count=3,
        character_count=20,
        start_char_offset=50,
        end_char_offset=70,
    )
    repr_str = repr(chunk)
    assert str(chunk_id) in repr_str
    assert str(doc_id) in repr_str
    assert "chunk_index=2" in repr_str
    assert "char_span=(50:70)" in repr_str


def test_alembic_migration_0003_metadata() -> None:
    """Verify Alembic migration 0003 has correct revision identifier and dependency on 0002."""
    migration_file = (
        Path(__file__).resolve().parents[1]
        / "alembic"
        / "versions"
        / "0003_create_chunks_table.py"
    )
    assert migration_file.exists(), f"Migration file not found at {migration_file}"

    spec = importlib.util.spec_from_file_location("migration_0003", migration_file)
    assert spec is not None and spec.loader is not None
    mod = importlib.util.module_from_spec(spec)
    spec.loader.exec_module(mod)

    assert mod.revision == "0003_create_chunks_table"
    assert mod.down_revision == "0002_create_documents_table"
    assert hasattr(mod, "upgrade")
    assert hasattr(mod, "downgrade")


def test_alembic_migration_0003_upgrade_and_downgrade() -> None:
    """Verify migration 0003 upgrade and downgrade execute cleanly."""
    migration_file_0002 = (
        Path(__file__).resolve().parents[1]
        / "alembic"
        / "versions"
        / "0002_create_documents_table.py"
    )
    migration_file_0003 = (
        Path(__file__).resolve().parents[1]
        / "alembic"
        / "versions"
        / "0003_create_chunks_table.py"
    )

    spec_0002 = importlib.util.spec_from_file_location("migration_0002_mod", migration_file_0002)
    assert spec_0002 is not None and spec_0002.loader is not None
    mod_0002 = importlib.util.module_from_spec(spec_0002)
    spec_0002.loader.exec_module(mod_0002)

    spec_0003 = importlib.util.spec_from_file_location("migration_0003_mod", migration_file_0003)
    assert spec_0003 is not None and spec_0003.loader is not None
    mod_0003 = importlib.util.module_from_spec(spec_0003)
    spec_0003.loader.exec_module(mod_0003)

    engine = sa.create_engine("sqlite:///:memory:")
    with engine.begin() as conn:
        ctx = MigrationContext.configure(conn)
        with Operations.context(ctx):
            # Setup prerequisite: documents table from 0002
            mod_0002.upgrade()

            # Upgrade: creates chunks table, constraints, and indexes
            mod_0003.upgrade()
            inspector = sa.inspect(conn)
            tables = inspector.get_table_names()
            assert "chunks" in tables

            columns = {col["name"]: col for col in inspector.get_columns("chunks")}
            assert "id" in columns
            assert "document_id" in columns
            assert "chunk_index" in columns
            assert "text" in columns
            assert "token_count" in columns
            assert "character_count" in columns
            assert "start_char_offset" in columns
            assert "end_char_offset" in columns
            assert "source_format" in columns
            assert "page_number" in columns
            assert "page_numbers" in columns
            assert "heading" in columns
            assert "heading_level" in columns
            assert "metadata" in columns
            assert "created_at" in columns
            assert "updated_at" in columns

            # Downgrade: drops chunks table and indexes
            mod_0003.downgrade()
            inspector = sa.inspect(conn)
            assert "chunks" not in inspector.get_table_names()
            # Parent table remains intact
            assert "documents" in inspector.get_table_names()
