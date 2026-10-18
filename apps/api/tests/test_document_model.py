"""Tests for Document persistence model, constraints, and migration schema.

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
from app.models.document import Document, DocumentStatus


@pytest.fixture
def sqlite_engine() -> sa.Engine:
    """Provide an in-memory SQLite engine with table schema created."""
    engine = sa.create_engine("sqlite:///:memory:")
    Base.metadata.create_all(engine)
    return engine


@pytest.fixture
def db_session(sqlite_engine: sa.Engine) -> Session:
    """Provide a scoped database session for model testing."""
    with Session(sqlite_engine) as session:
        yield session


def test_document_metadata_registered() -> None:
    """Verify Document model is registered in Base.metadata with expected columns and constraints."""
    assert "documents" in Base.metadata.tables
    table = Base.metadata.tables["documents"]

    column_names = {col.name for col in table.columns}
    expected_columns = {
        "id",
        "original_filename",
        "content_type",
        "size_bytes",
        "sha256",
        "storage_key",
        "status",
        "created_at",
        "updated_at",
    }
    assert expected_columns.issubset(column_names)

    # Check primary key
    assert table.columns["id"].primary_key is True

    # Check nullability
    assert table.columns["original_filename"].nullable is False
    assert table.columns["content_type"].nullable is False
    assert table.columns["size_bytes"].nullable is False
    assert table.columns["sha256"].nullable is False
    assert table.columns["storage_key"].nullable is False
    assert table.columns["status"].nullable is False
    assert table.columns["created_at"].nullable is False
    assert table.columns["updated_at"].nullable is False

    # Check unique constraint on storage_key
    assert table.columns["storage_key"].unique is True

    # Check indexes on sha256 and status
    index_columns = {col.name for idx in table.indexes for col in idx.columns}
    assert "sha256" in index_columns
    assert "status" in index_columns
    assert "created_at" in index_columns

    # Check named check constraints
    constraint_names = {c.name for c in table.constraints if isinstance(c, sa.CheckConstraint)}
    assert "check_document_size_non_negative" in constraint_names
    assert "check_document_status_valid" in constraint_names


def test_create_and_retrieve_document(db_session: Session) -> None:
    """Verify inserting and querying a valid document works with default values."""
    doc = Document(
        original_filename="sample_guide.pdf",
        content_type="application/pdf",
        size_bytes=2048576,
        sha256="e3b0c44298fc1c149afbf4c8996fb92427ae41e4649b934ca495991b7852b855",
        storage_key="raw/2026/10/guide.pdf",
    )
    db_session.add(doc)
    db_session.commit()

    retrieved = db_session.get(Document, doc.id)
    assert retrieved is not None
    assert isinstance(retrieved.id, uuid.UUID)
    assert retrieved.original_filename == "sample_guide.pdf"
    assert retrieved.content_type == "application/pdf"
    assert retrieved.size_bytes == 2048576
    assert retrieved.sha256 == "e3b0c44298fc1c149afbf4c8996fb92427ae41e4649b934ca495991b7852b855"
    assert retrieved.storage_key == "raw/2026/10/guide.pdf"
    assert retrieved.status == DocumentStatus.UPLOADED
    assert isinstance(retrieved.created_at, datetime)
    assert isinstance(retrieved.updated_at, datetime)


def test_custom_uuid_and_lifecycle_statuses(db_session: Session) -> None:
    """Verify custom UUID and distinct lifecycle statuses can be persisted."""
    custom_id = uuid.uuid4()
    doc = Document(
        id=custom_id,
        original_filename="notes.md",
        content_type="text/markdown",
        size_bytes=512,
        sha256="1234567890abcdef1234567890abcdef1234567890abcdef1234567890abcdef",
        storage_key="raw/notes.md",
        status=DocumentStatus.PROCESSING,
    )
    db_session.add(doc)
    db_session.commit()

    retrieved = db_session.get(Document, custom_id)
    assert retrieved is not None
    assert retrieved.id == custom_id
    assert retrieved.status == DocumentStatus.PROCESSING

    # Transition to READY
    retrieved.status = DocumentStatus.READY
    db_session.commit()

    updated = db_session.get(Document, custom_id)
    assert updated is not None
    assert updated.status == DocumentStatus.READY


def test_updated_at_advancement_on_mutation(db_session: Session) -> None:
    """Verify created_at remains immutable while updated_at is refreshed on update."""
    doc = Document(
        original_filename="initial.txt",
        content_type="text/plain",
        size_bytes=100,
        sha256="0" * 64,
        storage_key="raw/initial.txt",
        status=DocumentStatus.UPLOADED,
    )
    db_session.add(doc)
    db_session.commit()

    initial_created = doc.created_at
    initial_updated = doc.updated_at
    assert initial_created is not None
    assert initial_updated is not None

    # Mutate a field and commit
    doc.status = DocumentStatus.READY
    db_session.commit()
    db_session.refresh(doc)

    # created_at must remain constant across mutations
    assert doc.created_at == initial_created
    # updated_at must be present and >= initial_updated
    assert doc.updated_at is not None
    assert doc.updated_at >= initial_updated


def test_zero_size_document_allowed(db_session: Session) -> None:
    """Verify that a zero-byte document satisfies the non-negative check constraint."""
    doc = Document(
        original_filename="empty.txt",
        content_type="text/plain",
        size_bytes=0,
        sha256="e3b0c44298fc1c149afbf4c8996fb92427ae41e4649b934ca495991b7852b855",
        storage_key="raw/empty.txt",
    )
    db_session.add(doc)
    db_session.commit()

    retrieved = db_session.get(Document, doc.id)
    assert retrieved is not None
    assert retrieved.size_bytes == 0


def test_negative_size_bytes_rejected(db_session: Session) -> None:
    """Verify database-level check constraint rejects negative size_bytes."""
    doc = Document(
        original_filename="corrupted.pdf",
        content_type="application/pdf",
        size_bytes=-100,
        sha256="0" * 64,
        storage_key="raw/corrupted.pdf",
    )
    db_session.add(doc)
    with pytest.raises(IntegrityError):
        db_session.commit()
    db_session.rollback()


def test_invalid_lifecycle_status_rejected_at_db_level(sqlite_engine: sa.Engine) -> None:
    """Verify database-level check constraint rejects unapproved lifecycle status values."""
    table = Base.metadata.tables["documents"]
    with sqlite_engine.begin() as conn:
        with pytest.raises(IntegrityError):
            conn.execute(
                table.insert().values(
                    id=uuid.uuid4(),
                    original_filename="bad_status.txt",
                    content_type="text/plain",
                    size_bytes=100,
                    sha256="0" * 64,
                    storage_key="raw/bad_status.txt",
                    status="unsupported_status",
                )
            )


def test_storage_key_unique_constraint(db_session: Session) -> None:
    """Verify unique constraint on storage_key prevents collision."""
    doc1 = Document(
        original_filename="doc1.pdf",
        content_type="application/pdf",
        size_bytes=100,
        sha256="a" * 64,
        storage_key="raw/same_key.pdf",
    )
    doc2 = Document(
        original_filename="doc2.pdf",
        content_type="application/pdf",
        size_bytes=200,
        sha256="b" * 64,
        storage_key="raw/same_key.pdf",
    )
    db_session.add(doc1)
    db_session.commit()

    db_session.add(doc2)
    with pytest.raises(IntegrityError):
        db_session.commit()
    db_session.rollback()


def test_required_fields_null_rejected(db_session: Session) -> None:
    """Verify non-null constraints reject missing required values."""
    doc = Document(
        original_filename=None,  # type: ignore[arg-type]
        content_type="text/plain",
        size_bytes=100,
        sha256="a" * 64,
        storage_key="raw/missing_name.txt",
    )
    db_session.add(doc)
    with pytest.raises(IntegrityError):
        db_session.commit()
    db_session.rollback()


def test_document_repr() -> None:
    """Verify Document string representation contains key diagnostic attributes."""
    doc_id = uuid.uuid4()
    doc = Document(
        id=doc_id,
        original_filename="specs.pdf",
        content_type="application/pdf",
        size_bytes=4096,
        sha256="c" * 64,
        storage_key="raw/specs.pdf",
        status=DocumentStatus.READY,
    )
    repr_str = repr(doc)
    assert str(doc_id) in repr_str
    assert "specs.pdf" in repr_str
    assert "ready" in repr_str
    assert "4096" in repr_str


def test_alembic_migration_metadata() -> None:
    """Verify Alembic migration 0002 has correct revision identifier and dependency on 0001."""
    migration_file = (
        Path(__file__).resolve().parents[1]
        / "alembic"
        / "versions"
        / "0002_create_documents_table.py"
    )
    assert migration_file.exists(), f"Migration file not found at {migration_file}"

    spec = importlib.util.spec_from_file_location("migration_0002", migration_file)
    assert spec is not None and spec.loader is not None
    mod = importlib.util.module_from_spec(spec)
    spec.loader.exec_module(mod)

    assert mod.revision == "0002_create_documents_table"
    assert mod.down_revision == "0001_enable_vector"
    assert hasattr(mod, "upgrade")
    assert hasattr(mod, "downgrade")


def test_alembic_migration_upgrade_and_downgrade() -> None:
    """Verify migration upgrade and downgrade execute cleanly on a fresh database connection."""
    migration_file = (
        Path(__file__).resolve().parents[1]
        / "alembic"
        / "versions"
        / "0002_create_documents_table.py"
    )
    spec = importlib.util.spec_from_file_location("migration_0002_exec", migration_file)
    assert spec is not None and spec.loader is not None
    mod = importlib.util.module_from_spec(spec)
    spec.loader.exec_module(mod)

    engine = sa.create_engine("sqlite:///:memory:")
    with engine.begin() as conn:
        ctx = MigrationContext.configure(conn)
        with Operations.context(ctx):
            # Test upgrade: creates documents table and indexes
            mod.upgrade()
            inspector = sa.inspect(conn)
            tables = inspector.get_table_names()
            assert "documents" in tables

            columns = {col["name"]: col for col in inspector.get_columns("documents")}
            assert "id" in columns
            assert "original_filename" in columns
            assert "content_type" in columns
            assert "size_bytes" in columns
            assert "sha256" in columns
            assert "storage_key" in columns
            assert "status" in columns
            assert "created_at" in columns
            assert "updated_at" in columns

            # Test downgrade: drops documents table and indexes
            mod.downgrade()
            inspector = sa.inspect(conn)
            assert "documents" not in inspector.get_table_names()
