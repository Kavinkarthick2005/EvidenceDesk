"""Document persistence model and lifecycle states for EvidenceDesk."""

from datetime import datetime
from enum import StrEnum
import uuid

from typing import TYPE_CHECKING

import sqlalchemy as sa
from sqlalchemy.orm import Mapped, mapped_column, relationship

from app.db.base import Base

if TYPE_CHECKING:
    from app.models.chunk import Chunk


class DocumentStatus(StrEnum):
    """Lifecycle states of an ingested document."""

    UPLOADED = "uploaded"
    PROCESSING = "processing"
    READY = "ready"
    FAILED = "failed"


class Document(Base):
    """Represents a persisted document tracked by EvidenceDesk.

    Physical file contents are stored externally in blob/object storage identified
    by `storage_key`. This model tracks file metadata, content hash, ingestion status,
    and timestamps.
    """

    __tablename__ = "documents"

    id: Mapped[uuid.UUID] = mapped_column(
        sa.UUID(as_uuid=True),
        primary_key=True,
        default=uuid.uuid4,
        doc="Primary UUID identifier for the document.",
    )
    original_filename: Mapped[str] = mapped_column(
        sa.String(255),
        nullable=False,
        doc="Original uploaded filename as provided by the client.",
    )
    content_type: Mapped[str] = mapped_column(
        sa.String(100),
        nullable=False,
        doc="Validated MIME media type (e.g., application/pdf, text/markdown, text/plain).",
    )
    size_bytes: Mapped[int] = mapped_column(
        sa.BigInteger,
        nullable=False,
        doc="Size of the raw document file in bytes (must be non-negative).",
    )
    sha256: Mapped[str] = mapped_column(
        sa.String(64),
        nullable=False,
        index=True,
        doc="Hex-encoded SHA-256 digest of file contents for deduplication and integrity verification.",
    )
    storage_key: Mapped[str] = mapped_column(
        sa.String(512),
        nullable=False,
        unique=True,
        doc="Unique internal storage path/key decoupling file persistence from untrusted filenames.",
    )
    status: Mapped[DocumentStatus] = mapped_column(
        sa.Enum(
            DocumentStatus,
            native_enum=False,
            length=32,
            create_constraint=False,
            values_callable=lambda obj: [e.value for e in obj],
        ),
        nullable=False,
        default=DocumentStatus.UPLOADED,
        index=True,
        doc="Current lifecycle state of the document.",
    )
    created_at: Mapped[datetime] = mapped_column(
        sa.DateTime(timezone=True),
        server_default=sa.func.now(),
        nullable=False,
        index=True,
        doc="Timezone-aware timestamp recording when the record was created.",
    )
    updated_at: Mapped[datetime] = mapped_column(
        sa.DateTime(timezone=True),
        server_default=sa.func.now(),
        onupdate=sa.func.now(),
        nullable=False,
        doc="Timezone-aware timestamp recording when the record was last modified.",
    )

    # Relationships
    chunks: Mapped[list["Chunk"]] = relationship(
        "Chunk",
        back_populates="document",
        cascade="all, delete-orphan",
        order_by="Chunk.chunk_index",
    )

    __table_args__ = (
        sa.CheckConstraint(
            "size_bytes >= 0",
            name="check_document_size_non_negative",
        ),
        sa.CheckConstraint(
            "status IN ('uploaded', 'processing', 'ready', 'failed')",
            name="check_document_status_valid",
        ),
    )

    def __repr__(self) -> str:
        return (
            f"<Document(id={self.id}, filename='{self.original_filename}', "
            f"status='{self.status}', size_bytes={self.size_bytes})>"
        )
