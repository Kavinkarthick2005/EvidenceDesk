"""Chunk persistence model for EvidenceDesk structure-aware chunks."""

from datetime import datetime
from typing import TYPE_CHECKING, Any
import uuid

import sqlalchemy as sa
from sqlalchemy.dialects import postgresql
from sqlalchemy.orm import Mapped, mapped_column, relationship

from app.db.base import Base

if TYPE_CHECKING:
    from app.models.document import Document


class Chunk(Base):
    """Represents a discrete, source-traceable passage extracted from a document.

    Associated with a parent Document record. Deletion of the document cascades
    to all associated chunks.
    """

    __tablename__ = "chunks"

    id: Mapped[uuid.UUID] = mapped_column(
        sa.UUID(as_uuid=True),
        primary_key=True,
        default=uuid.uuid4,
        doc="Primary UUID identifier for the chunk.",
    )
    document_id: Mapped[uuid.UUID] = mapped_column(
        sa.UUID(as_uuid=True),
        sa.ForeignKey("documents.id", ondelete="CASCADE", name="fk_chunks_document_id_documents"),
        nullable=False,
        index=True,
        doc="Foreign key to the parent Document.",
    )
    chunk_index: Mapped[int] = mapped_column(
        sa.Integer,
        nullable=False,
        doc="Zero-based sequential index of the chunk within the document.",
    )
    text: Mapped[str] = mapped_column(
        sa.Text,
        nullable=False,
        doc="Exact textual content of the chunk.",
    )
    token_count: Mapped[int] = mapped_column(
        sa.Integer,
        nullable=False,
        doc="Token count measured with the standard tokenizer.",
    )
    character_count: Mapped[int] = mapped_column(
        sa.Integer,
        nullable=False,
        doc="Character count of the chunk text.",
    )
    start_char_offset: Mapped[int] = mapped_column(
        sa.Integer,
        nullable=False,
        doc="Exact character offset where the chunk starts in the extracted source text.",
    )
    end_char_offset: Mapped[int] = mapped_column(
        sa.Integer,
        nullable=False,
        doc="Exact character offset where the chunk ends in the extracted source text.",
    )
    source_format: Mapped[str] = mapped_column(
        sa.String(100),
        nullable=False,
        default="text/plain",
        doc="MIME format of the source document from which chunk was extracted.",
    )
    page_number: Mapped[int | None] = mapped_column(
        sa.Integer,
        nullable=True,
        doc="Optional 1-indexed primary page number (for PDF sources).",
    )
    page_numbers: Mapped[list[int]] = mapped_column(
        sa.JSON().with_variant(postgresql.JSONB, "postgresql"),
        nullable=False,
        default=list,
        doc="List of 1-indexed page numbers spanned by the chunk.",
    )
    heading: Mapped[str | None] = mapped_column(
        sa.String(500),
        nullable=True,
        doc="Optional Markdown heading text enclosing this chunk.",
    )
    heading_level: Mapped[int | None] = mapped_column(
        sa.Integer,
        nullable=True,
        doc="Optional heading depth level (e.g., 1 for #, 2 for ##).",
    )
    metadata_: Mapped[dict[str, Any]] = mapped_column(
        "metadata",
        sa.JSON().with_variant(postgresql.JSONB, "postgresql"),
        nullable=False,
        default=dict,
        doc="Format-specific structural metadata dictionary.",
    )
    created_at: Mapped[datetime] = mapped_column(
        sa.DateTime(timezone=True),
        server_default=sa.func.now(),
        nullable=False,
        index=True,
        doc="Timezone-aware timestamp recording when the chunk record was created.",
    )
    updated_at: Mapped[datetime] = mapped_column(
        sa.DateTime(timezone=True),
        server_default=sa.func.now(),
        onupdate=sa.func.now(),
        nullable=False,
        doc="Timezone-aware timestamp recording when the chunk record was last modified.",
    )

    # Relationships
    document: Mapped["Document"] = relationship(
        "Document",
        back_populates="chunks",
    )

    __table_args__ = (
        sa.UniqueConstraint(
            "document_id",
            "chunk_index",
            name="uq_chunks_document_chunk_index",
        ),
        sa.CheckConstraint(
            "chunk_index >= 0",
            name="check_chunk_index_non_negative",
        ),
        sa.CheckConstraint(
            "token_count >= 0",
            name="check_chunk_token_count_non_negative",
        ),
        sa.CheckConstraint(
            "character_count >= 0",
            name="check_chunk_character_count_non_negative",
        ),
        sa.CheckConstraint(
            "start_char_offset >= 0",
            name="check_chunk_start_char_offset_non_negative",
        ),
        sa.CheckConstraint(
            "end_char_offset >= start_char_offset",
            name="check_chunk_end_char_offset_valid",
        ),
        sa.CheckConstraint(
            "end_char_offset - start_char_offset = character_count",
            name="check_chunk_offset_span_equals_character_count",
        ),
        sa.CheckConstraint(
            "page_number IS NULL OR page_number >= 1",
            name="check_chunk_page_number_positive",
        ),
        sa.CheckConstraint(
            "heading_level IS NULL OR (heading_level >= 1 AND heading_level <= 6)",
            name="check_chunk_heading_level_valid",
        ),
    )

    def __repr__(self) -> str:
        return (
            f"<Chunk(id={self.id}, document_id={self.document_id}, "
            f"chunk_index={self.chunk_index}, token_count={self.token_count}, "
            f"char_span=({self.start_char_offset}:{self.end_char_offset}))>"
        )
