"""ChunkEmbedding persistence model for vector representations of document chunks."""

from datetime import datetime
from typing import TYPE_CHECKING
import uuid

from pgvector.sqlalchemy import Vector
import sqlalchemy as sa
from sqlalchemy.orm import Mapped, mapped_column, relationship

from app.db.base import Base

if TYPE_CHECKING:
    from app.models.chunk import Chunk
    from app.models.document import Document


class ChunkEmbedding(Base):
    """Represents a dense vector embedding generated from a document chunk passage.

    Associated with a parent Chunk record and parent Document record. Deletion of the parent
    document or chunk cascades to all associated embeddings.
    """

    __tablename__ = "chunk_embeddings"

    id: Mapped[uuid.UUID] = mapped_column(
        sa.UUID(as_uuid=True),
        primary_key=True,
        default=uuid.uuid4,
        doc="Primary UUID identifier for the embedding.",
    )
    chunk_id: Mapped[uuid.UUID] = mapped_column(
        sa.UUID(as_uuid=True),
        sa.ForeignKey("chunks.id", ondelete="CASCADE", name="fk_chunk_embeddings_chunk_id_chunks"),
        nullable=False,
        index=True,
        doc="Foreign key to the parent Chunk.",
    )
    document_id: Mapped[uuid.UUID] = mapped_column(
        sa.UUID(as_uuid=True),
        sa.ForeignKey("documents.id", ondelete="CASCADE", name="fk_chunk_embeddings_document_id_documents"),
        nullable=False,
        index=True,
        doc="Foreign key to the parent Document.",
    )
    model_id: Mapped[str] = mapped_column(
        sa.String(100),
        nullable=False,
        index=True,
        doc="Canonical identifier of the model that generated this vector.",
    )
    dimension: Mapped[int] = mapped_column(
        sa.Integer,
        nullable=False,
        doc="Dimensionality of the embedding vector (must be 384).",
    )
    embedding: Mapped[list[float]] = mapped_column(
        sa.JSON().with_variant(Vector(384), "postgresql"),
        nullable=False,
        doc="Unit-normalized float vector embedding.",
    )
    created_at: Mapped[datetime] = mapped_column(
        sa.DateTime(timezone=True),
        server_default=sa.func.now(),
        nullable=False,
        index=True,
        doc="Timezone-aware timestamp recording when the embedding was created.",
    )

    # Relationships
    chunk: Mapped["Chunk"] = relationship(
        "Chunk",
        back_populates="embeddings",
    )
    document: Mapped["Document"] = relationship(
        "Document",
    )

    __table_args__ = (
        sa.UniqueConstraint(
            "chunk_id",
            "model_id",
            name="uq_chunk_embeddings_chunk_model",
        ),
        sa.CheckConstraint(
            "dimension = 384",
            name="check_chunk_embedding_dimension_384",
        ),
    )

    def __repr__(self) -> str:
        return (
            f"<ChunkEmbedding(id={self.id}, chunk_id={self.chunk_id}, "
            f"model_id='{self.model_id}', dimension={self.dimension})>"
        )
