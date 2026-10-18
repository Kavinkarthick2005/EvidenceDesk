"""Embedding service coordinating vector generation and persistence."""

from collections.abc import Sequence
import logging
import uuid

import sqlalchemy as sa
from sqlalchemy.exc import OperationalError
from sqlalchemy.orm import Session

from app.core.config import Settings, get_settings
from app.embeddings import EmbeddingProvider, get_embedding_provider
from app.embeddings.exceptions import (
    EmbeddingDimensionMismatchError,
    EmbeddingError,
    EmbeddingInferenceError,
)
from app.models.chunk import Chunk as ChunkModel
from app.models.embedding import ChunkEmbedding
from app.services.exceptions import ChunkPersistenceError

logger = logging.getLogger(__name__)


class EmbeddingService:
    """Application service coordinating chunk vector generation and persistence."""

    def __init__(
        self,
        db: Session,
        provider: EmbeddingProvider | None = None,
        settings: Settings | None = None,
    ) -> None:
        """Initialize EmbeddingService with dependencies.

        Args:
            db: Scoped SQLAlchemy database session.
            provider: Optional embedding provider instance; defaults to configured singleton.
            settings: Optional application settings.
        """
        self.db = db
        self.settings = settings or get_settings()
        self.provider = provider or get_embedding_provider(self.settings)

    def generate_embeddings(self, texts: Sequence[str]) -> list[list[float]]:
        """Generate unit-normalized float vectors for a list of chunk texts in memory.

        Performs pure in-memory compute without holding database transactions or locks.

        Args:
            texts: Sequence of chunk texts to embed.

        Returns:
            List of float vectors, where len(vector) == self.provider.dimension.

        Raises:
            EmbeddingInferenceError: If model inference fails.
            EmbeddingDimensionMismatchError: If generated vector dimension does not match schema.
        """
        if not texts:
            return []

        vectors = self.provider.embed_texts(texts)

        if len(vectors) != len(texts):
            raise EmbeddingInferenceError(
                f"Provider returned {len(vectors)} vectors for {len(texts)} text inputs."
            )

        expected_dim = self.provider.dimension
        for idx, vec in enumerate(vectors):
            if len(vec) != expected_dim:
                raise EmbeddingDimensionMismatchError(
                    f"Vector at index {idx} has dimension {len(vec)}, expected {expected_dim}."
                )

        return vectors

    def store_embeddings(
        self,
        document_id: uuid.UUID | str,
        chunks: Sequence[ChunkModel],
        vectors: Sequence[list[float]],
        commit: bool = True,
    ) -> list[ChunkEmbedding]:
        """Stage and optionally commit ChunkEmbedding records for persisted chunks.

        Args:
            document_id: Parent document UUID.
            chunks: Sequence of persisted or flushed Chunk ORM records.
            vectors: Corresponding pre-computed float vectors matching chunks.
            commit: If True, commits the transaction. If False (default for coordinated ingestion),
                flushes records to enforce constraints and populate IDs without committing.

        Returns:
            List of ChunkEmbedding ORM instances.

        Raises:
            ChunkPersistenceError: If flush or commit fails.
        """
        doc_uuid = uuid.UUID(str(document_id)) if not isinstance(document_id, uuid.UUID) else document_id

        if len(chunks) != len(vectors):
            raise ChunkPersistenceError(
                f"Mismatch between chunks ({len(chunks)}) and vectors ({len(vectors)}) for document '{doc_uuid}'."
            )

        model_id = self.provider.model_id
        dimension = self.provider.dimension

        if dimension != 384:
            raise EmbeddingDimensionMismatchError(
                f"Provider dimension {dimension} does not match required schema dimension 384."
            )

        orm_embeddings: list[ChunkEmbedding] = []
        for chunk, vector in zip(chunks, vectors, strict=True):
            embedding_record = ChunkEmbedding(
                chunk_id=chunk.id,
                document_id=doc_uuid,
                model_id=model_id,
                dimension=dimension,
                embedding=vector,
            )
            self.db.add(embedding_record)
            orm_embeddings.append(embedding_record)

        try:
            self.db.flush()
        except Exception as flush_err:
            try:
                self.db.rollback()
            except Exception:
                pass
            logger.error(
                "Embedding persistence failed during flush for document '%s'; transaction rolled back: %s",
                doc_uuid,
                flush_err,
            )
            raise ChunkPersistenceError(
                f"Database persistence failed during embedding flush for document '{doc_uuid}': {flush_err}"
            ) from flush_err

        if not commit:
            return orm_embeddings

        try:
            self.db.commit()
        except OperationalError as conn_err:
            try:
                self.db.rollback()
            except Exception:
                pass
            logger.error(
                "Database commit failed with ambiguous operational error while persisting embeddings for doc '%s': %s",
                doc_uuid,
                conn_err,
            )
            raise ChunkPersistenceError(
                f"Database commit failed with ambiguous connection error for document '{doc_uuid}': {conn_err}. "
                f"The transaction outcome is ambiguous and changes may have been committed."
            ) from conn_err
        except Exception as commit_err:
            try:
                self.db.rollback()
            except Exception:
                pass
            logger.error(
                "Database commit failed for embeddings on document '%s'; transaction rolled back: %s",
                doc_uuid,
                commit_err,
            )
            raise ChunkPersistenceError(
                f"Database persistence failed during embedding commit for document '{doc_uuid}': {commit_err}"
            ) from commit_err

        for record in orm_embeddings:
            try:
                self.db.refresh(record)
            except Exception as refresh_err:
                logger.warning(
                    "Post-commit refresh failed for embedding '%s': %s",
                    record.id,
                    refresh_err,
                )

        return orm_embeddings

    def get_embeddings_for_document(
        self,
        document_id: uuid.UUID | str,
        model_id: str | None = None,
    ) -> list[ChunkEmbedding]:
        """Retrieve all persisted embeddings for a document, optionally filtered by model_id."""
        doc_uuid = uuid.UUID(str(document_id)) if not isinstance(document_id, uuid.UUID) else document_id
        stmt = sa.select(ChunkEmbedding).where(ChunkEmbedding.document_id == doc_uuid)
        if model_id is not None:
            stmt = stmt.where(ChunkEmbedding.model_id == model_id)
        stmt = stmt.order_by(ChunkEmbedding.created_at)
        return list(self.db.scalars(stmt).all())
