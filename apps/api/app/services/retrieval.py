"""Semantic retrieval service for EvidenceDesk."""

from collections.abc import Sequence
import logging
import math
from typing import Any
import uuid

from pydantic import BaseModel
import sqlalchemy as sa
from sqlalchemy.ext.compiler import compiles
from sqlalchemy.orm import Session
from sqlalchemy.sql.expression import BinaryExpression, ColumnElement

from app.core.config import Settings, get_settings
from app.embeddings import EmbeddingProvider, get_embedding_provider
from app.embeddings.exceptions import EmbeddingError
from app.models.chunk import Chunk
from app.models.document import Document, DocumentStatus
from app.models.embedding import ChunkEmbedding
from app.schemas.retrieval import RetrievalResponse, RetrievedChunk
from app.services.exceptions import (
    InvalidQueryError,
    InvalidScoreThresholdError,
    InvalidTopKError,
    RetrievalError,
)

logger = logging.getLogger(__name__)


class CosineDistance(BinaryExpression):
    """Expression representing cosine distance between a vector column and a query vector.

    Compiles to native pgvector `<=>` operator on PostgreSQL:
        chunk_embeddings.embedding <=> :query_vector
    Compiles to `cosine_distance(chunk_embeddings.embedding, :query_vector)` on SQLite for fast unit testing.
    """

    inherit_cache = True

    def __init__(self, left: ColumnElement[Any], right: Any) -> None:
        right_clause = right if hasattr(right, "__clause_element__") else sa.literal(right)
        super().__init__(
            left,
            right_clause,
            sa.sql.operators.custom_op("<=>"),
            type_=sa.Float(),
        )


@compiles(CosineDistance, "postgresql")
def compile_cosine_distance_pg(element: CosineDistance, compiler: Any, **kw: Any) -> str:
    """Compile native pgvector <=> operator for PostgreSQL with explicit vector cast."""
    return f"{compiler.process(element.left, **kw)} <=> CAST({compiler.process(element.right, **kw)} AS vector)"


@compiles(CosineDistance, "sqlite")
def compile_cosine_distance_sqlite(element: CosineDistance, compiler: Any, **kw: Any) -> str:
    """Compile cosine_distance() user function for SQLite."""
    return f"cosine_distance({compiler.process(element.left, **kw)}, {compiler.process(element.right, **kw)})"


class RetrievalService:
    """Service providing semantic similarity search over ingested document chunks."""

    def __init__(
        self,
        db: Session,
        provider: EmbeddingProvider | None = None,
        settings: Settings | None = None,
    ) -> None:
        """Initialize RetrievalService with database session and embedding provider.

        Args:
            db: Scoped SQLAlchemy database session.
            provider: Optional embedding provider instance; defaults to configured singleton.
            settings: Optional application settings.
        """
        self.db = db
        self.settings = settings or get_settings()
        self.provider = provider or get_embedding_provider(self.settings)

    def retrieve(
        self,
        query: str,
        top_k: int = 5,
        document_ids: Sequence[uuid.UUID | str] | None = None,
        score_threshold: float | None = None,
    ) -> RetrievalResponse:
        """Retrieve ranked document chunks most semantically relevant to the input query.

        Args:
            query: Natural language query string (1 to 2,000 characters).
            top_k: Maximum candidate chunks to return (1 to 50, defaults to 5).
            document_ids: Optional document UUIDs to scope retrieval to.
            score_threshold: Optional minimum cosine similarity score in [-1.0, 1.0].

        Returns:
            RetrievalResponse containing the ranked list of RetrievedChunk instances.

        Raises:
            InvalidQueryError: If query is empty, whitespace-only, not a string, or > 2,000 characters.
            InvalidTopKError: If top_k is outside [1, 50] or not an integer.
            InvalidScoreThresholdError: If score_threshold is outside [-1.0, 1.0] or non-finite.
            RetrievalError: If query embedding or database search fails.
            ValueError: If an item in document_ids cannot be parsed as a UUID.
        """
        # Validate query
        if not isinstance(query, str):
            raise InvalidQueryError(f"Query must be a string, got {type(query).__name__}.")

        stripped_query = query.strip()
        if len(stripped_query) < 1:
            raise InvalidQueryError("Query cannot be empty or whitespace-only.")

        if len(stripped_query) > 2000:
            raise InvalidQueryError(
                f"Query length ({len(stripped_query)}) exceeds maximum permitted limit of 2,000 characters."
            )

        # Validate top_k (reject bool because bool subclasses int)
        if not isinstance(top_k, int) or isinstance(top_k, bool):
            raise InvalidTopKError(f"top_k must be an integer, got {type(top_k).__name__}.")

        if top_k < 1 or top_k > 50:
            raise InvalidTopKError(f"top_k ({top_k}) must be between 1 and 50.")

        # Validate score_threshold
        if score_threshold is not None:
            if not isinstance(score_threshold, (int, float)) or isinstance(score_threshold, bool):
                raise InvalidScoreThresholdError(
                    f"score_threshold must be a float or int, got {type(score_threshold).__name__}."
                )
            if not math.isfinite(score_threshold) or score_threshold < -1.0 or score_threshold > 1.0:
                raise InvalidScoreThresholdError(
                    f"score_threshold ({score_threshold}) must be a finite number between -1.0 and 1.0."
                )

        # Validate and normalize document_ids
        normalized_doc_ids: list[uuid.UUID] | None = None
        if document_ids is not None:
            normalized_doc_ids = []
            for item in document_ids:
                if isinstance(item, uuid.UUID):
                    normalized_doc_ids.append(item)
                else:
                    try:
                        normalized_doc_ids.append(uuid.UUID(str(item)))
                    except (ValueError, TypeError) as err:
                        raise ValueError(f"Invalid document UUID '{item}': {err}") from err

        # Generate query vector in memory
        try:
            query_vector = self.provider.embed_query(stripped_query)
        except EmbeddingError as emb_err:
            logger.error("Embedding generation failed for query: %s", emb_err)
            raise RetrievalError(f"Failed to generate query embedding: {emb_err}") from emb_err
        except Exception as generic_err:
            logger.error("Unexpected error during query embedding: %s", generic_err)
            raise RetrievalError(f"Unexpected query embedding failure: {generic_err}") from generic_err

        active_model_id = self.provider.model_id
        active_dimension = self.provider.dimension

        if len(query_vector) != active_dimension:
            raise RetrievalError(
                f"Query vector dimensionality ({len(query_vector)}) does not match provider dimension ({active_dimension})."
            )

        # Build single SQL query with joins and filtering
        dist_expr = CosineDistance(ChunkEmbedding.embedding, query_vector).label("distance")

        stmt = (
            sa.select(
                Chunk,
                Document.original_filename,
                dist_expr,
                ChunkEmbedding.model_id,
            )
            .join(Chunk, ChunkEmbedding.chunk_id == Chunk.id)
            .join(Document, ChunkEmbedding.document_id == Document.id)
            .where(
                Document.status == DocumentStatus.READY,
                ChunkEmbedding.model_id == active_model_id,
                ChunkEmbedding.dimension == active_dimension,
            )
        )

        if normalized_doc_ids is not None:
            stmt = stmt.where(Document.id.in_(normalized_doc_ids))

        if score_threshold is not None:
            # score = 1.0 - distance >= score_threshold  <=>  distance <= 1.0 - score_threshold
            max_distance = 1.0 - float(score_threshold)
            stmt = stmt.where(dist_expr <= max_distance)

        # Deterministic ranking: primary cosine distance ASC, secondary document_id and chunk_index ASC
        stmt = stmt.order_by(
            dist_expr.asc(),
            Chunk.document_id.asc(),
            Chunk.chunk_index.asc(),
        ).limit(top_k)

        try:
            rows = self.db.execute(stmt).all()
        except Exception as db_err:
            logger.error("Database query execution failed during retrieval: %s", db_err)
            raise RetrievalError(f"Database query failed during semantic retrieval: {db_err}") from db_err

        results: list[RetrievedChunk] = []
        for chunk, original_filename, distance, emb_model_id in rows:
            distance_val = float(distance)
            score_val = 1.0 - distance_val
            results.append(
                RetrievedChunk(
                    chunk_id=chunk.id,
                    document_id=chunk.document_id,
                    document_filename=original_filename,
                    chunk_index=chunk.chunk_index,
                    text=chunk.text,
                    score=score_val,
                    cosine_distance=distance_val,
                    page_number=chunk.page_number,
                    page_numbers=chunk.page_numbers or [],
                    heading=chunk.heading,
                    heading_level=chunk.heading_level,
                    metadata=chunk.metadata_ or {},
                    model_id=emb_model_id,
                )
            )

        return RetrievalResponse(
            query=stripped_query,
            model_id=active_model_id,
            returned_count=len(results),
            results=results,
        )
