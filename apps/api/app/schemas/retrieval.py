"""Pydantic schemas for semantic retrieval requests and responses."""

from typing import Any
import uuid

from pydantic import BaseModel, ConfigDict, Field


class RetrievedChunk(BaseModel):
    """Source passage evidence retrieved from vector similarity search."""

    model_config = ConfigDict(frozen=True)

    chunk_id: uuid.UUID = Field(description="UUID identifier of the retrieved chunk")
    document_id: uuid.UUID = Field(description="UUID identifier of the parent document")
    document_filename: str = Field(description="Original filename of the parent document")
    chunk_index: int = Field(description="Sequential zero-based index of the chunk within the document")
    text: str = Field(description="Exact extracted textual content of the chunk")
    score: float = Field(description="Cosine similarity score (1.0 - cosine_distance), bounded in [-1.0, 1.0]")
    cosine_distance: float = Field(description="Raw cosine distance from pgvector <=> operator, bounded in [0.0, 2.0]")
    page_number: int | None = Field(default=None, description="Primary 1-indexed page number if available")
    page_numbers: list[int] = Field(default_factory=list, description="List of page numbers spanned by the chunk")
    heading: str | None = Field(default=None, description="Enclosing heading if available")
    heading_level: int | None = Field(default=None, description="Heading level depth if available")
    metadata: dict[str, Any] = Field(default_factory=dict, description="Format-specific structural metadata")
    model_id: str = Field(description="Canonical identifier of the model that generated the embedding")


class RetrievalRequest(BaseModel):
    """Request schema for semantic similarity search."""

    model_config = ConfigDict(frozen=True)

    query: str = Field(min_length=1, max_length=2000, description="Natural language search query")
    top_k: int = Field(default=5, ge=1, le=50, description="Maximum number of candidate chunks to retrieve")
    document_ids: list[uuid.UUID] | None = Field(default=None, description="Optional document UUID filter")
    score_threshold: float | None = Field(
        default=None,
        ge=-1.0,
        le=1.0,
        description="Optional minimum cosine similarity score threshold",
    )


class RetrievalResponse(BaseModel):
    """Result payload from semantic retrieval."""

    model_config = ConfigDict(frozen=True)

    query: str = Field(description="Processed query string")
    model_id: str = Field(description="Canonical model identifier used for query embedding")
    returned_count: int = Field(description="Number of candidate chunks returned (len(results))")
    results: list[RetrievedChunk] = Field(description="Ranked candidate chunks ordered by cosine similarity")
