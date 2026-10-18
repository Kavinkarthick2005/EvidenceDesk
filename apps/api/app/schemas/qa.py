"""Pydantic schemas for Question Answering (QA) requests and responses.

Defines the contract for the /qa/answer endpoint, including query parameters,
evidence citation metadata, abstention indicators, and verification diagnostics.
"""

from typing import Any
import uuid

from pydantic import BaseModel, ConfigDict, Field, field_validator


class QARequest(BaseModel):
    """Request payload for evidence-grounded question answering."""

    model_config = ConfigDict(frozen=True)

    query: str = Field(
        min_length=1,
        max_length=2000,
        description="Natural language question to answer from retrieved documents.",
    )
    top_k: int = Field(
        default=5,
        ge=1,
        le=50,
        description="Maximum number of candidate evidence chunks to retrieve.",
    )
    document_ids: list[uuid.UUID] | None = Field(
        default=None,
        description="Optional list of document UUIDs to restrict retrieval to.",
    )
    score_threshold: float | None = Field(
        default=None,
        ge=-1.0,
        le=1.0,
        description="Optional minimum cosine similarity score threshold for evidence retrieval.",
    )

    @field_validator("query")
    @classmethod
    def validate_query_not_whitespace(cls, v: str) -> str:
        """Reject empty or whitespace-only queries."""
        stripped = v.strip()
        if not stripped:
            raise ValueError("Query cannot be empty or whitespace-only.")
        return stripped


class QACitation(BaseModel):
    """Resolved source citation linking a marker in the answer to its source chunk metadata.

    NOTE: Resolving a citation to a retrieved source chunk confirms reference existence
    and exact metadata traceability. It does NOT prove semantic entailment of the claim.
    """

    model_config = ConfigDict(frozen=True)

    marker: str = Field(
        description="Formatted bracketed citation marker as it appears in the answer (e.g., '[1]').",
    )
    marker_id: str = Field(
        description="Unique citation identifier string (e.g., '1', '2').",
    )
    chunk_id: uuid.UUID = Field(
        description="UUID of the cited chunk.",
    )
    document_id: uuid.UUID = Field(
        description="UUID of the parent source document.",
    )
    document_filename: str = Field(
        description="Original filename of the parent document.",
    )
    chunk_index: int = Field(
        description="Sequential zero-based index of the chunk within the document.",
    )
    text: str = Field(
        description="Exact extracted text of the cited chunk passage.",
    )
    score: float = Field(
        description="Cosine similarity score from vector retrieval.",
    )
    page_number: int | None = Field(
        default=None,
        description="Primary page number of the cited passage if available.",
    )
    page_numbers: list[int] = Field(
        default_factory=list,
        description="List of page numbers spanned by the chunk.",
    )
    heading: str | None = Field(
        default=None,
        description="Enclosing heading of the cited chunk if available.",
    )


class QAResponse(BaseModel):
    """Response payload from evidence-grounded question answering."""

    model_config = ConfigDict(frozen=True)

    query: str = Field(
        description="The normalized query string that was processed.",
    )
    answer: str = Field(
        description="Generated answer text or safe abstention statement.",
    )
    is_abstained: bool = Field(
        description="True if the system abstained from answering due to missing evidence or invalid citations.",
    )
    abstention_reason: str | None = Field(
        default=None,
        description="Reason code if abstained (e.g., 'no_evidence_found', 'missing_citations', 'citation_validation_failed').",
    )
    has_citations: bool = Field(
        description="True if at least one valid citation was identified.",
    )
    is_valid_citations: bool = Field(
        description="True if all parsed citations were valid without unknown or malformed references.",
    )
    citations: list[QACitation] = Field(
        default_factory=list,
        description="List of resolved citations with full source metadata.",
    )
    unknown_citations: list[str] = Field(
        default_factory=list,
        description="Citation markers or IDs found in generation that did not exist in the evidence context.",
    )
    malformed_citations: list[str] = Field(
        default_factory=list,
        description="Malformed citation markers detected in the answer text.",
    )
    retrieved_count: int = Field(
        description="Total number of chunks retrieved from semantic search before context building.",
    )
    model_id: str = Field(
        description="Canonical identifier of the generation model used.",
    )
    retrieval_model_id: str = Field(
        description="Canonical identifier of the embedding model used for retrieval.",
    )
