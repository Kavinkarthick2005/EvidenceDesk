"""Typed domain models for document chunking."""

from typing import Any
import uuid

from pydantic import BaseModel, Field, model_validator

from app.chunking.exceptions import ChunkingConfigError


class ChunkingConfig(BaseModel):
    """Configuration governing structure-aware document chunking.

    Default baseline:
        - chunk_size_tokens: 500
        - chunk_overlap_tokens: 75
    """

    chunk_size_tokens: int = 500
    chunk_overlap_tokens: int = 75

    @model_validator(mode="after")
    def validate_config(self) -> "ChunkingConfig":
        """Enforce strict constraints on token windowing and overlap."""
        if self.chunk_size_tokens <= 0:
            raise ChunkingConfigError(
                f"chunk_size_tokens must be positive (got {self.chunk_size_tokens})"
            )
        if self.chunk_overlap_tokens < 0:
            raise ChunkingConfigError(
                f"chunk_overlap_tokens must be non-negative (got {self.chunk_overlap_tokens})"
            )
        if self.chunk_overlap_tokens >= self.chunk_size_tokens:
            raise ChunkingConfigError(
                f"chunk_overlap_tokens ({self.chunk_overlap_tokens}) must be strictly less "
                f"than chunk_size_tokens ({self.chunk_size_tokens})"
            )
        return self


class Chunk(BaseModel):
    """Represents a discrete, source-traceable passage extracted from a document.

    Preserves exact token metrics, character offsets, source format, and format-specific
    structural metadata (such as PDF page numbers and Markdown headings).
    """

    chunk_index: int
    text: str
    token_count: int
    character_count: int
    start_char_offset: int
    end_char_offset: int
    document_id: uuid.UUID | None = None
    source_format: str = "text/plain"
    page_number: int | None = None
    page_numbers: list[int] = Field(default_factory=list)
    heading: str | None = None
    heading_level: int | None = None
    metadata: dict[str, Any] = Field(default_factory=dict)


class ChunkingResult(BaseModel):
    """Container for the complete list of chunks produced from an extracted document."""

    chunks: list[Chunk] = Field(default_factory=list)
    total_chunks: int = 0
    total_tokens: int = 0
    document_id: uuid.UUID | None = None
    config: ChunkingConfig = Field(default_factory=ChunkingConfig)

    def model_post_init(self, __context: Any) -> None:
        """Derive total_chunks and total_tokens if not explicitly provided."""
        if self.total_chunks == 0 and self.chunks:
            self.total_chunks = len(self.chunks)
        if self.total_tokens == 0 and self.chunks:
            self.total_tokens = sum(c.token_count for c in self.chunks)
