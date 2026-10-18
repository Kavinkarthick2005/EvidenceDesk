"""Interfaces and protocols for tokenization and document chunking."""

from typing import Protocol, runtime_checkable
import uuid

from app.chunking.models import ChunkingConfig, ChunkingResult
from app.extraction.models import ExtractionResult


@runtime_checkable
class Tokenizer(Protocol):
    """Protocol for tokenizing and detokenizing text."""

    def encode(self, text: str) -> list[int]:
        """Convert a string into a list of token identifiers."""
        ...

    def decode(self, tokens: list[int]) -> str:
        """Convert a list of token identifiers back into text."""
        ...

    def count_tokens(self, text: str) -> int:
        """Return the exact number of tokens in the given text."""
        ...


@runtime_checkable
class Chunker(Protocol):
    """Protocol implemented by structure-aware chunkers."""

    def chunk(
        self,
        extraction_result: ExtractionResult,
        document_id: uuid.UUID | None = None,
        config: ChunkingConfig | None = None,
    ) -> ChunkingResult:
        """Convert an ExtractionResult into deterministic, source-traceable chunks.

        Args:
            extraction_result: Output from the document extraction layer.
            document_id: Optional UUID identifying the document record.
            config: Optional ChunkingConfig overriding default token sizes and overlap.

        Returns:
            ChunkingResult containing ordered, source-attributed chunks.

        Raises:
            ChunkingEmptyError: If document text is empty or whitespace only.
            ChunkingConfigError: If chunking configuration is invalid.
        """
        ...
