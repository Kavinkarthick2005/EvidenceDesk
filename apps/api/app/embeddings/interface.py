"""Provider-independent embedding interface for EvidenceDesk."""

from collections.abc import Sequence
from typing import Protocol, runtime_checkable


@runtime_checkable
class EmbeddingProvider(Protocol):
    """Abstract interface defining text embedding capabilities.

    Implementations must introspect their active model to derive vector dimensions
    and maximum sequence length, apply unit-normalization to all output vectors,
    and detect and report tokenizer truncation.
    """

    @property
    def model_id(self) -> str:
        """Canonical identifier of the active embedding model."""
        ...

    @property
    def dimension(self) -> int:
        """Vector dimensionality derived dynamically from the active model."""
        ...

    @property
    def max_sequence_length(self) -> int:
        """Maximum token context window supported by the model before truncation."""
        ...

    @property
    def truncated_chunks_count(self) -> int:
        """Count of text passages that exceeded max_sequence_length and were truncated."""
        ...

    def embed_texts(self, texts: Sequence[str]) -> list[list[float]]:
        """Batch-embed a sequence of text passages, returning unit-normalized float vectors.

        Args:
            texts: Sequence of chunk texts to embed.

        Returns:
            List of float vectors, where len(vector) == self.dimension and ||vector||_2 == 1.0.

        Raises:
            EmbeddingInferenceError: If model inference fails.
            EmbeddingNonFiniteError: If output vectors contain NaN or Inf values.
            EmbeddingDimensionMismatchError: If output dimensionality violates self.dimension.
        """
        ...

    def embed_query(self, query: str) -> list[float]:
        """Embed a single search query string, returning a unit-normalized float vector.

        Args:
            query: Natural language query string.

        Returns:
            Float vector of length self.dimension with ||vector||_2 == 1.0.

        Raises:
            EmbeddingInferenceError: If query embedding fails.
            EmbeddingNonFiniteError: If output vector contains NaN or Inf values.
        """
        ...
