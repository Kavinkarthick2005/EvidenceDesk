"""Deterministic in-memory mock embedding provider for fast, offline CI testing."""

from collections.abc import Sequence
import hashlib
import logging
import math
import random

from app.embeddings.exceptions import (
    EmbeddingDimensionMismatchError,
    EmbeddingInferenceError,
    EmbeddingNonFiniteError,
)

logger = logging.getLogger(__name__)


class DeterministicMockEmbeddingProvider:
    """Mock embedding provider producing deterministic, unit-normalized vectors.

    Uses SHA-256 digests as pseudo-random seeds to generate reproducible 384-dimensional
    vectors without downloading models or requiring network access.
    """

    def __init__(
        self,
        model_id: str = "mock-bge-small-384",
        dimension: int = 384,
        max_sequence_length: int = 512,
        should_fail: bool = False,
        simulate_error: Exception | None = None,
    ) -> None:
        """Initialize mock provider with configurable dimensions and error simulation.

        Args:
            model_id: Canonical model identifier.
            dimension: Dimensionality of output vectors (defaults to 384).
            max_sequence_length: Maximum allowed sequence token limit (defaults to 512).
            should_fail: If True, raises EmbeddingInferenceError on embedding requests.
            simulate_error: Optional explicit exception to raise on embedding requests.
        """
        self._model_id = model_id
        self._dimension = dimension
        self._max_sequence_length = max_sequence_length
        self._truncated_chunks_count = 0
        self.should_fail = should_fail
        self.simulate_error = simulate_error

    @property
    def model_id(self) -> str:
        """Canonical model identifier."""
        return self._model_id

    @property
    def dimension(self) -> int:
        """Configured vector dimensionality."""
        return self._dimension

    @property
    def max_sequence_length(self) -> int:
        """Maximum token context window before truncation."""
        return self._max_sequence_length

    @property
    def truncated_chunks_count(self) -> int:
        """Number of text chunks that exceeded max_sequence_length and were truncated."""
        return self._truncated_chunks_count

    def _generate_vector(self, text: str) -> list[float]:
        """Generate a deterministic unit-normalized vector for a given string."""
        seed_bytes = hashlib.sha256(text.encode("utf-8")).digest()
        rng = random.Random(seed_bytes)

        # Generate pseudo-random float components
        raw_components = [rng.uniform(-1.0, 1.0) for _ in range(self._dimension)]

        # Validate finiteness before normalization
        for val in raw_components:
            if not math.isfinite(val):
                raise EmbeddingNonFiniteError(f"Generated non-finite float '{val}' during mock vectorization.")

        # L2 unit normalization
        norm = math.sqrt(sum(c * c for c in raw_components))
        if norm == 0.0:
            norm = 1.0

        normalized = [c / norm for c in raw_components]

        # Double check post-normalization dimensions and finiteness
        if len(normalized) != self._dimension:
            raise EmbeddingDimensionMismatchError(
                f"Generated vector length {len(normalized)} does not match provider dimension {self._dimension}."
            )

        for val in normalized:
            if not math.isfinite(val):
                raise EmbeddingNonFiniteError(f"Normalized vector contains non-finite float '{val}'.")

        return normalized

    def _check_truncation(self, text: str) -> None:
        """Simulate subword token measurement and report truncation if context is exceeded."""
        # Simple whitespace token estimation with special token simulation (+2 for [CLS] and [SEP])
        estimated_tokens = len(text.split()) + 2
        if estimated_tokens > self._max_sequence_length:
            excess = estimated_tokens - self._max_sequence_length
            self._truncated_chunks_count += 1
            logger.warning(
                "Chunk text exceeded model context window (%d > %d tokens). "
                "Truncated %d trailing tokens for model '%s'.",
                estimated_tokens,
                self._max_sequence_length,
                excess,
                self._model_id,
            )

    def embed_texts(self, texts: Sequence[str]) -> list[list[float]]:
        """Batch-embed text sequences with error simulation and truncation tracking."""
        if self.simulate_error is not None:
            raise self.simulate_error
        if self.should_fail:
            raise EmbeddingInferenceError(f"Mock inference failure simulated for model '{self._model_id}'.")

        results: list[list[float]] = []
        for text in texts:
            self._check_truncation(text)
            results.append(self._generate_vector(text))

        return results

    def embed_query(self, query: str) -> list[float]:
        """Embed a single query string."""
        if self.simulate_error is not None:
            raise self.simulate_error
        if self.should_fail:
            raise EmbeddingInferenceError(f"Mock query inference failure simulated for model '{self._model_id}'.")

        self._check_truncation(query)
        return self._generate_vector(query)
