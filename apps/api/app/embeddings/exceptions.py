"""Structured exceptions for embedding operations."""


class EmbeddingError(Exception):
    """Base exception for all embedding-related failures."""


class EmbeddingModelLoadError(EmbeddingError):
    """Raised when an embedding model fails to initialize or load weights."""


class EmbeddingInferenceError(EmbeddingError):
    """Raised when an embedding model fails during inference."""


class EmbeddingDimensionMismatchError(EmbeddingError):
    """Raised when vector dimensionality does not match the active schema expectation."""


class EmbeddingConfigurationError(EmbeddingError):
    """Raised when embedding configuration settings are invalid or unsupported."""


class EmbeddingNonFiniteError(EmbeddingError):
    """Raised when model inference outputs contain non-finite numbers (NaN or Inf)."""
