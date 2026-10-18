"""EvidenceDesk embedding package providing interfaces, implementations, and factory."""

import logging
from typing import TYPE_CHECKING

from app.core.config import Settings, get_settings
from app.embeddings.exceptions import (
    EmbeddingConfigurationError,
    EmbeddingDimensionMismatchError,
    EmbeddingError,
    EmbeddingInferenceError,
    EmbeddingModelLoadError,
    EmbeddingNonFiniteError,
)
from app.embeddings.interface import EmbeddingProvider
from app.embeddings.mock import DeterministicMockEmbeddingProvider

logger = logging.getLogger(__name__)

_embedding_provider_instance: EmbeddingProvider | None = None


def get_embedding_provider(settings: Settings | None = None) -> EmbeddingProvider:
    """Return a shared or configured EmbeddingProvider instance.

    Inspects settings.EMBEDDING_PROVIDER ('mock' or 'sentence_transformers').
    Defaults to DeterministicMockEmbeddingProvider for fast, offline operation.
    """
    global _embedding_provider_instance
    app_settings = settings or get_settings()

    # If explicit settings are passed, or no singleton exists, create provider
    if settings is not None or _embedding_provider_instance is None:
        provider_type = app_settings.EMBEDDING_PROVIDER.lower().strip()

        if provider_type in ("mock", "deterministic_mock"):
            provider: EmbeddingProvider = DeterministicMockEmbeddingProvider(
                model_id=app_settings.EMBEDDING_MODEL,
                dimension=app_settings.EMBEDDING_DIMENSION,
            )
        elif provider_type in ("sentence_transformers", "sentence-transformers"):
            from app.embeddings.sentence_transformers import SentenceTransformersEmbeddingProvider

            provider = SentenceTransformersEmbeddingProvider(
                model_name_or_path=app_settings.EMBEDDING_MODEL,
                expected_dimension=app_settings.EMBEDDING_DIMENSION,
                device=app_settings.EMBEDDING_DEVICE,
                batch_size=app_settings.EMBEDDING_BATCH_SIZE,
                num_threads=app_settings.EMBEDDING_NUM_THREADS,
            )
        else:
            raise EmbeddingConfigurationError(
                f"Unsupported EMBEDDING_PROVIDER '{app_settings.EMBEDDING_PROVIDER}'. "
                f"Supported providers: 'mock', 'sentence_transformers'."
            )

        # Enforce that discovered provider dimension strictly satisfies the schema expectation
        if provider.dimension != app_settings.EMBEDDING_DIMENSION:
            raise EmbeddingDimensionMismatchError(
                f"Configured provider '{provider.model_id}' dimension {provider.dimension} "
                f"does not match required schema dimension {app_settings.EMBEDDING_DIMENSION}."
            )

        if settings is not None:
            return provider

        _embedding_provider_instance = provider

    return _embedding_provider_instance


def reset_embedding_state() -> None:
    """Reset cached embedding provider instance (used for test isolation)."""
    global _embedding_provider_instance
    _embedding_provider_instance = None


__all__ = [
    "DeterministicMockEmbeddingProvider",
    "EmbeddingConfigurationError",
    "EmbeddingDimensionMismatchError",
    "EmbeddingError",
    "EmbeddingInferenceError",
    "EmbeddingModelLoadError",
    "EmbeddingNonFiniteError",
    "EmbeddingProvider",
    "get_embedding_provider",
    "reset_embedding_state",
]
