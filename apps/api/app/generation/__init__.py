"""EvidenceDesk generation package providing interfaces, implementations, and factory."""

import logging
from typing import TYPE_CHECKING

from app.core.config import Settings, get_settings
from app.generation.citation import (
    CitationValidationResult,
    CitationValidator,
    ResolvedCitation,
)
from app.generation.context import (
    ContextBuilder,
    EvidenceContext,
    EvidenceSourceMapping,
)
from app.generation.exceptions import (
    CitationValidationError,
    ContextBuilderError,
    DuplicateEvidenceError,
    GenerationConfigurationError,
    GenerationError,
    GenerationInferenceError,
    GenerationTimeoutError,
    InvalidEvidenceError,
    InvalidPromptError,
)
from app.generation.interface import (
    EvidenceItem,
    GenerationProvider,
    GenerationResult,
)
from app.generation.mock import DeterministicMockGenerator

logger = logging.getLogger(__name__)

_generation_provider_instance: GenerationProvider | None = None


def get_generation_provider(settings: Settings | None = None) -> GenerationProvider:
    """Return a shared or configured GenerationProvider instance.

    Inspects settings.GENERATION_PROVIDER ('mock'). Defaults to
    DeterministicMockGenerator for fast, offline, and predictable operation.

    Args:
        settings: Optional explicit Settings instance. If omitted, uses get_settings().

    Returns:
        Configured GenerationProvider instance.

    Raises:
        GenerationConfigurationError: If configured provider is unsupported.
    """
    global _generation_provider_instance
    app_settings = settings or get_settings()

    if settings is not None or _generation_provider_instance is None:
        provider_type = app_settings.GENERATION_PROVIDER.lower().strip()

        if provider_type in ("mock", "deterministic_mock"):
            provider: GenerationProvider = DeterministicMockGenerator(
                model_id=app_settings.GENERATION_MODEL or "mock-generator-v1",
            )
        else:
            raise GenerationConfigurationError(
                f"Unsupported GENERATION_PROVIDER '{app_settings.GENERATION_PROVIDER}'. "
                f"Supported providers: 'mock'."
            )

        if settings is not None:
            return provider

        _generation_provider_instance = provider

    return _generation_provider_instance


def reset_generation_state() -> None:
    """Reset cached generation provider instance (used for test isolation)."""
    global _generation_provider_instance
    _generation_provider_instance = None


__all__ = [
    "CitationValidationError",
    "CitationValidationResult",
    "CitationValidator",
    "ContextBuilder",
    "ContextBuilderError",
    "DeterministicMockGenerator",
    "DuplicateEvidenceError",
    "EvidenceContext",
    "EvidenceItem",
    "EvidenceSourceMapping",
    "GenerationConfigurationError",
    "GenerationError",
    "GenerationInferenceError",
    "GenerationProvider",
    "GenerationResult",
    "GenerationTimeoutError",
    "InvalidEvidenceError",
    "InvalidPromptError",
    "ResolvedCitation",
    "get_generation_provider",
    "reset_generation_state",
]
