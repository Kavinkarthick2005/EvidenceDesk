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
from app.generation.ollama import OllamaGenerationProvider

logger = logging.getLogger(__name__)

_generation_provider_instance: GenerationProvider | None = None


def get_generation_provider(settings: Settings | None = None) -> GenerationProvider:
    """Return a shared or configured GenerationProvider instance.

    Inspects settings.GENERATION_PROVIDER ('mock' or 'ollama'). Defaults to
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
        elif provider_type == "ollama":
            model_name = (
                app_settings.GENERATION_MODEL
                if app_settings.GENERATION_MODEL and app_settings.GENERATION_MODEL != "mock-generator-v1"
                else "qwen3:4b"
            )
            provider = OllamaGenerationProvider(
                base_url=app_settings.OLLAMA_BASE_URL,
                model_id=model_name,
                timeout=app_settings.OLLAMA_TIMEOUT_SECONDS,
                keep_alive=app_settings.OLLAMA_KEEP_ALIVE,
            )
        else:
            raise GenerationConfigurationError(
                f"Unsupported GENERATION_PROVIDER '{app_settings.GENERATION_PROVIDER}'. "
                f"Supported providers: 'mock', 'ollama'."
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
    "OllamaGenerationProvider",
    "ResolvedCitation",
    "get_generation_provider",
    "reset_generation_state",
]
