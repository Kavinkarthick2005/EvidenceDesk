"""Exceptions for the EvidenceDesk generation subsystem."""


class GenerationError(Exception):
    """Base exception for all generation-related failures."""


class GenerationConfigurationError(GenerationError):
    """Raised when generation provider configuration is invalid or unsupported."""


class InvalidPromptError(GenerationError):
    """Raised when a query or generation prompt is invalid, empty, or malformed."""


class GenerationInferenceError(GenerationError):
    """Raised when model inference or generation fails."""


class GenerationTimeoutError(GenerationInferenceError):
    """Raised when generation request exceeds configured timeout."""
