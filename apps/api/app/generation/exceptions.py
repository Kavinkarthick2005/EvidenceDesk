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


class GenerationConnectionError(GenerationInferenceError):
    """Raised when upstream generation service connection fails or service is unavailable."""


class ContextBuilderError(GenerationError):
    """Base exception for context builder failures."""


class DuplicateEvidenceError(ContextBuilderError):
    """Raised when duplicate evidence items or chunk UUIDs are supplied to context builder."""


class InvalidEvidenceError(ContextBuilderError):
    """Raised when supplied evidence items are malformed, missing required fields, or of invalid type."""


class CitationValidationError(GenerationError):
    """Raised when citation validation fails due to malformed input or unrecoverable error."""
