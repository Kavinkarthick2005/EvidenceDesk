"""Domain exceptions for document chunking."""


class ChunkingError(Exception):
    """Base exception for all document chunking errors."""


class ChunkingConfigError(ChunkingError):
    """Raised when chunking parameters are invalid (e.g. non-positive size or overlap >= size)."""


class ChunkingEmptyError(ChunkingError):
    """Raised when document content is empty or contains only whitespace."""
