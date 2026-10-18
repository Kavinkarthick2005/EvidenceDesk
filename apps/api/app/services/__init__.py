"""Application domain services for EvidenceDesk."""

from app.services.chunk import ChunkPersistenceService
from app.services.document import DocumentService
from app.services.exceptions import (
    ChunkPersistenceError,
    ChunkServiceError,
    ChunkValidationError,
    DocumentNotFoundError,
    DocumentPersistenceError,
    DocumentServiceError,
    DocumentSizeLimitError,
    DocumentStorageError,
    DocumentValidationError,
    EmptyChunkSetError,
    EmptyDocumentError,
    UnsupportedDocumentTypeError,
)

__all__ = [
    "ChunkPersistenceError",
    "ChunkPersistenceService",
    "ChunkServiceError",
    "ChunkValidationError",
    "DocumentNotFoundError",
    "DocumentPersistenceError",
    "DocumentService",
    "DocumentServiceError",
    "DocumentSizeLimitError",
    "DocumentStorageError",
    "DocumentValidationError",
    "EmptyChunkSetError",
    "EmptyDocumentError",
    "UnsupportedDocumentTypeError",
]
