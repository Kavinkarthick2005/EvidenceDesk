"""Application domain services for EvidenceDesk."""

from app.services.chunk import ChunkPersistenceService
from app.services.document import DocumentService
from app.services.embedding import EmbeddingService
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
    IngestionChunkingError,
    IngestionError,
    IngestionExtractionError,
    IngestionPersistenceError,
    UnsupportedDocumentTypeError,
)
from app.services.ingestion import IngestionResult, IngestionService

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
    "EmbeddingService",
    "EmptyChunkSetError",
    "EmptyDocumentError",
    "IngestionChunkingError",
    "IngestionError",
    "IngestionExtractionError",
    "IngestionPersistenceError",
    "IngestionResult",
    "IngestionService",
    "UnsupportedDocumentTypeError",
]
