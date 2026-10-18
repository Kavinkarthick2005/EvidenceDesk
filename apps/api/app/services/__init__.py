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
    InvalidQueryError,
    InvalidScoreThresholdError,
    InvalidTopKError,
    RetrievalError,
    UnsupportedDocumentTypeError,
)
from app.services.ingestion import IngestionResult, IngestionService
from app.services.qa import QuestionAnsweringService
from app.services.retrieval import RetrievalService

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
    "InvalidQueryError",
    "InvalidScoreThresholdError",
    "InvalidTopKError",
    "QuestionAnsweringService",
    "RetrievalError",
    "RetrievalService",
    "UnsupportedDocumentTypeError",
]
