"""Exceptions for EvidenceDesk document application services."""


class DocumentServiceError(Exception):
    """Base exception for all document service errors."""


class DocumentNotFoundError(DocumentServiceError):
    """Raised when a document record or its underlying stored file does not exist."""


class DocumentValidationError(DocumentServiceError):
    """Raised when document metadata, filename, or payload fails validation."""


class EmptyDocumentError(DocumentValidationError):
    """Raised when document content or required metadata is empty."""


class UnsupportedDocumentTypeError(DocumentValidationError):
    """Raised when the document MIME type or extension is not permitted."""


class DocumentSizeLimitError(DocumentValidationError):
    """Raised when document content exceeds the configured maximum upload limit."""


class DocumentStorageError(DocumentServiceError):
    """Raised when an underlying file storage operation fails."""


class DocumentPersistenceError(DocumentServiceError):
    """Raised when database persistence or transaction commit fails."""


class ChunkServiceError(Exception):
    """Base exception for all chunk service errors."""


class ChunkValidationError(ChunkServiceError):
    """Raised when chunk payload, offsets, ordering, or metadata fails validation."""


class EmptyChunkSetError(ChunkValidationError):
    """Raised when an empty chunk set is provided where chunks are required."""


class ChunkPersistenceError(ChunkServiceError):
    """Raised when chunk persistence, atomic replacement, or transaction commit fails."""


class IngestionError(Exception):
    """Base exception for all ingestion pipeline errors."""


class IngestionExtractionError(IngestionError):
    """Raised when document text or structure extraction fails during ingestion."""


class IngestionChunkingError(IngestionError):
    """Raised when partitioning extracted content into chunks fails during ingestion."""


class IngestionPersistenceError(IngestionError):
    """Raised when persisting or replacing chunks in the database fails during ingestion."""
