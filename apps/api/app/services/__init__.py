"""Application domain services for EvidenceDesk."""

from app.services.document import DocumentService
from app.services.exceptions import (
    DocumentNotFoundError,
    DocumentPersistenceError,
    DocumentServiceError,
    DocumentSizeLimitError,
    DocumentStorageError,
    DocumentValidationError,
    EmptyDocumentError,
    UnsupportedDocumentTypeError,
)

__all__ = [
    "DocumentNotFoundError",
    "DocumentPersistenceError",
    "DocumentService",
    "DocumentServiceError",
    "DocumentSizeLimitError",
    "DocumentStorageError",
    "DocumentValidationError",
    "EmptyDocumentError",
    "UnsupportedDocumentTypeError",
]
