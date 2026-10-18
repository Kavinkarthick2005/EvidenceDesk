"""Domain exceptions for document extraction errors."""


class ExtractionError(Exception):
    """Base exception for all document text extraction errors."""


class ExtractionEmptyError(ExtractionError):
    """Raised when a document is empty (0 bytes) or contains only whitespace."""


class ExtractionMalformedError(ExtractionError):
    """Raised when a document is corrupted, truncated, or structurally invalid."""


class ExtractionEncryptedError(ExtractionError):
    """Raised when a document is encrypted or password-protected."""


class ExtractionImageOnlyError(ExtractionError):
    """Raised when a document contains images but no extractable text.

    EvidenceDesk does not perform OCR in the initial release; image-only
    or scanned documents cannot produce extractable text.
    """


class ExtractionEncodingError(ExtractionError):
    """Raised when text content cannot be decoded using supported encodings."""


class ExtractionUnsupportedTypeError(ExtractionError):
    """Raised when the document content type or format is unsupported."""
