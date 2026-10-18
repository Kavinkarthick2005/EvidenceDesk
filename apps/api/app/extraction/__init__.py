"""EvidenceDesk document extraction package."""

from app.extraction.exceptions import (
    ExtractionEmptyError,
    ExtractionEncodingError,
    ExtractionEncryptedError,
    ExtractionError,
    ExtractionImageOnlyError,
    ExtractionMalformedError,
    ExtractionUnsupportedTypeError,
)
from app.extraction.interface import DocumentExtractor
from app.extraction.models import ExtractedElement, ExtractionResult
from app.extraction.parsers import MarkdownExtractor, PdfExtractor, PlainTextExtractor
from app.extraction.service import (
    ExtractionService,
    extract_document,
    get_extraction_service,
)

__all__ = [
    "DocumentExtractor",
    "ExtractedElement",
    "ExtractionEmptyError",
    "ExtractionEncodingError",
    "ExtractionEncryptedError",
    "ExtractionError",
    "ExtractionImageOnlyError",
    "ExtractionMalformedError",
    "ExtractionResult",
    "ExtractionService",
    "ExtractionUnsupportedTypeError",
    "MarkdownExtractor",
    "PdfExtractor",
    "PlainTextExtractor",
    "extract_document",
    "get_extraction_service",
]
