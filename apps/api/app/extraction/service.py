"""Unified extraction service coordinating format-specific extractors."""

from pathlib import Path
from typing import Mapping

from app.extraction.exceptions import ExtractionUnsupportedTypeError
from app.extraction.interface import DocumentExtractor
from app.extraction.models import ExtractionResult
from app.extraction.parsers.markdown import MarkdownExtractor
from app.extraction.parsers.pdf import PdfExtractor
from app.extraction.parsers.text import PlainTextExtractor


class ExtractionService:
    """Dispatches raw document bytes to appropriate format extractors.

    Maintains separation between extraction logic, database access, storage,
    and API routes.
    """

    MIME_MAP: Mapping[str, str] = {
        "application/pdf": "application/pdf",
        "text/markdown": "text/markdown",
        "text/x-markdown": "text/markdown",
        "text/plain": "text/plain",
    }

    EXTENSION_MAP: Mapping[str, str] = {
        ".pdf": "application/pdf",
        ".md": "text/markdown",
        ".markdown": "text/markdown",
        ".txt": "text/plain",
    }

    def __init__(self) -> None:
        self._pdf_extractor = PdfExtractor()
        self._markdown_extractor = MarkdownExtractor()
        self._text_extractor = PlainTextExtractor()

    def extract(
        self,
        content: bytes,
        content_type: str | None = None,
        filename: str | None = None,
        **kwargs: object,
    ) -> ExtractionResult:
        """Resolve format and extract structured content from document bytes.

        Args:
            content: Raw document bytes.
            content_type: MIME type (e.g. application/pdf, text/markdown, text/plain).
            filename: Original document filename used for extension fallback.
            **kwargs: Extra parameters passed to the format extractor.

        Returns:
            ExtractionResult containing structured text, elements, and metadata.

        Raises:
            ExtractionUnsupportedTypeError: If format cannot be resolved or is unsupported.
            ExtractionEmptyError: If document is empty or whitespace-only.
            ExtractionMalformedError: If document structure is corrupted.
            ExtractionEncryptedError: If document is encrypted.
            ExtractionImageOnlyError: If document contains images but no extractable text.
            ExtractionEncodingError: If text cannot be decoded.
        """
        resolved_mime = self._resolve_mime_type(content_type, filename)
        extractor = self._get_extractor(resolved_mime)
        return extractor.extract(content, **kwargs)

    GENERIC_MIME_TYPES = frozenset({
        "application/octet-stream",
        "binary/octet-stream",
    })

    def _resolve_mime_type(
        self,
        content_type: str | None,
        filename: str | None,
    ) -> str:
        """Determine canonical MIME type from provided media type and filename.

        Guarantees that:
        - Mismatched MIME types and extensions raise ExtractionUnsupportedTypeError.
        - Unsupported extensions cannot bypass validation even with allowed content_type.
        - Unsupported content_type cannot bypass validation even with allowed extension.
        - Generic binary/octet-stream MIME types fall back to extension inference.
        """
        cleaned_mime: str | None = None
        if content_type and content_type.strip():
            raw_mime = content_type.split(";")[0].strip().lower()
            if raw_mime in self.GENERIC_MIME_TYPES:
                cleaned_mime = None  # Fall back to extension inference for generic stream
            elif raw_mime in self.MIME_MAP:
                cleaned_mime = self.MIME_MAP[raw_mime]
            else:
                # Explicitly unsupported MIME type
                raise ExtractionUnsupportedTypeError(
                    f"Unsupported content-type '{content_type}' for document extraction."
                )

        ext: str | None = None
        if filename and filename.strip():
            ext = Path(filename).suffix.lower()

        # If extension is provided, validate it
        if ext:
            if ext not in self.EXTENSION_MAP:
                raise ExtractionUnsupportedTypeError(
                    f"Unsupported file extension '{ext}' for filename '{filename}'."
                )
            ext_canonical_mime = self.EXTENSION_MAP[ext]
            if cleaned_mime is not None and cleaned_mime != ext_canonical_mime:
                raise ExtractionUnsupportedTypeError(
                    f"Mismatched content-type '{content_type}' and file extension '{ext}'."
                )
            return ext_canonical_mime

        # If no extension was provided or filename is None, rely on validated cleaned_mime
        if cleaned_mime is not None:
            return cleaned_mime

        raise ExtractionUnsupportedTypeError(
            f"Unsupported format for extraction (content_type='{content_type}', filename='{filename}')"
        )

    def _get_extractor(self, mime_type: str) -> DocumentExtractor:
        """Return the extractor instance corresponding to the resolved MIME type."""
        if mime_type == "application/pdf":
            return self._pdf_extractor
        if mime_type == "text/markdown":
            return self._markdown_extractor
        if mime_type == "text/plain":
            return self._text_extractor
        raise ExtractionUnsupportedTypeError(f"No extractor registered for MIME type: {mime_type}")


_default_extraction_service: ExtractionService | None = None


def get_extraction_service() -> ExtractionService:
    """Return a shared singleton ExtractionService instance."""
    global _default_extraction_service
    if _default_extraction_service is None:
        _default_extraction_service = ExtractionService()
    return _default_extraction_service


def extract_document(
    content: bytes,
    content_type: str | None = None,
    filename: str | None = None,
    **kwargs: object,
) -> ExtractionResult:
    """Convenience helper to extract document text using the shared service."""
    return get_extraction_service().extract(
        content=content,
        content_type=content_type,
        filename=filename,
        **kwargs,
    )
