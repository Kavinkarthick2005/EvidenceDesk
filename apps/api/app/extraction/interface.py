"""Protocol and base interface for format-specific document extractors."""

from typing import Protocol, runtime_checkable

from app.extraction.models import ExtractionResult


@runtime_checkable
class DocumentExtractor(Protocol):
    """Protocol for document extractors converting raw bytes into structured text."""

    def extract(self, content: bytes, **kwargs: object) -> ExtractionResult:
        """Extract text and structural elements from document bytes.

        Args:
            content: Raw document bytes.
            **kwargs: Optional parser-specific configuration.

        Returns:
            ExtractionResult containing structured text, elements, and metadata.

        Raises:
            ExtractionEmptyError: When document contains zero bytes or whitespace only.
            ExtractionMalformedError: When document syntax is corrupted or unparseable.
            ExtractionEncryptedError: When document requires a decryption password.
            ExtractionImageOnlyError: When document has images but no extractable text.
            ExtractionEncodingError: When text cannot be decoded into valid characters.
            ExtractionUnsupportedTypeError: When the content type cannot be parsed.
        """
        ...
