"""Plain text document extractor with deliberate encoding handling."""

import unicodedata

from app.extraction.exceptions import (
    ExtractionEmptyError,
    ExtractionEncodingError,
)
from app.extraction.models import ExtractedElement, ExtractionResult


class PlainTextExtractor:
    """Extracts text from plain text files with explicit encoding detection.

    Supported encodings:
    - UTF-8 (with or without BOM)
    - UTF-16-LE / UTF-16-BE (with BOM)
    - Windows-1252 / ISO-8859-1 (Latin-1)
    - ASCII

    Ensures conservative normalization (standardizing line endings to \\n)
    without silently corrupting or discarding content. Rejects binary payloads
    containing null bytes or unprintable control sequences.
    """

    ALLOWED_CONTROL_CHARS = frozenset({"\t", "\n", "\r", "\f"})

    def extract(self, content: bytes, **kwargs: object) -> ExtractionResult:
        """Decode raw text bytes and produce structured ExtractionResult."""
        if not content or not content.strip():
            raise ExtractionEmptyError("Text document is empty or contains only whitespace.")

        # Reject binary null bytes
        if b"\x00" in content:
            # Check if this could be valid UTF-16 with BOM
            if content.startswith((b"\xff\xfe", b"\xfe\xff")):
                pass  # Legitimate UTF-16 null bytes in character code points
            else:
                raise ExtractionEncodingError(
                    "Binary data or null bytes detected in plain text document."
                )

        text, encoding = self._decode_content(content)
        self._validate_text_sanity(text)

        # Check for whitespace-only after decoding
        if not text.strip():
            raise ExtractionEmptyError("Text document contains only whitespace characters.")

        # Conservative line ending normalization
        normalized_text = self._normalize_text(text)

        element = ExtractedElement(
            text=normalized_text,
            metadata={
                "encoding": encoding,
                "line_count": len(normalized_text.splitlines()),
            },
        )

        return ExtractionResult(
            text=normalized_text,
            elements=[element],
            content_type="text/plain",
            total_pages=None,
            character_count=len(normalized_text),
            word_count=len(normalized_text.split()),
            is_ocr_applied=False,
            has_images=False,
            metadata={"encoding": encoding},
        )

    def _decode_content(self, content: bytes) -> tuple[str, str]:
        """Decode bytes using deliberate encoding resolution without data loss."""
        # 1. Check for explicit BOM markers
        if content.startswith(b"\xef\xbb\xbf"):
            try:
                return content[3:].decode("utf-8"), "utf-8-sig"
            except UnicodeDecodeError as err:
                raise ExtractionEncodingError("Invalid UTF-8 with BOM.") from err

        if content.startswith(b"\xff\xfe"):
            try:
                # Strip leading BOM character if decoded as utf-16-le
                decoded = content.decode("utf-16-le").lstrip("\ufeff")
                return decoded, "utf-16-le"
            except UnicodeDecodeError as err:
                raise ExtractionEncodingError("Invalid UTF-16-LE content.") from err

        if content.startswith(b"\xfe\xff"):
            try:
                decoded = content.decode("utf-16-be").lstrip("\ufeff")
                return decoded, "utf-16-be"
            except UnicodeDecodeError as err:
                raise ExtractionEncodingError("Invalid UTF-16-BE content.") from err

        # 2. Try UTF-8 strict
        try:
            return content.decode("utf-8"), "utf-8"
        except UnicodeDecodeError:
            pass

        # 3. Try Windows-1252 (strict)
        try:
            decoded = content.decode("cp1252")
            self._validate_text_sanity(decoded)
            return decoded, "windows-1252"
        except (UnicodeDecodeError, ExtractionEncodingError):
            pass

        # 4. Try ISO-8859-1 (Latin-1) with sanity validation
        try:
            decoded = content.decode("iso-8859-1")
            self._validate_text_sanity(decoded)
            return decoded, "iso-8859-1"
        except (UnicodeDecodeError, ExtractionEncodingError):
            pass

        raise ExtractionEncodingError(
            "Unable to decode text using supported encodings (UTF-8, UTF-16, Windows-1252, ISO-8859-1)."
        )

    def _validate_text_sanity(self, text: str) -> None:
        """Ensure decoded text does not contain binary control characters or null bytes."""
        if "\x00" in text:
            raise ExtractionEncodingError("Decoded text contains null characters.")
        for c in text:
            if unicodedata.category(c) == "Cc" and c not in self.ALLOWED_CONTROL_CHARS:
                raise ExtractionEncodingError(
                    f"Binary control character U+{ord(c):04X} detected in plain text document."
                )

    @staticmethod
    def _normalize_text(text: str) -> str:
        """Conservatively normalize line breaks to \\n without touching content."""
        return text.replace("\r\n", "\n").replace("\r", "\n")
