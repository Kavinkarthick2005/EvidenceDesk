"""PDF document extractor using pypdf for deterministic text extraction."""

import io
from typing import Any

import pypdf
from pypdf.errors import EmptyFileError, FileNotDecryptedError, PdfReadError, PdfStreamError

from app.extraction.exceptions import (
    ExtractionEmptyError,
    ExtractionEncryptedError,
    ExtractionImageOnlyError,
    ExtractionMalformedError,
)
from app.extraction.models import ExtractedElement, ExtractionResult


class PdfExtractor:
    """Extracts text and page metadata from PDF documents using pypdf.

    Preserves 1-indexed page-number metadata for every page.
    Detects encrypted, malformed, empty, and image-only documents cleanly.
    Never claims OCR has occurred (is_ocr_applied is always False).
    """

    def extract(self, content: bytes, **kwargs: object) -> ExtractionResult:
        """Parse PDF bytes into page elements and aggregate text."""
        if not content or not content.strip():
            raise ExtractionEmptyError("PDF document is empty.")

        # Validate PDF magic header
        if b"%PDF-" not in content[:1024]:
            raise ExtractionMalformedError("Document does not contain a valid %PDF header.")

        stream = io.BytesIO(content)

        try:
            reader = pypdf.PdfReader(stream, strict=False)
        except EmptyFileError as err:
            raise ExtractionEmptyError("PDF file is empty.") from err
        except FileNotDecryptedError as err:
            raise ExtractionEncryptedError("PDF file is encrypted or password-protected.") from err
        except (PdfReadError, PdfStreamError, ValueError, KeyError, IndexError, TypeError, RecursionError, MemoryError) as err:
            raise ExtractionMalformedError(f"PDF syntax is corrupted or unparseable: {err}") from err
        except Exception as err:
            raise ExtractionMalformedError(f"Unexpected failure reading PDF structure: {err}") from err

        # Check encryption
        if reader.is_encrypted:
            raise ExtractionEncryptedError("PDF file is encrypted or password-protected.")

        # Check page count
        try:
            total_pages = len(reader.pages)
        except Exception as err:
            raise ExtractionMalformedError(f"Unable to read PDF pages: {err}") from err

        if total_pages == 0:
            raise ExtractionEmptyError("PDF contains no pages.")

        elements: list[ExtractedElement] = []
        total_text_parts: list[str] = []
        total_images = 0

        for idx, page in enumerate(reader.pages):
            page_num = idx + 1
            try:
                page_text = page.extract_text() or ""
            except Exception as err:
                raise ExtractionMalformedError(
                    f"Failed to extract text from page {page_num}: {err}"
                ) from err

            normalized_page_text = page_text.replace("\r\n", "\n").replace("\r", "\n").strip()

            try:
                img_count = len(page.images)
            except Exception:
                img_count = 0

            total_images += img_count

            if normalized_page_text:
                total_text_parts.append(normalized_page_text)

            elements.append(
                ExtractedElement(
                    text=normalized_page_text,
                    page_number=page_num,
                    heading=None,
                    heading_level=None,
                    metadata={"page_number": page_num, "image_count": img_count},
                )
            )

        full_text = "\n\n".join(total_text_parts).strip()

        # Handle documents where no text could be extracted
        if not full_text:
            if total_images > 0:
                raise ExtractionImageOnlyError(
                    "PDF contains images but no extractable text. "
                    "Scanned documents requiring OCR are not supported."
                )
            raise ExtractionEmptyError("PDF contains no extractable text or content.")

        # Safe extraction of PDF document metadata
        pdf_metadata: dict[str, Any] = {}
        try:
            if reader.metadata:
                for key, val in reader.metadata.items():
                    clean_key = str(key).lstrip("/").lower()
                    pdf_metadata[clean_key] = str(val)
        except Exception:
            pass

        return ExtractionResult(
            text=full_text,
            elements=elements,
            content_type="application/pdf",
            total_pages=total_pages,
            character_count=len(full_text),
            word_count=len(full_text.split()),
            is_ocr_applied=False,
            has_images=total_images > 0,
            metadata={
                "pdf_metadata": pdf_metadata,
                "total_images": total_images,
            },
        )
