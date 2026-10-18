"""Data structures and representations for extracted document content."""

from typing import Any

from pydantic import BaseModel, Field


class ExtractedElement(BaseModel):
    """Represents a discrete structural unit (page, heading section, or paragraph).

    Attributes:
        text: Normalized text content of the element.
        page_number: 1-indexed page number if extracted from a paginated format (e.g. PDF).
        heading: Heading text if extracted from a structured document (e.g. Markdown).
        heading_level: Integer level of the heading (e.g. 1 for #, 2 for ##).
        metadata: Format-specific structural metadata (e.g. image count, line ranges).
    """

    text: str
    page_number: int | None = None
    heading: str | None = None
    heading_level: int | None = None
    metadata: dict[str, Any] = Field(default_factory=dict)


class ExtractionResult(BaseModel):
    """Normalized, deterministic extraction output across all document formats.

    Attributes:
        text: Full normalized document text.
        elements: Ordered list of structural units (pages, sections, paragraphs).
        content_type: MIME media type of the extracted document.
        total_pages: Total page count for paginated documents, or None.
        character_count: Total character count of full text.
        word_count: Total word count of full text.
        is_ocr_applied: Always False in the initial release (never claim OCR occurred).
        has_images: True if embedded images or visual graphics were detected.
        metadata: Document-level metadata (e.g. author, title, creation date, encoding).
    """

    text: str
    elements: list[ExtractedElement] = Field(default_factory=list)
    content_type: str
    total_pages: int | None = None
    character_count: int = 0
    word_count: int = 0
    is_ocr_applied: bool = False
    has_images: bool = False
    metadata: dict[str, Any] = Field(default_factory=dict)

    def model_post_init(self, __context: Any) -> None:
        """Ensure character_count and word_count accurately reflect full text."""
        self.character_count = len(self.text)
        self.word_count = len(self.text.split())
