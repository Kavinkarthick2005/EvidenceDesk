"""Markdown document extractor preserving structure, headings, and raw text."""

from markdown_it import MarkdownIt

from app.extraction.exceptions import (
    ExtractionEmptyError,
    ExtractionEncodingError,
)
from app.extraction.models import ExtractedElement, ExtractionResult
from app.extraction.parsers.text import PlainTextExtractor


class MarkdownExtractor:
    """Extracts text and structural heading hierarchy from Markdown documents.

    Uses MarkdownIt to parse headings (H1-H6) into distinct structural elements
    while preserving the complete original text content and order.
    """

    def __init__(self) -> None:
        self._md = MarkdownIt()
        self._text_extractor = PlainTextExtractor()

    def extract(self, content: bytes, **kwargs: object) -> ExtractionResult:
        """Parse markdown bytes into structured elements with heading hierarchy."""
        if not content or not content.strip():
            raise ExtractionEmptyError("Markdown document is empty or contains only whitespace.")

        try:
            raw_text, encoding = self._text_extractor._decode_content(content)
        except Exception as err:
            raise ExtractionEncodingError(
                f"Failed to decode markdown content: {err}"
            ) from err

        if not raw_text.strip():
            raise ExtractionEmptyError("Markdown document contains only whitespace characters.")

        normalized_text = raw_text.replace("\r\n", "\n").replace("\r", "\n")
        lines = normalized_text.splitlines()

        tokens = self._md.parse(normalized_text)

        # Collect heading positions and check for images
        headings: list[dict[str, object]] = []
        has_images = False

        for i, token in enumerate(tokens):
            if token.type == "heading_open":
                level = int(token.tag[1:]) if len(token.tag) > 1 and token.tag[1:].isdigit() else 1
                title = tokens[i + 1].content if i + 1 < len(tokens) else ""
                start_line = token.map[0] if token.map else 0
                headings.append({
                    "level": level,
                    "title": title,
                    "start_line": start_line,
                })

            if token.children:
                for child in token.children:
                    if child.type == "image":
                        has_images = True

        elements: list[ExtractedElement] = []

        if not headings:
            # Document has no headings; entire content is a single element
            elements.append(
                ExtractedElement(
                    text=normalized_text,
                    heading=None,
                    heading_level=None,
                    metadata={"line_count": len(lines), "encoding": encoding},
                )
            )
        else:
            # Section before the first heading (if any)
            first_start = int(headings[0]["start_line"])  # type: ignore[arg-type]
            if first_start > 0:
                preamble = "\n".join(lines[:first_start]).strip()
                if preamble:
                    elements.append(
                        ExtractedElement(
                            text=preamble,
                            heading=None,
                            heading_level=None,
                            metadata={"start_line": 0, "end_line": first_start},
                        )
                    )

            # Sections under each heading
            for idx, h in enumerate(headings):
                start = int(h["start_line"])  # type: ignore[arg-type]
                end = (
                    int(headings[idx + 1]["start_line"])  # type: ignore[arg-type]
                    if idx + 1 < len(headings)
                    else len(lines)
                )
                section_lines = lines[start:end]
                section_text = "\n".join(section_lines).strip()
                elements.append(
                    ExtractedElement(
                        text=section_text,
                        heading=str(h["title"]),
                        heading_level=int(h["level"]),  # type: ignore[arg-type]
                        metadata={
                            "start_line": start,
                            "end_line": end,
                            "heading_level": int(h["level"]),  # type: ignore[arg-type]
                        },
                    )
                )

        heading_metadata = [
            {"level": h["level"], "title": h["title"], "start_line": h["start_line"]}
            for h in headings
        ]

        return ExtractionResult(
            text=normalized_text,
            elements=elements,
            content_type="text/markdown",
            total_pages=None,
            character_count=len(normalized_text),
            word_count=len(normalized_text.split()),
            is_ocr_applied=False,
            has_images=has_images,
            metadata={
                "encoding": encoding,
                "headings": heading_metadata,
                "heading_count": len(headings),
            },
        )
