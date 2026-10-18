"""Tests for EvidenceDesk document extraction layer."""

import io

from pypdf import PdfWriter
from pypdf.generic import DecodedStreamObject, DictionaryObject, NameObject, NumberObject
import pytest

from app.extraction import (
    ExtractedElement,
    ExtractionEmptyError,
    ExtractionEncodingError,
    ExtractionEncryptedError,
    ExtractionError,
    ExtractionImageOnlyError,
    ExtractionMalformedError,
    ExtractionResult,
    ExtractionService,
    ExtractionUnsupportedTypeError,
    MarkdownExtractor,
    PdfExtractor,
    PlainTextExtractor,
    extract_document,
    get_extraction_service,
)


def _build_pdf(pages: list[str], title: str | None = None, author: str | None = None) -> bytes:
    """Helper to construct a valid in-memory multi-page PDF with text."""
    writer = PdfWriter()
    for text in pages:
        page = writer.add_blank_page(width=400, height=400)
        content = DecodedStreamObject()
        # BT = Begin Text, Tf = Font, Td = Move to coords, Tj = Show text, ET = End Text
        content.set_data(f"BT /F1 12 Tf 50 350 Td ({text}) Tj ET".encode("latin-1"))
        page[NameObject("/Contents")] = writer._add_object(content)

        font_obj = DictionaryObject()
        font_obj[NameObject("/Type")] = NameObject("/Font")
        font_obj[NameObject("/Subtype")] = NameObject("/Type1")
        font_obj[NameObject("/BaseFont")] = NameObject("/Helvetica")
        fonts = DictionaryObject()
        fonts[NameObject("/F1")] = writer._add_object(font_obj)

        resources = DictionaryObject()
        resources[NameObject("/Font")] = fonts
        page[NameObject("/Resources")] = resources

    if title or author:
        meta: dict[str, str] = {}
        if title:
            meta["/Title"] = title
        if author:
            meta["/Author"] = author
        writer.add_metadata(meta)

    out = io.BytesIO()
    writer.write(out)
    return out.getvalue()


def _build_encrypted_pdf(pages: list[str], password: str = "secret123") -> bytes:
    """Helper to construct an encrypted/password-protected PDF."""
    writer = PdfWriter()
    for text in pages:
        page = writer.add_blank_page(width=300, height=300)
        content = DecodedStreamObject()
        content.set_data(f"BT /F1 12 Tf 50 250 Td ({text}) Tj ET".encode("latin-1"))
        page[NameObject("/Contents")] = writer._add_object(content)
        font_obj = DictionaryObject()
        font_obj[NameObject("/Type")] = NameObject("/Font")
        font_obj[NameObject("/Subtype")] = NameObject("/Type1")
        font_obj[NameObject("/BaseFont")] = NameObject("/Helvetica")
        fonts = DictionaryObject()
        fonts[NameObject("/F1")] = writer._add_object(font_obj)
        resources = DictionaryObject()
        resources[NameObject("/Font")] = fonts
        page[NameObject("/Resources")] = resources

    writer.encrypt(password)
    out = io.BytesIO()
    writer.write(out)
    return out.getvalue()


def _build_image_only_pdf() -> bytes:
    """Helper to construct a valid PDF containing an image XObject but zero text."""
    writer = PdfWriter()
    page = writer.add_blank_page(width=300, height=300)

    # 2x2 8-bit RGB image
    img = DecodedStreamObject()
    img[NameObject("/Type")] = NameObject("/XObject")
    img[NameObject("/Subtype")] = NameObject("/Image")
    img[NameObject("/Width")] = NumberObject(2)
    img[NameObject("/Height")] = NumberObject(2)
    img[NameObject("/ColorSpace")] = NameObject("/DeviceRGB")
    img[NameObject("/BitsPerComponent")] = NumberObject(8)
    img.set_data(b"\xff\x00\x00\x00\xff\x00\x00\x00\xff\xff\xff\xff")
    img_ref = writer._add_object(img)

    xobjects = DictionaryObject()
    xobjects[NameObject("/Im1")] = img_ref
    resources = DictionaryObject()
    resources[NameObject("/XObject")] = xobjects
    page[NameObject("/Resources")] = resources

    content = DecodedStreamObject()
    content.set_data(b"q 100 0 0 100 50 50 cm /Im1 Do Q")
    page[NameObject("/Contents")] = writer._add_object(content)

    out = io.BytesIO()
    writer.write(out)
    return out.getvalue()


def _build_pdf_with_text_and_image(text: str = "EvidenceDesk Grounded Text") -> bytes:
    """Helper to construct a valid PDF containing both text and an embedded image."""
    writer = PdfWriter()
    page = writer.add_blank_page(width=400, height=400)

    # 2x2 8-bit RGB image
    img = DecodedStreamObject()
    img[NameObject("/Type")] = NameObject("/XObject")
    img[NameObject("/Subtype")] = NameObject("/Image")
    img[NameObject("/Width")] = NumberObject(2)
    img[NameObject("/Height")] = NumberObject(2)
    img[NameObject("/ColorSpace")] = NameObject("/DeviceRGB")
    img[NameObject("/BitsPerComponent")] = NumberObject(8)
    img.set_data(b"\xff\x00\x00\x00\xff\x00\x00\x00\xff\xff\xff\xff")
    img_ref = writer._add_object(img)

    xobjects = DictionaryObject()
    xobjects[NameObject("/Im1")] = img_ref

    font_obj = DictionaryObject()
    font_obj[NameObject("/Type")] = NameObject("/Font")
    font_obj[NameObject("/Subtype")] = NameObject("/Type1")
    font_obj[NameObject("/BaseFont")] = NameObject("/Helvetica")
    fonts = DictionaryObject()
    fonts[NameObject("/F1")] = writer._add_object(font_obj)

    resources = DictionaryObject()
    resources[NameObject("/Font")] = fonts
    resources[NameObject("/XObject")] = xobjects
    page[NameObject("/Resources")] = resources

    content = DecodedStreamObject()
    stream_data = f"q 100 0 0 100 50 50 cm /Im1 Do Q BT /F1 12 Tf 50 250 Td ({text}) Tj ET"
    content.set_data(stream_data.encode("latin-1"))
    page[NameObject("/Contents")] = writer._add_object(content)

    out = io.BytesIO()
    writer.write(out)
    return out.getvalue()


def _build_blank_pdf(page_count: int = 1) -> bytes:
    """Helper to construct a PDF with blank pages (no text, no images)."""
    writer = PdfWriter()
    for _ in range(page_count):
        writer.add_blank_page(width=200, height=200)
    out = io.BytesIO()
    writer.write(out)
    return out.getvalue()


# ==============================================================================
# PDF Extraction Tests
# ==============================================================================


def test_pdf_normal_multipage_preserves_page_numbers_and_text() -> None:
    extractor = PdfExtractor()
    pdf_bytes = _build_pdf(
        ["EvidenceDesk Page One Content", "EvidenceDesk Page Two Content"],
        title="Architecture Document",
        author="EvidenceDesk Team",
    )

    result = extractor.extract(pdf_bytes)

    assert isinstance(result, ExtractionResult)
    assert result.content_type == "application/pdf"
    assert result.total_pages == 2
    assert result.is_ocr_applied is False
    assert len(result.elements) == 2

    # Check page-number metadata and text preservation
    first_el = result.elements[0]
    assert first_el.page_number == 1
    assert "EvidenceDesk Page One Content" in first_el.text
    assert first_el.metadata["page_number"] == 1

    second_el = result.elements[1]
    assert second_el.page_number == 2
    assert "EvidenceDesk Page Two Content" in second_el.text
    assert second_el.metadata["page_number"] == 2

    assert "EvidenceDesk Page One Content" in result.text
    assert "EvidenceDesk Page Two Content" in result.text
    assert result.character_count > 0
    assert result.word_count > 0

    # Metadata check
    doc_meta = result.metadata.get("pdf_metadata", {})
    assert doc_meta.get("title") == "Architecture Document"
    assert doc_meta.get("author") == "EvidenceDesk Team"


def test_pdf_empty_bytes_raises_empty_error() -> None:
    extractor = PdfExtractor()
    with pytest.raises(ExtractionEmptyError) as exc_info:
        extractor.extract(b"")
    assert "empty" in str(exc_info.value).lower()


def test_pdf_whitespace_bytes_raises_empty_error() -> None:
    extractor = PdfExtractor()
    with pytest.raises(ExtractionEmptyError):
        extractor.extract(b"   \r\n\t  ")


def test_pdf_blank_pages_raises_empty_error() -> None:
    extractor = PdfExtractor()
    blank_pdf = _build_blank_pdf(page_count=2)
    with pytest.raises(ExtractionEmptyError) as exc_info:
        extractor.extract(blank_pdf)
    assert "no extractable text" in str(exc_info.value).lower()


def test_pdf_missing_header_raises_malformed_error() -> None:
    extractor = PdfExtractor()
    with pytest.raises(ExtractionMalformedError) as exc_info:
        extractor.extract(b"Not a real PDF document at all")
    assert "valid %pdf header" in str(exc_info.value).lower()


def test_pdf_corrupted_structure_raises_malformed_error() -> None:
    extractor = PdfExtractor()
    # Has %PDF header but corrupted trailing bytes
    corrupted = b"%PDF-1.4\n1 0 obj << /Type /Catalog >> endobj\ncorrupted syntax%%EOF"
    with pytest.raises(ExtractionMalformedError):
        extractor.extract(corrupted)


def test_pdf_encrypted_raises_encrypted_error() -> None:
    extractor = PdfExtractor()
    encrypted_pdf = _build_encrypted_pdf(["Secret Page Content"])
    with pytest.raises(ExtractionEncryptedError) as exc_info:
        extractor.extract(encrypted_pdf)
    assert "encrypted" in str(exc_info.value).lower()


def test_pdf_image_only_raises_image_only_error() -> None:
    extractor = PdfExtractor()
    image_pdf = _build_image_only_pdf()
    with pytest.raises(ExtractionImageOnlyError) as exc_info:
        extractor.extract(image_pdf)
    error_msg = str(exc_info.value).lower()
    assert "ocr" in error_msg
    assert "images" in error_msg


def test_pdf_never_claims_ocr() -> None:
    extractor = PdfExtractor()
    pdf_bytes = _build_pdf(["Sample factual claim grounded in evidence"])
    result = extractor.extract(pdf_bytes)
    assert result.is_ocr_applied is False


def test_pdf_with_both_text_and_images() -> None:
    extractor = PdfExtractor()
    pdf_bytes = _build_pdf_with_text_and_image("EvidenceDesk Hybrid Page")
    result = extractor.extract(pdf_bytes)

    assert result.content_type == "application/pdf"
    assert result.total_pages == 1
    assert result.is_ocr_applied is False
    assert result.has_images is True
    assert "EvidenceDesk Hybrid Page" in result.text
    assert len(result.elements) == 1
    assert result.elements[0].page_number == 1
    assert result.elements[0].metadata["image_count"] == 1
    assert "EvidenceDesk Hybrid Page" in result.elements[0].text


# ==============================================================================
# Markdown Extraction Tests
# ==============================================================================


def test_markdown_headings_and_structure_hierarchy() -> None:
    extractor = MarkdownExtractor()
    md_content = """# EvidenceDesk Overview
This is introductory text.

## Architecture
Modular monolith design with clear module boundaries.

### Ingestion Pipeline
Text extraction precedes chunking and embedding.

## Conclusion
Traceability is preserved.
""".encode()

    result = extractor.extract(md_content)

    assert isinstance(result, ExtractionResult)
    assert result.content_type == "text/markdown"
    assert result.total_pages is None
    assert result.is_ocr_applied is False
    assert result.character_count > 0

    # Elements structure:
    # 1. H1: EvidenceDesk Overview
    # 2. H2: Architecture
    # 3. H3: Ingestion Pipeline
    # 4. H2: Conclusion
    assert len(result.elements) == 4

    h1_el = result.elements[0]
    assert h1_el.heading == "EvidenceDesk Overview"
    assert h1_el.heading_level == 1
    assert "This is introductory text." in h1_el.text

    h2_el = result.elements[1]
    assert h2_el.heading == "Architecture"
    assert h2_el.heading_level == 2
    assert "Modular monolith design" in h2_el.text

    h3_el = result.elements[2]
    assert h3_el.heading == "Ingestion Pipeline"
    assert h3_el.heading_level == 3
    assert "Text extraction precedes chunking" in h3_el.text

    h2_concl = result.elements[3]
    assert h2_concl.heading == "Conclusion"
    assert h2_concl.heading_level == 2
    assert "Traceability is preserved." in h2_concl.text

    # Complete text is preserved in result.text
    assert "# EvidenceDesk Overview" in result.text
    assert "## Architecture" in result.text
    assert "Traceability is preserved." in result.text


def test_markdown_with_preamble_before_headings() -> None:
    extractor = MarkdownExtractor()
    md_content = """Preamble line 1.
Preamble line 2.

# First Section
Section content here.
""".encode()

    result = extractor.extract(md_content)
    assert len(result.elements) == 2
    preamble_el = result.elements[0]
    assert preamble_el.heading is None
    assert preamble_el.heading_level is None
    assert "Preamble line 1." in preamble_el.text

    first_el = result.elements[1]
    assert first_el.heading == "First Section"
    assert first_el.heading_level == 1
    assert "Section content here." in first_el.text


def test_markdown_without_headings() -> None:
    extractor = MarkdownExtractor()
    md_content = """Just plain markdown without any headers.
A second paragraph with some **bold** and *italic* text.
""".encode()

    result = extractor.extract(md_content)
    assert len(result.elements) == 1
    assert result.elements[0].heading is None
    assert "Just plain markdown" in result.elements[0].text
    assert result.text == result.elements[0].text


def test_markdown_with_embedded_image_flagged() -> None:
    extractor = MarkdownExtractor()
    md_content = """# Diagram Section
Here is the system architecture:
![Architecture Diagram](https://example.com/arch.png)
""".encode()

    result = extractor.extract(md_content)
    assert result.has_images is True


def test_markdown_empty_bytes_raises_empty_error() -> None:
    extractor = MarkdownExtractor()
    with pytest.raises(ExtractionEmptyError):
        extractor.extract(b"")


def test_markdown_whitespace_only_raises_empty_error() -> None:
    extractor = MarkdownExtractor()
    with pytest.raises(ExtractionEmptyError):
        extractor.extract(b"  \n\t  \r\n  ")


def test_markdown_normalizes_crlf() -> None:
    extractor = MarkdownExtractor()
    md_content = b"# Title\r\nLine 1\r\nLine 2\r\n"
    result = extractor.extract(md_content)
    assert "\r\n" not in result.text
    assert "\n" in result.text


def test_markdown_deeply_nested_headings_and_line_metadata() -> None:
    extractor = MarkdownExtractor()
    md_content = """# Level 1
H1 Content.

## Level 2
H2 Content.

### Level 3
H3 Content.

#### Level 4
H4 Content.

## Second Level 2
More H2 content.
""".encode()

    result = extractor.extract(md_content)
    assert len(result.elements) == 5
    levels = [el.heading_level for el in result.elements]
    assert levels == [1, 2, 3, 4, 2]
    titles = [el.heading for el in result.elements]
    assert titles == ["Level 1", "Level 2", "Level 3", "Level 4", "Second Level 2"]
    for el in result.elements:
        assert "start_line" in el.metadata
        assert "end_line" in el.metadata
        assert el.metadata["start_line"] < el.metadata["end_line"]


def test_markdown_orphan_heading_coalesces_with_subsequent_section() -> None:
    extractor = MarkdownExtractor()
    md_content = """# EvidenceDesk Platform Reference Manual

## 1. Storage and Document Limits
The maximum permitted upload size is 25 MB.
""".encode()

    result = extractor.extract(md_content)

    # The orphan H1 should not produce an independent content-free element.
    # It must coalesce into the subsequent section that has body content.
    assert len(result.elements) == 1
    el = result.elements[0]
    assert el.heading == "1. Storage and Document Limits"
    assert el.heading_level == 2
    assert "# EvidenceDesk Platform Reference Manual" in el.text
    assert "The maximum permitted upload size is 25 MB." in el.text
    assert "parent_headings" in el.metadata
    assert el.metadata["parent_headings"] == [
        {"title": "EvidenceDesk Platform Reference Manual", "level": 1, "start_line": 0}
    ]


def test_markdown_multiple_consecutive_orphan_headings() -> None:
    extractor = MarkdownExtractor()
    md_content = """# System Architecture Survey
## Distributed Databases
### 1. Vector Search Engines
Passages are indexed into dense vectors.
""".encode()

    result = extractor.extract(md_content)

    assert len(result.elements) == 1
    el = result.elements[0]
    assert el.heading == "1. Vector Search Engines"
    assert el.heading_level == 3
    assert "# System Architecture Survey" in el.text
    assert "## Distributed Databases" in el.text
    assert "Passages are indexed into dense vectors." in el.text
    assert "parent_headings" in el.metadata
    assert len(el.metadata["parent_headings"]) == 2
    assert el.metadata["parent_headings"][0]["title"] == "System Architecture Survey"
    assert el.metadata["parent_headings"][1]["title"] == "Distributed Databases"


def test_markdown_trailing_orphan_heading_coalesces_with_preceding_section() -> None:
    extractor = MarkdownExtractor()
    md_content = """# Section One
Body text of section one.

## Empty Trailing Section
""".encode()

    result = extractor.extract(md_content)

    # The trailing orphan heading must not produce a content-free title-only chunk.
    # It coalesces with the preceding section and is recorded in trailing_headings metadata.
    assert len(result.elements) == 1
    el = result.elements[0]
    assert el.heading == "Section One"
    assert "Body text of section one." in el.text
    assert "## Empty Trailing Section" in el.text
    assert "trailing_headings" in el.metadata
    assert el.metadata["trailing_headings"] == [
        {"title": "Empty Trailing Section", "level": 2, "start_line": 3}
    ]


def test_markdown_document_containing_only_headings() -> None:
    extractor = MarkdownExtractor()
    md_content = """# Main Title
## Subheading 1
### Subheading 2
""".encode()

    result = extractor.extract(md_content)

    # Document with only headings produces a single element covering all headings
    assert len(result.elements) == 1
    el = result.elements[0]
    assert el.heading == "Main Title"
    assert "# Main Title" in el.text
    assert "## Subheading 1" in el.text
    assert "### Subheading 2" in el.text


def test_markdown_heading_followed_by_blank_lines_before_next_heading() -> None:
    extractor = MarkdownExtractor()
    md_content = """# Main Heading



## Section Two
Body text of section two.
""".encode()

    result = extractor.extract(md_content)

    assert len(result.elements) == 1
    el = result.elements[0]
    assert el.heading == "Section Two"
    assert el.heading_level == 2
    assert "# Main Heading" in el.text
    assert "## Section Two" in el.text
    assert "Body text of section two." in el.text
    assert "parent_headings" in el.metadata
    assert el.metadata["parent_headings"][0]["title"] == "Main Heading"



# ==============================================================================
# Plain Text Extraction Tests
# ==============================================================================


def test_text_utf8_success() -> None:
    extractor = PlainTextExtractor()
    content = "EvidenceDesk local-first document retrieval platform.".encode("utf-8")
    result = extractor.extract(content)

    assert result.content_type == "text/plain"
    assert result.text == "EvidenceDesk local-first document retrieval platform."
    assert result.metadata["encoding"] == "utf-8"
    assert len(result.elements) == 1
    assert result.is_ocr_applied is False
    assert result.has_images is False


def test_text_utf8_sig_bom_success() -> None:
    extractor = PlainTextExtractor()
    content = "\ufeffEvidenceDesk with UTF-8 BOM marker".encode("utf-8-sig")
    result = extractor.extract(content)

    assert "EvidenceDesk with UTF-8 BOM marker" in result.text
    assert result.metadata["encoding"] == "utf-8-sig"


def test_text_utf16_le_bom_success() -> None:
    extractor = PlainTextExtractor()
    content = "EvidenceDesk encoded in UTF-16 LE".encode("utf-16-le")
    # Prepend LE BOM
    full_bytes = b"\xff\xfe" + content
    result = extractor.extract(full_bytes)

    assert "EvidenceDesk encoded in UTF-16 LE" in result.text
    assert result.metadata["encoding"] == "utf-16-le"


def test_text_utf16_be_bom_success() -> None:
    extractor = PlainTextExtractor()
    content = "EvidenceDesk encoded in UTF-16 BE".encode("utf-16-be")
    # Prepend BE BOM
    full_bytes = b"\xfe\xff" + content
    result = extractor.extract(full_bytes)

    assert "EvidenceDesk encoded in UTF-16 BE" in result.text
    assert result.metadata["encoding"] == "utf-16-be"


def test_text_windows_1252_success() -> None:
    extractor = PlainTextExtractor()
    # 0x93 and 0x94 are Windows-1252 left and right smart quotes (invalid in strict UTF-8)
    content = b"EvidenceDesk \x93smart quotes\x94 test"
    result = extractor.extract(content)

    assert "smart quotes" in result.text
    assert result.metadata["encoding"] in ("windows-1252", "iso-8859-1")


def test_text_empty_bytes_raises_empty_error() -> None:
    extractor = PlainTextExtractor()
    with pytest.raises(ExtractionEmptyError):
        extractor.extract(b"")


def test_text_whitespace_only_raises_empty_error() -> None:
    extractor = PlainTextExtractor()
    with pytest.raises(ExtractionEmptyError):
        extractor.extract(b"   \r\n\t   \n")


def test_text_binary_null_bytes_raises_encoding_error() -> None:
    extractor = PlainTextExtractor()
    # Arbitrary binary payload with null bytes
    binary_content = b"\x7fELF\x02\x01\x01\x00\x00\x00\x00\x00\x00\x00\x00\x00"
    with pytest.raises(ExtractionEncodingError) as exc_info:
        extractor.extract(binary_content)
    assert "binary" in str(exc_info.value).lower() or "null" in str(exc_info.value).lower()


def test_text_rejects_binary_control_characters_in_utf8() -> None:
    extractor = PlainTextExtractor()
    content = b"Valid text prefix \x01\x02\x03 corrupted binary suffix"
    with pytest.raises(ExtractionEncodingError) as exc_info:
        extractor.extract(content)
    error_msg = str(exc_info.value).lower()
    assert "binary" in error_msg or "control" in error_msg


def test_text_newline_normalization() -> None:
    extractor = PlainTextExtractor()
    content = b"Line 1\r\nLine 2\rLine 3\nLine 4"
    result = extractor.extract(content)
    assert "\r" not in result.text
    assert result.text == "Line 1\nLine 2\nLine 3\nLine 4"


# ==============================================================================
# Unified ExtractionService Tests
# ==============================================================================


def test_service_dispatches_pdf() -> None:
    service = ExtractionService()
    pdf_bytes = _build_pdf(["Single page test"])
    result = service.extract(pdf_bytes, content_type="application/pdf")
    assert result.content_type == "application/pdf"
    assert result.total_pages == 1
    assert "Single page test" in result.text


def test_service_dispatches_markdown() -> None:
    service = ExtractionService()
    md_bytes = b"# Header\nMarkdown content."
    result = service.extract(md_bytes, content_type="text/markdown")
    assert result.content_type == "text/markdown"
    assert len(result.elements) == 1
    assert result.elements[0].heading == "Header"


def test_service_dispatches_text() -> None:
    service = ExtractionService()
    txt_bytes = b"Plain text note."
    result = service.extract(txt_bytes, content_type="text/plain")
    assert result.content_type == "text/plain"
    assert result.text == "Plain text note."


def test_service_infers_from_filename_when_content_type_missing() -> None:
    service = ExtractionService()
    txt_bytes = b"Text by filename."
    result = service.extract(txt_bytes, filename="notes.txt")
    assert result.content_type == "text/plain"

    md_bytes = b"# Heading\nMarkdown by filename."
    result_md = service.extract(md_bytes, filename="guide.markdown")
    assert result_md.content_type == "text/markdown"

    pdf_bytes = _build_pdf(["PDF by filename"])
    result_pdf = service.extract(pdf_bytes, filename="paper.pdf")
    assert result_pdf.content_type == "application/pdf"


def test_service_cleans_mime_parameters() -> None:
    service = ExtractionService()
    txt_bytes = b"Plain text with charset in content-type."
    result = service.extract(txt_bytes, content_type="text/plain; charset=utf-8")
    assert result.content_type == "text/plain"


def test_service_unsupported_format_raises_error() -> None:
    service = ExtractionService()
    with pytest.raises(ExtractionUnsupportedTypeError) as exc_info:
        service.extract(b"Some data", content_type="application/vnd.ms-excel", filename="data.xls")
    assert "unsupported" in str(exc_info.value).lower()


def test_service_rejects_mismatched_mime_and_extension() -> None:
    service = ExtractionService()
    with pytest.raises(ExtractionUnsupportedTypeError) as exc_info:
        service.extract(b"Some data", content_type="application/pdf", filename="document.txt")
    assert "mismatched" in str(exc_info.value).lower()


def test_service_rejects_unsupported_extension_even_if_mime_supported() -> None:
    service = ExtractionService()
    with pytest.raises(ExtractionUnsupportedTypeError) as exc_info:
        service.extract(b"Some data", content_type="application/pdf", filename="exploit.exe")
    assert "unsupported file extension" in str(exc_info.value).lower()


def test_service_rejects_unsupported_mime_even_if_extension_supported() -> None:
    service = ExtractionService()
    with pytest.raises(ExtractionUnsupportedTypeError) as exc_info:
        service.extract(b"Some data", content_type="application/x-sh", filename="script.pdf")
    assert "unsupported content-type" in str(exc_info.value).lower()


def test_service_infers_from_generic_octet_stream_mime() -> None:
    service = ExtractionService()
    txt_bytes = b"Generic stream text."
    result = service.extract(
        txt_bytes,
        content_type="application/octet-stream",
        filename="notes.txt",
    )
    assert result.content_type == "text/plain"
    assert result.text == "Generic stream text."


def test_convenience_extract_document_helper() -> None:
    txt_bytes = b"Convenience helper text."
    result = extract_document(txt_bytes, filename="helper.txt")
    assert result.text == "Convenience helper text."
    assert result.content_type == "text/plain"


def test_extraction_deterministic_for_identical_input() -> None:
    service = get_extraction_service()
    pdf_bytes = _build_pdf(["Deterministic test page 1", "Deterministic test page 2"])

    res1 = service.extract(pdf_bytes, content_type="application/pdf")
    res2 = service.extract(pdf_bytes, content_type="application/pdf")

    assert res1.text == res2.text
    assert res1.total_pages == res2.total_pages
    assert len(res1.elements) == len(res2.elements)
    for el1, el2 in zip(res1.elements, res2.elements):
        assert el1.page_number == el2.page_number
        assert el1.text == el2.text
        assert el1.heading == el2.heading


def test_hierarchy_of_domain_exceptions() -> None:
    assert issubclass(ExtractionEmptyError, ExtractionError)
    assert issubclass(ExtractionMalformedError, ExtractionError)
    assert issubclass(ExtractionEncryptedError, ExtractionError)
    assert issubclass(ExtractionImageOnlyError, ExtractionError)
    assert issubclass(ExtractionEncodingError, ExtractionError)
    assert issubclass(ExtractionUnsupportedTypeError, ExtractionError)
