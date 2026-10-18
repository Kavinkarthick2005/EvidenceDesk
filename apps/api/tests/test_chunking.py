"""Unit tests for structure-aware document chunking."""

from pathlib import Path
import socket
import uuid

import pytest

from app.chunking import (
    Chunk,
    ChunkingConfig,
    ChunkingConfigError,
    ChunkingEmptyError,
    ChunkingError,
    ChunkingResult,
    StructureAwareChunker,
    TiktokenTokenizer,
    chunk_document,
    get_chunker,
)
from app.extraction.models import ExtractedElement, ExtractionResult


@pytest.fixture
def tokenizer() -> TiktokenTokenizer:
    return TiktokenTokenizer()


@pytest.fixture
def chunker(tokenizer: TiktokenTokenizer) -> StructureAwareChunker:
    return StructureAwareChunker(tokenizer=tokenizer)


# ==============================================================================
# Configuration Validation Tests
# ==============================================================================


def test_invalid_config_nonpositive_chunk_size() -> None:
    with pytest.raises(ChunkingConfigError) as exc_info:
        ChunkingConfig(chunk_size_tokens=0)
    assert "positive" in str(exc_info.value).lower()

    with pytest.raises(ChunkingConfigError):
        ChunkingConfig(chunk_size_tokens=-50)


def test_invalid_config_negative_overlap() -> None:
    with pytest.raises(ChunkingConfigError) as exc_info:
        ChunkingConfig(chunk_size_tokens=100, chunk_overlap_tokens=-1)
    assert "non-negative" in str(exc_info.value).lower()


def test_invalid_config_overlap_greater_than_or_equal_to_size() -> None:
    with pytest.raises(ChunkingConfigError) as exc_info:
        ChunkingConfig(chunk_size_tokens=100, chunk_overlap_tokens=100)
    assert "strictly less than" in str(exc_info.value).lower()

    with pytest.raises(ChunkingConfigError):
        ChunkingConfig(chunk_size_tokens=100, chunk_overlap_tokens=150)


def test_valid_default_config() -> None:
    cfg = ChunkingConfig()
    assert cfg.chunk_size_tokens == 500
    assert cfg.chunk_overlap_tokens == 75


# ==============================================================================
# Empty & Whitespace Input Tests
# ==============================================================================


def test_empty_input_raises_chunking_empty_error(chunker: StructureAwareChunker) -> None:
    empty_result = ExtractionResult(
        text="",
        elements=[],
        content_type="text/plain",
        character_count=0,
        word_count=0,
    )
    with pytest.raises(ChunkingEmptyError) as exc_info:
        chunker.chunk(empty_result)
    assert "empty" in str(exc_info.value).lower()


def test_whitespace_only_input_raises_chunking_empty_error(
    chunker: StructureAwareChunker,
) -> None:
    whitespace_result = ExtractionResult(
        text="   \n\t  \r\n  ",
        elements=[ExtractedElement(text="   \n\t  ")],
        content_type="text/plain",
        character_count=13,
        word_count=0,
    )
    with pytest.raises(ChunkingEmptyError):
        chunker.chunk(whitespace_result)


# ==============================================================================
# Basic Chunking & Size Limit Tests
# ==============================================================================


def test_short_document_fits_in_one_chunk(chunker: StructureAwareChunker) -> None:
    text = "EvidenceDesk is a local-first, evidence-grounded document question-answering platform."
    doc_result = ExtractionResult(
        text=text,
        elements=[ExtractedElement(text=text)],
        content_type="text/plain",
        character_count=len(text),
        word_count=len(text.split()),
    )
    doc_id = uuid.uuid4()
    result = chunker.chunk(doc_result, document_id=doc_id)

    assert isinstance(result, ChunkingResult)
    assert result.total_chunks == 1
    assert len(result.chunks) == 1
    assert result.document_id == doc_id

    chunk = result.chunks[0]
    assert chunk.chunk_index == 0
    assert chunk.text == text
    assert chunk.token_count > 0
    assert chunk.token_count <= 500
    assert chunk.start_char_offset == 0
    assert chunk.end_char_offset == len(text)
    assert chunk.source_format == "text/plain"


def test_multiple_chunks_and_overlap_correctness(chunker: StructureAwareChunker) -> None:
    # Build text with multiple sentences totaling ~150 tokens
    sentences = [
        f"Sentence number {i} describes a distinct unit of factual evidence in EvidenceDesk."
        for i in range(1, 16)
    ]
    full_text = " ".join(sentences)
    doc_result = ExtractionResult(
        text=full_text,
        elements=[ExtractedElement(text=full_text)],
        content_type="text/plain",
        character_count=len(full_text),
        word_count=len(full_text.split()),
    )

    # Configure small chunks: size 40 tokens, overlap 15 tokens
    cfg = ChunkingConfig(chunk_size_tokens=40, chunk_overlap_tokens=15)
    result = chunker.chunk(doc_result, config=cfg)

    assert result.total_chunks > 1
    assert len(result.chunks) == result.total_chunks

    # Check index continuity and bounds
    for idx, c in enumerate(result.chunks):
        assert c.chunk_index == idx
        assert c.token_count <= 40
        assert c.token_count > 0
        assert c.start_char_offset < c.end_char_offset

    # Check overlap between consecutive chunks
    for i in range(len(result.chunks) - 1):
        c1 = result.chunks[i]
        c2 = result.chunks[i + 1]
        assert c2.start_char_offset > c1.start_char_offset, "Start offsets must strictly advance"
        assert c2.start_char_offset < c1.end_char_offset, "Consecutive chunks should overlap"


def test_exact_token_limit_boundary(chunker: StructureAwareChunker, tokenizer: TiktokenTokenizer) -> None:
    # Construct text whose tokens match chunk_size exactly
    cfg = ChunkingConfig(chunk_size_tokens=20, chunk_overlap_tokens=5)
    sentence = "One two three four five six seven eight nine ten."
    tokens = tokenizer.encode(sentence)
    # Pad to exact 20 tokens
    needed = 20 - len(tokens)
    extra_words = " word" * needed
    exact_text = sentence + extra_words
    assert tokenizer.count_tokens(exact_text) == 20

    doc_result = ExtractionResult(
        text=exact_text,
        elements=[ExtractedElement(text=exact_text)],
        content_type="text/plain",
        character_count=len(exact_text),
        word_count=len(exact_text.split()),
    )

    result = chunker.chunk(doc_result, config=cfg)
    assert result.total_chunks == 1
    assert result.chunks[0].token_count == 20


# ==============================================================================
# Metadata Preservation Tests (Markdown & PDF)
# ==============================================================================


def test_markdown_heading_metadata_propagation(chunker: StructureAwareChunker) -> None:
    section1 = "First section content under introduction."
    section2 = "Second section content under architecture with more detailed information."
    full_text = f"# Introduction\n{section1}\n\n## Architecture\n{section2}"

    doc_result = ExtractionResult(
        text=full_text,
        elements=[
            ExtractedElement(
                text=section1,
                heading="Introduction",
                heading_level=1,
                metadata={"heading": "Introduction", "start_line": 0, "end_line": 2},
            ),
            ExtractedElement(
                text=section2,
                heading="Architecture",
                heading_level=2,
                metadata={"heading": "Architecture", "start_line": 3, "end_line": 5},
            ),
        ],
        content_type="text/markdown",
        character_count=len(full_text),
        word_count=len(full_text.split()),
    )

    result = chunker.chunk(doc_result)
    assert len(result.chunks) == 2

    chunk_intro = result.chunks[0]
    assert chunk_intro.heading == "Introduction"
    assert chunk_intro.heading_level == 1
    assert chunk_intro.source_format == "text/markdown"
    assert section1 in chunk_intro.text

    chunk_arch = result.chunks[1]
    assert chunk_arch.heading == "Architecture"
    assert chunk_arch.heading_level == 2
    assert chunk_arch.source_format == "text/markdown"
    assert section2 in chunk_arch.text


def test_pdf_page_metadata_propagation(chunker: StructureAwareChunker) -> None:
    p1_text = "Page 1: EvidenceDesk ingestion pipeline."
    p2_text = "Page 2: Vector search and retrieval mechanisms."
    full_text = f"{p1_text}\n\n{p2_text}"

    doc_result = ExtractionResult(
        text=full_text,
        elements=[
            ExtractedElement(text=p1_text, page_number=1, metadata={"page_number": 1}),
            ExtractedElement(text=p2_text, page_number=2, metadata={"page_number": 2}),
        ],
        content_type="application/pdf",
        total_pages=2,
        character_count=len(full_text),
        word_count=len(full_text.split()),
    )

    result = chunker.chunk(doc_result)
    assert len(result.chunks) == 2

    c1 = result.chunks[0]
    assert c1.page_number == 1
    assert c1.page_numbers == [1]
    assert c1.source_format == "application/pdf"
    assert p1_text in c1.text

    c2 = result.chunks[1]
    assert c2.page_number == 2
    assert c2.page_numbers == [2]
    assert c2.source_format == "application/pdf"
    assert p2_text in c2.text


# ==============================================================================
# Long Paragraph & Oversized Element Tests
# ==============================================================================


def test_long_unbroken_element_sub_splits_safely(chunker: StructureAwareChunker) -> None:
    # A single continuous sentence with 100 repeated words without sentence terminators
    unbroken = "word " * 120
    doc_result = ExtractionResult(
        text=unbroken.strip(),
        elements=[ExtractedElement(text=unbroken.strip())],
        content_type="text/plain",
        character_count=len(unbroken.strip()),
        word_count=120,
    )

    cfg = ChunkingConfig(chunk_size_tokens=30, chunk_overlap_tokens=10)
    result = chunker.chunk(doc_result, config=cfg)

    assert result.total_chunks > 1
    for c in result.chunks:
        assert c.token_count <= 30
        assert c.token_count > 0


# ==============================================================================
# Determinism & Unicode Tests
# ==============================================================================


def test_chunking_deterministic_for_identical_input(chunker: StructureAwareChunker) -> None:
    text = (
        "EvidenceDesk ensures traceability. Every answer references retrieved passages. "
        "Evaluation results are reproducible across multiple runs."
    )
    doc_result = ExtractionResult(
        text=text,
        elements=[ExtractedElement(text=text)],
        content_type="text/plain",
        character_count=len(text),
        word_count=len(text.split()),
    )
    cfg = ChunkingConfig(chunk_size_tokens=20, chunk_overlap_tokens=5)

    res1 = chunker.chunk(doc_result, config=cfg)
    res2 = chunker.chunk(doc_result, config=cfg)

    assert res1.total_chunks == res2.total_chunks
    assert res1.total_tokens == res2.total_tokens
    for c1, c2 in zip(res1.chunks, res2.chunks):
        assert c1.chunk_index == c2.chunk_index
        assert c1.text == c2.text
        assert c1.token_count == c2.token_count
        assert c1.start_char_offset == c2.start_char_offset
        assert c1.end_char_offset == c2.end_char_offset


def test_unicode_and_punctuation_handling(chunker: StructureAwareChunker) -> None:
    unicode_text = (
        "EvidenceDesk vérifie les données: «citation 1» & «modèle 2» — 100% exactitude! "
        "Japanese text: 日本語のテスト文章です。Emoji test: 🚀 🔍 📊. German umlauts: Äpfel, Ökologie, Überprüfung."
    )
    doc_result = ExtractionResult(
        text=unicode_text,
        elements=[ExtractedElement(text=unicode_text)],
        content_type="text/plain",
        character_count=len(unicode_text),
        word_count=len(unicode_text.split()),
    )
    result = chunker.chunk(doc_result)

    assert result.total_chunks >= 1
    assert "日本語" in result.chunks[0].text
    assert "🚀" in result.chunks[0].text
    assert "Überprüfung" in result.chunks[0].text


def test_convenience_chunk_document_function() -> None:
    text = "Convenience function test passage."
    doc_result = ExtractionResult(
        text=text,
        elements=[ExtractedElement(text=text)],
        content_type="text/plain",
        character_count=len(text),
        word_count=len(text.split()),
    )
    result = chunk_document(doc_result)
    assert result.total_chunks == 1
    assert result.chunks[0].text == text


def test_character_offset_accuracy(chunker: StructureAwareChunker) -> None:
    text = (
        "EvidenceDesk grounds claims in retrieved evidence. "
        "Every citation refers to a real passage. "
        "Evaluations compare dense and lexical retrieval."
    )
    doc_result = ExtractionResult(
        text=text,
        elements=[ExtractedElement(text=text)],
        content_type="text/plain",
        character_count=len(text),
        word_count=len(text.split()),
    )
    cfg = ChunkingConfig(chunk_size_tokens=15, chunk_overlap_tokens=5)
    result = chunker.chunk(doc_result, config=cfg)

    assert result.total_chunks > 1
    for c in result.chunks:
        # Verify exact character substring match against the source document text
        source_slice = text[c.start_char_offset : c.end_char_offset]
        assert source_slice == c.text


def test_zero_overlap_allowed(chunker: StructureAwareChunker) -> None:
    text = "First block of content. Second block of content. Third block of content."
    doc_result = ExtractionResult(
        text=text,
        elements=[ExtractedElement(text=text)],
        content_type="text/plain",
        character_count=len(text),
        word_count=len(text.split()),
    )
    cfg = ChunkingConfig(chunk_size_tokens=10, chunk_overlap_tokens=0)
    result = chunker.chunk(doc_result, config=cfg)

    assert result.total_chunks > 1
    # Check that chunks do not overlap
    for i in range(len(result.chunks) - 1):
        c1 = result.chunks[i]
        c2 = result.chunks[i + 1]
        assert c2.start_char_offset >= c1.end_char_offset


def test_document_id_preservation(chunker: StructureAwareChunker) -> None:
    doc_id = uuid.uuid4()
    text = "Document identity must be preserved across all chunks."
    doc_result = ExtractionResult(
        text=text,
        elements=[ExtractedElement(text=text)],
        content_type="text/plain",
        character_count=len(text),
        word_count=len(text.split()),
    )
    result = chunker.chunk(doc_result, document_id=doc_id)
    assert result.document_id == doc_id
    assert all(c.document_id == doc_id for c in result.chunks)


def test_pdf_multipage_oversized_page_splitting(chunker: StructureAwareChunker) -> None:
    short_page = "Short title on page 1."
    long_page = " ".join([f"Detailed evidence item {i} on page 2." for i in range(25)])
    full_text = f"{short_page}\n\n{long_page}"

    doc_result = ExtractionResult(
        text=full_text,
        elements=[
            ExtractedElement(text=short_page, page_number=1, metadata={"page_number": 1}),
            ExtractedElement(text=long_page, page_number=2, metadata={"page_number": 2}),
        ],
        content_type="application/pdf",
        total_pages=2,
        character_count=len(full_text),
        word_count=len(full_text.split()),
    )

    cfg = ChunkingConfig(chunk_size_tokens=30, chunk_overlap_tokens=10)
    result = chunker.chunk(doc_result, config=cfg)

    # Page 1 should produce 1 chunk; Page 2 should produce multiple chunks
    page_1_chunks = [c for c in result.chunks if c.page_number == 1]
    page_2_chunks = [c for c in result.chunks if c.page_number == 2]

    assert len(page_1_chunks) == 1
    assert len(page_2_chunks) > 1
    assert page_1_chunks[0].chunk_index == 0
    assert page_2_chunks[0].chunk_index == 1
    assert page_1_chunks[0].page_numbers == [1]
    assert all(c.page_numbers == [2] for c in page_2_chunks)


def test_hierarchy_of_chunking_exceptions() -> None:
    assert issubclass(ChunkingConfigError, ChunkingError)
    assert issubclass(ChunkingEmptyError, ChunkingError)


# ==============================================================================
# Offline Tokenizer & Integrity Tests
# ==============================================================================


def test_tokenizer_offline_with_blocked_network_and_empty_cache(
    monkeypatch: pytest.MonkeyPatch, tmp_path: Path
) -> None:
    """Verify tokenizer initializes offline without network access and with empty cache."""
    # Point cache dirs to empty temp dir
    monkeypatch.setenv("TIKTOKEN_CACHE_DIR", str(tmp_path / "tiktoken_cache"))
    monkeypatch.setenv("DATA_GYM_CACHE_DIR", str(tmp_path / "data_gym_cache"))

    # Block socket connections to guarantee zero network access
    def block_connect(*args: object, **kwargs: object) -> None:
        raise RuntimeError("Blocked network access attempted during offline tokenizer initialization")

    monkeypatch.setattr(socket.socket, "connect", block_connect)

    tok = TiktokenTokenizer()
    text = "EvidenceDesk strictly enforces local-first offline execution."
    encoded = tok.encode(text)
    assert len(encoded) > 0
    decoded = tok.decode(encoded)
    assert decoded == text
    assert tok.count_tokens(text) == len(encoded)

    # Confirm cache directories remained completely empty
    cache_dir = tmp_path / "tiktoken_cache"
    assert not cache_dir.exists() or len(list(cache_dir.iterdir())) == 0


def test_tokenizer_asset_integrity_check(tmp_path: Path) -> None:
    """Corrupted vocabulary asset must trigger an integrity validation error."""
    corrupted_asset = tmp_path / "corrupted_cl100k.tiktoken"
    corrupted_asset.write_bytes(b"corrupted invalid token vocabulary content")

    with pytest.raises(ChunkingError) as exc_info:
        TiktokenTokenizer(asset_path=corrupted_asset)
    assert "integrity check failed" in str(exc_info.value).lower()


def test_tokenizer_missing_asset_error(tmp_path: Path) -> None:
    """Missing vocabulary asset must raise ChunkingError rather than attempting network download."""
    missing_asset = tmp_path / "non_existent.tiktoken"
    with pytest.raises(ChunkingError) as exc_info:
        TiktokenTokenizer(asset_path=missing_asset)
    assert "not found" in str(exc_info.value).lower()


# ==============================================================================
# Character Offset Fidelity Invariant Tests
# ==============================================================================


def test_offset_fidelity_case1_oversized_unbroken_text(chunker: StructureAwareChunker) -> None:
    """Case 1: Oversized unbroken text split into token windows."""
    unbroken = "word " * 120
    doc_result = ExtractionResult(
        text=unbroken.strip(),
        elements=[ExtractedElement(text=unbroken.strip())],
        content_type="text/plain",
        character_count=len(unbroken.strip()),
        word_count=120,
    )
    cfg = ChunkingConfig(chunk_size_tokens=30, chunk_overlap_tokens=10)
    result = chunker.chunk(doc_result, config=cfg)

    assert result.total_chunks > 1
    for c in result.chunks:
        assert doc_result.text[c.start_char_offset : c.end_char_offset] == c.text
        assert c.token_count <= 30
        assert c.token_count > 0


def test_offset_fidelity_case2_whitespace_around_split_boundaries(chunker: StructureAwareChunker) -> None:
    """Case 2: Text with leading/trailing whitespace around split boundaries."""
    text = (
        "   \n\nFirst paragraph with leading spaces.   \n\n"
        "   Second paragraph surrounded by whitespace.   \n\n\t"
        "  Third paragraph with trailing whitespace.   \n\n"
    )
    doc_result = ExtractionResult(
        text=text,
        elements=[ExtractedElement(text=text)],
        content_type="text/plain",
        character_count=len(text),
        word_count=len(text.split()),
    )
    cfg = ChunkingConfig(chunk_size_tokens=12, chunk_overlap_tokens=3)
    result = chunker.chunk(doc_result, config=cfg)

    assert result.total_chunks > 1
    for c in result.chunks:
        assert doc_result.text[c.start_char_offset : c.end_char_offset] == c.text
        assert c.text == c.text.strip()


def test_offset_fidelity_case3_unicode_and_punctuation(chunker: StructureAwareChunker) -> None:
    """Case 3: Unicode and punctuation across token-window boundaries."""
    unicode_text = (
        "EvidenceDesk vérifie les données: «citation 1» & «modèle 2» — 100% exactitude! "
        "Japanese text: 日本語のテスト文章です。Emoji test: 🚀 🔍 📊. German umlauts: Äpfel, Ökologie, Überprüfung."
    )
    doc_result = ExtractionResult(
        text=unicode_text,
        elements=[ExtractedElement(text=unicode_text)],
        content_type="text/plain",
        character_count=len(unicode_text),
        word_count=len(unicode_text.split()),
    )
    cfg = ChunkingConfig(chunk_size_tokens=15, chunk_overlap_tokens=5)
    result = chunker.chunk(doc_result, config=cfg)

    assert result.total_chunks > 1
    for c in result.chunks:
        assert doc_result.text[c.start_char_offset : c.end_char_offset] == c.text
        assert c.token_count <= 15


def test_offset_fidelity_case4_repeated_identical_passages(chunker: StructureAwareChunker) -> None:
    """Case 4: Repeated identical passages sequentially located."""
    passage = "Identical factual statement repeated for consistency testing. "
    full_text = passage * 6
    doc_result = ExtractionResult(
        text=full_text,
        elements=[ExtractedElement(text=full_text)],
        content_type="text/plain",
        character_count=len(full_text),
        word_count=len(full_text.split()),
    )
    cfg = ChunkingConfig(chunk_size_tokens=25, chunk_overlap_tokens=8)
    result = chunker.chunk(doc_result, config=cfg)

    assert result.total_chunks > 1
    for c in result.chunks:
        assert doc_result.text[c.start_char_offset : c.end_char_offset] == c.text


def test_offset_fidelity_case5_multiple_structural_elements(chunker: StructureAwareChunker) -> None:
    """Case 5: Multiple structural elements with separators and offsets."""
    sec1 = "Section 1 provides an overview of the system."
    sec2 = "Section 2 provides deep technical details on chunking."
    sec3 = "Section 3 provides evaluation benchmarks."
    full_text = f"# Intro\n\n{sec1}\n\n# Design\n\n{sec2}\n\n# Benchmarks\n\n{sec3}"

    doc_result = ExtractionResult(
        text=full_text,
        elements=[
            ExtractedElement(text=sec1, heading="Intro", heading_level=1),
            ExtractedElement(text=sec2, heading="Design", heading_level=1),
            ExtractedElement(text=sec3, heading="Benchmarks", heading_level=1),
        ],
        content_type="text/markdown",
        character_count=len(full_text),
        word_count=len(full_text.split()),
    )
    cfg = ChunkingConfig(chunk_size_tokens=15, chunk_overlap_tokens=4)
    result = chunker.chunk(doc_result, config=cfg)

    assert result.total_chunks >= 3
    for c in result.chunks:
        assert full_text[c.start_char_offset : c.end_char_offset] == c.text


def test_offset_fidelity_case6_overlapping_and_zero_overlap(chunker: StructureAwareChunker) -> None:
    """Case 6: Overlapping chunks and zero-overlap chunks."""
    text = (
        "First sentence in the test document. Second sentence with distinct information. "
        "Third sentence describing findings. Fourth sentence concluding the section."
    )
    doc_result = ExtractionResult(
        text=text,
        elements=[ExtractedElement(text=text)],
        content_type="text/plain",
        character_count=len(text),
        word_count=len(text.split()),
    )

    # 1. Overlapping
    cfg_overlap = ChunkingConfig(chunk_size_tokens=12, chunk_overlap_tokens=4)
    res_overlap = chunker.chunk(doc_result, config=cfg_overlap)
    assert res_overlap.total_chunks > 1
    for c in res_overlap.chunks:
        assert text[c.start_char_offset : c.end_char_offset] == c.text

    # 2. Zero-overlap
    cfg_zero = ChunkingConfig(chunk_size_tokens=12, chunk_overlap_tokens=0)
    res_zero = chunker.chunk(doc_result, config=cfg_zero)
    assert res_zero.total_chunks > 1
    for c in res_zero.chunks:
        assert text[c.start_char_offset : c.end_char_offset] == c.text
    for i in range(len(res_zero.chunks) - 1):
        assert res_zero.chunks[i + 1].start_char_offset >= res_zero.chunks[i].end_char_offset
