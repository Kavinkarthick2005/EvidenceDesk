"""Unit tests for the EvidenceDesk CitationValidator component."""

import uuid

import pytest

from app.generation.citation import (
    CitationValidationResult,
    CitationValidator,
    ResolvedCitation,
)
from app.generation.context import ContextBuilder
from app.generation.interface import GenerationResult
from app.generation.mock import DeterministicMockGenerator
from app.schemas.retrieval import RetrievedChunk


def make_retrieved_chunk(
    chunk_index: int = 0,
    text: str = "Quarterly revenue reached $12.5M, representing an 18% YoY growth.",
    document_filename: str = "q3_report.pdf",
    heading: str | None = "Financial Overview",
    page_number: int | None = 2,
    score: float = 0.9100,
    chunk_id: uuid.UUID | None = None,
    document_id: uuid.UUID | None = None,
) -> RetrievedChunk:
    """Helper creating realistic RetrievedChunk instances for citation validation tests."""
    return RetrievedChunk(
        chunk_id=chunk_id or uuid.uuid4(),
        document_id=document_id or uuid.uuid4(),
        document_filename=document_filename,
        chunk_index=chunk_index,
        text=text,
        score=score,
        cosine_distance=round(1.0 - score, 4),
        page_number=page_number,
        page_numbers=[page_number] if page_number is not None else [],
        heading=heading,
        heading_level=2 if heading else None,
        metadata={"source": "sec_filing"},
        model_id="bge-small-en-v1.5",
    )


def test_citation_validator_valid_single_and_multiple_citations() -> None:
    """Verify validator resolves valid single and multiple citations to exact source metadata."""
    c1 = make_retrieved_chunk(
        chunk_index=0,
        text="Revenue was $12.5M.",
        document_filename="finances.pdf",
    )
    c2 = make_retrieved_chunk(
        chunk_index=1,
        text="Expenses were $8.0M.",
        document_filename="expenses.pdf",
    )
    ctx = ContextBuilder().build_context([c1, c2])
    validator = CitationValidator()

    answer = "Revenue reached $12.5M [1], while operating expenses were $8.0M [2]."
    res = validator.validate(answer, ctx)

    assert isinstance(res, CitationValidationResult)
    assert res.is_valid is True
    assert res.has_citations is True
    assert res.cited_evidence_ids == ["1", "2"]
    assert len(res.citations) == 2
    assert res.unknown_citation_ids == []
    assert res.malformed_citations == []
    assert res.duplicate_citation_ids == []

    # Verify source traceability on resolved citations
    assert res.citations[0].marker_id == "1"
    assert res.citations[0].citation_marker == "[1]"
    assert res.citations[0].chunk_id == c1.chunk_id
    assert res.citations[0].document_filename == "finances.pdf"
    assert res.citations[0].score == c1.score

    assert res.citations[1].marker_id == "2"
    assert res.citations[1].citation_marker == "[2]"
    assert res.citations[1].chunk_id == c2.chunk_id
    assert res.citations[1].document_filename == "expenses.pdf"


def test_citation_validator_grouped_citations() -> None:
    """Verify validator correctly parses comma-separated citations inside a single bracket."""
    c1 = make_retrieved_chunk(chunk_index=0, text="First factor.")
    c2 = make_retrieved_chunk(chunk_index=1, text="Second factor.")
    ctx = ContextBuilder().build_context([c1, c2])
    validator = CitationValidator()

    answer = "Both factors contributed to growth [1, 2]."
    res = validator.validate(answer, ctx)

    assert res.is_valid is True
    assert res.has_citations is True
    assert res.cited_evidence_ids == ["1", "2"]
    assert len(res.citations) == 2
    assert res.unknown_citation_ids == []
    assert res.malformed_citations == []


def test_citation_validator_unknown_marker() -> None:
    """Verify validator flags unknown markers and invalidates the result."""
    c1 = make_retrieved_chunk(chunk_index=0, text="Only valid passage.")
    ctx = ContextBuilder().build_context([c1])
    validator = CitationValidator()

    answer = "The policy applies to all employees [99]."
    res = validator.validate(answer, ctx)

    assert res.is_valid is False
    assert res.has_citations is False
    assert res.cited_evidence_ids == []
    assert res.unknown_citation_ids == ["99"]
    assert res.malformed_citations == []


def test_citation_validator_malformed_syntax_detection() -> None:
    """Verify validator detects various malformed citation patterns without crashing."""
    c1 = make_retrieved_chunk(chunk_index=0, text="Valid passage.")
    ctx = ContextBuilder().build_context([c1])
    validator = CitationValidator()

    # Empty brackets
    res_empty = validator.validate("Information [] was missing.", ctx)
    assert res_empty.is_valid is False
    assert "[]" in res_empty.malformed_citations

    # Zero marker
    res_zero = validator.validate("Using base index [0].", ctx)
    assert res_zero.is_valid is False
    assert "[0]" in res_zero.malformed_citations

    # Negative marker
    res_neg = validator.validate("Offset [-1] was applied.", ctx)
    assert res_neg.is_valid is False
    assert "[-1]" in res_neg.malformed_citations

    # Hash prefix with spaces
    res_hash = validator.validate("See section [#1] and [ #1 ] and [# 2].", ctx)
    assert res_hash.is_valid is False
    assert "[#1]" in res_hash.malformed_citations
    assert "[ #1 ]" in res_hash.malformed_citations
    assert "[# 2]" in res_hash.malformed_citations

    # Broken comma
    res_comma = validator.validate("Refer to [1,] for details.", ctx)
    assert res_comma.is_valid is False
    assert "[1,]" in res_comma.malformed_citations

    # Bare keyword
    res_keyword = validator.validate("Omitted citation [ref].", ctx)
    assert res_keyword.is_valid is False
    assert "[ref]" in res_keyword.malformed_citations


def test_citation_validator_preserves_ordinary_bracketed_prose() -> None:
    """Verify validator does NOT misclassify ordinary bracketed prose as citations or malformed syntax."""
    c1 = make_retrieved_chunk(
        chunk_index=0,
        text="Audited numbers confirm 12% margin.",
        document_filename="audit.pdf",
    )
    ctx = ContextBuilder().build_context([c1])
    validator = CitationValidator()

    # Simulated text containing mock prefixes, editorial notes, table references, and valid citation
    answer = (
        "[Mock Answer] In response to 'What was the margin?', audit.pdf [1] states: "
        "'[Table 1] [see note 3] [sic]'."
    )
    res = validator.validate(answer, ctx)

    assert res.is_valid is True
    assert res.has_citations is True
    assert res.cited_evidence_ids == ["1"]
    assert len(res.citations) == 1
    assert res.citations[0].document_filename == "audit.pdf"
    assert res.unknown_citation_ids == []
    assert res.malformed_citations == []


def test_citation_validator_duplicate_citations() -> None:
    """Verify validator tracks duplicate citation occurrences while maintaining validity."""
    c1 = make_retrieved_chunk(chunk_index=0, text="Recurring observation.")
    ctx = ContextBuilder().build_context([c1])
    validator = CitationValidator()

    answer = "As stated in [1], growth continued. Furthermore, [1] confirms the trend persisted."
    res = validator.validate(answer, ctx)

    assert res.is_valid is True
    assert res.has_citations is True
    assert res.cited_evidence_ids == ["1"]
    assert len(res.citations) == 1
    assert res.duplicate_citation_ids == ["1"]
    assert res.citation_counts == {"1": 2}


def test_citation_validator_no_citations() -> None:
    """Verify validator deterministically handles answers containing no citations."""
    c1 = make_retrieved_chunk(chunk_index=0, text="Background text.")
    ctx = ContextBuilder().build_context([c1])
    validator = CitationValidator()

    answer = "I cannot answer this question based on the provided evidence."
    res = validator.validate(answer, ctx)

    assert res.is_valid is True
    assert res.has_citations is False
    assert res.cited_evidence_ids == []
    assert res.citations == []
    assert res.unknown_citation_ids == []
    assert res.malformed_citations == []
    assert res.duplicate_citation_ids == []
    assert res.citation_counts == {}


def test_citation_validator_with_mock_generator_end_to_end() -> None:
    """Verify validator validates GenerationResult produced by DeterministicMockGenerator."""
    c1 = make_retrieved_chunk(
        chunk_index=0,
        text="Alpha Corp acquired Beta LLC in August 2022.",
        document_filename="acquisitions.md",
    )
    ctx = ContextBuilder().build_context([c1])

    generator = DeterministicMockGenerator()
    gen_result = generator.generate_answer(
        query="When did Alpha Corp acquire Beta LLC?",
        evidence=ctx.evidence_items,
    )

    validator = CitationValidator()
    val_result = validator.validate(gen_result, ctx)

    assert val_result.is_valid is True
    assert val_result.has_citations is True
    assert val_result.cited_evidence_ids == ["1"]
    assert val_result.cited_evidence_ids == gen_result.cited_evidence_ids
    assert val_result.citations[0].document_filename == "acquisitions.md"
    assert val_result.citations[0].chunk_id == c1.chunk_id


def test_citation_validator_catches_mock_hallucinated_citation() -> None:
    """Verify validator catches unsupplied citations simulated by DeterministicMockGenerator."""
    c1 = make_retrieved_chunk(
        chunk_index=0,
        text="Ground truth text.",
        document_filename="truth.md",
    )
    ctx = ContextBuilder().build_context([c1])

    # Generator configured to inject hallucinated citation ID
    generator = DeterministicMockGenerator(simulate_hallucinated_citation=True)
    gen_result = generator.generate_answer(
        query="Test query?",
        evidence=ctx.evidence_items,
    )

    validator = CitationValidator()
    val_result = validator.validate(gen_result, ctx)

    assert val_result.is_valid is False
    assert "unsupplied-evidence-999" in val_result.unknown_citation_ids
    # Valid chunk citation was still identified
    assert "1" in val_result.cited_evidence_ids


def test_citation_validator_immutability_and_type_checks() -> None:
    """Verify immutability of CitationValidationResult and validation of input arguments."""
    validator = CitationValidator()
    c1 = make_retrieved_chunk(chunk_index=0, text="Text")
    ctx = ContextBuilder().build_context([c1])

    res = validator.validate("Answer [1]", ctx)

    # Immutability check
    with pytest.raises(Exception):
        res.is_valid = False  # type: ignore[misc]

    # Type validation on target
    with pytest.raises(TypeError, match="Validation target must be a str or GenerationResult"):
        validator.validate(12345, ctx)  # type: ignore[arg-type]

    # Type validation on context
    with pytest.raises(TypeError, match="Context must be an EvidenceContext or Mapping"):
        validator.validate("Answer", "invalid_context")  # type: ignore[arg-type]


def test_citation_validator_ignores_code_spans_and_fenced_blocks() -> None:
    """Verify citation-like tokens inside inline code and fenced blocks are ignored.

    Regression test for:
    - Inline code containing `items[0]`.
    - A fenced code block containing `[1, 2]`.
    - A real citation such as `[1]` outside the code span or block in the same answer.
    """
    c1 = make_retrieved_chunk(
        chunk_index=0,
        text="Indexing starts at zero in Python.",
        document_filename="python_guide.md",
    )
    ctx = ContextBuilder().build_context([c1])
    validator = CitationValidator()

    answer = (
        "To get the first element, call `items[0]` on the list.\n"
        "```python\n"
        "# Example list initialization\n"
        "values = [1, 2]\n"
        "```\n"
        "This convention is defined in the language documentation [1]."
    )

    res = validator.validate(answer, ctx)

    # Neither `items[0]` nor `[1, 2]` should trigger malformed or unknown citations
    assert res.is_valid is True
    assert res.has_citations is True
    assert res.cited_evidence_ids == ["1"]
    assert len(res.citations) == 1
    assert res.citations[0].document_filename == "python_guide.md"
    assert res.unknown_citation_ids == []
    assert res.malformed_citations == []
    # Original answer text is preserved verbatim
    assert res.answer == answer


def test_citation_validator_nested_brackets() -> None:
    """Verify nested brackets like [[1]] resolve the innermost un-bracketed token '[1]'."""
    c1 = make_retrieved_chunk(chunk_index=0, text="Nested citation reference.")
    ctx = ContextBuilder().build_context([c1])
    validator = CitationValidator()

    res_nested = validator.validate("See double bracket [[1]] for source.", ctx)
    assert res_nested.is_valid is True
    assert res_nested.has_citations is True
    assert res_nested.cited_evidence_ids == ["1"]


def test_citation_grounding_zero_overlap_detected() -> None:
    """Verify zero content token overlap between a cited sentence and chunk is flagged in unsupported_citation_ids."""
    c1 = make_retrieved_chunk(
        chunk_index=0,
        text="Employees are entitled to 15 days of annual paid leave.",
        heading="Annual Leave",
        document_filename="handbook.pdf",
    )
    ctx = ContextBuilder().build_context([c1])
    validator = CitationValidator()

    # Statement has zero lexical overlap with chunk text, heading, or filename
    answer = "The administrative root password is SuperSecretAdmin123 [1]."
    res = validator.validate(answer, ctx)

    assert res.is_valid is True
    assert res.has_citations is True
    assert res.cited_evidence_ids == ["1"]
    assert res.unsupported_citation_ids == ["1"]


def test_citation_grounding_numeric_facts_anchor_claim() -> None:
    """Verify numeric strings and amounts establish lexical grounding without false rejection."""
    c1 = make_retrieved_chunk(
        chunk_index=0,
        text="Total capital expenditure for the cluster upgrade was $50,000.",
        heading="Budget Allocation",
        document_filename="financials.pdf",
    )
    ctx = ContextBuilder().build_context([c1])
    validator = CitationValidator()

    answer = "Expenditure totaled $50,000 [1]."
    res = validator.validate(answer, ctx)

    assert res.is_valid is True
    assert res.unsupported_citation_ids == []


def test_citation_grounding_heading_fallback_prevents_false_rejection() -> None:
    """Verify words in chunk heading or document title provide grounding for section-level statements."""
    c1 = make_retrieved_chunk(
        chunk_index=0,
        text="Employees receive 20 days off annually.",
        heading="Vacation Policy",
        document_filename="handbook.pdf",
    )
    ctx = ContextBuilder().build_context([c1])
    validator = CitationValidator()

    # Answer uses "vacation policy" which appears only in heading, not in passage text
    answer = "The vacation policy governs time off [1]."
    res = validator.validate(answer, ctx)

    assert res.is_valid is True
    assert res.unsupported_citation_ids == []


def test_citation_grounding_micro_sentences_exempt() -> None:
    """Verify micro-sentences with <= 2 content tokens are not flagged as unsupported."""
    c1 = make_retrieved_chunk(
        chunk_index=0,
        text="The database migration finished without errors.",
        heading="Database Status",
        document_filename="status.md",
    )
    ctx = ContextBuilder().build_context([c1])
    validator = CitationValidator()

    answer = "It completed [1]."
    res = validator.validate(answer, ctx)

    assert res.is_valid is True
    assert res.unsupported_citation_ids == []


def test_citation_grounding_multi_sentence_mixed_grounding() -> None:
    """Verify multi-citation answers accurately identify which citation failed grounding."""
    c1 = make_retrieved_chunk(
        chunk_index=0,
        text="Company was founded in 2012 by Jane Doe.",
        heading="Company History",
        document_filename="about.md",
    )
    c2 = make_retrieved_chunk(
        chunk_index=1,
        text="PTO policy provides 15 days of annual leave.",
        heading="PTO Guidelines",
        document_filename="pto.md",
    )
    ctx = ContextBuilder().build_context([c1, c2])
    validator = CitationValidator()

    answer = "Jane Doe founded the company in 2012 [1]. The server root password is confidential [2]."
    res = validator.validate(answer, ctx)

    assert res.is_valid is True
    assert res.has_citations is True
    assert set(res.cited_evidence_ids) == {"1", "2"}
    assert res.unsupported_citation_ids == ["2"]
