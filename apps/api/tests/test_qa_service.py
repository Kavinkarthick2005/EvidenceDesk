"""Unit tests for QuestionAnsweringService orchestration."""

import logging
from typing import Any
from unittest.mock import MagicMock
import uuid

import pytest

from app.generation.exceptions import GenerationInferenceError
from app.generation.interface import EvidenceItem, GenerationProvider, GenerationResult
from app.generation.mock import DeterministicMockGenerator
from app.schemas.qa import QAResponse
from app.schemas.retrieval import RetrievalResponse, RetrievedChunk
from app.services.exceptions import InvalidQueryError, RetrievalError
from app.services.qa import QuestionAnsweringService
from app.services.retrieval import RetrievalService


def make_retrieved_chunk(
    chunk_index: int = 0,
    text: str = "Quarterly revenue increased by 15% due to enterprise expansions.",
    document_filename: str = "financial_report.pdf",
    heading: str | None = "Revenue Overview",
    page_number: int | None = 3,
    score: float = 0.9200,
    chunk_id: uuid.UUID | None = None,
    document_id: uuid.UUID | None = None,
) -> RetrievedChunk:
    """Helper creating realistic RetrievedChunk objects for QA service testing."""
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
        metadata={"category": "finance"},
        model_id="bge-small-en-v1.5",
    )


def make_mock_retrieval_service(chunks: list[RetrievedChunk]) -> RetrievalService:
    """Create a mock RetrievalService returning the specified chunks."""
    service = MagicMock(spec=RetrievalService)
    service.retrieve.return_value = RetrievalResponse(
        query="test query",
        model_id="bge-small-en-v1.5",
        returned_count=len(chunks),
        results=chunks,
    )
    return service


def test_qa_service_grounded_happy_path() -> None:
    """Verify standard grounded workflow: retrieval -> context -> generation -> valid citations."""
    c1 = make_retrieved_chunk(chunk_index=0, text="Revenue grew 15%.", score=0.92)
    c2 = make_retrieved_chunk(chunk_index=1, text="Operating margins were 22%.", score=0.88)
    retrieval_service = make_mock_retrieval_service([c1, c2])

    qa_service = QuestionAnsweringService(
        retrieval_service=retrieval_service,
        generation_provider=DeterministicMockGenerator(),
    )

    response = qa_service.answer_question(query="What was the revenue growth?")

    assert isinstance(response, QAResponse)
    assert response.is_abstained is False
    assert response.abstention_reason is None
    assert response.has_citations is True
    assert response.is_valid_citations is True
    assert response.retrieved_count == 2
    assert len(response.citations) == 2

    # Verify source traceability
    c0 = response.citations[0]
    assert c0.marker == "[1]"
    assert c0.marker_id == "1"
    assert c0.chunk_id == c1.chunk_id
    assert c0.document_filename == c1.document_filename
    assert c0.score == c1.score

    assert response.unknown_citations == []
    assert response.malformed_citations == []


def test_qa_service_zero_evidence_skips_generator() -> None:
    """Verify that zero retrieved evidence immediately yields honest abstention without calling LLM."""
    retrieval_service = make_mock_retrieval_service([])

    # Mock provider that raises an exception if called
    failing_provider = MagicMock(spec=GenerationProvider)
    failing_provider.model_id = "test-fail-provider"
    failing_provider.generate_answer.side_effect = RuntimeError("Generator should not be called!")

    qa_service = QuestionAnsweringService(
        retrieval_service=retrieval_service,
        generation_provider=failing_provider,
    )

    response = qa_service.answer_question(query="What is the undisclosed secret?")

    # Generator was NOT invoked
    failing_provider.generate_answer.assert_not_called()

    assert response.is_abstained is True
    assert response.abstention_reason == "no_evidence_found"
    assert response.has_citations is False
    assert response.is_valid_citations is True
    assert response.citations == []
    assert response.retrieved_count == 0
    assert "no supporting evidence was found" in response.answer


def test_qa_service_propagates_generator_abstention() -> None:
    """Verify generator-level abstention is propagated with empty citations."""
    c1 = make_retrieved_chunk(chunk_index=0, text="General background text.")
    retrieval_service = make_mock_retrieval_service([c1])

    abstaining_generator = DeterministicMockGenerator(
        should_abstain=True,
        abstention_reason="out_of_domain",
    )

    qa_service = QuestionAnsweringService(
        retrieval_service=retrieval_service,
        generation_provider=abstaining_generator,
    )

    response = qa_service.answer_question(query="How to cook pasta?")

    assert response.is_abstained is True
    assert response.abstention_reason == "out_of_domain"
    assert response.has_citations is False
    assert response.is_valid_citations is True
    assert response.citations == []


def test_qa_service_rejects_hallucinated_citations() -> None:
    """Verify answer with unsupplied citation ID is rejected and never returned as authoritative."""
    c1 = make_retrieved_chunk(chunk_index=0, text="Legitimate evidence text.")
    retrieval_service = make_mock_retrieval_service([c1])

    hallucinating_generator = DeterministicMockGenerator(simulate_hallucinated_citation=True)

    qa_service = QuestionAnsweringService(
        retrieval_service=retrieval_service,
        generation_provider=hallucinating_generator,
    )

    response = qa_service.answer_question(query="Tell me about executive compensation.")

    # Rejection semantics
    assert response.is_abstained is True
    assert response.abstention_reason == "citation_validation_failed"
    assert response.is_valid_citations is False
    assert response.citations == []
    assert "unsupplied-evidence-999" in response.unknown_citations
    # The generated ungrounded text is NOT returned as authoritative
    assert "invalid or unverified citations" in response.answer


def test_qa_service_rejects_malformed_citation_syntax() -> None:
    """Verify answer containing malformed syntax (e.g., [0]) is rejected."""
    c1 = make_retrieved_chunk(chunk_index=0, text="Passage text.")
    retrieval_service = make_mock_retrieval_service([c1])

    class MalformedProvider:
        model_id = "malformed-provider"

        def generate_answer(self, query: str, evidence: Any) -> GenerationResult:
            return GenerationResult(
                answer="Invalid reference syntax used here [0].",
                cited_evidence_ids=["0"],
                is_abstained=False,
                abstention_reason=None,
                model_id=self.model_id,
            )

    qa_service = QuestionAnsweringService(
        retrieval_service=retrieval_service,
        generation_provider=MalformedProvider(),
    )

    response = qa_service.answer_question(query="What is the index?")

    assert response.is_abstained is True
    assert response.abstention_reason == "citation_validation_failed"
    assert response.is_valid_citations is False
    assert "[0]" in response.malformed_citations
    assert response.citations == []


def test_qa_service_rejects_ungrounded_answer_lacking_citations() -> None:
    """Verify substantive answer with zero citations is rejected as ungrounded."""
    c1 = make_retrieved_chunk(chunk_index=0, text="Company policies on leave.")
    retrieval_service = make_mock_retrieval_service([c1])

    class UncitedProvider:
        model_id = "uncited-provider"

        def generate_answer(self, query: str, evidence: Any) -> GenerationResult:
            return GenerationResult(
                answer="Leave policy grants 20 days annually without any citation.",
                cited_evidence_ids=[],
                is_abstained=False,
                abstention_reason=None,
                model_id=self.model_id,
            )

    qa_service = QuestionAnsweringService(
        retrieval_service=retrieval_service,
        generation_provider=UncitedProvider(),
    )

    response = qa_service.answer_question(query="How much leave is granted?")

    assert response.is_abstained is True
    assert response.abstention_reason == "missing_citations"
    assert response.has_citations is False
    assert response.is_valid_citations is True
    assert response.citations == []
    assert "did not cite supporting evidence" in response.answer


def test_qa_service_query_validation() -> None:
    """Verify query validation handles empty, whitespace, and oversized inputs."""
    retrieval_service = make_mock_retrieval_service([])
    qa_service = QuestionAnsweringService(retrieval_service=retrieval_service)

    with pytest.raises(InvalidQueryError, match="empty or whitespace-only"):
        qa_service.answer_question(query="")

    with pytest.raises(InvalidQueryError, match="empty or whitespace-only"):
        qa_service.answer_question(query="   \t\n  ")

    with pytest.raises(InvalidQueryError, match="must be a string"):
        qa_service.answer_question(query=123)  # type: ignore[arg-type]

    oversized_query = "a" * 2001
    with pytest.raises(InvalidQueryError, match="exceeds maximum permitted limit"):
        qa_service.answer_question(query=oversized_query)


def test_qa_service_passthrough_parameters() -> None:
    """Verify top_k, document_ids, and score_threshold are forwarded unmodified to RetrievalService."""
    c1 = make_retrieved_chunk(chunk_index=0, text="Passage text.")
    retrieval_service = make_mock_retrieval_service([c1])

    qa_service = QuestionAnsweringService(
        retrieval_service=retrieval_service,
        generation_provider=DeterministicMockGenerator(),
    )

    doc_ids = [uuid.uuid4(), uuid.uuid4()]
    qa_service.answer_question(
        query="Filter query",
        top_k=7,
        document_ids=doc_ids,
        score_threshold=0.85,
    )

    retrieval_service.retrieve.assert_called_once_with(
        query="Filter query",
        top_k=7,
        document_ids=doc_ids,
        score_threshold=0.85,
    )


def test_qa_service_rejects_mixed_valid_and_invalid_citations() -> None:
    """Verify an answer citing both valid [1] and invalid [99] evidence is rejected."""
    c1 = make_retrieved_chunk(chunk_index=0, text="Revenue grew 10%.")
    retrieval_service = make_mock_retrieval_service([c1])

    class MixedCitationProvider:
        model_id = "mixed-citation-provider"

        def generate_answer(self, query: str, evidence: Any) -> GenerationResult:
            return GenerationResult(
                answer="Revenue grew 10% [1], but secret costs were hidden [99].",
                cited_evidence_ids=["1", "99"],
                is_abstained=False,
                abstention_reason=None,
                model_id=self.model_id,
            )

    qa_service = QuestionAnsweringService(
        retrieval_service=retrieval_service,
        generation_provider=MixedCitationProvider(),
    )

    response = qa_service.answer_question(query="What is the financial breakdown?")

    assert response.is_abstained is True
    assert response.abstention_reason == "citation_validation_failed"
    assert response.is_valid_citations is False
    assert response.citations == []
    assert "99" in response.unknown_citations
    assert "invalid or unverified citations" in response.answer
    assert "Revenue grew 10% [1]" not in response.answer


def test_qa_service_propagates_generation_error() -> None:
    """Verify generation inference failure propagates cleanly without being swallowed."""
    c1 = make_retrieved_chunk(chunk_index=0, text="Revenue grew 10%.")
    retrieval_service = make_mock_retrieval_service([c1])

    failing_generator = DeterministicMockGenerator(should_fail=True)
    qa_service = QuestionAnsweringService(
        retrieval_service=retrieval_service,
        generation_provider=failing_generator,
    )

    with pytest.raises(GenerationInferenceError, match="Mock generation failure simulated"):
        qa_service.answer_question(query="Failing query")


def test_qa_service_propagates_retrieval_error() -> None:
    """Verify retrieval database errors propagate cleanly."""
    service = MagicMock(spec=RetrievalService)
    service.retrieve.side_effect = RetrievalError("Database connection lost")

    qa_service = QuestionAnsweringService(retrieval_service=service)

    with pytest.raises(RetrievalError, match="Database connection lost"):
        qa_service.answer_question(query="Failing query")


def test_qa_service_safe_logging(caplog: pytest.LogCaptureFixture) -> None:
    """Verify sensitive query text, document passages, and prompts are NEVER logged."""
    secret_query = "What is the secret passphrase for account #999888?"
    secret_chunk_text = "The super secret passphrase is 'Swordfish-2026'."
    c1 = make_retrieved_chunk(chunk_index=0, text=secret_chunk_text)
    retrieval_service = make_mock_retrieval_service([c1])

    qa_service = QuestionAnsweringService(
        retrieval_service=retrieval_service,
        generation_provider=DeterministicMockGenerator(),
    )

    with caplog.at_level(logging.INFO):
        qa_service.answer_question(query=secret_query)

    log_output = caplog.text

    # Operational metrics must be logged
    assert "Starting QA orchestration" in log_output
    assert "query_length=" in log_output
    assert "Retrieval completed" in log_output

    # Sensitive data MUST NOT be logged
    assert secret_query not in log_output
    assert secret_chunk_text not in log_output
    assert "Swordfish-2026" not in log_output
