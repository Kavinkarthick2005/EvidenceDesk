"""Integration tests for the /qa/answer API endpoint."""

import logging
from typing import Any
from unittest.mock import MagicMock
import uuid

from fastapi import status
from fastapi.testclient import TestClient
import pytest

from app.api.deps import get_qa_service
from app.generation.exceptions import (
    GenerationConnectionError,
    GenerationInferenceError,
    GenerationTimeoutError,
)
from app.generation.interface import GenerationResult
from app.generation.mock import DeterministicMockGenerator
from app.schemas.qa import QACitation, QAResponse
from app.schemas.retrieval import RetrievalResponse, RetrievedChunk
from app.services.exceptions import RetrievalError
from app.services.qa import QuestionAnsweringService
from app.services.retrieval import RetrievalService


def make_retrieved_chunk(
    chunk_index: int = 0,
    text: str = "Quarterly revenue increased by 15% due to enterprise expansions.",
    document_filename: str = "financial_report.pdf",
    score: float = 0.9200,
) -> RetrievedChunk:
    """Helper creating realistic RetrievedChunk objects for route tests."""
    return RetrievedChunk(
        chunk_id=uuid.uuid4(),
        document_id=uuid.uuid4(),
        document_filename=document_filename,
        chunk_index=chunk_index,
        text=text,
        score=score,
        cosine_distance=round(1.0 - score, 4),
        page_number=3,
        page_numbers=[3],
        heading="Financial Highlights",
        heading_level=2,
        metadata={"category": "finance"},
        model_id="bge-small-en-v1.5",
    )


def test_qa_route_happy_path(client: TestClient) -> None:
    """Verify POST /qa/answer returns HTTP 200 and a complete QAResponse on success."""
    c1 = make_retrieved_chunk(chunk_index=0, text="Revenue increased 15%.")
    c2 = make_retrieved_chunk(chunk_index=1, text="Operating margin reached 22%.")

    mock_retrieval = MagicMock(spec=RetrievalService)
    mock_retrieval.retrieve.return_value = RetrievalResponse(
        query="What is revenue?",
        model_id="bge-small-en-v1.5",
        returned_count=2,
        results=[c1, c2],
    )

    mock_qa_service = QuestionAnsweringService(
        retrieval_service=mock_retrieval,
        generation_provider=DeterministicMockGenerator(),
    )

    client.app.dependency_overrides[get_qa_service] = lambda: mock_qa_service
    try:
        response = client.post(
            "/qa/answer",
            json={"query": "What was the revenue growth?", "top_k": 3},
        )

        assert response.status_code == status.HTTP_200_OK
        data = response.json()

        assert data["query"] == "What was the revenue growth?"
        assert data["is_abstained"] is False
        assert data["abstention_reason"] is None
        assert data["has_citations"] is True
        assert data["is_valid_citations"] is True
        assert data["retrieved_count"] == 2
        assert len(data["citations"]) == 2

        # Verify citation structure
        citation0 = data["citations"][0]
        assert citation0["marker"] == "[1]"
        assert citation0["chunk_id"] == str(c1.chunk_id)
        assert citation0["document_filename"] == "financial_report.pdf"
        assert citation0["score"] == 0.9200
    finally:
        client.app.dependency_overrides.pop(get_qa_service, None)


def test_qa_route_empty_query_rejected(client: TestClient) -> None:
    """Verify empty or whitespace-only query returns HTTP 422 Unprocessable Entity."""
    # Empty string
    res_empty = client.post("/qa/answer", json={"query": ""})
    assert res_empty.status_code == status.HTTP_422_UNPROCESSABLE_ENTITY

    # Whitespace only
    res_ws = client.post("/qa/answer", json={"query": "   \n\t  "})
    assert res_ws.status_code == status.HTTP_422_UNPROCESSABLE_ENTITY

    # Missing query field entirely
    res_missing = client.post("/qa/answer", json={})
    assert res_missing.status_code == status.HTTP_422_UNPROCESSABLE_ENTITY


def test_qa_route_out_of_bounds_top_k(client: TestClient) -> None:
    """Verify out-of-bounds top_k values return HTTP 422 Unprocessable Entity."""
    res_zero = client.post("/qa/answer", json={"query": "Valid question?", "top_k": 0})
    assert res_zero.status_code == status.HTTP_422_UNPROCESSABLE_ENTITY

    res_too_high = client.post("/qa/answer", json={"query": "Valid question?", "top_k": 51})
    assert res_too_high.status_code == status.HTTP_422_UNPROCESSABLE_ENTITY


def test_qa_route_invalid_score_threshold(client: TestClient) -> None:
    """Verify invalid score_threshold values return HTTP 422 Unprocessable Entity."""
    res_high = client.post("/qa/answer", json={"query": "Valid?", "score_threshold": 1.5})
    assert res_high.status_code == status.HTTP_422_UNPROCESSABLE_ENTITY

    res_low = client.post("/qa/answer", json={"query": "Valid?", "score_threshold": -1.5})
    assert res_low.status_code == status.HTTP_422_UNPROCESSABLE_ENTITY


def test_qa_route_invalid_document_uuid(client: TestClient) -> None:
    """Verify malformed UUID string in document_ids returns HTTP 422."""
    res = client.post(
        "/qa/answer",
        json={"query": "Valid question?", "document_ids": ["not-a-valid-uuid"]},
    )
    assert res.status_code == status.HTTP_422_UNPROCESSABLE_ENTITY


def test_qa_route_honest_abstention_on_no_evidence(client: TestClient) -> None:
    """Verify route returns HTTP 200 with explicit abstention when 0 evidence chunks are found."""
    mock_retrieval = MagicMock(spec=RetrievalService)
    mock_retrieval.retrieve.return_value = RetrievalResponse(
        query="Unanswerable question",
        model_id="bge-small-en-v1.5",
        returned_count=0,
        results=[],
    )

    mock_qa_service = QuestionAnsweringService(
        retrieval_service=mock_retrieval,
        generation_provider=DeterministicMockGenerator(),
    )

    client.app.dependency_overrides[get_qa_service] = lambda: mock_qa_service
    try:
        response = client.post(
            "/qa/answer",
            json={"query": "What is the secret recipe?"},
        )

        assert response.status_code == status.HTTP_200_OK
        data = response.json()
        assert data["is_abstained"] is True
        assert data["abstention_reason"] == "no_evidence_found"
        assert data["has_citations"] is False
        assert data["citations"] == []
        assert "no supporting evidence was found" in data["answer"]
    finally:
        client.app.dependency_overrides.pop(get_qa_service, None)


def test_qa_route_hallucinated_citation_rejected(client: TestClient) -> None:
    """Verify route returns HTTP 200 with safe rejection when generator produces unknown citations."""
    c1 = make_retrieved_chunk(chunk_index=0, text="Legitimate text.")
    mock_retrieval = MagicMock(spec=RetrievalService)
    mock_retrieval.retrieve.return_value = RetrievalResponse(
        query="Test query",
        model_id="bge-small-en-v1.5",
        returned_count=1,
        results=[c1],
    )

    generator = DeterministicMockGenerator(simulate_hallucinated_citation=True)
    mock_qa_service = QuestionAnsweringService(
        retrieval_service=mock_retrieval,
        generation_provider=generator,
    )

    client.app.dependency_overrides[get_qa_service] = lambda: mock_qa_service
    try:
        response = client.post("/qa/answer", json={"query": "Test query?"})
        assert response.status_code == status.HTTP_200_OK
        data = response.json()

        assert data["is_abstained"] is True
        assert data["abstention_reason"] == "citation_validation_failed"
        assert data["is_valid_citations"] is False
        assert "unsupplied-evidence-999" in data["unknown_citations"]
        assert "invalid or unverified citations" in data["answer"]
    finally:
        client.app.dependency_overrides.pop(get_qa_service, None)


def test_qa_route_operational_failure_returns_safe_500(client: TestClient) -> None:
    """Verify operational retrieval errors yield safe HTTP 500 without leaking stack traces."""
    mock_retrieval = MagicMock(spec=RetrievalService)
    mock_retrieval.retrieve.side_effect = RetrievalError("Internal DB connection dropped")

    mock_qa_service = QuestionAnsweringService(
        retrieval_service=mock_retrieval,
        generation_provider=DeterministicMockGenerator(),
    )

    client.app.dependency_overrides[get_qa_service] = lambda: mock_qa_service
    try:
        response = client.post("/qa/answer", json={"query": "Test query?"})
        assert response.status_code == status.HTTP_500_INTERNAL_SERVER_ERROR
        data = response.json()

        # Check generic safe message
        assert data["detail"] == "Retrieval operation failed."
        # No internal SQL / connection details leaked
        assert "connection dropped" not in data["detail"]
    finally:
        client.app.dependency_overrides.pop(get_qa_service, None)


def test_qa_route_generation_failure_returns_safe_500(client: TestClient) -> None:
    """Verify operational generation errors yield safe HTTP 500 without leaking internal details."""
    c1 = make_retrieved_chunk(chunk_index=0, text="Legitimate text.")
    mock_retrieval = MagicMock(spec=RetrievalService)
    mock_retrieval.retrieve.return_value = RetrievalResponse(
        query="Test query",
        model_id="bge-small-en-v1.5",
        returned_count=1,
        results=[c1],
    )

    failing_generator = DeterministicMockGenerator(should_fail=True)
    mock_qa_service = QuestionAnsweringService(
        retrieval_service=mock_retrieval,
        generation_provider=failing_generator,
    )

    client.app.dependency_overrides[get_qa_service] = lambda: mock_qa_service
    try:
        response = client.post("/qa/answer", json={"query": "Test query?"})
        assert response.status_code == status.HTTP_500_INTERNAL_SERVER_ERROR
        data = response.json()

        assert data["detail"] == "Answer generation failed."
        assert "Mock generation failure" not in data["detail"]
    finally:
        client.app.dependency_overrides.pop(get_qa_service, None)


def test_qa_route_generation_timeout_returns_504(client: TestClient) -> None:
    """Verify GenerationTimeoutError returns HTTP 504 with sanitized detail."""
    c1 = make_retrieved_chunk(chunk_index=0, text="Legitimate text.")
    mock_retrieval = MagicMock(spec=RetrievalService)
    mock_retrieval.retrieve.return_value = RetrievalResponse(
        query="Timeout query",
        model_id="bge-small-en-v1.5",
        returned_count=1,
        results=[c1],
    )

    class TimeoutProvider:
        model_id = "timeout-provider"

        def generate_answer(self, query: str, evidence: Any) -> GenerationResult:
            raise GenerationTimeoutError("Ollama generation timed out after 60.0s at http://127.0.0.1:11434")

    mock_qa_service = QuestionAnsweringService(
        retrieval_service=mock_retrieval,
        generation_provider=TimeoutProvider(),
    )

    client.app.dependency_overrides[get_qa_service] = lambda: mock_qa_service
    try:
        response = client.post("/qa/answer", json={"query": "Timeout query?"})
        assert response.status_code == status.HTTP_504_GATEWAY_TIMEOUT
        data = response.json()
        assert data["detail"] == "Answer generation timed out."
        assert "60.0s" not in response.text
        assert "11434" not in response.text
        assert "127.0.0.1" not in response.text
    finally:
        client.app.dependency_overrides.pop(get_qa_service, None)


def test_qa_route_generation_connection_refusal_returns_503(client: TestClient) -> None:
    """Verify GenerationConnectionError returns HTTP 503 with sanitized detail."""
    c1 = make_retrieved_chunk(chunk_index=0, text="Legitimate text.")
    mock_retrieval = MagicMock(spec=RetrievalService)
    mock_retrieval.retrieve.return_value = RetrievalResponse(
        query="Connection query",
        model_id="bge-small-en-v1.5",
        returned_count=1,
        results=[c1],
    )

    class ConnectionErrorProvider:
        model_id = "connection-error-provider"

        def generate_answer(self, query: str, evidence: Any) -> GenerationResult:
            raise GenerationConnectionError("Unable to connect to Ollama server at 'http://localhost:11434'")

    mock_qa_service = QuestionAnsweringService(
        retrieval_service=mock_retrieval,
        generation_provider=ConnectionErrorProvider(),
    )

    client.app.dependency_overrides[get_qa_service] = lambda: mock_qa_service
    try:
        response = client.post("/qa/answer", json={"query": "Connection query?"})
        assert response.status_code == status.HTTP_503_SERVICE_UNAVAILABLE
        data = response.json()
        assert data["detail"] == "Generation service is currently unavailable."
        assert "localhost" not in response.text
        assert "11434" not in response.text
    finally:
        client.app.dependency_overrides.pop(get_qa_service, None)


def test_qa_route_unexpected_inference_failure_returns_500(client: TestClient) -> None:
    """Verify unexpected GenerationInferenceError returns HTTP 500 with sanitized detail."""
    c1 = make_retrieved_chunk(chunk_index=0, text="Legitimate text.")
    mock_retrieval = MagicMock(spec=RetrievalService)
    mock_retrieval.retrieve.return_value = RetrievalResponse(
        query="Inference query",
        model_id="bge-small-en-v1.5",
        returned_count=1,
        results=[c1],
    )

    class InferenceErrorProvider:
        model_id = "inference-error-provider"

        def generate_answer(self, query: str, evidence: Any) -> GenerationResult:
            raise GenerationInferenceError("Ollama model 'qwen3:4b' produced an empty answer after sanitization.")

    mock_qa_service = QuestionAnsweringService(
        retrieval_service=mock_retrieval,
        generation_provider=InferenceErrorProvider(),
    )

    client.app.dependency_overrides[get_qa_service] = lambda: mock_qa_service
    try:
        response = client.post("/qa/answer", json={"query": "Inference query?"})
        assert response.status_code == status.HTTP_500_INTERNAL_SERVER_ERROR
        data = response.json()
        assert data["detail"] == "Answer generation failed."
        assert "qwen3:4b" not in response.text
        assert "empty answer" not in response.text
    finally:
        client.app.dependency_overrides.pop(get_qa_service, None)


def test_qa_route_safe_logging(client: TestClient, caplog: pytest.LogCaptureFixture) -> None:
    """Verify HTTP requests do not log sensitive query text or document snippets."""
    c1 = make_retrieved_chunk(chunk_index=0, text="TOP_SECRET_PASSPHRASE_ABC_123")
    mock_retrieval = MagicMock(spec=RetrievalService)
    mock_retrieval.retrieve.return_value = RetrievalResponse(
        query="CONFIDENTIAL_USER_QUERY_XYZ",
        model_id="bge-small-en-v1.5",
        returned_count=1,
        results=[c1],
    )

    mock_qa_service = QuestionAnsweringService(
        retrieval_service=mock_retrieval,
        generation_provider=DeterministicMockGenerator(),
    )

    client.app.dependency_overrides[get_qa_service] = lambda: mock_qa_service
    try:
        with caplog.at_level(logging.INFO):
            client.post("/qa/answer", json={"query": "CONFIDENTIAL_USER_QUERY_XYZ"})

        log_text = caplog.text

        # Verify sensitive terms are ABSENT
        assert "CONFIDENTIAL_USER_QUERY_XYZ" not in log_text
        assert "TOP_SECRET_PASSPHRASE_ABC_123" not in log_text
    finally:
        client.app.dependency_overrides.pop(get_qa_service, None)
