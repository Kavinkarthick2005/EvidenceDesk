"""Offline unit tests for local Ollama generation provider.

Uses httpx.MockTransport to simulate all local HTTP scenarios completely in-memory
with zero network access, GPU requirements, or external server dependencies.
"""

from collections.abc import Generator
import json
from typing import Any
import uuid

import httpx
import pytest

from app.core.config import Settings
from app.generation import (
    DeterministicMockGenerator,
    EvidenceItem,
    GenerationConfigurationError,
    GenerationInferenceError,
    GenerationProvider,
    GenerationResult,
    GenerationTimeoutError,
    InvalidPromptError,
    OllamaGenerationProvider,
    get_generation_provider,
    reset_generation_state,
)
from app.generation.context import ContextBuilder
from app.generation.citation import CitationValidator
from app.schemas.retrieval import RetrievalResponse, RetrievedChunk
from app.services.qa import QuestionAnsweringService


@pytest.fixture(autouse=True)
def isolate_generation_env() -> Generator[None, None, None]:
    """Ensure generation provider singleton state is reset before and after each test."""
    reset_generation_state()
    yield
    reset_generation_state()


def _make_evidence(marker_id: str = "1", text: str = "Test passage.") -> EvidenceItem:
    """Helper to create a valid EvidenceItem."""
    return EvidenceItem(
        id=marker_id,
        text=text,
        document_title="test_doc.md",
        heading="Introduction",
        chunk_id=uuid.uuid4(),
    )


def test_ollama_provider_protocol_conformance() -> None:
    """Verify OllamaGenerationProvider conforms to GenerationProvider protocol."""
    provider = OllamaGenerationProvider(
        base_url="http://localhost:11434",
        model_id="qwen3:4b",
        timeout=30.0,
    )
    assert isinstance(provider, GenerationProvider)
    assert provider.model_id == "qwen3:4b"
    assert provider.base_url == "http://localhost:11434"
    assert provider.timeout == 30.0
    provider.close()


def test_ollama_request_payload_and_options() -> None:
    """Verify endpoint, payload structure, stream=false, temperature=0, and keep_alive."""
    captured_requests: list[httpx.Request] = []

    def handler(request: httpx.Request) -> httpx.Response:
        captured_requests.append(request)
        return httpx.Response(
            status_code=200,
            json={
                "model": "qwen3:4b",
                "message": {
                    "role": "assistant",
                    "content": "The policy states 15 days of PTO [1].",
                },
                "done": True,
            },
        )

    client = httpx.Client(
        transport=httpx.MockTransport(handler),
        base_url="http://localhost:11434",
    )
    provider = OllamaGenerationProvider(
        base_url="http://localhost:11434",
        model_id="qwen3:4b",
        timeout=45.0,
        keep_alive="5m",
        client=client,
    )

    evidence = [
        _make_evidence("1", "Full-time employees receive 15 days PTO per year."),
    ]
    result = provider.generate_answer("How much PTO is provided?", evidence)

    assert len(captured_requests) == 1
    req = captured_requests[0]
    assert req.method == "POST"
    assert req.url.path == "/api/chat"

    payload = json.loads(req.content.decode("utf-8"))
    assert payload["model"] == "qwen3:4b"
    assert payload["stream"] is False
    assert payload["options"]["temperature"] == 0.0
    assert payload["keep_alive"] == "5m"

    # Verify messages
    messages = payload["messages"]
    assert len(messages) == 2
    assert messages[0]["role"] == "system"
    assert "EvidenceDesk" in messages[0]["content"]
    assert "bracketed numbers" in messages[0]["content"]

    assert messages[1]["role"] == "user"
    assert "--- BEGIN RETRIEVED EVIDENCE PASSAGES ---" in messages[1]["content"]
    assert "[1] Document: test_doc.md" in messages[1]["content"]
    assert "Question: How much PTO is provided?" in messages[1]["content"]

    # Verify result
    assert result.answer == "The policy states 15 days of PTO [1]."
    assert result.cited_evidence_ids == ["1"]
    assert result.is_abstained is False
    assert result.model_id == "qwen3:4b"


def test_ollama_ignores_thinking_field() -> None:
    """Verify provider extracts only message.content and completely ignores message.thinking."""
    def handler(request: httpx.Request) -> httpx.Response:
        return httpx.Response(
            status_code=200,
            json={
                "model": "qwen3:4b",
                "message": {
                    "role": "assistant",
                    "content": "Eligible employees receive $1,200 [1].",
                    "thinking": "User is asking about reimbursement. Let me ponder the internal reasoning...",
                },
                "done": True,
            },
        )

    client = httpx.Client(transport=httpx.MockTransport(handler), base_url="http://localhost:11434")
    provider = OllamaGenerationProvider(client=client)

    result = provider.generate_answer("What is the reimbursement?", [_make_evidence("1", "$1,200 reimbursement.")])
    assert result.answer == "Eligible employees receive $1,200 [1]."
    assert "ponder" not in result.answer
    assert "internal reasoning" not in result.answer


def test_ollama_sanitizes_closed_and_unclosed_reasoning_tags() -> None:
    """Verify defensive stripping of <think> and <thought> tags (both closed and unclosed)."""
    test_cases = [
        (
            "<think>Analyzing passages carefully.</think>The answer is 42 [1].",
            "The answer is 42 [1].",
        ),
        (
            "<thought>Considering the question...</thought>Confirmed answer is 42 [1].",
            "Confirmed answer is 42 [1].",
        ),
        (
            "<THINK>Case-insensitive check.</THINK>Result is verified [1].",
            "Result is verified [1].",
        ),
        (
            "Final result is confirmed [1]. <think>Trailing reasoning without closing tag",
            "Final result is confirmed [1].",
        ),
        (
            "Final result is confirmed [1]. <thought>Trailing unclosed thought",
            "Final result is confirmed [1].",
        ),
    ]

    for raw_content, expected_clean in test_cases:
        def handler(request: httpx.Request, content: str = raw_content) -> httpx.Response:
            return httpx.Response(
                status_code=200,
                json={"message": {"role": "assistant", "content": content}},
            )

        client = httpx.Client(transport=httpx.MockTransport(handler), base_url="http://localhost:11434")
        provider = OllamaGenerationProvider(client=client)
        res = provider.generate_answer("What is the result?", [_make_evidence("1", "Result is 42.")])
        assert res.answer == expected_clean
        assert "<think>" not in res.answer
        assert "<thought>" not in res.answer


def test_ollama_extracts_multiple_and_composite_citations() -> None:
    """Verify ordered deduplication of candidate numeric citation tokens."""
    def handler(request: httpx.Request) -> httpx.Response:
        return httpx.Response(
            status_code=200,
            json={
                "message": {
                    "role": "assistant",
                    "content": "According to [1] and [2], vectors are indexed. Furthermore [1, 3] also confirms this.",
                }
            },
        )

    client = httpx.Client(transport=httpx.MockTransport(handler), base_url="http://localhost:11434")
    provider = OllamaGenerationProvider(client=client)

    evidence = [
        _make_evidence("1", "Passage 1"),
        _make_evidence("2", "Passage 2"),
        _make_evidence("3", "Passage 3"),
    ]
    result = provider.generate_answer("How are vectors indexed?", evidence)
    assert result.cited_evidence_ids == ["1", "2", "3"]
    assert result.is_abstained is False


def test_ollama_zero_evidence_honest_abstention_no_http() -> None:
    """Verify zero evidence triggers instant abstention without making an HTTP request."""
    http_called = False

    def handler(request: httpx.Request) -> httpx.Response:
        nonlocal http_called
        http_called = True
        return httpx.Response(200, json={})

    client = httpx.Client(transport=httpx.MockTransport(handler), base_url="http://localhost:11434")
    provider = OllamaGenerationProvider(client=client)

    result = provider.generate_answer("Any query?", [])
    assert http_called is False
    assert result.is_abstained is True
    assert result.abstention_reason == "insufficient_evidence"
    assert result.cited_evidence_ids == []
    assert "no supporting evidence was provided" in result.answer


def test_ollama_model_abstention_detection() -> None:
    """Verify standard model refusal phrases are detected as abstention."""
    refusal_phrases = [
        "I cannot answer this question based on the provided evidence.",
        "The provided evidence does not contain information about pet policies.",
        "There is no information is provided in the reference material.",
        "The evidence provided is insufficient to determine the policy.",
    ]

    for phrase in refusal_phrases:
        def handler(request: httpx.Request, text: str = phrase) -> httpx.Response:
            return httpx.Response(
                status_code=200,
                json={"message": {"role": "assistant", "content": text}},
            )

        client = httpx.Client(transport=httpx.MockTransport(handler), base_url="http://localhost:11434")
        provider = OllamaGenerationProvider(client=client)

        result = provider.generate_answer("What is the pet policy?", [_make_evidence("1", "Unrelated passage.")])
        assert result.is_abstained is True
        assert result.abstention_reason == "insufficient_evidence"
        assert result.cited_evidence_ids == []


def test_ollama_input_validation() -> None:
    """Verify invalid queries and evidence raise InvalidPromptError."""
    client = httpx.Client(transport=httpx.MockTransport(lambda r: httpx.Response(200)))
    provider = OllamaGenerationProvider(client=client)

    valid_ev = [_make_evidence("1", "Valid text")]

    # Empty or whitespace query
    with pytest.raises(InvalidPromptError, match="Query cannot be empty or whitespace-only"):
        provider.generate_answer("", valid_ev)

    with pytest.raises(InvalidPromptError, match="Query cannot be empty or whitespace-only"):
        provider.generate_answer("   \n\t  ", valid_ev)

    # Non-string query
    with pytest.raises(InvalidPromptError, match="Query must be a string"):
        provider.generate_answer(12345, valid_ev)  # type: ignore[arg-type]

    # Non-sequence evidence
    with pytest.raises(InvalidPromptError, match="Evidence must be a sequence"):
        provider.generate_answer("Valid query?", "not-a-sequence")  # type: ignore[arg-type]

    # Malformed evidence items
    with pytest.raises(InvalidPromptError, match="must be an EvidenceItem instance"):
        provider.generate_answer("Valid query?", ["invalid item"])  # type: ignore[list-item]

    # Duplicate evidence IDs
    dup_evidence = [
        _make_evidence("1", "Passage A"),
        _make_evidence("1", "Passage B"),
    ]
    with pytest.raises(InvalidPromptError, match="Duplicate evidence ID '1' detected"):
        provider.generate_answer("Valid query?", dup_evidence)


def test_ollama_connection_error_raises_generation_inference_error() -> None:
    """Verify connection refused raises GenerationInferenceError with actionable message."""
    def handler(request: httpx.Request) -> httpx.Response:
        raise httpx.ConnectError("Connection refused by daemon")

    client = httpx.Client(transport=httpx.MockTransport(handler), base_url="http://localhost:11434")
    provider = OllamaGenerationProvider(client=client)

    with pytest.raises(GenerationInferenceError, match="Unable to connect to Ollama server"):
        provider.generate_answer("Question?", [_make_evidence()])


def test_ollama_timeout_raises_generation_timeout_error() -> None:
    """Verify request timeout raises GenerationTimeoutError."""
    def handler(request: httpx.Request) -> httpx.Response:
        raise httpx.ReadTimeout("Timed out waiting for model tokens")

    client = httpx.Client(transport=httpx.MockTransport(handler), base_url="http://localhost:11434")
    provider = OllamaGenerationProvider(client=client, timeout=10.0)

    with pytest.raises(GenerationTimeoutError, match="Ollama generation timed out after 10.0s"):
        provider.generate_answer("Question?", [_make_evidence()])


def test_ollama_http_404_raises_generation_inference_error() -> None:
    """Verify HTTP 404 (model missing or bad endpoint) raises GenerationInferenceError."""
    def handler(request: httpx.Request) -> httpx.Response:
        return httpx.Response(404, json={"error": "model 'qwen3:4b' not found"})

    client = httpx.Client(transport=httpx.MockTransport(handler), base_url="http://localhost:11434")
    provider = OllamaGenerationProvider(client=client, model_id="qwen3:4b")

    with pytest.raises(GenerationInferenceError, match="Ollama model 'qwen3:4b' not found"):
        provider.generate_answer("Question?", [_make_evidence()])


def test_ollama_http_500_raises_generation_inference_error() -> None:
    """Verify HTTP 500 raises GenerationInferenceError without leaking secrets."""
    def handler(request: httpx.Request) -> httpx.Response:
        return httpx.Response(500, json={"error": "internal crash"})

    client = httpx.Client(transport=httpx.MockTransport(handler), base_url="http://localhost:11434")
    provider = OllamaGenerationProvider(client=client)

    with pytest.raises(GenerationInferenceError, match="Ollama HTTP error 500"):
        provider.generate_answer("Question?", [_make_evidence()])


def test_ollama_malformed_json_raises_generation_inference_error() -> None:
    """Verify non-JSON response payload raises GenerationInferenceError."""
    def handler(request: httpx.Request) -> httpx.Response:
        return httpx.Response(200, content=b"<!DOCTYPE html><html>Gateway error</html>")

    client = httpx.Client(transport=httpx.MockTransport(handler), base_url="http://localhost:11434")
    provider = OllamaGenerationProvider(client=client)

    with pytest.raises(GenerationInferenceError, match="Unexpected failure communicating with Ollama"):
        provider.generate_answer("Question?", [_make_evidence()])


def test_ollama_missing_message_or_content_raises_error() -> None:
    """Verify missing expected response keys raises GenerationInferenceError."""
    # Missing 'message'
    def handler_no_msg(request: httpx.Request) -> httpx.Response:
        return httpx.Response(200, json={"done": True})

    client = httpx.Client(transport=httpx.MockTransport(handler_no_msg), base_url="http://localhost:11434")
    provider = OllamaGenerationProvider(client=client)
    with pytest.raises(GenerationInferenceError, match="missing expected 'message' object"):
        provider.generate_answer("Question?", [_make_evidence()])

    # Missing 'content'
    def handler_no_content(request: httpx.Request) -> httpx.Response:
        return httpx.Response(200, json={"message": {"role": "assistant"}})

    client = httpx.Client(transport=httpx.MockTransport(handler_no_content), base_url="http://localhost:11434")
    provider = OllamaGenerationProvider(client=client)
    with pytest.raises(GenerationInferenceError, match="missing expected 'message.content' string"):
        provider.generate_answer("Question?", [_make_evidence()])


def test_ollama_empty_answer_rejected() -> None:
    """Verify empty or reasoning-only content raises GenerationInferenceError."""
    # Whitespace only
    def handler_whitespace(request: httpx.Request) -> httpx.Response:
        return httpx.Response(200, json={"message": {"content": "   \n\t  "}})

    client = httpx.Client(transport=httpx.MockTransport(handler_whitespace), base_url="http://localhost:11434")
    provider = OllamaGenerationProvider(client=client)
    with pytest.raises(GenerationInferenceError, match="produced an empty answer after sanitization"):
        provider.generate_answer("Question?", [_make_evidence()])

    # Reasoning only
    def handler_reasoning_only(request: httpx.Request) -> httpx.Response:
        return httpx.Response(200, json={"message": {"content": "<think>Thinking through every option.</think>"}})

    client = httpx.Client(transport=httpx.MockTransport(handler_reasoning_only), base_url="http://localhost:11434")
    provider = OllamaGenerationProvider(client=client)
    with pytest.raises(GenerationInferenceError, match="produced an empty answer after sanitization"):
        provider.generate_answer("Question?", [_make_evidence()])


def test_ollama_factory_registration() -> None:
    """Verify factory returns OllamaGenerationProvider when configured and preserves mock default."""
    # 1. Default settings returns DeterministicMockGenerator
    default_settings = Settings()
    default_provider = get_generation_provider(default_settings)
    assert isinstance(default_provider, DeterministicMockGenerator)
    assert default_provider.model_id == "mock-generator-v1"

    # 2. Configured GENERATION_PROVIDER="ollama" returns OllamaGenerationProvider
    ollama_settings = Settings(
        GENERATION_PROVIDER="ollama",
        OLLAMA_BASE_URL="http://127.0.0.1:11434",
        OLLAMA_TIMEOUT_SECONDS=45.0,
    )
    ollama_provider = get_generation_provider(ollama_settings)
    assert isinstance(ollama_provider, OllamaGenerationProvider)
    assert ollama_provider.model_id == "qwen3:4b"
    assert ollama_provider.base_url == "http://127.0.0.1:11434"
    assert ollama_provider.timeout == 45.0
    ollama_provider.close()

    # 3. Custom model override
    custom_settings = Settings(
        GENERATION_PROVIDER="ollama",
        GENERATION_MODEL="custom-llm:8b",
    )
    custom_provider = get_generation_provider(custom_settings)
    assert isinstance(custom_provider, OllamaGenerationProvider)
    assert custom_provider.model_id == "custom-llm:8b"
    custom_provider.close()

    # 4. Unsupported provider raises GenerationConfigurationError
    bad_settings = Settings(GENERATION_PROVIDER="unsupported_cloud_provider")
    with pytest.raises(GenerationConfigurationError, match="Supported providers: 'mock', 'ollama'"):
        get_generation_provider(bad_settings)


def test_ollama_qa_orchestration_end_to_end_mocked() -> None:
    """Verify QuestionAnsweringService seamlessly validates citations produced by OllamaGenerationProvider."""
    doc_id = uuid.uuid4()
    chunk_id = uuid.uuid4()

    def handler(request: httpx.Request) -> httpx.Response:
        return httpx.Response(
            status_code=200,
            json={
                "message": {
                    "role": "assistant",
                    "content": "The annual health stipend is $1,200 [1].",
                }
            },
        )

    client = httpx.Client(transport=httpx.MockTransport(handler), base_url="http://localhost:11434")
    ollama_provider = OllamaGenerationProvider(client=client)

    # Mock retrieval service returning a single candidate chunk
    class MockRetrievalService:
        def retrieve(self, **kwargs: Any) -> RetrievalResponse:
            return RetrievalResponse(
                query="stipend",
                results=[
                    RetrievedChunk(
                        chunk_id=chunk_id,
                        document_id=doc_id,
                        document_filename="handbook.pdf",
                        chunk_index=0,
                        text="The annual health stipend is $1,200.",
                        score=0.92,
                        cosine_distance=0.08,
                        page_number=1,
                        model_id="mock-embed",
                    )
                ],
                returned_count=1,
                model_id="mock-embed",
            )

    qa_service = QuestionAnsweringService(
        retrieval_service=MockRetrievalService(),  # type: ignore[arg-type]
        generation_provider=ollama_provider,
    )

    response = qa_service.answer_question("What is the health stipend?")
    assert response.is_abstained is False
    assert response.has_citations is True
    assert response.is_valid_citations is True
    assert len(response.citations) == 1
    assert response.citations[0].marker == "[1]"
    assert response.citations[0].chunk_id == chunk_id
    assert response.citations[0].document_filename == "handbook.pdf"


def test_ollama_qa_service_rejects_hallucinated_citation() -> None:
    """Verify downstream QA validation rejects answers if Ollama hallucinated an unsupplied citation marker."""
    def handler(request: httpx.Request) -> httpx.Response:
        return httpx.Response(
            status_code=200,
            json={
                "message": {
                    "role": "assistant",
                    "content": "Grounded claim [1], but also ungrounded claim [99].",
                }
            },
        )

    client = httpx.Client(transport=httpx.MockTransport(handler), base_url="http://localhost:11434")
    ollama_provider = OllamaGenerationProvider(client=client)

    class MockRetrievalService:
        def retrieve(self, **kwargs: Any) -> RetrievalResponse:
            return RetrievalResponse(
                query="claim",
                results=[
                    RetrievedChunk(
                        chunk_id=uuid.uuid4(),
                        document_id=uuid.uuid4(),
                        document_filename="doc.md",
                        chunk_index=0,
                        text="Valid passage text.",
                        score=0.88,
                        cosine_distance=0.12,
                        model_id="mock-embed",
                    )
                ],
                returned_count=1,
                model_id="mock-embed",
            )

    qa_service = QuestionAnsweringService(
        retrieval_service=MockRetrievalService(),  # type: ignore[arg-type]
        generation_provider=ollama_provider,
    )

    response = qa_service.answer_question("Test query")
    # Downstream citation validation detects [99] is not in context and forces safe abstention
    assert response.is_abstained is True
    assert response.abstention_reason == "citation_validation_failed"
    assert response.is_valid_citations is False
    assert "99" in response.unknown_citations
