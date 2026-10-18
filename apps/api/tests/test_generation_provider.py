"""Unit tests for generation provider interface, deterministic mock, and factory.

Runs fast and completely in-memory with zero network calls, API keys, or GPU dependencies.
"""

from collections.abc import Generator
import uuid

from pydantic import ValidationError
import pytest

from app.core.config import Settings
from app.generation import (
    DeterministicMockGenerator,
    EvidenceItem,
    GenerationConfigurationError,
    GenerationError,
    GenerationInferenceError,
    GenerationProvider,
    GenerationResult,
    GenerationTimeoutError,
    InvalidPromptError,
    get_generation_provider,
    reset_generation_state,
)


@pytest.fixture(autouse=True)
def isolate_generation_env() -> Generator[None, None, None]:
    """Ensure generation provider state is reset before and after each test."""
    reset_generation_state()
    yield
    reset_generation_state()


def test_generation_provider_protocol_conformance() -> None:
    """Verify DeterministicMockGenerator conforms to runtime GenerationProvider protocol."""
    provider = DeterministicMockGenerator(model_id="mock-test-model")
    assert isinstance(provider, GenerationProvider)
    assert provider.model_id == "mock-test-model"


def test_evidence_item_and_generation_result_immutability() -> None:
    """Verify EvidenceItem and GenerationResult models are frozen and validated."""
    chunk_uuid = uuid.uuid4()
    item = EvidenceItem(
        id="1",
        text="Sample retrieved passage text.",
        document_title="guide.md",
        heading="Getting Started",
        chunk_id=chunk_uuid,
    )
    assert item.id == "1"
    assert item.text == "Sample retrieved passage text."
    assert item.document_title == "guide.md"
    assert item.heading == "Getting Started"
    assert item.chunk_id == chunk_uuid

    # Immutable: mutating attributes raises ValidationError
    with pytest.raises(ValidationError):
        item.text = "New text"  # type: ignore[misc]

    # Rejects empty or whitespace-only id or text
    with pytest.raises(ValidationError):
        EvidenceItem(id="", text="valid text")

    with pytest.raises(ValidationError):
        EvidenceItem(id="   ", text="valid text")

    with pytest.raises(ValidationError):
        EvidenceItem(id="1", text="")

    with pytest.raises(ValidationError):
        EvidenceItem(id="1", text="   \t\n  ")

    # Result immutability
    result = GenerationResult(
        answer="Valid grounded answer.",
        cited_evidence_ids=["1"],
        is_abstained=False,
        model_id="mock-test",
    )
    with pytest.raises(ValidationError):
        result.answer = "Modified answer"  # type: ignore[misc]


def test_generation_result_enforces_abstention_consistency() -> None:
    """Verify GenerationResult model validator rejects contradictory abstention states."""
    # 1. is_abstained=True requires an abstention_reason
    with pytest.raises(ValidationError, match="Abstention requires an explicit, non-empty abstention_reason"):
        GenerationResult(
            answer="Refusal.",
            is_abstained=True,
            abstention_reason=None,
            model_id="m",
        )

    # 2. is_abstained=True cannot cite evidence IDs
    with pytest.raises(ValidationError, match="Abstained result cannot cite evidence IDs"):
        GenerationResult(
            answer="Refusal.",
            is_abstained=True,
            abstention_reason="refusal",
            cited_evidence_ids=["ev-1"],
            model_id="m",
        )

    # 3. is_abstained=False cannot have an abstention_reason
    with pytest.raises(ValidationError, match="Non-abstained result cannot have an abstention_reason"):
        GenerationResult(
            answer="Normal answer.",
            is_abstained=False,
            abstention_reason="contradictory_reason",
            cited_evidence_ids=["ev-1"],
            model_id="m",
        )


def test_mock_generator_deterministic_output() -> None:
    """Verify repeated identical queries and evidence produce bitwise identical results."""
    provider = DeterministicMockGenerator(model_id="mock-gen")
    evidence = [
        EvidenceItem(
            id="1",
            text="PostgreSQL 16 with pgvector handles vector search.",
            document_title="manual.md",
            heading="Storage",
        ),
        EvidenceItem(
            id="2",
            text="Chunking splits text into 500-token segments.",
            document_title="manual.md",
            heading="Extraction",
        ),
    ]

    res1 = provider.generate_answer("What database is used?", evidence)
    res2 = provider.generate_answer("What database is used?", evidence)

    assert res1 == res2
    assert res1.answer == res2.answer
    assert res1.cited_evidence_ids == res2.cited_evidence_ids
    assert res1.is_abstained is False
    assert res1.model_id == "mock-gen"
    assert "[Mock Answer]" in res1.answer
    assert "[1]" in res1.answer
    assert res1.cited_evidence_ids == ["1", "2"]


def test_mock_generator_cites_only_supplied_evidence() -> None:
    """Verify mock generator cites only IDs present in supplied evidence and never invents IDs."""
    provider = DeterministicMockGenerator()
    evidence = [
        EvidenceItem(id="doc-chunk-alpha", text="First evidence passage."),
        EvidenceItem(id="doc-chunk-beta", text="Second evidence passage."),
    ]

    result = provider.generate_answer("Explain the architecture.", evidence)
    assert not result.is_abstained
    assert result.cited_evidence_ids == ["doc-chunk-alpha", "doc-chunk-beta"]

    # Invariant: cited_evidence_ids must be a strict subset of supplied IDs
    supplied_ids = {e.id for e in evidence}
    assert set(result.cited_evidence_ids).issubset(supplied_ids)


def test_mock_generator_abstains_on_empty_evidence() -> None:
    """Verify provider honestly abstains with insufficient_evidence when evidence list is empty."""
    provider = DeterministicMockGenerator()
    result = provider.generate_answer("What is the secret key?", evidence=[])

    assert result.is_abstained is True
    assert result.abstention_reason == "insufficient_evidence"
    assert result.cited_evidence_ids == []
    assert "no supporting evidence was provided" in result.answer.lower()


def test_mock_generator_forced_abstention() -> None:
    """Verify provider supports explicit forced abstention for testing abstention pipelines."""
    provider = DeterministicMockGenerator(
        should_abstain=True,
        abstention_reason="out_of_domain",
    )
    evidence = [EvidenceItem(id="1", text="Some passage.")]
    result = provider.generate_answer("Unanswerable question?", evidence)

    assert result.is_abstained is True
    assert result.abstention_reason == "out_of_domain"
    assert result.cited_evidence_ids == []
    assert "cannot answer" in result.answer.lower()


def test_mock_generator_simulates_hallucinated_citation() -> None:
    """Verify simulate_hallucinated_citation appends unsupplied citation for validation testing."""
    provider = DeterministicMockGenerator(simulate_hallucinated_citation=True)
    evidence = [EvidenceItem(id="ev-1", text="Legitimate passage.")]
    result = provider.generate_answer("Question", evidence)

    assert "unsupplied-evidence-999" in result.cited_evidence_ids
    assert "unsupplied-evidence-999" in result.answer


def test_mock_generator_rejects_duplicate_evidence_ids() -> None:
    """Verify generator rejects duplicate evidence IDs with InvalidPromptError."""
    provider = DeterministicMockGenerator()
    duplicate_evidence = [
        EvidenceItem(id="ev-1", text="First passage."),
        EvidenceItem(id="ev-1", text="Second passage with duplicate ID."),
    ]
    with pytest.raises(InvalidPromptError, match="Duplicate evidence ID 'ev-1' detected"):
        provider.generate_answer("What is the system?", duplicate_evidence)


def test_mock_generator_rejects_invalid_evidence_sequence() -> None:
    """Verify generator rejects non-sequence evidence or items not matching EvidenceItem."""
    provider = DeterministicMockGenerator()

    # Non-sequence
    with pytest.raises(InvalidPromptError, match="Evidence must be a sequence"):
        provider.generate_answer("Question", "not-a-sequence")  # type: ignore[arg-type]

    # Malformed element in sequence
    with pytest.raises(InvalidPromptError, match="must be an EvidenceItem instance"):
        provider.generate_answer("Question", ["not-an-evidence-item"])  # type: ignore[list-item]


def test_mock_generator_empty_evidence_with_hallucinated_flag_still_abstains() -> None:
    """Verify empty evidence strictly abstains with zero citations even if simulate_hallucinated_citation=True."""
    provider = DeterministicMockGenerator(simulate_hallucinated_citation=True)
    result = provider.generate_answer("Query with zero evidence", evidence=[])

    assert result.is_abstained is True
    assert result.abstention_reason == "insufficient_evidence"
    assert result.cited_evidence_ids == []


def test_mock_generator_query_validation() -> None:
    """Verify generator rejects empty, whitespace, and non-string queries with InvalidPromptError."""
    provider = DeterministicMockGenerator()
    evidence = [EvidenceItem(id="1", text="Text")]

    with pytest.raises(InvalidPromptError, match="cannot be empty or whitespace-only"):
        provider.generate_answer("", evidence)

    with pytest.raises(InvalidPromptError, match="cannot be empty or whitespace-only"):
        provider.generate_answer("   \n\t  ", evidence)

    with pytest.raises(InvalidPromptError, match="Query must be a string"):
        provider.generate_answer(None, evidence)  # type: ignore[arg-type]

    with pytest.raises(InvalidPromptError, match="Query must be a string"):
        provider.generate_answer(12345, evidence)  # type: ignore[arg-type]


def test_mock_generator_simulates_failure() -> None:
    """Verify should_fail and simulate_error trigger GenerationInferenceError."""
    # 1. should_fail flag
    failing_provider = DeterministicMockGenerator(should_fail=True)
    with pytest.raises(GenerationInferenceError, match="Mock generation failure simulated"):
        failing_provider.generate_answer("Query", [EvidenceItem(id="1", text="T")])

    # 2. Explicit simulate_error exception
    custom_err = GenerationTimeoutError("Simulated LLM gateway timeout (30s).")
    custom_provider = DeterministicMockGenerator(simulate_error=custom_err)
    with pytest.raises(GenerationTimeoutError, match="Simulated LLM gateway timeout"):
        custom_provider.generate_answer("Query", [EvidenceItem(id="1", text="T")])


def test_get_generation_provider_factory_default() -> None:
    """Verify provider factory returns DeterministicMockGenerator with default settings."""
    settings = Settings(GENERATION_PROVIDER="mock", _env_file=None)
    provider = get_generation_provider(settings)
    assert isinstance(provider, DeterministicMockGenerator)
    assert provider.model_id == "mock-generator-v1"


def test_get_generation_provider_factory_custom_model() -> None:
    """Verify provider factory respects custom GENERATION_MODEL."""
    settings = Settings(
        GENERATION_PROVIDER="mock",
        GENERATION_MODEL="custom-mock-v2",
        _env_file=None,
    )
    provider = get_generation_provider(settings)
    assert isinstance(provider, DeterministicMockGenerator)
    assert provider.model_id == "custom-mock-v2"


def test_get_generation_provider_factory_rejects_unsupported() -> None:
    """Verify provider factory raises GenerationConfigurationError for unsupported providers."""
    settings = Settings(
        GENERATION_PROVIDER="unsupported_cloud_provider",
        _env_file=None,
    )
    with pytest.raises(GenerationConfigurationError, match="Unsupported GENERATION_PROVIDER"):
        get_generation_provider(settings)


def test_generation_provider_singleton_caching_and_reset() -> None:
    """Verify factory caches singleton when settings is None and reset_generation_state clears it."""
    # 1. First retrieval caches singleton
    p1 = get_generation_provider()
    p2 = get_generation_provider()
    assert p1 is p2

    # 2. Reset clears singleton
    reset_generation_state()
    p3 = get_generation_provider()
    assert p3 is not p1
