"""Deterministic in-memory mock generation provider for fast, offline CI testing."""

from collections.abc import Sequence
import logging

from app.generation.exceptions import (
    GenerationInferenceError,
    InvalidPromptError,
)
from app.generation.interface import EvidenceItem, GenerationResult

logger = logging.getLogger(__name__)


class DeterministicMockGenerator:
    """Mock generation provider producing reproducible, evidence-grounded answers.

    Makes zero external network calls and requires no GPU or API keys. Designed for
    testing citation resolution, abstention pathways, and service orchestration.
    """

    def __init__(
        self,
        model_id: str = "mock-generator-v1",
        should_fail: bool = False,
        simulate_error: Exception | None = None,
        should_abstain: bool = False,
        abstention_reason: str | None = None,
        simulate_hallucinated_citation: bool = False,
    ) -> None:
        """Initialize mock generator with configurable test behaviors.

        Args:
            model_id: Canonical identifier of the simulated model.
            should_fail: If True, raises GenerationInferenceError on generation.
            simulate_error: Optional explicit exception to raise on generation.
            should_abstain: If True, forces abstention even if evidence is provided.
            abstention_reason: Custom reason code when abstaining.
            simulate_hallucinated_citation: If True, appends an unsupplied citation ID
                to test citation validation defenses.
        """
        self._model_id = model_id
        self.should_fail = should_fail
        self.simulate_error = simulate_error
        self.should_abstain = should_abstain
        self.abstention_reason = abstention_reason
        self.simulate_hallucinated_citation = simulate_hallucinated_citation

    @property
    def model_id(self) -> str:
        """Canonical identifier of the active generation model."""
        return self._model_id

    def generate_answer(
        self,
        query: str,
        evidence: Sequence[EvidenceItem],
    ) -> GenerationResult:
        """Generate a deterministic mock response grounded strictly in supplied evidence.

        Args:
            query: Natural language question string.
            evidence: Explicit sequence of evidence items available to the model.

        Returns:
            GenerationResult containing answer text, cited evidence IDs, and abstention status.

        Raises:
            InvalidPromptError: If query is empty, whitespace-only, or not a string.
            GenerationInferenceError: If should_fail is True or simulated error triggers.
        """
        # Validate query prompt
        if not isinstance(query, str):
            raise InvalidPromptError(f"Query must be a string, got {type(query).__name__}.")

        stripped_query = query.strip()
        if not stripped_query:
            raise InvalidPromptError("Query cannot be empty or whitespace-only.")

        # Simulate provider failure if configured
        if self.simulate_error is not None:
            raise self.simulate_error

        if self.should_fail:
            raise GenerationInferenceError(
                f"Mock generation failure simulated for model '{self._model_id}'."
            )

        # Handle explicit or forced abstention
        if self.should_abstain:
            reason = self.abstention_reason or "forced_abstention"
            return GenerationResult(
                answer="I cannot answer this question based on the provided evidence.",
                cited_evidence_ids=[],
                is_abstained=True,
                abstention_reason=reason,
                model_id=self._model_id,
            )

        # Validate evidence sequence and enforce unique evidence IDs
        if isinstance(evidence, (str, bytes)) or not isinstance(evidence, Sequence):
            raise InvalidPromptError(f"Evidence must be a sequence, got {type(evidence).__name__}.")

        seen_ids: set[str] = set()
        for idx, item in enumerate(evidence):
            if not isinstance(item, EvidenceItem):
                raise InvalidPromptError(
                    f"Evidence item at index {idx} must be an EvidenceItem instance, got {type(item).__name__}."
                )
            if item.id in seen_ids:
                raise InvalidPromptError(
                    f"Duplicate evidence ID '{item.id}' detected at index {idx}. "
                    "Evidence IDs must be uniquely identifiable."
                )
            seen_ids.add(item.id)

        # Handle zero-evidence condition (honest abstention)
        if not evidence:
            return GenerationResult(
                answer="I cannot answer this question because no supporting evidence was provided.",
                cited_evidence_ids=[],
                is_abstained=True,
                abstention_reason="insufficient_evidence",
                model_id=self._model_id,
            )

        # Deterministic generation using supplied evidence
        # Select up to the first 2 evidence passages
        selected_evidence = list(evidence[:2])
        cited_ids = [e.id for e in selected_evidence]

        citations_str = " ".join(f"[{e.id}]" for e in selected_evidence)
        source_doc = selected_evidence[0].document_title or "the supplied evidence"

        # Truncate passage snippet for clean formatting
        passage_snippets = "; ".join(
            f"'{e.text.strip()[:100]}...'" if len(e.text.strip()) > 100 else f"'{e.text.strip()}'"
            for e in selected_evidence
        )

        answer_text = (
            f"[Mock Answer] In response to '{stripped_query}', {source_doc} {citations_str} "
            f"states: {passage_snippets}"
        )

        # Optional test simulation: inject a citation ID that was NOT in the supplied evidence
        if self.simulate_hallucinated_citation:
            hallucinated_id = "unsupplied-evidence-999"
            cited_ids.append(hallucinated_id)
            answer_text += f" [{hallucinated_id}]"

        return GenerationResult(
            answer=answer_text,
            cited_evidence_ids=cited_ids,
            is_abstained=False,
            abstention_reason=None,
            model_id=self._model_id,
        )
