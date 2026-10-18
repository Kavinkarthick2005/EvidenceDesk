"""Provider-independent generation interface and evidence contract for EvidenceDesk."""

from collections.abc import Sequence
from typing import Protocol, Self, runtime_checkable
import uuid

from pydantic import BaseModel, ConfigDict, Field, field_validator, model_validator


class EvidenceItem(BaseModel):
    """A discrete unit of retrieved passage evidence presented to a generation provider.

    Decoupled from database models and HTTP requests. Represents a self-contained
    passage anchor that the generator can reference in citations.
    """

    model_config = ConfigDict(frozen=True)

    id: str = Field(
        min_length=1,
        description="Unique citation identifier for the evidence passage (e.g., '1', 'chunk-1').",
    )
    text: str = Field(
        min_length=1,
        description="Exact extracted textual content of the evidence passage.",
    )
    document_title: str | None = Field(
        default=None,
        description="Optional title or original filename of the source document.",
    )
    heading: str | None = Field(
        default=None,
        description="Optional Markdown section heading enclosing the passage.",
    )
    chunk_id: uuid.UUID | None = Field(
        default=None,
        description="Optional UUID of the underlying stored chunk for end-to-end traceability.",
    )

    @field_validator("id", "text")
    @classmethod
    def validate_non_whitespace(cls, v: str) -> str:
        """Reject empty or whitespace-only strings."""
        if not v.strip():
            raise ValueError("Value cannot be empty or whitespace-only.")
        return v


class GenerationResult(BaseModel):
    """Result of generating an evidence-grounded answer.

    NOTE: cited_evidence_ids identifies the evidence items the model referenced
    in its response. This does NOT imply or guarantee that the model factually
    entails the claims; semantic entailment must be verified independently.
    """

    model_config = ConfigDict(frozen=True)

    answer: str = Field(
        description="Generated answer text or explicit refusal/abstention message.",
    )
    cited_evidence_ids: list[str] = Field(
        default_factory=list,
        description="List of evidence item IDs cited by the generator as supporting material.",
    )
    is_abstained: bool = Field(
        default=False,
        description="True if the generator abstained due to insufficient or conflicting evidence.",
    )
    abstention_reason: str | None = Field(
        default=None,
        description="Diagnostic reason code if is_abstained is True (e.g., 'insufficient_evidence').",
    )
    model_id: str = Field(
        description="Canonical identifier of the generation model that produced this result.",
    )

    @model_validator(mode="after")
    def validate_abstention_consistency(self) -> Self:
        """Enforce logical consistency across abstention state, reason, and citations.

        - If is_abstained is True:
            * An abstention_reason must be provided.
            * cited_evidence_ids must be empty (cannot cite evidence for an abstained response).
        - If is_abstained is False:
            * abstention_reason must be None.
        """
        if self.is_abstained:
            if not self.abstention_reason or not self.abstention_reason.strip():
                raise ValueError("Abstention requires an explicit, non-empty abstention_reason.")
            if self.cited_evidence_ids:
                raise ValueError(
                    f"Abstained result cannot cite evidence IDs (found {len(self.cited_evidence_ids)} citations)."
                )
        else:
            if self.abstention_reason is not None:
                raise ValueError(
                    f"Non-abstained result cannot have an abstention_reason (got '{self.abstention_reason}')."
                )
        return self


@runtime_checkable
class GenerationProvider(Protocol):
    """Abstract interface defining question-answering generation capabilities.

    Implementations must accept a question and an explicit sequence of EvidenceItem
    records, returning a structured GenerationResult without relying on global state,
    FastAPI request contexts, or direct database access.
    """

    @property
    def model_id(self) -> str:
        """Canonical identifier of the active generation model."""
        ...

    def generate_answer(
        self,
        query: str,
        evidence: Sequence[EvidenceItem],
    ) -> GenerationResult:
        """Generate an evidence-grounded answer for the query using supplied evidence.

        Args:
            query: Natural language question string (must be non-empty, non-whitespace).
            evidence: Explicit sequence of evidence items available to the model.

        Returns:
            GenerationResult containing the generated answer, cited evidence IDs,
            and abstention status.

        Raises:
            InvalidPromptError: If query is empty, whitespace-only, or invalid.
            GenerationInferenceError: If model inference fails.
            GenerationTimeoutError: If generation exceeds configured execution timeout.
            GenerationError: Base exception for all generation failures.
        """
        ...
