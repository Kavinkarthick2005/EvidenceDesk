"""Pydantic schemas for versioned evaluation datasets and cases.

Defines the contract for evaluation dataset definitions, case metadata,
ground-truth evidence references, reference claims, negative constraints,
and split partitioning (development vs. held-out test).
"""

from typing import Any, Literal
import re

from pydantic import BaseModel, ConfigDict, Field, field_validator, model_validator

EvaluationSplit = Literal["dev", "test"]

_HEX_64_REGEX = re.compile(r"^[0-9a-f]{64}$")


class CorpusDocumentRef(BaseModel):
    """Reference to a source document within the evaluation corpus."""

    model_config = ConfigDict(frozen=True)

    filename: str = Field(
        min_length=1,
        description="Original filename of the fixture document in the evaluation corpus.",
    )
    sha256: str = Field(
        min_length=64,
        max_length=64,
        description="Hex-encoded SHA-256 digest of the fixture document bytes.",
    )
    size_bytes: int = Field(
        gt=0,
        description="Exact file size in bytes.",
    )

    @field_validator("sha256")
    @classmethod
    def validate_sha256_format(cls, v: str) -> str:
        """Enforce strict 64-character lowercase hex string."""
        lowered = v.strip().lower()
        if not _HEX_64_REGEX.match(lowered):
            raise ValueError(f"sha256 must be a 64-character lowercase hexadecimal string, got '{v}'.")
        return lowered


class ExpectedEvidenceRef(BaseModel):
    """Structured reference to an expected supporting evidence passage."""

    model_config = ConfigDict(frozen=True)

    document_filename: str = Field(
        min_length=1,
        description="Filename of the target document containing the evidence.",
    )
    text: str = Field(
        min_length=1,
        description="Exact ground-truth passage or sentence that supports the answer.",
    )
    heading: str | None = Field(
        default=None,
        description="Enclosing section heading where the evidence resides, if available.",
    )


class EvaluationCase(BaseModel):
    """A single evaluation case with question, ground truth, constraints, and split assignment."""

    model_config = ConfigDict(frozen=True)

    schema_version: str = Field(
        default="1.0.0",
        description="Semantic version of the evaluation case schema.",
    )
    dataset_id: str = Field(
        min_length=1,
        description="Identifier of the parent evaluation dataset.",
    )
    case_id: str = Field(
        min_length=1,
        description="Unique identifier for the evaluation case (e.g., 'eval-001').",
    )
    split: EvaluationSplit = Field(
        description="Dataset partition: 'dev' for tuning/calibration, 'test' for held-out evaluation.",
    )
    question: str = Field(
        min_length=1,
        max_length=2000,
        description="Natural language question text submitted to the system.",
    )
    question_type: str = Field(
        min_length=1,
        description="Taxonomy category: 'direct_factual', 'paraphrased', 'lexical_distractor', 'unanswerable_in_domain', etc.",
    )
    answerable: bool = Field(
        description="True if the question can be fully answered from the corpus; False if it warrants abstention.",
    )
    target_document_filename: str = Field(
        min_length=1,
        description="Primary document filename associated with the case or unanswerable query target.",
    )
    target_section: str | None = Field(
        default=None,
        description="Document section or heading group used to mechanically enforce cross-split isolation.",
    )
    expected_evidence_snippets: list[str] = Field(
        default_factory=list,
        description="Exact ground-truth text passages that must be retrieved to support the answer.",
    )
    reference_claims: list[str] = Field(
        default_factory=list,
        description="Core factual propositions that must be conveyed in an accurate, substantive answer.",
    )
    forbidden_claims: list[str] = Field(
        default_factory=list,
        description="Negative constraints: claims, entities, or measurements that must NOT appear in the answer.",
    )
    acceptable_abstention_reasons: list[str] = Field(
        default_factory=list,
        description="List of valid abstention reason codes when answerable is False (e.g. 'no_evidence_found').",
    )
    provenance: str | None = Field(
        default=None,
        description="Source tracking noting legacy dataset origins or manual curation notes.",
    )
    needs_manual_review: bool = Field(
        default=False,
        description="True if the ground-truth evidence or reference claims require further human verification.",
    )

    @field_validator("question", "case_id", "target_document_filename")
    @classmethod
    def validate_non_whitespace(cls, v: str) -> str:
        """Reject empty or whitespace-only string values."""
        stripped = v.strip()
        if not stripped:
            raise ValueError("Field cannot be empty or whitespace-only.")
        return stripped

    @model_validator(mode="after")
    def validate_answerability_consistency(self) -> "EvaluationCase":
        """Enforce strict consistency between answerability, evidence, claims, and abstention reasons."""
        if self.answerable:
            if not self.expected_evidence_snippets:
                raise ValueError(
                    f"Answerable case '{self.case_id}' must provide at least one expected_evidence_snippet."
                )
            if not self.reference_claims:
                raise ValueError(
                    f"Answerable case '{self.case_id}' must provide at least one reference_claim."
                )
            if self.acceptable_abstention_reasons:
                raise ValueError(
                    f"Answerable case '{self.case_id}' cannot specify acceptable_abstention_reasons."
                )
        else:
            if self.expected_evidence_snippets:
                raise ValueError(
                    f"Unanswerable case '{self.case_id}' cannot have expected_evidence_snippets."
                )
            if self.reference_claims:
                raise ValueError(
                    f"Unanswerable case '{self.case_id}' cannot have reference_claims."
                )
            if not self.acceptable_abstention_reasons:
                raise ValueError(
                    f"Unanswerable case '{self.case_id}' must specify at least one acceptable_abstention_reason."
                )
        return self


class EvaluationDataset(BaseModel):
    """Versioned collection of evaluation cases tied to a verified corpus."""

    model_config = ConfigDict(frozen=True)

    schema_version: str = Field(
        default="1.0.0",
        description="Semantic version of the dataset schema format.",
    )
    dataset_id: str = Field(
        min_length=1,
        description="Unique identifier for the dataset (e.g., 'evidencedesk-rag-v1.0.0').",
    )
    name: str = Field(
        min_length=1,
        description="Human-readable title of the evaluation dataset.",
    )
    version: str = Field(
        min_length=1,
        description="Dataset version identifier.",
    )
    description: str = Field(
        min_length=1,
        description="Description of the dataset purpose, scope, and evaluation methodology.",
    )
    corpus_documents: list[CorpusDocumentRef] = Field(
        min_length=1,
        description="Inventory of verified fixture documents and their cryptographic digests.",
    )
    cases: list[EvaluationCase] = Field(
        min_length=1,
        description="Complete list of evaluation cases across dev and test splits.",
    )

    @field_validator("dataset_id", "name", "version")
    @classmethod
    def validate_non_whitespace(cls, v: str) -> str:
        """Reject empty or whitespace-only string values."""
        stripped = v.strip()
        if not stripped:
            raise ValueError("Field cannot be empty or whitespace-only.")
        return stripped

    @model_validator(mode="after")
    def validate_dataset_invariants(self) -> "EvaluationDataset":
        """Validate ID uniqueness, split representation, and document references."""
        # 1. Verify case ID uniqueness
        seen_ids: set[str] = set()
        for c in self.cases:
            if c.case_id in seen_ids:
                raise ValueError(f"Duplicate case_id '{c.case_id}' detected in dataset '{self.dataset_id}'.")
            seen_ids.add(c.case_id)

        # 2. Verify split representation (both dev and test must have cases)
        dev_cases = [c for c in self.cases if c.split == "dev"]
        test_cases = [c for c in self.cases if c.split == "test"]
        if not dev_cases:
            raise ValueError(f"Dataset '{self.dataset_id}' must contain at least one 'dev' case.")
        if not test_cases:
            raise ValueError(f"Dataset '{self.dataset_id}' must contain at least one 'test' case.")

        # 3. Verify all target documents exist in corpus_documents
        known_docs = {doc.filename for doc in self.corpus_documents}
        for c in self.cases:
            if c.target_document_filename not in known_docs:
                raise ValueError(
                    f"Case '{c.case_id}' references target document '{c.target_document_filename}' "
                    f"which is not listed in corpus_documents."
                )

        return self
