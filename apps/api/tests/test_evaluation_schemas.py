"""Unit and integrity tests for Evaluation Schemas and Dataset Splits (Milestone 9.0 Task 9.1).

Verifies:
1. Strict Pydantic validation rules for EvaluationCase, CorpusDocumentRef, and EvaluationDataset.
2. Complete integrity of the versioned evaluation_dataset_v1.json file against corpus fixtures.
3. Cryptographic digest verification of corpus documents.
4. Mechanical zero-leakage guarantee between dev and test partitions.
5. Immutability of the legacy baseline_retrieval_cases.json benchmark.
"""

import hashlib
import json
from pathlib import Path
import pytest
from pydantic import ValidationError

from app.schemas.evaluation import (
    CorpusDocumentRef,
    EvaluationCase,
    EvaluationDataset,
    ExpectedEvidenceRef,
)

FIXTURES_DIR = Path(__file__).parent / "evaluation" / "fixtures"
DATASET_V1_PATH = Path(__file__).parent / "evaluation" / "evaluation_dataset_v1.json"
LEGACY_CASES_PATH = Path(__file__).parent / "evaluation" / "baseline_retrieval_cases.json"


# ==============================================================================
# 1. Pydantic Schema Unit Validation Tests
# ==============================================================================


def test_corpus_document_ref_valid() -> None:
    """Valid CorpusDocumentRef instances pass validation."""
    doc = CorpusDocumentRef(
        filename="platform_reference_manual.md",
        sha256="cf25023fb0db45e82a00bef69ce27c72c46e2df28aaa4ba2b343e577dbc06913",
        size_bytes=3771,
    )
    assert doc.filename == "platform_reference_manual.md"
    assert doc.size_bytes == 3771


def test_corpus_document_ref_rejects_invalid_sha256() -> None:
    """CorpusDocumentRef strictly rejects non-64-char or non-hex sha256 digests."""
    with pytest.raises(ValidationError):
        CorpusDocumentRef(
            filename="doc.md",
            sha256="short_hash",
            size_bytes=100,
        )

    with pytest.raises(ValidationError):
        CorpusDocumentRef(
            filename="doc.md",
            sha256="z" * 64,  # 'z' is not a hex character
            size_bytes=100,
        )


def test_evaluation_case_answerable_consistency() -> None:
    """Answerable cases require evidence snippets and reference claims, and prohibit abstention reasons."""
    # Valid answerable case
    case = EvaluationCase(
        dataset_id="test-ds",
        case_id="case-001",
        split="dev",
        question="What is the answer?",
        question_type="direct_factual",
        answerable=True,
        target_document_filename="doc.md",
        expected_evidence_snippets=["The answer is 42."],
        reference_claims=["The answer is 42."],
        forbidden_claims=["The answer is 99."],
    )
    assert case.answerable is True
    assert case.expected_evidence_snippets == ["The answer is 42."]

    # Missing evidence snippets
    with pytest.raises(ValidationError, match="must provide at least one expected_evidence_snippet"):
        EvaluationCase(
            dataset_id="test-ds",
            case_id="case-002",
            split="dev",
            question="What is the answer?",
            question_type="direct_factual",
            answerable=True,
            target_document_filename="doc.md",
            expected_evidence_snippets=[],
            reference_claims=["Some claim."],
        )

    # Missing reference claims
    with pytest.raises(ValidationError, match="must provide at least one reference_claim"):
        EvaluationCase(
            dataset_id="test-ds",
            case_id="case-003",
            split="dev",
            question="What is the answer?",
            question_type="direct_factual",
            answerable=True,
            target_document_filename="doc.md",
            expected_evidence_snippets=["Some snippet."],
            reference_claims=[],
        )

    # Answerable case specifying abstention reasons
    with pytest.raises(ValidationError, match="cannot specify acceptable_abstention_reasons"):
        EvaluationCase(
            dataset_id="test-ds",
            case_id="case-004",
            split="dev",
            question="What is the answer?",
            question_type="direct_factual",
            answerable=True,
            target_document_filename="doc.md",
            expected_evidence_snippets=["Some snippet."],
            reference_claims=["Some claim."],
            acceptable_abstention_reasons=["no_evidence_found"],
        )


def test_evaluation_case_unanswerable_consistency() -> None:
    """Unanswerable cases prohibit evidence/claims and require at least one acceptable abstention reason."""
    # Valid unanswerable case
    case = EvaluationCase(
        dataset_id="test-ds",
        case_id="case-unans",
        split="test",
        question="What is the secret?",
        question_type="unanswerable_in_domain",
        answerable=False,
        target_document_filename="doc.md",
        expected_evidence_snippets=[],
        reference_claims=[],
        acceptable_abstention_reasons=["no_evidence_found", "insufficient_evidence"],
    )
    assert case.answerable is False

    # Prohibits expected evidence
    with pytest.raises(ValidationError, match="cannot have expected_evidence_snippets"):
        EvaluationCase(
            dataset_id="test-ds",
            case_id="case-bad",
            split="test",
            question="What is the secret?",
            question_type="unanswerable_in_domain",
            answerable=False,
            target_document_filename="doc.md",
            expected_evidence_snippets=["Unexpected text."],
            reference_claims=[],
            acceptable_abstention_reasons=["no_evidence_found"],
        )

    # Missing abstention reasons
    with pytest.raises(ValidationError, match="must specify at least one acceptable_abstention_reason"):
        EvaluationCase(
            dataset_id="test-ds",
            case_id="case-bad2",
            split="test",
            question="What is the secret?",
            question_type="unanswerable_in_domain",
            answerable=False,
            target_document_filename="doc.md",
            expected_evidence_snippets=[],
            reference_claims=[],
            acceptable_abstention_reasons=[],
        )


def test_evaluation_case_rejects_invalid_split() -> None:
    """EvaluationCase strictly rejects split values other than 'dev' and 'test'."""
    with pytest.raises(ValidationError):
        EvaluationCase(
            dataset_id="test-ds",
            case_id="case-001",
            split="train",  # Only 'dev' and 'test' are permitted
            question="Question?",
            question_type="direct_factual",
            answerable=True,
            target_document_filename="doc.md",
            expected_evidence_snippets=["Text"],
            reference_claims=["Claim"],
        )


def test_evaluation_dataset_rejects_duplicate_case_ids() -> None:
    """EvaluationDataset rejects cases with duplicate case_id values."""
    c1 = EvaluationCase(
        dataset_id="ds-1",
        case_id="eval-dup",
        split="dev",
        question="Q1",
        question_type="direct_factual",
        answerable=True,
        target_document_filename="doc.md",
        expected_evidence_snippets=["Text"],
        reference_claims=["Claim"],
    )
    c2 = EvaluationCase(
        dataset_id="ds-1",
        case_id="eval-dup",
        split="test",
        question="Q2",
        question_type="direct_factual",
        answerable=True,
        target_document_filename="doc.md",
        expected_evidence_snippets=["Text"],
        reference_claims=["Claim"],
    )
    doc = CorpusDocumentRef(filename="doc.md", sha256="a" * 64, size_bytes=100)

    with pytest.raises(ValidationError, match="Duplicate case_id 'eval-dup' detected"):
        EvaluationDataset(
            dataset_id="ds-1",
            name="Test Dataset",
            version="1.0.0",
            description="Test",
            corpus_documents=[doc],
            cases=[c1, c2],
        )


def test_evaluation_dataset_requires_both_splits() -> None:
    """EvaluationDataset rejects datasets missing either the dev or test split."""
    c1 = EvaluationCase(
        dataset_id="ds-1",
        case_id="eval-001",
        split="dev",
        question="Q1",
        question_type="direct_factual",
        answerable=True,
        target_document_filename="doc.md",
        expected_evidence_snippets=["Text"],
        reference_claims=["Claim"],
    )
    doc = CorpusDocumentRef(filename="doc.md", sha256="a" * 64, size_bytes=100)

    with pytest.raises(ValidationError, match="must contain at least one 'test' case"):
        EvaluationDataset(
            dataset_id="ds-1",
            name="Test Dataset",
            version="1.0.0",
            description="Test",
            corpus_documents=[doc],
            cases=[c1],
        )


def test_evaluation_dataset_rejects_unlisted_target_document() -> None:
    """EvaluationDataset rejects cases targeting documents not present in corpus_documents."""
    c1 = EvaluationCase(
        dataset_id="ds-1",
        case_id="eval-001",
        split="dev",
        question="Q1",
        question_type="direct_factual",
        answerable=True,
        target_document_filename="unregistered_document.md",
        expected_evidence_snippets=["Text"],
        reference_claims=["Claim"],
    )
    c2 = EvaluationCase(
        dataset_id="ds-1",
        case_id="eval-002",
        split="test",
        question="Q2",
        question_type="direct_factual",
        answerable=True,
        target_document_filename="registered.md",
        expected_evidence_snippets=["Text"],
        reference_claims=["Claim"],
    )
    doc = CorpusDocumentRef(filename="registered.md", sha256="a" * 64, size_bytes=100)

    with pytest.raises(ValidationError, match="references target document 'unregistered_document.md'"):
        EvaluationDataset(
            dataset_id="ds-1",
            name="Test Dataset",
            version="1.0.0",
            description="Test",
            corpus_documents=[doc],
            cases=[c1, c2],
        )


# ==============================================================================
# 2. Dataset V1 File & Corpus Integrity Tests
# ==============================================================================


@pytest.fixture(scope="module")
def dataset_v1() -> EvaluationDataset:
    """Load and validate the evaluation_dataset_v1.json artifact."""
    assert DATASET_V1_PATH.is_file(), f"Dataset file not found at {DATASET_V1_PATH}"
    data = json.loads(DATASET_V1_PATH.read_text(encoding="utf-8"))
    return EvaluationDataset.model_validate(data)


def test_dataset_v1_metadata_and_corpus_digests(dataset_v1: EvaluationDataset) -> None:
    """Verify that dataset metadata and fixture document SHA-256 digests match disk reality."""
    assert dataset_v1.schema_version == "1.0.0"
    assert dataset_v1.dataset_id == "evidencedesk-rag-v1.0.0"
    assert dataset_v1.version == "1.0.0"
    assert len(dataset_v1.corpus_documents) == 4

    for doc_ref in dataset_v1.corpus_documents:
        fixture_path = FIXTURES_DIR / doc_ref.filename
        assert fixture_path.is_file(), f"Fixture file {doc_ref.filename} does not exist on disk."
        file_bytes = fixture_path.read_bytes()
        actual_sha256 = hashlib.sha256(file_bytes).hexdigest()
        assert actual_sha256 == doc_ref.sha256, (
            f"Digest mismatch for {doc_ref.filename}: expected {doc_ref.sha256}, got {actual_sha256}"
        )
        assert len(file_bytes) == doc_ref.size_bytes, (
            f"Size mismatch for {doc_ref.filename}: expected {doc_ref.size_bytes}, got {len(file_bytes)}"
        )


def test_dataset_v1_total_cases_and_ids(dataset_v1: EvaluationDataset) -> None:
    """Verify total case count, case ID uniqueness, and valid provenance references."""
    assert len(dataset_v1.cases) == 38, f"Expected 38 migrated cases, got {len(dataset_v1.cases)}"

    case_ids = [c.case_id for c in dataset_v1.cases]
    assert len(case_ids) == len(set(case_ids)), "Case IDs must be strictly unique."

    for c in dataset_v1.cases:
        assert c.provenance is not None
        assert c.provenance.startswith("migrated from baseline_retrieval_cases.json:")
        assert c.dataset_id == "evidencedesk-rag-v1.0.0"
        assert c.schema_version == "1.0.0"


def test_dataset_v1_evidence_containment_in_fixtures(dataset_v1: EvaluationDataset) -> None:
    """Verify every answerable case's expected evidence snippets exist verbatim in the fixture file."""
    fixture_texts = {p.name: p.read_text(encoding="utf-8") for p in FIXTURES_DIR.glob("*.md")}

    for c in dataset_v1.cases:
        if c.answerable:
            assert len(c.expected_evidence_snippets) >= 1
            doc_text = fixture_texts[c.target_document_filename]
            for snippet in c.expected_evidence_snippets:
                assert snippet in doc_text, (
                    f"Evidence snippet in case '{c.case_id}' was not found in {c.target_document_filename}."
                )
        else:
            assert c.expected_evidence_snippets == []
            assert c.reference_claims == []
            assert len(c.acceptable_abstention_reasons) >= 1


# ==============================================================================
# 3. Mechanical Split Partitioning & Zero-Leakage Tests
# ==============================================================================


def test_dataset_v1_split_balance(dataset_v1: EvaluationDataset) -> None:
    """Verify partition counts: 26 dev cases (~68%) and 12 test cases (~32%)."""
    dev_cases = [c for c in dataset_v1.cases if c.split == "dev"]
    test_cases = [c for c in dataset_v1.cases if c.split == "test"]

    assert len(dev_cases) == 26, f"Expected 26 dev cases, found {len(dev_cases)}"
    assert len(test_cases) == 12, f"Expected 12 test cases, found {len(test_cases)}"
    assert len(dev_cases) + len(test_cases) == 38

    # Answerability distribution
    dev_ans = [c for c in dev_cases if c.answerable]
    dev_unans = [c for c in dev_cases if not c.answerable]
    test_ans = [c for c in test_cases if c.answerable]
    test_unans = [c for c in test_cases if not c.answerable]

    assert len(dev_ans) == 22
    assert len(dev_unans) == 4
    assert len(test_ans) == 10
    assert len(test_unans) == 2


def test_dataset_v1_cross_split_section_isolation(dataset_v1: EvaluationDataset) -> None:
    """Mechanically verify that zero document sections or fact clusters overlap across splits."""
    dev_sections = {c.target_section for c in dataset_v1.cases if c.split == "dev" and c.target_section}
    test_sections = {c.target_section for c in dataset_v1.cases if c.split == "test" and c.target_section}

    overlap = dev_sections & test_sections
    assert len(overlap) == 0, f"Cross-split section leakage detected: {overlap}"


def test_dataset_v1_cross_split_evidence_leakage(dataset_v1: EvaluationDataset) -> None:
    """Mechanically verify that no evidence text in the test split appears in the dev split."""
    dev_snippets = {
        s.strip().lower()
        for c in dataset_v1.cases
        if c.split == "dev"
        for s in c.expected_evidence_snippets
    }
    test_snippets = {
        s.strip().lower()
        for c in dataset_v1.cases
        if c.split == "test"
        for s in c.expected_evidence_snippets
    }

    overlap = dev_snippets & test_snippets
    assert len(overlap) == 0, f"Cross-split evidence snippet leakage detected: {overlap}"


def test_dataset_v1_near_duplicate_pairs_grouped_together(dataset_v1: EvaluationDataset) -> None:
    """Mechanically verify that known direct/paraphrased fact pairs are held together in the same split."""
    cases_by_id = {c.case_id: c for c in dataset_v1.cases}

    # Pairs targeting the exact same section or fact:
    # 1. 25 MB max upload size (Sec 1)
    assert cases_by_id["eval-001"].split == cases_by_id["eval-013"].split == "dev"
    assert cases_by_id["eval-019"].split == "dev"

    # 2. Failed reprocessing transaction rollback (Sec 4)
    assert cases_by_id["eval-004"].split == cases_by_id["eval-014"].split == "dev"

    # 3. Vector similarity score range [-1.0, 1.0] (Sec 6)
    assert cases_by_id["eval-006"].split == cases_by_id["eval-016"].split == "dev"

    # 4. Tokenizer context 512 tokens & chunking 500/75 (Sec 7)
    assert cases_by_id["eval-007"].split == cases_by_id["eval-015"].split == "test"

    # 5. Document extraction formats & structure-aware chunking (Sec 8)
    assert cases_by_id["eval-008"].split == cases_by_id["eval-022"].split == "dev"

    # 6. Top-k bounds & query length bounds (Sec 11)
    assert cases_by_id["eval-011"].split == cases_by_id["eval-017"].split == "dev"

    # 7. BGE-small 384 dimensions & SentenceTransformers CPU inference (Sec 3)
    assert cases_by_id["eval-003"].split == cases_by_id["eval-021"].split == "test"


# ==============================================================================
# 4. Legacy Dataset Immutability Verification
# ==============================================================================


def test_legacy_baseline_cases_file_preserved() -> None:
    """Verify that baseline_retrieval_cases.json is preserved unchanged with 38 legacy cases."""
    assert LEGACY_CASES_PATH.is_file(), f"Legacy cases file missing at {LEGACY_CASES_PATH}"
    data = json.loads(LEGACY_CASES_PATH.read_text(encoding="utf-8"))
    assert isinstance(data, list)
    assert len(data) == 38

    # Verify essential fields remain present
    for item in data:
        assert "case_id" in item
        assert "question" in item
        assert "question_type" in item
        assert "answerable" in item
        assert "target_document_filename" in item
        assert "evidence_text" in item
