"""Unit tests for reusable RetrievalEvaluator (Milestone 9.0 Task 9.2).

Verifies:
1. Exact mathematical calculations of Recall@1, Recall@3, Recall@5, and MRR.
2. Ground-truth relevance matching across EvaluationCase and legacy dict formats.
3. Unanswerable query False Acceptance Rate (FAR) and Correct Rejection Rate (CRR).
4. Deliberate handling of empty answerable and unanswerable groups with explicit denominators.
5. Input bounds and configuration validation.
6. Threshold trade-off sweeps (FAR vs. FRR).
7. Native integration with Dataset V1 (evaluation_dataset_v1.json).
8. Backward compatibility with legacy benchmark cases (baseline_retrieval_cases.json).
"""

import json
from pathlib import Path
import uuid
import pytest

from app.evaluation.retrieval import (
    RetrievalCaseEvaluation,
    RetrievalEvaluationMetrics,
    RetrievalEvaluationReport,
    RetrievalEvaluator,
    ThresholdTradeoffPoint,
    is_chunk_relevant,
)
from app.schemas.evaluation import EvaluationCase, EvaluationDataset
from app.schemas.retrieval import RetrievedChunk

DATASET_V1_PATH = Path(__file__).parent / "evaluation" / "evaluation_dataset_v1.json"
LEGACY_CASES_PATH = Path(__file__).parent / "evaluation" / "baseline_retrieval_cases.json"


def _make_chunk(
    doc_filename: str,
    text: str,
    score: float,
    chunk_index: int = 0,
) -> RetrievedChunk:
    """Helper to build a deterministic RetrievedChunk."""
    return RetrievedChunk(
        chunk_id=uuid.uuid4(),
        document_id=uuid.uuid4(),
        document_filename=doc_filename,
        chunk_index=chunk_index,
        text=text,
        score=score,
        cosine_distance=1.0 - score,
        model_id="test-embed-model",
    )


# ==============================================================================
# 1. Relevance Matching Tests
# ==============================================================================


def test_is_chunk_relevant_matching_rules() -> None:
    """Verify document matching, substring containment, and unanswerable handling."""
    case = EvaluationCase(
        dataset_id="test-ds",
        case_id="eval-001",
        split="dev",
        question="What is the limit?",
        question_type="direct_factual",
        answerable=True,
        target_document_filename="manual.md",
        expected_evidence_snippets=["The limit is 25 MB by default."],
        reference_claims=["Limit is 25 MB."],
    )

    # 1. Exact match
    chunk_match = _make_chunk("manual.md", "Section 1: The limit is 25 MB by default.", 0.85)
    assert is_chunk_relevant(chunk_match, case) is True

    # 2. Case-insensitive substring match
    chunk_case_insensitive = _make_chunk("manual.md", "section 1: the limit is 25 mb by default.", 0.80)
    assert is_chunk_relevant(chunk_case_insensitive, case) is True

    # 3. Document mismatch (text matches but wrong document)
    chunk_wrong_doc = _make_chunk("wrong_manual.md", "The limit is 25 MB by default.", 0.90)
    assert is_chunk_relevant(chunk_wrong_doc, case) is False

    # 4. Text mismatch (right document but wrong text)
    chunk_wrong_text = _make_chunk("manual.md", "Irrelevant content about logging.", 0.70)
    assert is_chunk_relevant(chunk_wrong_text, case) is False

    # 5. Unanswerable case always returns False
    unans_case = EvaluationCase(
        dataset_id="test-ds",
        case_id="eval-unans",
        split="dev",
        question="What is the password?",
        question_type="unanswerable_in_domain",
        answerable=False,
        target_document_filename="manual.md",
        expected_evidence_snippets=[],
        reference_claims=[],
        acceptable_abstention_reasons=["no_evidence_found"],
    )
    assert is_chunk_relevant(chunk_match, unans_case) is False


def test_is_chunk_relevant_legacy_dict_compatibility() -> None:
    """Verify is_chunk_relevant supports legacy baseline_retrieval_cases dictionary format."""
    legacy_case = {
        "case_id": "legacy-001",
        "question": "What is the port?",
        "question_type": "direct_factual",
        "answerable": True,
        "target_document_filename": "runbook.md",
        "evidence_text": "Connects on port 5432.",
    }

    matching_chunk = _make_chunk("runbook.md", "The database connects on port 5432.", 0.95)
    non_matching_chunk = _make_chunk("runbook.md", "The database connects on port 8080.", 0.90)

    assert is_chunk_relevant(matching_chunk, legacy_case) is True
    assert is_chunk_relevant(non_matching_chunk, legacy_case) is False


# ==============================================================================
# 2. Deterministic IR Metrics: Recall@k & MRR
# ==============================================================================


def test_recall_and_mrr_deterministic_fixture() -> None:
    """Verify Recall@1, Recall@3, Recall@5, and MRR calculations on a controlled fixture.

    Fixture Setup: 4 Answerable Cases:
    - Case 1: First relevant at Rank 1 -> hits: 1, 3, 5; RR = 1/1 = 1.0
    - Case 2: First relevant at Rank 3 -> hits: 0, 3, 5; RR = 1/3 ≈ 0.33333333
    - Case 3: First relevant at Rank 5 -> hits: 0, 0, 5; RR = 1/5 = 0.2
    - Case 4: No relevant chunk in top 5 -> hits: 0, 0, 0; RR = 0.0

    Expected Aggregate Metrics (N_ans = 4):
    - Recall@1: 1 / 4 = 0.25
    - Recall@3: 2 / 4 = 0.50
    - Recall@5: 3 / 4 = 0.75
    - MRR: (1.0 + 1/3 + 0.2 + 0.0) / 4 = 1.53333333 / 4 ≈ 0.38333333
    """
    evaluator = RetrievalEvaluator(default_top_k=5, default_acceptance_threshold=0.55)

    cases_with_results = [
        # Case 1: Relevant at rank 1
        (
            EvaluationCase(
                dataset_id="test-ds",
                case_id="case-1",
                split="dev",
                question="Q1",
                question_type="direct_factual",
                answerable=True,
                target_document_filename="doc.md",
                expected_evidence_snippets=["Fact 1"],
                reference_claims=["Claim 1"],
            ),
            [
                _make_chunk("doc.md", "Fact 1 evidence", 0.90),
                _make_chunk("doc.md", "Irrelevant", 0.80),
                _make_chunk("doc.md", "Irrelevant", 0.70),
            ],
        ),
        # Case 2: Relevant at rank 3
        (
            EvaluationCase(
                dataset_id="test-ds",
                case_id="case-2",
                split="dev",
                question="Q2",
                question_type="direct_factual",
                answerable=True,
                target_document_filename="doc.md",
                expected_evidence_snippets=["Fact 2"],
                reference_claims=["Claim 2"],
            ),
            [
                _make_chunk("doc.md", "Distractor A", 0.92),
                _make_chunk("doc.md", "Distractor B", 0.88),
                _make_chunk("doc.md", "Fact 2 evidence", 0.85),
                _make_chunk("doc.md", "Irrelevant", 0.70),
            ],
        ),
        # Case 3: Relevant at rank 5
        (
            EvaluationCase(
                dataset_id="test-ds",
                case_id="case-3",
                split="dev",
                question="Q3",
                question_type="direct_factual",
                answerable=True,
                target_document_filename="doc.md",
                expected_evidence_snippets=["Fact 3"],
                reference_claims=["Claim 3"],
            ),
            [
                _make_chunk("doc.md", "Distractor 1", 0.91),
                _make_chunk("doc.md", "Distractor 2", 0.89),
                _make_chunk("doc.md", "Distractor 3", 0.87),
                _make_chunk("doc.md", "Distractor 4", 0.83),
                _make_chunk("doc.md", "Fact 3 evidence", 0.81),
            ],
        ),
        # Case 4: No relevant chunk found
        (
            EvaluationCase(
                dataset_id="test-ds",
                case_id="case-4",
                split="dev",
                question="Q4",
                question_type="direct_factual",
                answerable=True,
                target_document_filename="doc.md",
                expected_evidence_snippets=["Fact 4"],
                reference_claims=["Claim 4"],
            ),
            [
                _make_chunk("doc.md", "Distractor X", 0.85),
                _make_chunk("doc.md", "Distractor Y", 0.82),
            ],
        ),
    ]

    report = evaluator.evaluate_cases(cases_with_results, top_k=5)
    metrics = report.metrics

    assert metrics.total_cases == 4
    assert metrics.answerable_count == 4
    assert metrics.unanswerable_count == 0

    assert metrics.hit_at_1_count == 1
    assert metrics.hit_at_3_count == 2
    assert metrics.hit_at_5_count == 3

    assert metrics.recall_at_1 == pytest.approx(0.25, abs=1e-6)
    assert metrics.recall_at_3 == pytest.approx(0.50, abs=1e-6)
    assert metrics.recall_at_5 == pytest.approx(0.75, abs=1e-6)
    assert metrics.mrr == pytest.approx((1.0 + 1.0 / 3.0 + 0.2 + 0.0) / 4.0, abs=1e-6)

    # Check individual case details
    c1 = report.case_evaluations[0]
    assert c1.first_relevant_rank == 1
    assert c1.hit_at_1 is True
    assert c1.reciprocal_rank == 1.0

    c2 = report.case_evaluations[1]
    assert c2.first_relevant_rank == 3
    assert c2.hit_at_1 is False
    assert c2.hit_at_3 is True
    assert c2.reciprocal_rank == pytest.approx(1.0 / 3.0)

    c4 = report.case_evaluations[3]
    assert c4.first_relevant_rank is None
    assert c4.hit_at_5 is False
    assert c4.reciprocal_rank == 0.0


# ==============================================================================
# 3. Unanswerable Query Rejection: FAR & Correct-Reject Rate
# ==============================================================================


def test_unanswerable_queries_far_and_correct_reject() -> None:
    """Verify False Acceptance Rate and Correct Rejection Rate calculations.

    Setup: 3 Unanswerable Cases with acceptance threshold tau = 0.60:
    - Case A: max candidate score = 0.75 >= 0.60 -> False Accept
    - Case B: max candidate score = 0.50 < 0.60  -> Correct Reject
    - Case C: 0 candidate chunks returned        -> Correct Reject

    Expected Metrics (N_unans = 3, tau = 0.60):
    - false_accept_count: 1
    - correct_reject_count: 2
    - false_accept_rate (FAR): 1 / 3 ≈ 0.33333333
    - correct_reject_rate (CRR): 2 / 3 ≈ 0.66666667
    """
    evaluator = RetrievalEvaluator(default_top_k=5, default_acceptance_threshold=0.60)

    cases_with_results = [
        # Case A: Falsely accepted (score 0.75 >= 0.60)
        (
            EvaluationCase(
                dataset_id="test-ds",
                case_id="unans-A",
                split="dev",
                question="Unans A",
                question_type="unanswerable_in_domain",
                answerable=False,
                target_document_filename="doc.md",
                expected_evidence_snippets=[],
                reference_claims=[],
                acceptable_abstention_reasons=["no_evidence_found"],
            ),
            [_make_chunk("doc.md", "Irrelevant top candidate", 0.75)],
        ),
        # Case B: Correctly rejected (score 0.50 < 0.60)
        (
            EvaluationCase(
                dataset_id="test-ds",
                case_id="unans-B",
                split="dev",
                question="Unans B",
                question_type="unanswerable_in_domain",
                answerable=False,
                target_document_filename="doc.md",
                expected_evidence_snippets=[],
                reference_claims=[],
                acceptable_abstention_reasons=["no_evidence_found"],
            ),
            [_make_chunk("doc.md", "Weak candidate", 0.50)],
        ),
        # Case C: Correctly rejected (empty results)
        (
            EvaluationCase(
                dataset_id="test-ds",
                case_id="unans-C",
                split="dev",
                question="Unans C",
                question_type="unanswerable_in_domain",
                answerable=False,
                target_document_filename="doc.md",
                expected_evidence_snippets=[],
                reference_claims=[],
                acceptable_abstention_reasons=["no_evidence_found"],
            ),
            [],
        ),
    ]

    report = evaluator.evaluate_cases(cases_with_results, acceptance_threshold=0.60)
    metrics = report.metrics

    assert metrics.total_cases == 3
    assert metrics.answerable_count == 0
    assert metrics.unanswerable_count == 3
    assert metrics.acceptance_threshold == 0.60

    assert metrics.false_accept_count == 1
    assert metrics.correct_reject_count == 2
    assert metrics.false_accept_rate == pytest.approx(1.0 / 3.0)
    assert metrics.correct_reject_rate == pytest.approx(2.0 / 3.0)

    # Per-case checks
    cA = report.case_evaluations[0]
    assert cA.is_false_accept is True
    assert cA.is_correct_reject is False

    cB = report.case_evaluations[1]
    assert cB.is_false_accept is False
    assert cB.is_correct_reject is True

    cC = report.case_evaluations[2]
    assert cC.is_false_accept is False
    assert cC.is_correct_reject is True
    assert cC.max_candidate_score is None


# ==============================================================================
# 4. Empty Groups & Edge Cases
# ==============================================================================


def test_empty_answerable_group_handling() -> None:
    """When a dataset has zero answerable queries, answerable metrics report 0.0 with denominator 0."""
    evaluator = RetrievalEvaluator()
    unans_case = EvaluationCase(
        dataset_id="test-ds",
        case_id="unans-only",
        split="dev",
        question="Unans query",
        question_type="unanswerable_in_domain",
        answerable=False,
        target_document_filename="doc.md",
        expected_evidence_snippets=[],
        reference_claims=[],
        acceptable_abstention_reasons=["no_evidence_found"],
    )

    report = evaluator.evaluate_cases([(unans_case, [])])
    metrics = report.metrics

    assert metrics.answerable_count == 0
    assert metrics.recall_at_1 == 0.0
    assert metrics.recall_at_3 == 0.0
    assert metrics.recall_at_5 == 0.0
    assert metrics.mrr == 0.0
    assert metrics.avg_first_relevant_score is None
    assert metrics.avg_answerable_top1_score is None

    assert metrics.unanswerable_count == 1
    assert metrics.correct_reject_rate == 1.0


def test_empty_unanswerable_group_handling() -> None:
    """When a dataset has zero unanswerable queries, rejection metrics report 0.0 with denominator 0."""
    evaluator = RetrievalEvaluator()
    ans_case = EvaluationCase(
        dataset_id="test-ds",
        case_id="ans-only",
        split="dev",
        question="Ans query",
        question_type="direct_factual",
        answerable=True,
        target_document_filename="doc.md",
        expected_evidence_snippets=["Evidence text"],
        reference_claims=["Claim text"],
    )
    chunk = _make_chunk("doc.md", "Evidence text is here.", 0.85)

    report = evaluator.evaluate_cases([(ans_case, [chunk])])
    metrics = report.metrics

    assert metrics.unanswerable_count == 0
    assert metrics.false_accept_count == 0
    assert metrics.correct_reject_count == 0
    assert metrics.false_accept_rate == 0.0
    assert metrics.correct_reject_rate == 0.0
    assert metrics.avg_unanswerable_max_score is None

    assert metrics.answerable_count == 1
    assert metrics.recall_at_1 == 1.0


def test_completely_empty_evaluation_input() -> None:
    """An empty evaluation list returns zero denominators and zero metrics without crashing."""
    evaluator = RetrievalEvaluator()
    report = evaluator.evaluate_cases([])
    metrics = report.metrics

    assert metrics.total_cases == 0
    assert metrics.answerable_count == 0
    assert metrics.unanswerable_count == 0
    assert metrics.recall_at_1 == 0.0
    assert metrics.mrr == 0.0
    assert metrics.false_accept_rate == 0.0


# ==============================================================================
# 5. Bounds & Input Validation
# ==============================================================================


def test_evaluator_rejects_invalid_bounds() -> None:
    """RetrievalEvaluator rejects out-of-bounds or non-finite parameters."""
    with pytest.raises(ValueError, match="top_k"):
        RetrievalEvaluator(default_top_k=0)

    with pytest.raises(ValueError, match="top_k"):
        RetrievalEvaluator(default_top_k=51)

    with pytest.raises(ValueError, match="score_threshold"):
        RetrievalEvaluator(default_acceptance_threshold=1.5)

    with pytest.raises(ValueError, match="score_threshold"):
        RetrievalEvaluator(default_acceptance_threshold=-1.5)

    evaluator = RetrievalEvaluator()
    with pytest.raises(ValueError, match="top_k"):
        evaluator.evaluate_cases([], top_k=0)

    with pytest.raises(ValueError, match="score_threshold"):
        evaluator.evaluate_cases([], acceptance_threshold=2.0)


# ==============================================================================
# 6. Threshold Trade-Off Sweep
# ==============================================================================


def test_compute_threshold_sweep_monotonic_properties() -> None:
    """Verify threshold trade-off sweep: FAR is non-increasing and FRR is non-decreasing with tau."""
    evaluator = RetrievalEvaluator()

    # Case 1: Answerable with relevant score = 0.68
    c_ans = EvaluationCase(
        dataset_id="test-ds",
        case_id="case-ans",
        split="dev",
        question="Q Ans",
        question_type="direct_factual",
        answerable=True,
        target_document_filename="doc.md",
        expected_evidence_snippets=["Target Evidence"],
        reference_claims=["Claim"],
    )
    chunks_ans = [_make_chunk("doc.md", "Target Evidence text", 0.68)]

    # Case 2: Unanswerable with max candidate score = 0.62
    c_unans = EvaluationCase(
        dataset_id="test-ds",
        case_id="case-unans",
        split="dev",
        question="Q Unans",
        question_type="unanswerable_in_domain",
        answerable=False,
        target_document_filename="doc.md",
        expected_evidence_snippets=[],
        reference_claims=[],
        acceptable_abstention_reasons=["no_evidence_found"],
    )
    chunks_unans = [_make_chunk("doc.md", "Distractor text", 0.62)]

    cases_with_results = [(c_ans, chunks_ans), (c_unans, chunks_unans)]

    sweep = evaluator.compute_threshold_sweep(
        cases_with_candidates=cases_with_results,
        thresholds=[0.55, 0.60, 0.65, 0.70],
    )

    assert len(sweep) == 4

    # FAR at tau=0.55: 0.62 >= 0.55 -> FAR = 1.0
    # FAR at tau=0.60: 0.62 >= 0.60 -> FAR = 1.0
    # FAR at tau=0.65: 0.62 < 0.65  -> FAR = 0.0
    # FAR at tau=0.70: 0.62 < 0.70  -> FAR = 0.0
    assert sweep[0].false_accept_rate == 1.0
    assert sweep[1].false_accept_rate == 1.0
    assert sweep[2].false_accept_rate == 0.0
    assert sweep[3].false_accept_rate == 0.0

    # FRR at tau=0.55: 0.68 >= 0.55 -> FRR = 0.0
    # FRR at tau=0.60: 0.68 >= 0.60 -> FRR = 0.0
    # FRR at tau=0.65: 0.68 >= 0.65 -> FRR = 0.0
    # FRR at tau=0.70: 0.68 < 0.70  -> FRR = 1.0 (rejected!)
    assert sweep[0].false_rejection_rate == 0.0
    assert sweep[1].false_rejection_rate == 0.0
    assert sweep[2].false_rejection_rate == 0.0
    assert sweep[3].false_rejection_rate == 1.0


# ==============================================================================
# 7. Dataset V1 Integration & Legacy Compatibility
# ==============================================================================


def test_evaluator_integration_with_dataset_v1() -> None:
    """Verify evaluator seamlessly consumes EvaluationDataset v1 cases."""
    assert DATASET_V1_PATH.is_file()
    dataset = EvaluationDataset.model_validate(json.loads(DATASET_V1_PATH.read_text(encoding="utf-8")))

    dev_cases = [c for c in dataset.cases if c.split == "dev"]
    assert len(dev_cases) == 26

    # Create synthetic candidate chunks for each dev case:
    # - If answerable: rank 1 contains expected evidence
    # - If unanswerable: rank 1 contains low-scoring distractor (0.40)
    mocked_pairs = []
    for case in dev_cases:
        if case.answerable:
            chunk = _make_chunk(
                doc_filename=case.target_document_filename,
                text=case.expected_evidence_snippets[0],
                score=0.88,
            )
            mocked_pairs.append((case, [chunk]))
        else:
            chunk = _make_chunk(
                doc_filename=case.target_document_filename,
                text="Out of domain noise.",
                score=0.40,
            )
            mocked_pairs.append((case, [chunk]))

    evaluator = RetrievalEvaluator(default_acceptance_threshold=0.55)
    report = evaluator.evaluate_cases(
        mocked_pairs,
        dataset_id=dataset.dataset_id,
        split="dev",
    )

    assert report.dataset_id == "evidencedesk-rag-v1.0.0"
    assert report.split == "dev"
    assert report.metrics.total_cases == 26
    assert report.metrics.answerable_count == 22
    assert report.metrics.unanswerable_count == 4

    # Perfect retrieval simulated:
    assert report.metrics.recall_at_1 == 1.0
    assert report.metrics.mrr == 1.0

    # All unanswerable scores were 0.40 < 0.55:
    assert report.metrics.false_accept_rate == 0.0
    assert report.metrics.correct_reject_rate == 1.0


def test_evaluator_legacy_benchmark_compatibility() -> None:
    """Verify evaluator seamlessly consumes legacy baseline_retrieval_cases.json dictionaries."""
    assert LEGACY_CASES_PATH.is_file()
    legacy_cases = json.loads(LEGACY_CASES_PATH.read_text(encoding="utf-8"))
    assert len(legacy_cases) == 38

    # Simulate realistic retrieval responses
    mocked_pairs = []
    for c in legacy_cases:
        if c.get("answerable", True):
            chunk = _make_chunk(
                doc_filename=c["target_document_filename"],
                text=f"Prefix text. {c['evidence_text']} Suffix text.",
                score=0.82,
            )
            mocked_pairs.append((c, [chunk]))
        else:
            chunk = _make_chunk(
                doc_filename=c["target_document_filename"],
                text="Completely irrelevant passage.",
                score=0.45,
            )
            mocked_pairs.append((c, [chunk]))

    evaluator = RetrievalEvaluator(default_acceptance_threshold=0.55)
    report = evaluator.evaluate_cases(mocked_pairs)

    assert report.metrics.total_cases == 38
    assert report.metrics.answerable_count == 32
    assert report.metrics.unanswerable_count == 6

    assert report.metrics.recall_at_1 == 1.0
    assert report.metrics.mrr == 1.0
    assert report.metrics.false_accept_rate == 0.0
    assert report.metrics.correct_reject_rate == 1.0
