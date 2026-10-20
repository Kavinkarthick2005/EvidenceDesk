"""Unit tests for reusable GenerationEvaluator (Milestone 9.0 Task 9.3).

Verifies:
1. Exact mathematical calculation of Abstention Precision, Abstention Recall,
   False Abstention Rate, and Answer Coverage.
2. Robust handling of empty groups and explicit denominators (no ZeroDivisionError).
3. Distinction between citation syntax validity and reference existence in context.
4. Reference claim heuristic support, partial coverage, and excluded cases.
5. Negative constraint compliance, forbidden claim violation detection, and missing ground truth.
6. Separation of retrieval failure from generator failure in End-to-End QA mode.
7. Deterministic offline evaluation with mock generator and error category capture.
8. Native compatibility with Dataset V1 (evaluation_dataset_v1.json).
"""

from collections.abc import Sequence
import json
from pathlib import Path
import pytest

from app.evaluation.generation import (
    ClaimEvaluation,
    GenerationCaseEvaluation,
    GenerationEvaluationMetrics,
    GenerationEvaluationReport,
    GenerationEvaluator,
)
from app.generation.citation import CitationValidator
from app.generation.interface import EvidenceItem, GenerationProvider, GenerationResult
from app.schemas.evaluation import EvaluationCase, EvaluationDataset

DATASET_V1_PATH = Path(__file__).parent / "evaluation" / "evaluation_dataset_v1.json"


# ==============================================================================
# Deterministic Test Helpers and Fixtures
# ==============================================================================


def _make_eval_case(
    case_id: str,
    question: str,
    answerable: bool,
    expected_snippets: list[str] | None = None,
    reference_claims: list[str] | None = None,
    forbidden_claims: list[str] | None = None,
    acceptable_abstention_reasons: list[str] | None = None,
    split: str = "dev",
    dataset_id: str = "test-dataset",
) -> EvaluationCase:
    """Helper to build a valid EvaluationCase for test fixtures."""
    snippets = (
        expected_snippets
        if expected_snippets is not None
        else (["Snippet about test topic"] if answerable else [])
    )
    claims = (
        reference_claims
        if reference_claims is not None
        else (["The test topic has a property."] if answerable else [])
    )
    constraints = forbidden_claims if forbidden_claims is not None else []
    abstention_reasons = (
        acceptable_abstention_reasons
        if acceptable_abstention_reasons is not None
        else ([] if answerable else ["insufficient_evidence"])
    )

    return EvaluationCase(
        dataset_id=dataset_id,
        case_id=case_id,
        split=split,
        question=question,
        question_type="factual" if answerable else "unanswerable",
        answerable=answerable,
        target_document_filename="test_doc.md",
        target_section="Section A",
        expected_evidence_snippets=snippets,
        reference_claims=claims,
        forbidden_claims=constraints,
        acceptable_abstention_reasons=abstention_reasons,
        acceptance_criteria=["Must answer correctly with citations"] if answerable else ["Must abstain"],
    )


class DeterministicMockGenerator:
    """Deterministic GenerationProvider for unit testing."""

    def __init__(
        self,
        model_id: str = "mock-eval-model",
        canned_answers: dict[str, GenerationResult] | None = None,
        should_raise: Exception | None = None,
    ) -> None:
        self._model_id = model_id
        self.canned_answers = canned_answers or {}
        self.should_raise = should_raise

    @property
    def model_id(self) -> str:
        return self._model_id

    def generate_answer(
        self,
        query: str,
        evidence: Sequence[EvidenceItem],
    ) -> GenerationResult:
        if self.should_raise:
            raise self.should_raise

        if query in self.canned_answers:
            return self.canned_answers[query]

        if not evidence:
            return GenerationResult(
                answer="I cannot answer this question because no relevant evidence was found.",
                cited_evidence_ids=[],
                is_abstained=True,
                abstention_reason="insufficient_evidence",
                model_id=self._model_id,
            )

        # Default answer referencing first evidence item
        return GenerationResult(
            answer=f"Based on the evidence, {evidence[0].text} [1]",
            cited_evidence_ids=["1"],
            is_abstained=False,
            abstention_reason=None,
            model_id=self._model_id,
        )


class MockQAService:
    """Deterministic QuestionAnsweringService mock for E2E decoupling tests."""

    def __init__(self, canned_response_fn) -> None:
        self.canned_response_fn = canned_response_fn
        self.generation_provider = DeterministicMockGenerator()

    def answer_question(self, query: str, top_k: int = 5, score_threshold: float | None = None):
        return self.canned_response_fn(query, top_k, score_threshold)


# ==============================================================================
# 1. Abstention & Answerability Metrics Calculation
# ==============================================================================


def test_abstention_metrics_calculation() -> None:
    """Verify exact mathematical calculation of TP, FP, TN, FN and rate metrics."""
    evaluator = GenerationEvaluator()

    # Construct 6 cases:
    # 2 Unanswerable cases:
    #   - Case 1: unanswerable, model abstained -> TP
    #   - Case 2: unanswerable, model gave answer -> FN
    # 4 Answerable cases:
    #   - Case 3: answerable, model falsely abstained -> FP
    #   - Case 4: answerable, model answered -> TN
    #   - Case 5: answerable, model answered -> TN
    #   - Case 6: answerable, model answered -> TN
    case1 = _make_eval_case("c1", "Unans Q1", answerable=False)
    case2 = _make_eval_case("c2", "Unans Q2", answerable=False)
    case3 = _make_eval_case("c3", "Ans Q3", answerable=True)
    case4 = _make_eval_case("c4", "Ans Q4", answerable=True)
    case5 = _make_eval_case("c5", "Ans Q5", answerable=True)
    case6 = _make_eval_case("c6", "Ans Q6", answerable=True)

    evals = [
        # TP: unanswerable + abstained
        evaluator.evaluate_response(case1, "I abstain [insufficient]", is_abstained=True),
        # FN: unanswerable + answered
        evaluator.evaluate_response(case2, "Hallucinated answer [1]", is_abstained=False),
        # FP: answerable + falsely abstained
        evaluator.evaluate_response(case3, "I abstain falsely", is_abstained=True),
        # TN: answerable + answered
        evaluator.evaluate_response(case4, "Good answer [1]", is_abstained=False),
        # TN: answerable + answered
        evaluator.evaluate_response(case5, "Good answer 2 [1]", is_abstained=False),
        # TN: answerable + answered
        evaluator.evaluate_response(case6, "Good answer 3 [1]", is_abstained=False),
    ]

    metrics = evaluator.aggregate_metrics(evals)

    # Denominators and counts
    assert metrics.total_cases == 6
    assert metrics.answerable_count == 4
    assert metrics.unanswerable_count == 2
    assert metrics.true_positive_abstentions == 1
    assert metrics.false_negative_answers == 1
    assert metrics.false_positive_abstentions == 1
    assert metrics.true_negative_answers == 3

    # Rates:
    # Abstention precision: TP / (TP + FP) = 1 / 2 = 0.50
    assert metrics.abstention_precision == pytest.approx(0.50)
    # Abstention recall: TP / (TP + FN) = 1 / 2 = 0.50
    assert metrics.abstention_recall == pytest.approx(0.50)
    # False abstention rate: FP / answerable_count = 1 / 4 = 0.25
    assert metrics.false_abstention_rate == pytest.approx(0.25)
    # Answer coverage: TN / answerable_count = 3 / 4 = 0.75
    assert metrics.answer_coverage == pytest.approx(0.75)

    # Invariants
    assert metrics.false_abstention_rate + metrics.answer_coverage == pytest.approx(1.0)
    assert metrics.true_positive_abstentions + metrics.false_negative_answers == metrics.unanswerable_count
    assert metrics.false_positive_abstentions + metrics.true_negative_answers == metrics.answerable_count


# ==============================================================================
# 2. Empty Groups and Explicit Denominators
# ==============================================================================


def test_empty_groups_handling() -> None:
    """Verify zero division is safely handled across all empty condition groups."""
    evaluator = GenerationEvaluator()

    # Scenario A: Completely empty case list
    empty_metrics = evaluator.aggregate_metrics([])
    assert empty_metrics.total_cases == 0
    assert empty_metrics.answerable_count == 0
    assert empty_metrics.unanswerable_count == 0
    assert empty_metrics.abstention_precision == 0.0
    assert empty_metrics.abstention_recall == 0.0
    assert empty_metrics.false_abstention_rate == 0.0
    assert empty_metrics.answer_coverage == 0.0
    assert empty_metrics.citation_validity_rate == 0.0
    assert empty_metrics.claim_support_rate == 0.0
    assert empty_metrics.constraint_compliance_rate == 0.0
    assert empty_metrics.mean_latency_seconds is None
    assert empty_metrics.p50_latency_seconds is None
    assert empty_metrics.p95_latency_seconds is None

    # Scenario B: Only answerable cases (0 unanswerable)
    ans_case = _make_eval_case("c1", "Q", answerable=True)
    ans_eval = evaluator.evaluate_response(ans_case, "Answer [1]", is_abstained=False)
    metrics_ans = evaluator.aggregate_metrics([ans_eval])
    assert metrics_ans.unanswerable_count == 0
    assert metrics_ans.abstention_recall == 0.0
    assert metrics_ans.true_positive_abstentions == 0
    assert metrics_ans.false_negative_answers == 0
    assert metrics_ans.answer_coverage == 1.0

    # Scenario C: Only unanswerable cases (0 answerable)
    unans_case = _make_eval_case("c2", "Q", answerable=False)
    unans_eval = evaluator.evaluate_response(unans_case, "I abstain", is_abstained=True)
    metrics_unans = evaluator.aggregate_metrics([unans_eval])
    assert metrics_unans.answerable_count == 0
    assert metrics_unans.false_abstention_rate == 0.0
    assert metrics_unans.answer_coverage == 0.0
    assert metrics_unans.abstention_recall == 1.0
    assert metrics_unans.abstention_precision == 1.0

    # Scenario D: Zero abstentions (all cases answered)
    ans_case_2 = _make_eval_case("c3", "Q2", answerable=True)
    eval_2 = evaluator.evaluate_response(ans_case_2, "Answer [1]", is_abstained=False)
    metrics_no_abstain = evaluator.aggregate_metrics([ans_eval, eval_2])
    assert metrics_no_abstain.true_positive_abstentions == 0
    assert metrics_no_abstain.false_positive_abstentions == 0
    assert metrics_no_abstain.abstention_precision == 0.0

    # Scenario E: Zero substantive answers (all cases abstained)
    eval_abstained = evaluator.evaluate_response(ans_case, "I abstain", is_abstained=True)
    metrics_all_abstained = evaluator.aggregate_metrics([eval_abstained, unans_eval])
    assert metrics_all_abstained.substantive_answers_count == 0
    assert metrics_all_abstained.citation_validity_rate == 0.0
    assert metrics_all_abstained.lexical_grounding_pass_rate == 0.0


# ==============================================================================
# 3. Citation Syntax vs. Reference Existence
# ==============================================================================


def test_citation_syntax_versus_reference_integrity() -> None:
    """Verify distinct tracking of malformed syntax vs non-existent reference IDs."""
    evaluator = GenerationEvaluator()
    case = _make_eval_case("c1", "What is X?", answerable=True)

    # 1. Syntactically valid citation referencing an unknown chunk ID
    ev_unknown = evaluator.evaluate_response(
        case=case,
        answer_text="The retention policy is 7 years [99].",
        is_abstained=False,
        unknown_citations=["99"],
        malformed_citations=[],
    )
    assert ev_unknown.has_citations is True
    assert ev_unknown.is_valid_citations is False
    assert ev_unknown.unknown_citations == ["99"]
    assert len(ev_unknown.malformed_citations) == 0

    # 2. Malformed citation marker syntax ([ref], [0], [1,])
    ev_malformed = evaluator.evaluate_response(
        case=case,
        answer_text="The retention policy is 7 years [ref] and [0].",
        is_abstained=False,
        unknown_citations=[],
        malformed_citations=["[ref]", "[0]"],
    )
    assert ev_malformed.is_valid_citations is False
    assert "[ref]" in ev_malformed.malformed_citations
    assert len(ev_malformed.unknown_citations) == 0

    # 3. Completely valid citation
    ev_valid = evaluator.evaluate_response(
        case=case,
        answer_text="The retention policy is 7 years [1].",
        is_abstained=False,
        unknown_citations=[],
        malformed_citations=[],
    )
    assert ev_valid.has_citations is True
    assert ev_valid.is_valid_citations is True
    assert len(ev_valid.unknown_citations) == 0
    assert len(ev_valid.malformed_citations) == 0

    # Aggregate metric verification
    metrics = evaluator.aggregate_metrics([ev_unknown, ev_malformed, ev_valid])
    assert metrics.substantive_answers_count == 3
    assert metrics.citation_syntax_valid_count == 2  # ev_unknown and ev_valid have valid syntax
    assert metrics.citation_reference_valid_count == 2  # ev_malformed and ev_valid have no unknown refs
    assert metrics.citation_validity_rate == pytest.approx(1 / 3)


# ==============================================================================
# 4. Reference Claim Support Heuristics & Exclusions
# ==============================================================================


def test_reference_claim_support_heuristics() -> None:
    """Verify heuristic claim matching, partial support, and ground-truth exclusions."""
    evaluator = GenerationEvaluator()

    # 1. Static claim evaluation method
    claim_1 = "Audit log retention period is 7 years."
    ev_supported = evaluator.evaluate_reference_claim(
        claim_text=claim_1,
        answer_text="The system mandates that audit log retention is strictly set to 7 years [1].",
    )
    assert ev_supported.heuristic_supported is True
    assert "audit" in ev_supported.matching_tokens
    assert "retention" in ev_supported.matching_tokens
    assert "years" in ev_supported.matching_tokens
    assert "7" in ev_supported.matching_tokens

    ev_unsupported = evaluator.evaluate_reference_claim(
        claim_text=claim_1,
        answer_text="Quantum encryption algorithm requires 256-bit prime modulus [1].",
    )
    assert ev_unsupported.heuristic_supported is False
    assert len(ev_unsupported.matching_tokens) == 0

    # Empty claim text handling
    ev_empty = evaluator.evaluate_reference_claim("", "Some answer text.")
    assert ev_empty.heuristic_supported is False
    assert ev_empty.total_claim_tokens == 0

    # 2. Multi-claim case evaluation
    case_multi = _make_eval_case(
        "c_claims",
        "Explain log retention and encryption",
        answerable=True,
        reference_claims=[
            "Audit logs are retained for 7 years.",
            "Encryption uses AES-256-GCM.",
        ],
    )
    # Answer that satisfies only the first claim
    ev_partial = evaluator.evaluate_response(
        case=case_multi,
        answer_text="Audit logs are retained for 7 years [1].",
        is_abstained=False,
    )
    assert len(ev_partial.claim_evaluations) == 2
    assert ev_partial.claim_evaluations[0].heuristic_supported is True
    assert ev_partial.claim_evaluations[1].heuristic_supported is False
    assert ev_partial.all_claims_supported is False

    # 3. Answerable case lacking reference claims (excluded from claims denominator)
    ev_no_claims = GenerationCaseEvaluation(
        case_id="c_noclaims",
        query="What is Y?",
        answerable=True,
        mode="direct_generation",
        answer_text="Answer Y [1]",
        is_abstained=False,
        abstention_verdict="true_negative",
        has_citations=True,
        is_valid_citations=True,
        claim_evaluations=[],
        all_claims_supported=None,
    )
    assert ev_no_claims.all_claims_supported is None
    assert len(ev_no_claims.claim_evaluations) == 0

    metrics = evaluator.aggregate_metrics([ev_partial, ev_no_claims])
    assert metrics.answerable_count == 2
    assert metrics.cases_with_reference_claims == 1
    assert metrics.excluded_cases_no_claims == 1
    assert metrics.total_reference_claims == 2
    assert metrics.supported_reference_claims == 1
    assert metrics.claim_support_rate == pytest.approx(0.50)
    assert metrics.full_claim_coverage_rate == pytest.approx(0.0)


# ==============================================================================
# 5. Negative Constraints & Missing Ground Truth
# ==============================================================================


def test_negative_constraints_and_missing_ground_truth() -> None:
    """Verify detection of forbidden claims and proper exclusion when constraints are absent."""
    evaluator = GenerationEvaluator()

    # Case 1: Has forbidden claims, answer violates constraint
    case_violated = _make_eval_case(
        "c_forbid1",
        "What compliance certs exist?",
        answerable=True,
        forbidden_claims=["SOC 2 Type II", "FedRAMP High"],
    )
    ev_violated = evaluator.evaluate_response(
        case=case_violated,
        answer_text="The platform maintains SOC 2 Type II certification [1].",
        is_abstained=False,
    )
    assert ev_violated.has_forbidden_violation is True
    assert "SOC 2 Type II" in ev_violated.forbidden_violations_detected

    # Case 2: Has forbidden claims, answer is compliant
    case_compliant = _make_eval_case(
        "c_forbid2",
        "What compliance certs exist?",
        answerable=True,
        forbidden_claims=["SOC 2 Type II", "FedRAMP High"],
    )
    ev_compliant = evaluator.evaluate_response(
        case=case_compliant,
        answer_text="The platform is ISO 27001 certified [1].",
        is_abstained=False,
    )
    assert ev_compliant.has_forbidden_violation is False
    assert len(ev_compliant.forbidden_violations_detected) == 0

    # Case 3: No forbidden constraints specified (unconstrained)
    case_unconstrained = _make_eval_case(
        "c_unconstrained",
        "Normal question?",
        answerable=True,
        forbidden_claims=[],
    )
    ev_unconstrained = evaluator.evaluate_response(
        case=case_unconstrained,
        answer_text="Normal answer [1].",
        is_abstained=False,
    )
    assert ev_unconstrained.has_forbidden_violation is False
    assert len(ev_unconstrained.forbidden_claims_checked) == 0

    metrics = evaluator.aggregate_metrics([ev_violated, ev_compliant, ev_unconstrained])
    assert metrics.total_cases == 3
    assert metrics.cases_with_forbidden_constraints == 2
    assert metrics.excluded_cases_no_constraints == 1
    assert metrics.forbidden_violations_count == 1
    # Compliance rate: 1 compliant out of 2 constrained = 0.50
    assert metrics.constraint_compliance_rate == pytest.approx(0.50)


# ==============================================================================
# 6. Separation of Retrieval Failure from Generator Failure
# ==============================================================================


def test_separation_of_retrieval_failure_from_generation_failure() -> None:
    """Verify that retrieval misses in E2E mode are isolated and not blamed on generator."""
    evaluator = GenerationEvaluator()

    class FakeQAResponse:
        def __init__(self, answer, is_abstained, abstention_reason, retrieved_count):
            self.answer = answer
            self.is_abstained = is_abstained
            self.abstention_reason = abstention_reason
            self.retrieved_count = retrieved_count
            self.unknown_citations = []
            self.malformed_citations = []
            self.unsupported_citations = []
            self.model_id = "test-e2e-model"

    # Case A: Retrieval failed (returned 0 chunks) for an answerable query
    case_ans_1 = _make_eval_case("c_miss", "Answerable query where retrieval missed", answerable=True)

    # Case B: Retrieval succeeded (returned 3 chunks) and model answered
    case_ans_2 = _make_eval_case("c_hit", "Answerable query where retrieval hit", answerable=True)

    def fake_qa_logic(query, top_k, score_threshold):
        if "missed" in query:
            return FakeQAResponse(
                answer="I abstain because no relevant evidence was found.",
                is_abstained=True,
                abstention_reason="insufficient_evidence",
                retrieved_count=0,
            )
        return FakeQAResponse(
            answer="Answer backed by evidence [1].",
            is_abstained=False,
            abstention_reason=None,
            retrieved_count=3,
        )

    qa_service = MockQAService(canned_response_fn=fake_qa_logic)
    report = evaluator.evaluate_qa_service(qa_service, [case_ans_1, case_ans_2])

    assert len(report.case_evaluations) == 2
    ev_miss = report.case_evaluations[0]
    ev_hit = report.case_evaluations[1]

    # Retrieval failure correctly flagged on the missed case
    assert ev_miss.retrieval_failed is True
    assert ev_miss.abstention_verdict == "false_positive"  # Abstained on an answerable query
    assert ev_hit.retrieval_failed is False
    assert ev_hit.abstention_verdict == "true_negative"

    # Overall metrics isolate retrieval failures
    assert report.metrics.retrieval_failures_count == 1
    assert report.metrics.answerable_count == 2


# ==============================================================================
# 7. Deterministic Direct Generation & Latency Tracking
# ==============================================================================


def test_deterministic_mock_direct_generation() -> None:
    """Verify direct generation evaluation using deterministic mock generator."""
    evaluator = GenerationEvaluator()

    case_ans = _make_eval_case(
        "c_ans",
        "What is the token limit?",
        answerable=True,
        expected_snippets=["The maximum token limit is 4096 tokens."],
        reference_claims=["The token limit is 4096 tokens."],
    )
    case_unans = _make_eval_case(
        "c_unans",
        "What is the company valuation?",
        answerable=False,
    )

    mock_provider = DeterministicMockGenerator(
        model_id="deterministic-mock-v1",
        canned_answers={
            "What is the token limit?": GenerationResult(
                answer="The maximum token limit is 4096 tokens [1].",
                cited_evidence_ids=["1"],
                is_abstained=False,
                abstention_reason=None,
                model_id="deterministic-mock-v1",
            ),
            "What is the company valuation?": GenerationResult(
                answer="I cannot answer this question as valuation is not stated.",
                cited_evidence_ids=[],
                is_abstained=True,
                abstention_reason="insufficient_evidence",
                model_id="deterministic-mock-v1",
            ),
        },
    )

    report = evaluator.evaluate_direct_generation(
        generation_provider=mock_provider,
        cases=[case_ans, case_unans],
        dataset_id="mock-dataset-v1",
        split="dev",
    )

    assert report.dataset_id == "mock-dataset-v1"
    assert report.split == "dev"
    assert report.mode == "direct_generation"
    assert report.model_id == "deterministic-mock-v1"
    assert report.metrics.total_cases == 2
    assert report.metrics.true_positive_abstentions == 1
    assert report.metrics.true_negative_answers == 1
    assert report.metrics.abstention_precision == 1.0
    assert report.metrics.abstention_recall == 1.0
    assert report.metrics.citation_validity_rate == 1.0
    assert report.metrics.claim_support_rate == 1.0
    assert report.metrics.full_claim_coverage_rate == 1.0
    assert report.metrics.latency_sample_count == 2
    assert report.metrics.mean_latency_seconds is not None
    assert report.metrics.p50_latency_seconds is not None
    assert report.metrics.p95_latency_seconds is not None


def test_direct_generation_exception_handling() -> None:
    """Verify that runtime errors during generation are safely captured without crashing."""
    evaluator = GenerationEvaluator()
    case = _make_eval_case("c_err", "Crash question", answerable=True)

    failing_provider = DeterministicMockGenerator(
        should_raise=RuntimeError("Simulated generation backend failure"),
    )

    report = evaluator.evaluate_direct_generation(failing_provider, [case])
    assert report.metrics.total_cases == 1
    ev = report.case_evaluations[0]
    assert ev.is_abstained is True
    assert ev.abstention_reason == "generation_error"
    assert ev.error_category == "RuntimeError"
    assert "Simulated generation backend failure" in ev.answer_text


# ==============================================================================
# 8. Sentence Grounding in Direct Generation
# ==============================================================================


def test_direct_generation_sentence_grounding() -> None:
    """Verify lexical grounding check detects unsupported citations in direct generation."""
    evaluator = GenerationEvaluator()
    case = _make_eval_case(
        "c_ground",
        "What is the policy?",
        answerable=True,
        expected_snippets=["The retention policy requires keeping data for 7 years."],
    )

    # Provider outputs a sentence citing [1] that has zero lexical overlap with snippet [1]
    hallucinating_provider = DeterministicMockGenerator(
        canned_answers={
            "What is the policy?": GenerationResult(
                answer="Quantum entanglement stabilizes galactic wormholes [1].",
                cited_evidence_ids=["1"],
                is_abstained=False,
                abstention_reason=None,
                model_id="hallucinating-mock",
            )
        }
    )

    report = evaluator.evaluate_direct_generation(hallucinating_provider, [case])
    ev = report.case_evaluations[0]
    assert "1" in ev.unsupported_citations
    assert report.metrics.lexical_grounding_pass_rate == 0.0


# ==============================================================================
# 9. Dataset V1 Compatibility
# ==============================================================================


def test_dataset_v1_compatibility_direct_generation() -> None:
    """Verify Dataset V1 JSON can be loaded and evaluated across dev and test splits."""
    assert DATASET_V1_PATH.exists(), f"Dataset V1 file missing at {DATASET_V1_PATH}"

    with open(DATASET_V1_PATH, encoding="utf-8") as f:
        data = json.load(f)

    dataset = EvaluationDataset.model_validate(data)
    assert dataset.dataset_id == "evidencedesk-rag-v1.0.0"
    assert len(dataset.cases) == 38

    evaluator = GenerationEvaluator()
    mock_provider = DeterministicMockGenerator()

    # 1. Dev Split
    dev_cases = [c for c in dataset.cases if c.split == "dev"]
    assert len(dev_cases) == 26
    dev_report = evaluator.evaluate_direct_generation(
        generation_provider=mock_provider,
        cases=dev_cases,
        dataset_id=dataset.dataset_id,
        split="dev",
    )
    assert dev_report.metrics.total_cases == 26
    assert dev_report.metrics.answerable_count + dev_report.metrics.unanswerable_count == 26

    # 2. Test Split (Held-out)
    test_cases = [c for c in dataset.cases if c.split == "test"]
    assert len(test_cases) == 12
    test_report = evaluator.evaluate_direct_generation(
        generation_provider=mock_provider,
        cases=test_cases,
        dataset_id=dataset.dataset_id,
        split="test",
    )
    assert test_report.metrics.total_cases == 12
    assert test_report.metrics.answerable_count + test_report.metrics.unanswerable_count == 12

    # 3. Serialization to dict/JSON without error
    report_dict = dev_report.model_dump()
    assert report_dict["metrics"]["total_cases"] == 26
    assert "claim_support_rate" in report_dict["metrics"]
    assert "abstention_precision" in report_dict["metrics"]
    assert "abstention_recall" in report_dict["metrics"]
