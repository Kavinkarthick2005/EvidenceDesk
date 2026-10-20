"""Reusable generation and end-to-end QA evaluator for EvidenceDesk.

Measures:
1. Answerability & Abstention: Abstention Precision, Abstention Recall, False Abstention Rate, Answer Coverage.
2. Citation Integrity: Syntactic marker validity and reference existence in context (without claiming proof of entailment).
3. Claim Support & Correctness: Reference claim containment heuristics vs. whole-answer correctness.
4. Negative Constraints: Compliance with forbidden claims and rejection of unsupported premises.
5. Operational Latency & Decoupling: Direct generation vs. End-to-end QA, isolating retrieval misses from generator failures.
"""

from collections.abc import Sequence
import logging
import math
import re
import statistics
import time
from typing import Any

from pydantic import BaseModel, ConfigDict, Field

from app.generation.citation import (
    CitationValidator,
    check_sentence_grounding,
    extract_grounding_tokens,
)
from app.generation.interface import EvidenceItem, GenerationProvider, GenerationResult
from app.schemas.evaluation import EvaluationCase
from app.schemas.qa import QAResponse
from app.schemas.retrieval import RetrievedChunk

logger = logging.getLogger(__name__)


# ==============================================================================
# Typed Evaluation Output Schemas
# ==============================================================================


class ClaimEvaluation(BaseModel):
    """Evaluation result for an atomic reference claim."""

    model_config = ConfigDict(frozen=True)

    claim_text: str = Field(description="The reference claim proposition.")
    heuristic_supported: bool = Field(
        description="True if lexical content tokens from the claim appear in the answer text.",
    )
    matching_tokens: list[str] = Field(
        default_factory=list,
        description="Non-stopword tokens from the claim that were identified in the answer.",
    )
    total_claim_tokens: int = Field(
        description="Total content tokens extracted from the reference claim.",
    )


class GenerationCaseEvaluation(BaseModel):
    """Detailed generation evaluation result for a single case."""

    model_config = ConfigDict(frozen=True)

    case_id: str = Field(description="Unique identifier of the evaluated case.")
    query: str = Field(description="Question string evaluated.")
    answerable: bool = Field(description="Whether the query was answerable from verified ground truth.")
    split: str | None = Field(default=None, description="Dataset partition ('dev' or 'test').")
    mode: str = Field(description="Evaluation mode: 'direct_generation' or 'end_to_end_qa'.")
    answer_text: str = Field(description="Generated answer or abstention text.")

    # Abstention diagnostics
    is_abstained: bool = Field(description="True if the response was an explicit refusal or abstention.")
    abstention_reason: str | None = Field(default=None, description="Reported abstention reason code.")
    abstention_verdict: str = Field(
        description="Abstention classification: 'true_positive', 'false_positive', 'true_negative', or 'false_negative'.",
    )
    retrieval_failed: bool = Field(
        default=False,
        description="True if an answerable query had zero chunks retrieved before generation (E2E mode).",
    )

    # Citation validity diagnostics
    has_citations: bool = Field(description="True if at least one citation marker was detected.")
    is_valid_citations: bool = Field(
        description="True if all citation markers adhere to numeric syntax and resolve to context chunks.",
    )
    unknown_citations: list[str] = Field(
        default_factory=list,
        description="Citation markers referencing IDs not present in the supplied evidence context.",
    )
    malformed_citations: list[str] = Field(
        default_factory=list,
        description="Bracketed markers identified as malformed citation attempts.",
    )
    unsupported_citations: list[str] = Field(
        default_factory=list,
        description="Cited evidence IDs where the cited passage shares zero lexical grounding with the claim.",
    )

    # Claim-level support
    claim_evaluations: list[ClaimEvaluation] = Field(
        default_factory=list,
        description="Per-claim heuristic evaluation results for verified reference claims.",
    )
    all_claims_supported: bool | None = Field(
        default=None,
        description="True if all reference claims passed heuristic support; None if no reference claims exist.",
    )

    # Negative constraints
    forbidden_claims_checked: list[str] = Field(
        default_factory=list,
        description="List of forbidden phrases checked against the answer text.",
    )
    forbidden_violations_detected: list[str] = Field(
        default_factory=list,
        description="Forbidden phrases that appeared verbatim in the answer text.",
    )
    has_forbidden_violation: bool = Field(
        default=False,
        description="True if at least one forbidden claim or fabricated entity appeared in the answer.",
    )

    # Operational
    latency_seconds: float | None = Field(default=None, description="Elapsed execution time in seconds.")
    error_category: str | None = Field(default=None, description="Operational failure classification if any.")


class GenerationEvaluationMetrics(BaseModel):
    """Aggregated generation quality metrics with explicit denominators."""

    model_config = ConfigDict(frozen=True)

    total_cases: int = Field(description="Total cases evaluated.")
    answerable_count: int = Field(description="Total answerable cases evaluated.")
    unanswerable_count: int = Field(description="Total unanswerable cases evaluated.")

    # 1. Answerability & Abstention
    true_positive_abstentions: int = Field(
        description="TP: Unanswerable cases correctly abstained on.",
    )
    false_positive_abstentions: int = Field(
        description="FP: Answerable cases falsely abstained on.",
    )
    true_negative_answers: int = Field(
        description="TN: Answerable cases where an answer was provided.",
    )
    false_negative_answers: int = Field(
        description="FN: Unanswerable cases where the model failed to abstain.",
    )
    abstention_precision: float = Field(
        description="Abstention Precision: TP / (TP + FP) (or 0.0 if zero abstentions).",
    )
    abstention_recall: float = Field(
        description="Abstention Recall: TP / (TP + FN) = TP / unanswerable_count (or 0.0 if zero unanswerables).",
    )
    false_abstention_rate: float = Field(
        description="False Abstention Rate: FP / answerable_count (or 0.0 if zero answerables).",
    )
    answer_coverage: float = Field(
        description="Answer Coverage: TN / answerable_count (or 0.0 if zero answerables).",
    )

    # 2. Citation Integrity (evaluated on substantive answers)
    substantive_answers_count: int = Field(
        description="Total substantive answers evaluated for citations (TN + FN).",
    )
    citation_syntax_valid_count: int = Field(
        description="Substantive answers with zero malformed citation markers.",
    )
    citation_reference_valid_count: int = Field(
        description="Substantive answers where all citations resolve to existing context passages.",
    )
    citation_validity_rate: float = Field(
        description="Fraction of substantive answers with completely valid citation syntax and references.",
    )

    # 3. Claim Support & Grounding
    cases_with_reference_claims: int = Field(
        description="Answerable cases evaluated that had verified reference claims.",
    )
    excluded_cases_no_claims: int = Field(
        description="Answerable cases excluded from claim metrics due to lacking reference claims.",
    )
    total_reference_claims: int = Field(
        description="Total atomic reference claims evaluated across cases.",
    )
    supported_reference_claims: int = Field(
        description="Reference claims passing heuristic lexical support in the answer.",
    )
    claim_support_rate: float = Field(
        description="Heuristic claim support rate: supported_reference_claims / total_reference_claims.",
    )
    full_claim_coverage_rate: float = Field(
        description="Fraction of cases where 100% of reference claims were heuristically supported.",
    )
    lexical_grounding_pass_rate: float = Field(
        description="Fraction of substantive answers where zero cited statements failed sentence grounding.",
    )

    # 4. Negative Constraints
    cases_with_forbidden_constraints: int = Field(
        description="Cases evaluated that had explicit forbidden claims specified.",
    )
    excluded_cases_no_constraints: int = Field(
        description="Cases excluded from constraint metrics because no forbidden constraints were defined.",
    )
    forbidden_violations_count: int = Field(
        description="Cases where one or more forbidden claims appeared in the answer.",
    )
    constraint_compliance_rate: float = Field(
        description="Fraction of constrained cases with zero forbidden claim violations.",
    )

    # 5. Retrieval vs. Generation Decoupling
    retrieval_failures_count: int = Field(
        description="Answerable cases where retrieval returned 0 chunks before generation (E2E mode).",
    )

    # 6. Operational Performance
    mean_latency_seconds: float | None = Field(default=None, description="Average response latency in seconds.")
    p50_latency_seconds: float | None = Field(default=None, description="Median response latency (50th percentile).")
    p95_latency_seconds: float | None = Field(default=None, description="95th percentile response latency.")
    latency_sample_count: int = Field(default=0, description="Total cases with recorded latency.")


class GenerationEvaluationReport(BaseModel):
    """Complete structured report of a generation evaluation run."""

    model_config = ConfigDict(frozen=True)

    dataset_id: str | None = Field(default=None, description="Identifier of the evaluation dataset.")
    split: str | None = Field(default=None, description="Dataset partition evaluated ('dev', 'test', 'full').")
    mode: str = Field(description="Evaluation mode ('direct_generation' or 'end_to_end_qa').")
    model_id: str | None = Field(default=None, description="Canonical identifier of the generation model evaluated.")
    metrics: GenerationEvaluationMetrics = Field(description="Aggregated generation metrics.")
    case_evaluations: list[GenerationCaseEvaluation] = Field(description="Per-case evaluation details.")


# ==============================================================================
# Evaluator Engine
# ==============================================================================


class GenerationEvaluator:
    """Evaluates question-answering generation, citation integrity, and abstention behavior.

    Supports:
    1. Direct generation evaluation using known, verified evidence passages.
    2. End-to-end QA evaluation using actual retrieval plus generation.
    3. Deterministic offline evaluation using mock generators or recorded responses.
    """

    def __init__(self, citation_validator: CitationValidator | None = None) -> None:
        """Initialize the GenerationEvaluator.

        Args:
            citation_validator: Optional CitationValidator instance for syntax and reference checks.
        """
        self.citation_validator = citation_validator or CitationValidator()

    @staticmethod
    def evaluate_reference_claim(
        claim_text: str,
        answer_text: str,
    ) -> ClaimEvaluation:
        """Heuristically check whether reference claim content tokens appear in the answer.

        EVALUATION NOTE:
        Lexical token matching is an automated heuristic signal and does NOT constitute
        semantic entailment or mathematical proof of factual correctness.
        """
        claim_tokens = extract_grounding_tokens(claim_text)
        if not claim_tokens:
            return ClaimEvaluation(
                claim_text=claim_text,
                heuristic_supported=False,
                matching_tokens=[],
                total_claim_tokens=0,
            )

        answer_tokens = extract_grounding_tokens(answer_text)
        matching = sorted(claim_tokens & answer_tokens)

        # Heuristic support rule: at least 50% of claim tokens, or at least 2 tokens for short claims
        threshold = max(2, math.ceil(len(claim_tokens) * 0.50)) if len(claim_tokens) > 2 else len(claim_tokens)
        is_supported = len(matching) >= threshold

        return ClaimEvaluation(
            claim_text=claim_text,
            heuristic_supported=is_supported,
            matching_tokens=matching,
            total_claim_tokens=len(claim_tokens),
        )

    def evaluate_response(
        self,
        case: EvaluationCase,
        answer_text: str,
        is_abstained: bool,
        abstention_reason: str | None = None,
        unknown_citations: Sequence[str] | None = None,
        malformed_citations: Sequence[str] | None = None,
        unsupported_citations: Sequence[str] | None = None,
        retrieval_failed: bool = False,
        mode: str = "direct_generation",
        latency_seconds: float | None = None,
        error_category: str | None = None,
    ) -> GenerationCaseEvaluation:
        """Evaluate a single generation response against an EvaluationCase."""
        # 1. Abstention classification
        if not case.answerable:
            if is_abstained:
                abstention_verdict = "true_positive"
            else:
                abstention_verdict = "false_negative"
        else:
            if is_abstained:
                abstention_verdict = "false_positive"
            else:
                abstention_verdict = "true_negative"

        # 2. Citation validity diagnostics
        unknown_cits = list(unknown_citations or [])
        malformed_cits = list(malformed_citations or [])
        unsupported_cits = list(unsupported_citations or [])

        # Parse bracketed tokens to detect citations in answer text if not supplied
        if not is_abstained:
            parsed_ids, parsed_malformed = self.citation_validator.parse_bracketed_tokens(answer_text)
            has_citations = len(parsed_ids) > 0
            if parsed_malformed and not malformed_cits:
                malformed_cits = parsed_malformed
            is_valid_citations = len(unknown_cits) == 0 and len(malformed_cits) == 0
        else:
            has_citations = False
            is_valid_citations = True

        # 3. Claim support heuristics (only evaluated on substantive answers for answerable cases)
        claim_evals: list[ClaimEvaluation] = []
        all_supported: bool | None = None

        if case.answerable and not is_abstained and case.reference_claims:
            for claim in case.reference_claims:
                ev = self.evaluate_reference_claim(claim, answer_text)
                claim_evals.append(ev)
            all_supported = all(c.heuristic_supported for c in claim_evals)
        elif case.answerable and is_abstained:
            # Falsely abstained: claims were not asserted
            all_supported = False

        # 4. Negative constraints evaluation
        forbidden_checked = list(case.forbidden_claims)
        forbidden_violations: list[str] = []
        lowered_answer = answer_text.lower()
        for forbidden in forbidden_checked:
            if forbidden.strip().lower() in lowered_answer:
                forbidden_violations.append(forbidden)

        has_forbidden_violation = len(forbidden_violations) > 0

        return GenerationCaseEvaluation(
            case_id=case.case_id,
            query=case.question,
            answerable=case.answerable,
            split=case.split,
            mode=mode,
            answer_text=answer_text,
            is_abstained=is_abstained,
            abstention_reason=abstention_reason,
            abstention_verdict=abstention_verdict,
            retrieval_failed=retrieval_failed,
            has_citations=has_citations,
            is_valid_citations=is_valid_citations,
            unknown_citations=unknown_cits,
            malformed_citations=malformed_cits,
            unsupported_citations=unsupported_cits,
            claim_evaluations=claim_evals,
            all_claims_supported=all_supported,
            forbidden_claims_checked=forbidden_checked,
            forbidden_violations_detected=forbidden_violations,
            has_forbidden_violation=has_forbidden_violation,
            latency_seconds=latency_seconds,
            error_category=error_category,
        )

    def aggregate_metrics(
        self,
        case_evaluations: Sequence[GenerationCaseEvaluation],
    ) -> GenerationEvaluationMetrics:
        """Aggregate per-case evaluations into typed evaluation metrics with explicit denominators."""
        total_cases = len(case_evaluations)
        ans_cases = [c for c in case_evaluations if c.answerable]
        unans_cases = [c for c in case_evaluations if not c.answerable]

        ans_count = len(ans_cases)
        unans_count = len(unans_cases)

        # 1. Abstention counts
        tp = sum(1 for c in unans_cases if c.abstention_verdict == "true_positive")
        fn = sum(1 for c in unans_cases if c.abstention_verdict == "false_negative")
        fp = sum(1 for c in ans_cases if c.abstention_verdict == "false_positive")
        tn = sum(1 for c in ans_cases if c.abstention_verdict == "true_negative")

        total_abstentions = tp + fp
        abstention_precision = tp / total_abstentions if total_abstentions > 0 else 0.0
        abstention_recall = tp / unans_count if unans_count > 0 else 0.0
        false_abstention_rate = fp / ans_count if ans_count > 0 else 0.0
        answer_coverage = tn / ans_count if ans_count > 0 else 0.0

        # 2. Citation validity (evaluated on substantive responses: tn + fn)
        substantive_cases = [c for c in case_evaluations if not c.is_abstained]
        substantive_count = len(substantive_cases)

        syntax_valid_count = sum(1 for c in substantive_cases if len(c.malformed_citations) == 0)
        ref_valid_count = sum(1 for c in substantive_cases if len(c.unknown_citations) == 0)
        fully_valid_count = sum(1 for c in substantive_cases if c.is_valid_citations)

        cit_syntax_rate = syntax_valid_count / substantive_count if substantive_count > 0 else 0.0
        cit_ref_rate = ref_valid_count / substantive_count if substantive_count > 0 else 0.0
        cit_validity_rate = fully_valid_count / substantive_count if substantive_count > 0 else 0.0

        # 3. Claims evaluation
        cases_with_claims = [c for c in ans_cases if c.claim_evaluations or (c.all_claims_supported is not None)]
        with_claims_count = len(cases_with_claims)
        excluded_claims_count = ans_count - with_claims_count

        all_claims = [ce for c in cases_with_claims for ce in c.claim_evaluations]
        total_claims_count = len(all_claims)
        supported_claims_count = sum(1 for ce in all_claims if ce.heuristic_supported)
        claim_support_rate = supported_claims_count / total_claims_count if total_claims_count > 0 else 0.0

        full_coverage_count = sum(1 for c in cases_with_claims if c.all_claims_supported is True)
        full_coverage_rate = full_coverage_count / with_claims_count if with_claims_count > 0 else 0.0

        # Lexical grounding pass rate on substantive answers
        grounding_pass_count = sum(1 for c in substantive_cases if len(c.unsupported_citations) == 0)
        grounding_pass_rate = grounding_pass_count / substantive_count if substantive_count > 0 else 0.0

        # 4. Negative constraints
        constrained_cases = [c for c in case_evaluations if c.forbidden_claims_checked]
        constrained_count = len(constrained_cases)
        excluded_constraints_count = total_cases - constrained_count

        violations_count = sum(1 for c in constrained_cases if c.has_forbidden_violation)
        compliance_rate = (constrained_count - violations_count) / constrained_count if constrained_count > 0 else 0.0

        # 5. Retrieval decoupling
        retrieval_failures = sum(1 for c in ans_cases if c.retrieval_failed)

        # 6. Latency statistics
        latencies = [c.latency_seconds for c in case_evaluations if c.latency_seconds is not None]
        mean_lat: float | None = None
        p50_lat: float | None = None
        p95_lat: float | None = None

        if latencies:
            mean_lat = sum(latencies) / len(latencies)
            sorted_lat = sorted(latencies)
            # 50th percentile (median)
            p50_lat = statistics.median(sorted_lat)
            # 95th percentile
            p95_idx = max(0, math.ceil(len(sorted_lat) * 0.95) - 1)
            p95_lat = sorted_lat[p95_idx]

        return GenerationEvaluationMetrics(
            total_cases=total_cases,
            answerable_count=ans_count,
            unanswerable_count=unans_count,
            true_positive_abstentions=tp,
            false_positive_abstentions=fp,
            true_negative_answers=tn,
            false_negative_answers=fn,
            abstention_precision=abstention_precision,
            abstention_recall=abstention_recall,
            false_abstention_rate=false_abstention_rate,
            answer_coverage=answer_coverage,
            substantive_answers_count=substantive_count,
            citation_syntax_valid_count=syntax_valid_count,
            citation_reference_valid_count=ref_valid_count,
            citation_validity_rate=cit_validity_rate,
            cases_with_reference_claims=with_claims_count,
            excluded_cases_no_claims=excluded_claims_count,
            total_reference_claims=total_claims_count,
            supported_reference_claims=supported_claims_count,
            claim_support_rate=claim_support_rate,
            full_claim_coverage_rate=full_coverage_rate,
            lexical_grounding_pass_rate=grounding_pass_rate,
            cases_with_forbidden_constraints=constrained_count,
            excluded_cases_no_constraints=excluded_constraints_count,
            forbidden_violations_count=violations_count,
            constraint_compliance_rate=compliance_rate,
            retrieval_failures_count=retrieval_failures,
            mean_latency_seconds=mean_lat,
            p50_latency_seconds=p50_lat,
            p95_latency_seconds=p95_lat,
            latency_sample_count=len(latencies),
        )

    def evaluate_direct_generation(
        self,
        generation_provider: GenerationProvider,
        cases: Sequence[EvaluationCase],
        dataset_id: str | None = None,
        split: str | None = None,
    ) -> GenerationEvaluationReport:
        """Evaluate generation provider directly using known, verified evidence passages.

        Eliminates retrieval failure noise by presenting verified evidence directly to the generator.
        """
        case_evaluations: list[GenerationCaseEvaluation] = []

        for case in cases:
            # Construct EvidenceItem instances from expected evidence snippets
            evidence_items: list[EvidenceItem] = []
            if case.answerable:
                for idx, snippet in enumerate(case.expected_evidence_snippets, start=1):
                    evidence_items.append(
                        EvidenceItem(
                            id=str(idx),
                            text=snippet,
                            document_title=case.target_document_filename,
                            heading=case.target_section,
                        )
                    )

            t0 = time.perf_counter()
            error_cat: str | None = None
            try:
                gen_result = generation_provider.generate_answer(
                    query=case.question,
                    evidence=evidence_items,
                )
                answer_text = gen_result.answer
                is_abstained = gen_result.is_abstained
                abstention_reason = gen_result.abstention_reason
                reported_ids = gen_result.cited_evidence_ids
            except Exception as exc:
                error_cat = type(exc).__name__
                answer_text = f"Error during generation: {exc}"
                is_abstained = True
                abstention_reason = "generation_error"
                reported_ids = []

            latency = time.perf_counter() - t0

            # Validate citations against supplied evidence
            context_map = {item.id: item for item in evidence_items}
            parsed_ids, malformed = self.citation_validator.parse_bracketed_tokens(
                answer_text, reported_ids=reported_ids
            )
            unknown = [cid for cid in parsed_ids if cid not in context_map]

            # Grounding check on substantive answers
            unsupported: list[str] = []
            if not is_abstained:
                raw_sentences = re.split(r"(?<=[.!?])\s+|\n+", answer_text)

                class _DirectMapping:
                    def __init__(self, text: str, heading: str | None, doc_name: str | None) -> None:
                        self.text = text
                        self.heading = heading
                        self.document_filename = doc_name

                d_map = {
                    item.id: _DirectMapping(item.text, item.heading, item.document_title)
                    for item in evidence_items
                }
                for s in raw_sentences:
                    s_clean = s.strip()
                    if not s_clean:
                        continue
                    s_cand_ids, _ = self.citation_validator.parse_bracketed_tokens(s_clean)
                    if s_cand_ids:
                        unsupp_s = check_sentence_grounding(s_clean, s_cand_ids, d_map)  # type: ignore[arg-type]
                        for u_id in unsupp_s:
                            if u_id not in unsupported:
                                unsupported.append(u_id)

            ev = self.evaluate_response(
                case=case,
                answer_text=answer_text,
                is_abstained=is_abstained,
                abstention_reason=abstention_reason,
                unknown_citations=unknown,
                malformed_citations=malformed,
                unsupported_citations=unsupported,
                retrieval_failed=False,
                mode="direct_generation",
                latency_seconds=latency,
                error_category=error_cat,
            )
            case_evaluations.append(ev)

        metrics = self.aggregate_metrics(case_evaluations)
        return GenerationEvaluationReport(
            dataset_id=dataset_id,
            split=split,
            mode="direct_generation",
            model_id=generation_provider.model_id,
            metrics=metrics,
            case_evaluations=case_evaluations,
        )

    def evaluate_qa_service(
        self,
        qa_service: Any,
        cases: Sequence[EvaluationCase],
        top_k: int = 5,
        score_threshold: float | None = None,
        dataset_id: str | None = None,
        split: str | None = None,
    ) -> GenerationEvaluationReport:
        """Evaluate complete pipeline via QuestionAnsweringService.answer_question.

        Distinguishes retrieval failure (missing candidate evidence) from generator failure.
        """
        case_evaluations: list[GenerationCaseEvaluation] = []

        for case in cases:
            t0 = time.perf_counter()
            error_cat: str | None = None
            try:
                qa_resp: QAResponse = qa_service.answer_question(
                    query=case.question,
                    top_k=top_k,
                    score_threshold=score_threshold,
                )
                answer_text = qa_resp.answer
                is_abstained = qa_resp.is_abstained
                abstention_reason = qa_resp.abstention_reason
                unknown_cits = qa_resp.unknown_citations
                malformed_cits = qa_resp.malformed_citations
                unsupported_cits = qa_resp.unsupported_citations
                retrieved_count = qa_resp.retrieved_count
                model_id = qa_resp.model_id
            except Exception as exc:
                error_cat = type(exc).__name__
                answer_text = f"Error during QA: {exc}"
                is_abstained = True
                abstention_reason = "qa_error"
                unknown_cits = []
                malformed_cits = []
                unsupported_cits = []
                retrieved_count = 0
                model_id = getattr(qa_service.generation_provider, "model_id", "unknown")

            latency = time.perf_counter() - t0

            # Tag retrieval failure if an answerable query retrieved 0 candidates
            retrieval_failed = case.answerable and (retrieved_count == 0)

            ev = self.evaluate_response(
                case=case,
                answer_text=answer_text,
                is_abstained=is_abstained,
                abstention_reason=abstention_reason,
                unknown_citations=unknown_cits,
                malformed_citations=malformed_cits,
                unsupported_citations=unsupported_cits,
                retrieval_failed=retrieval_failed,
                mode="end_to_end_qa",
                latency_seconds=latency,
                error_category=error_cat,
            )
            case_evaluations.append(ev)

        metrics = self.aggregate_metrics(case_evaluations)
        return GenerationEvaluationReport(
            dataset_id=dataset_id,
            split=split,
            mode="end_to_end_qa",
            model_id=model_id if case_evaluations else None,
            metrics=metrics,
            case_evaluations=case_evaluations,
        )
