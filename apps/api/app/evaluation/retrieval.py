"""Reusable semantic retrieval evaluator for EvidenceDesk.

Measures Recall@1, Recall@3, Recall@5, and Mean Reciprocal Rank (MRR)
over answerable queries, and evaluates False Acceptance Rate (FAR) and Correct
Rejection Rate over unanswerable queries against explicit score thresholds.

Designed to consume either the versioned EvaluationDataset schema (Dataset V1)
or legacy benchmark cases, operating over live RetrievalService instances or
deterministic ranked chunk fixtures.
"""

from collections.abc import Sequence
import math
from typing import Any

from pydantic import BaseModel, ConfigDict, Field

from app.schemas.evaluation import EvaluationCase
from app.schemas.retrieval import RetrievalResponse, RetrievedChunk


def _validate_threshold(threshold: float) -> float:
    """Validate that an acceptance threshold is a finite float in [-1.0, 1.0]."""
    if not isinstance(threshold, (int, float)) or isinstance(threshold, bool):
        raise ValueError(f"score_threshold must be a float or int, got {type(threshold).__name__}.")
    if not math.isfinite(threshold) or threshold < -1.0 or threshold > 1.0:
        raise ValueError(f"score_threshold ({threshold}) must be a finite number between -1.0 and 1.0.")
    return float(threshold)


def _validate_top_k(top_k: int) -> int:
    """Validate that top_k is an integer in [1, 50]."""
    if not isinstance(top_k, int) or isinstance(top_k, bool):
        raise ValueError(f"top_k must be an integer, got {type(top_k).__name__}.")
    if top_k < 1 or top_k > 50:
        raise ValueError(f"top_k ({top_k}) must be between 1 and 50.")
    return top_k


def is_chunk_relevant(
    chunk: RetrievedChunk,
    case: EvaluationCase | dict[str, Any],
) -> bool:
    """Determine ground-truth relevance using document filename and evidence text containment.

    A retrieved chunk is considered relevant if and only if:
    1. The case is answerable (answerable is True).
    2. The chunk's document_filename matches the case's target_document_filename.
    3. At least one non-empty expected evidence snippet from the case is contained
       verbatim in the chunk's text (case-insensitive comparison).

    Returns False for unanswerable cases.

    EVALUATION NOTE:
    Relevance confirms that the retrieved candidate chunk encloses the verified source
    evidence passage. It does NOT prove generated answer correctness or semantic entailment.
    """
    if isinstance(case, EvaluationCase):
        if not case.answerable:
            return False
        if chunk.document_filename != case.target_document_filename:
            return False
        snippets = case.expected_evidence_snippets
    elif isinstance(case, dict):
        if not case.get("answerable", True):
            return False
        if chunk.document_filename != case.get("target_document_filename"):
            return False
        snippets = case.get("expected_evidence_snippets")
        if not snippets and "evidence_text" in case:
            snippets = [case["evidence_text"]]
    else:
        raise TypeError(f"Case must be an EvaluationCase or dict, got {type(case).__name__}.")

    if not snippets:
        return False

    chunk_lower = chunk.text.lower()
    for snippet in snippets:
        s_clean = snippet.strip().lower()
        if s_clean and s_clean in chunk_lower:
            return True

    return False


# ==============================================================================
# Typed Evaluation Output Schemas
# ==============================================================================


class ThresholdTradeoffPoint(BaseModel):
    """Result of threshold evaluation at a specific cosine similarity score."""

    model_config = ConfigDict(frozen=True)

    threshold: float = Field(
        description="Cosine similarity threshold tau tested in [-1.0, 1.0].",
    )
    unanswerable_denominator: int = Field(
        description="Total unanswerable queries evaluated at this threshold.",
    )
    false_accept_count: int = Field(
        description="Number of unanswerable queries with max similarity score >= threshold.",
    )
    false_accept_rate: float = Field(
        description="False-accept rate: false_accept_count / unanswerable_denominator (or 0.0 if empty).",
    )
    correct_reject_count: int = Field(
        description="Number of unanswerable queries correctly rejected (max similarity score < threshold or empty).",
    )
    correct_reject_rate: float = Field(
        description="Correct-reject rate: 1.0 - false_accept_rate (or 0.0 if empty).",
    )
    answerable_denominator: int = Field(
        description="Total answerable queries evaluated at this threshold.",
    )
    false_rejection_count: int = Field(
        description="Number of answerable queries where first relevant score < threshold or no relevant chunk found.",
    )
    false_rejection_rate: float = Field(
        description="False-rejection rate: false_rejection_count / answerable_denominator (or 0.0 if empty).",
    )


class RetrievalCaseEvaluation(BaseModel):
    """Evaluation result for a single question case."""

    model_config = ConfigDict(frozen=True)

    case_id: str = Field(description="Unique identifier of the evaluated case.")
    query: str = Field(description="Question string evaluated.")
    answerable: bool = Field(description="Whether the query was answerable from the corpus.")
    split: str | None = Field(default=None, description="Dataset partition ('dev' or 'test').")
    target_document_filename: str = Field(description="Target document associated with the case.")
    returned_count: int = Field(description="Number of candidate chunks returned in top_k.")
    first_relevant_rank: int | None = Field(
        default=None,
        description="1-based rank of the first relevant chunk, or None if no relevant chunk found in top_k.",
    )
    first_relevant_score: float | None = Field(
        default=None,
        description="Cosine similarity score of the first relevant chunk, or None if not found.",
    )
    max_candidate_score: float | None = Field(
        default=None,
        description="Highest similarity score among returned chunks, or None if zero chunks returned.",
    )
    hit_at_1: bool = Field(default=False, description="True if a relevant chunk was found at rank 1.")
    hit_at_3: bool = Field(default=False, description="True if a relevant chunk was found at rank <= 3.")
    hit_at_5: bool = Field(default=False, description="True if a relevant chunk was found at rank <= 5.")
    reciprocal_rank: float = Field(
        default=0.0,
        description="1.0 / first_relevant_rank if relevant chunk found, else 0.0.",
    )
    is_false_accept: bool | None = Field(
        default=None,
        description="For unanswerable queries: True if max candidate score >= acceptance threshold.",
    )
    is_correct_reject: bool | None = Field(
        default=None,
        description="For unanswerable queries: True if max candidate score < acceptance threshold or 0 chunks.",
    )
    top_candidate_text: str | None = Field(
        default=None,
        description="Snippet of the top-ranked candidate chunk text for failure analysis.",
    )


class RetrievalEvaluationMetrics(BaseModel):
    """Aggregated information retrieval metrics with explicit denominators."""

    model_config = ConfigDict(frozen=True)

    total_cases: int = Field(description="Total number of cases evaluated.")
    top_k: int = Field(description="Maximum candidate chunks evaluated per query.")
    acceptance_threshold: float = Field(
        description="Configured cosine similarity threshold tau for unanswerable query rejection.",
    )

    # Answerable queries evaluation
    answerable_count: int = Field(description="Total answerable queries evaluated (denominator for Recall and MRR).")
    recall_at_1: float = Field(description="Recall@1: fraction of answerable queries with relevant chunk at rank 1.")
    recall_at_3: float = Field(description="Recall@3: fraction of answerable queries with relevant chunk at rank <= 3.")
    recall_at_5: float = Field(description="Recall@5: fraction of answerable queries with relevant chunk at rank <= 5.")
    mrr: float = Field(description="Mean Reciprocal Rank over answerable queries.")
    hit_at_1_count: int = Field(description="Absolute count of answerable queries with hit at rank 1.")
    hit_at_3_count: int = Field(description="Absolute count of answerable queries with hit at rank <= 3.")
    hit_at_5_count: int = Field(description="Absolute count of answerable queries with hit at rank <= 5.")
    avg_first_relevant_score: float | None = Field(
        default=None,
        description="Average cosine similarity score of first relevant chunks, or None if answerable_count == 0.",
    )
    avg_answerable_top1_score: float | None = Field(
        default=None,
        description="Average cosine similarity score of top-1 candidate chunk for answerables.",
    )

    # Unanswerable queries evaluation (retrieval rejection)
    unanswerable_count: int = Field(
        description="Total unanswerable queries evaluated (denominator for FAR and correct-reject rate).",
    )
    false_accept_count: int = Field(
        description="Unanswerable queries where max candidate score >= acceptance_threshold.",
    )
    correct_reject_count: int = Field(
        description="Unanswerable queries correctly rejected (max score < acceptance_threshold or empty).",
    )
    false_accept_rate: float = Field(
        description="False-accept rate (FAR): false_accept_count / unanswerable_count (or 0.0 if empty).",
    )
    correct_reject_rate: float = Field(
        description="Correct-reject rate: correct_reject_count / unanswerable_count (or 0.0 if empty).",
    )
    avg_unanswerable_max_score: float | None = Field(
        default=None,
        description="Average maximum candidate score across unanswerable queries, or None if unanswerable_count == 0.",
    )


class RetrievalEvaluationReport(BaseModel):
    """Complete structured report of a retrieval evaluation run."""

    model_config = ConfigDict(frozen=True)

    dataset_id: str | None = Field(default=None, description="Dataset identifier if evaluated from a dataset.")
    split: str | None = Field(default=None, description="Dataset partition evaluated ('dev', 'test', or 'full').")
    metrics: RetrievalEvaluationMetrics = Field(description="Aggregated retrieval metrics.")
    case_evaluations: list[RetrievalCaseEvaluation] = Field(description="Per-case evaluation details.")
    threshold_sweep: list[ThresholdTradeoffPoint] = Field(
        default_factory=list,
        description="Optional threshold trade-off sweep results across candidate acceptance thresholds.",
    )


# ==============================================================================
# Reusable Retrieval Evaluator Engine
# ==============================================================================


class RetrievalEvaluator:
    """Evaluates information retrieval performance against structured evaluation datasets.

    Calculates Recall@1, Recall@3, Recall@5, MRR, False Acceptance Rate (FAR),
    and Correct Rejection Rate. Reports explicit denominators and supports threshold sweeps.
    """

    def __init__(
        self,
        default_top_k: int = 5,
        default_acceptance_threshold: float = 0.55,
    ) -> None:
        """Initialize the evaluator with default operational bounds.

        Args:
            default_top_k: Default candidate limit (1 to 50).
            default_acceptance_threshold: Default similarity threshold tau for rejection (-1.0 to 1.0).
        """
        self.default_top_k = _validate_top_k(default_top_k)
        self.default_acceptance_threshold = _validate_threshold(default_acceptance_threshold)

    def evaluate_case_candidates(
        self,
        case: EvaluationCase | dict[str, Any],
        candidate_chunks: Sequence[RetrievedChunk],
        top_k: int | None = None,
        acceptance_threshold: float | None = None,
    ) -> RetrievalCaseEvaluation:
        """Evaluate ranked retrieval results for a single evaluation case.

        Args:
            case: EvaluationCase instance or legacy case dictionary.
            candidate_chunks: Ranked sequence of candidate chunks returned by retrieval.
            top_k: Maximum candidate rank to consider (defaults to instance setting).
            acceptance_threshold: Cosine similarity threshold for rejection check.

        Returns:
            RetrievalCaseEvaluation detailing rank, scores, hits, and rejection verdict.
        """
        k = _validate_top_k(top_k if top_k is not None else self.default_top_k)
        tau = _validate_threshold(
            acceptance_threshold if acceptance_threshold is not None else self.default_acceptance_threshold
        )

        case_id = case.case_id if isinstance(case, EvaluationCase) else str(case.get("case_id", "unknown"))
        query = case.question if isinstance(case, EvaluationCase) else str(case.get("question", ""))
        answerable = case.answerable if isinstance(case, EvaluationCase) else bool(case.get("answerable", True))
        split = case.split if isinstance(case, EvaluationCase) else case.get("split")
        doc_filename = (
            case.target_document_filename
            if isinstance(case, EvaluationCase)
            else str(case.get("target_document_filename", ""))
        )

        ranked_candidates = list(candidate_chunks[:k])
        returned_count = len(ranked_candidates)

        max_candidate_score: float | None = None
        top_candidate_text: str | None = None
        if ranked_candidates:
            max_candidate_score = ranked_candidates[0].score
            top_candidate_text = ranked_candidates[0].text[:120].replace("\n", " ").strip()

        if answerable:
            first_rel_rank: int | None = None
            first_rel_score: float | None = None

            for rank_idx, chunk in enumerate(ranked_candidates, start=1):
                if is_chunk_relevant(chunk, case):
                    first_rel_rank = rank_idx
                    first_rel_score = chunk.score
                    break

            hit_1 = first_rel_rank is not None and first_rel_rank <= 1
            hit_3 = first_rel_rank is not None and first_rel_rank <= 3
            hit_5 = first_rel_rank is not None and first_rel_rank <= 5
            rr = 1.0 / first_rel_rank if first_rel_rank is not None else 0.0

            return RetrievalCaseEvaluation(
                case_id=case_id,
                query=query,
                answerable=True,
                split=split,
                target_document_filename=doc_filename,
                returned_count=returned_count,
                first_relevant_rank=first_rel_rank,
                first_relevant_score=first_rel_score,
                max_candidate_score=max_candidate_score,
                hit_at_1=hit_1,
                hit_at_3=hit_3,
                hit_at_5=hit_5,
                reciprocal_rank=rr,
                is_false_accept=None,
                is_correct_reject=None,
                top_candidate_text=top_candidate_text,
            )
        else:
            # Unanswerable case
            is_false_accept = (max_candidate_score is not None) and (max_candidate_score >= tau)
            is_correct_reject = not is_false_accept

            return RetrievalCaseEvaluation(
                case_id=case_id,
                query=query,
                answerable=False,
                split=split,
                target_document_filename=doc_filename,
                returned_count=returned_count,
                first_relevant_rank=None,
                first_relevant_score=None,
                max_candidate_score=max_candidate_score,
                hit_at_1=False,
                hit_at_3=False,
                hit_at_5=False,
                reciprocal_rank=0.0,
                is_false_accept=is_false_accept,
                is_correct_reject=is_correct_reject,
                top_candidate_text=top_candidate_text,
            )

    def evaluate_cases(
        self,
        cases_with_candidates: Sequence[tuple[EvaluationCase | dict[str, Any], Sequence[RetrievedChunk]]],
        top_k: int | None = None,
        acceptance_threshold: float | None = None,
        dataset_id: str | None = None,
        split: str | None = None,
        sweep_thresholds: Sequence[float] | None = None,
    ) -> RetrievalEvaluationReport:
        """Evaluate a sequence of pre-computed (case, candidate_chunks) pairs.

        Args:
            cases_with_candidates: Sequence of (case, candidates) tuples.
            top_k: Candidate limit.
            acceptance_threshold: Acceptance threshold tau for unanswerable query checks.
            dataset_id: Optional dataset identifier.
            split: Optional dataset partition label.
            sweep_thresholds: Optional sequence of thresholds to compute a trade-off sweep.

        Returns:
            RetrievalEvaluationReport containing metrics, per-case evaluations, and optional sweep.
        """
        k = _validate_top_k(top_k if top_k is not None else self.default_top_k)
        tau = _validate_threshold(
            acceptance_threshold if acceptance_threshold is not None else self.default_acceptance_threshold
        )

        case_evaluations: list[RetrievalCaseEvaluation] = []
        for case, candidates in cases_with_candidates:
            res = self.evaluate_case_candidates(
                case=case,
                candidate_chunks=candidates,
                top_k=k,
                acceptance_threshold=tau,
            )
            case_evaluations.append(res)

        total_cases = len(case_evaluations)
        answerable_cases = [c for c in case_evaluations if c.answerable]
        unanswerable_cases = [c for c in case_evaluations if not c.answerable]

        ans_count = len(answerable_cases)
        unans_count = len(unanswerable_cases)

        # Answerable metric aggregation
        hit_1_count = sum(1 for c in answerable_cases if c.hit_at_1)
        hit_3_count = sum(1 for c in answerable_cases if c.hit_at_3)
        hit_5_count = sum(1 for c in answerable_cases if c.hit_at_5)
        mrr_sum = sum(c.reciprocal_rank for c in answerable_cases)

        recall_1 = hit_1_count / ans_count if ans_count > 0 else 0.0
        recall_3 = hit_3_count / ans_count if ans_count > 0 else 0.0
        recall_5 = hit_5_count / ans_count if ans_count > 0 else 0.0
        mrr = mrr_sum / ans_count if ans_count > 0 else 0.0

        rel_scores = [c.first_relevant_score for c in answerable_cases if c.first_relevant_score is not None]
        avg_first_rel = sum(rel_scores) / len(rel_scores) if rel_scores else None

        ans_top1_scores = [c.max_candidate_score for c in answerable_cases if c.max_candidate_score is not None]
        avg_ans_top1 = sum(ans_top1_scores) / len(ans_top1_scores) if ans_top1_scores else None

        # Unanswerable metric aggregation
        false_accept_count = sum(1 for c in unanswerable_cases if c.is_false_accept is True)
        correct_reject_count = sum(1 for c in unanswerable_cases if c.is_correct_reject is True)

        far = false_accept_count / unans_count if unans_count > 0 else 0.0
        crr = correct_reject_count / unans_count if unans_count > 0 else 0.0

        unans_scores = [c.max_candidate_score for c in unanswerable_cases if c.max_candidate_score is not None]
        avg_unans_max = sum(unans_scores) / len(unans_scores) if unans_scores else None

        metrics = RetrievalEvaluationMetrics(
            total_cases=total_cases,
            top_k=k,
            acceptance_threshold=tau,
            answerable_count=ans_count,
            recall_at_1=recall_1,
            recall_at_3=recall_3,
            recall_at_5=recall_5,
            mrr=mrr,
            hit_at_1_count=hit_1_count,
            hit_at_3_count=hit_3_count,
            hit_at_5_count=hit_5_count,
            avg_first_relevant_score=avg_first_rel,
            avg_answerable_top1_score=avg_ans_top1,
            unanswerable_count=unans_count,
            false_accept_count=false_accept_count,
            correct_reject_count=correct_reject_count,
            false_accept_rate=far,
            correct_reject_rate=crr,
            avg_unanswerable_max_score=avg_unans_max,
        )

        threshold_sweep: list[ThresholdTradeoffPoint] = []
        if sweep_thresholds is not None:
            threshold_sweep = self.compute_threshold_sweep(
                cases_with_candidates=cases_with_candidates,
                thresholds=sweep_thresholds,
                top_k=k,
            )

        return RetrievalEvaluationReport(
            dataset_id=dataset_id,
            split=split,
            metrics=metrics,
            case_evaluations=case_evaluations,
            threshold_sweep=threshold_sweep,
        )

    def evaluate_retrieval_service(
        self,
        retrieval_service: Any,
        cases: Sequence[EvaluationCase | dict[str, Any]],
        top_k: int | None = None,
        acceptance_threshold: float | None = None,
        service_score_threshold: float | None = None,
        dataset_id: str | None = None,
        split: str | None = None,
        sweep_thresholds: Sequence[float] | None = None,
    ) -> RetrievalEvaluationReport:
        """Execute retrieval queries via a production RetrievalService and evaluate results.

        Args:
            retrieval_service: RetrievalService instance providing .retrieve().
            cases: List of EvaluationCase or legacy case dictionaries.
            top_k: Candidate limit.
            acceptance_threshold: Acceptance threshold tau for rejection calculations.
            service_score_threshold: Optional similarity score_threshold passed to retrieve().
            dataset_id: Optional dataset ID.
            split: Optional dataset partition.
            sweep_thresholds: Optional list of thresholds for trade-off curve.

        Returns:
            RetrievalEvaluationReport containing evaluation results.
        """
        k = _validate_top_k(top_k if top_k is not None else self.default_top_k)

        cases_with_candidates: list[tuple[EvaluationCase | dict[str, Any], Sequence[RetrievedChunk]]] = []
        for case in cases:
            q = case.question if isinstance(case, EvaluationCase) else case["question"]
            resp = retrieval_service.retrieve(
                query=q,
                top_k=k,
                score_threshold=service_score_threshold,
            )
            chunks = resp.results if hasattr(resp, "results") else resp
            cases_with_candidates.append((case, chunks))

        return self.evaluate_cases(
            cases_with_candidates=cases_with_candidates,
            top_k=k,
            acceptance_threshold=acceptance_threshold,
            dataset_id=dataset_id,
            split=split,
            sweep_thresholds=sweep_thresholds,
        )

    def compute_threshold_sweep(
        self,
        cases_with_candidates: Sequence[tuple[EvaluationCase | dict[str, Any], Sequence[RetrievedChunk]]],
        thresholds: Sequence[float] = (0.55, 0.60, 0.65, 0.70, 0.75),
        top_k: int | None = None,
    ) -> list[ThresholdTradeoffPoint]:
        """Compute False Acceptance Rate (FAR) and False Rejection Rate (FRR) across thresholds.

        Args:
            cases_with_candidates: Sequence of (case, candidates) tuples.
            thresholds: Sequence of cosine similarity thresholds to sweep.
            top_k: Candidate limit.

        Returns:
            List of ThresholdTradeoffPoint instances ordered by ascending threshold.
        """
        k = _validate_top_k(top_k if top_k is not None else self.default_top_k)
        validated_thresholds = sorted(_validate_threshold(t) for t in thresholds)

        ans_cases = [item for item in cases_with_candidates if (
            item[0].answerable if isinstance(item[0], EvaluationCase) else item[0].get("answerable", True)
        )]
        unans_cases = [item for item in cases_with_candidates if not (
            item[0].answerable if isinstance(item[0], EvaluationCase) else item[0].get("answerable", True)
        )]

        ans_count = len(ans_cases)
        unans_count = len(unans_cases)

        # Pre-extract first relevant score per answerable query
        ans_first_rel_scores: list[float | None] = []
        for case, candidates in ans_cases:
            first_rel: float | None = None
            for chunk in candidates[:k]:
                if is_chunk_relevant(chunk, case):
                    first_rel = chunk.score
                    break
            ans_first_rel_scores.append(first_rel)

        # Pre-extract max candidate score per unanswerable query
        unans_max_scores: list[float | None] = []
        for _, candidates in unans_cases:
            max_sc: float | None = candidates[0].score if candidates[:k] else None
            unans_max_scores.append(max_sc)

        sweep_points: list[ThresholdTradeoffPoint] = []
        for tau in validated_thresholds:
            # FAR: unanswerable query whose max score >= tau
            fa_count = sum(1 for s in unans_max_scores if s is not None and s >= tau)
            far = fa_count / unans_count if unans_count > 0 else 0.0
            cr_count = unans_count - fa_count
            crr = cr_count / unans_count if unans_count > 0 else 0.0

            # FRR: answerable query for which no relevant retrieved chunk meets tau
            # (i.e. first relevant score is None or < tau)
            fr_count = sum(1 for s in ans_first_rel_scores if s is None or s < tau)
            frr = fr_count / ans_count if ans_count > 0 else 0.0

            sweep_points.append(
                ThresholdTradeoffPoint(
                    threshold=tau,
                    unanswerable_denominator=unans_count,
                    false_accept_count=fa_count,
                    false_accept_rate=far,
                    correct_reject_count=cr_count,
                    correct_reject_rate=crr,
                    answerable_denominator=ans_count,
                    false_rejection_count=fr_count,
                    false_rejection_rate=frr,
                )
            )

        return sweep_points
