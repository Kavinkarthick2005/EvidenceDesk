"""EvidenceDesk evaluation package for information retrieval and generation quality."""

from app.evaluation.generation import (
    ClaimEvaluation,
    GenerationCaseEvaluation,
    GenerationEvaluationMetrics,
    GenerationEvaluationReport,
    GenerationEvaluator,
)
from app.evaluation.retrieval import (
    RetrievalCaseEvaluation,
    RetrievalEvaluationMetrics,
    RetrievalEvaluationReport,
    RetrievalEvaluator,
    ThresholdTradeoffPoint,
    is_chunk_relevant,
)

__all__ = [
    "ClaimEvaluation",
    "GenerationCaseEvaluation",
    "GenerationEvaluationMetrics",
    "GenerationEvaluationReport",
    "GenerationEvaluator",
    "RetrievalCaseEvaluation",
    "RetrievalEvaluationMetrics",
    "RetrievalEvaluationReport",
    "RetrievalEvaluator",
    "ThresholdTradeoffPoint",
    "is_chunk_relevant",
]
