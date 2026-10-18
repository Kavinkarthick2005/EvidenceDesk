"""Question Answering (QA) orchestration service.

Coordinates semantic retrieval -> evidence context building -> generation -> citation validation.
Enforces that:
1. Zero retrieved evidence results in honest abstention without calling the LLM.
2. Generator abstentions are propagated cleanly.
3. Non-abstained responses with invalid, unknown, or malformed citations are rejected
   from being authoritative answers, returning safe abstentions with diagnostic details.
4. Non-abstained responses lacking citations are rejected with abstention reason 'missing_citations'.
5. Operational logs strictly omit user query text, prompts, chunk contents, and raw answers.
"""

from collections.abc import Sequence
import logging
import time
from typing import Any
import uuid

from app.core.config import Settings, get_settings
from app.generation.citation import CitationValidationResult, CitationValidator
from app.generation.context import ContextBuilder, EvidenceContext
from app.generation.exceptions import GenerationError
from app.generation.interface import GenerationProvider, GenerationResult
from app.generation import get_generation_provider
from app.schemas.qa import QACitation, QAResponse
from app.services.exceptions import InvalidQueryError, RetrievalError
from app.services.retrieval import RetrievalService

logger = logging.getLogger(__name__)


class QuestionAnsweringService:
    """Orchestrates evidence retrieval, context construction, answer generation, and citation validation."""

    def __init__(
        self,
        retrieval_service: RetrievalService,
        generation_provider: GenerationProvider | None = None,
        context_builder: ContextBuilder | None = None,
        citation_validator: CitationValidator | None = None,
        settings: Settings | None = None,
    ) -> None:
        """Initialize the QuestionAnsweringService.

        Args:
            retrieval_service: RetrievalService for semantic vector similarity search.
            generation_provider: Optional generation provider protocol implementation.
                Defaults to configured application provider (e.g., DeterministicMockGenerator).
            context_builder: Optional ContextBuilder for evidence assembly.
            citation_validator: Optional CitationValidator for reference integrity checks.
            settings: Optional application settings instance.
        """
        self.retrieval_service = retrieval_service
        self.settings = settings or get_settings()
        self.generation_provider = generation_provider or get_generation_provider(self.settings)
        self.context_builder = context_builder or ContextBuilder()
        self.citation_validator = citation_validator or CitationValidator()

    def answer_question(
        self,
        query: str,
        top_k: int = 5,
        document_ids: Sequence[uuid.UUID | str] | None = None,
        score_threshold: float | None = None,
    ) -> QAResponse:
        """Process a question by retrieving evidence, generating an answer, and verifying citations.

        Args:
            query: Natural language question to answer.
            top_k: Maximum candidate chunks to retrieve.
            document_ids: Optional document UUID filter.
            score_threshold: Optional cosine similarity score threshold.

        Returns:
            QAResponse containing answer text, abstention status, resolved citations,
            and validation diagnostics.

        Raises:
            InvalidQueryError: If query is empty, whitespace-only, or too long.
            RetrievalError: If database search fails.
            GenerationError: If answer generation encounters an operational failure.
        """
        if not isinstance(query, str):
            raise InvalidQueryError(f"Query must be a string, got {type(query).__name__}.")

        stripped_query = query.strip()
        if not stripped_query:
            raise InvalidQueryError("Query cannot be empty or whitespace-only.")

        if len(stripped_query) > 2000:
            raise InvalidQueryError(
                f"Query length ({len(stripped_query)}) exceeds maximum permitted limit of 2,000 characters."
            )

        logger.info(
            "Starting QA orchestration: query_length=%d, top_k=%d, doc_filter_count=%d, score_threshold=%s",
            len(stripped_query),
            top_k,
            len(document_ids) if document_ids else 0,
            f"{score_threshold:.4f}" if score_threshold is not None else "None",
        )

        # 1. Semantic Retrieval
        retrieval_start = time.perf_counter()
        retrieval_res = self.retrieval_service.retrieve(
            query=stripped_query,
            top_k=top_k,
            document_ids=document_ids,
            score_threshold=score_threshold,
        )
        retrieval_duration = time.perf_counter() - retrieval_start
        chunks = retrieval_res.results

        logger.info(
            "Retrieval completed in %.3fs: candidate_chunks=%d, retrieval_model=%s",
            retrieval_duration,
            len(chunks),
            retrieval_res.model_id,
        )

        # 2. Honest Abstention on Zero Evidence
        # If no chunks were retrieved, do not invoke the generator; return honest abstention immediately.
        if not chunks:
            logger.info("Zero candidate chunks retrieved. Returning honest abstention without calling generator.")
            return QAResponse(
                query=stripped_query,
                answer="I cannot answer this question because no supporting evidence was found.",
                is_abstained=True,
                abstention_reason="no_evidence_found",
                has_citations=False,
                is_valid_citations=True,
                citations=[],
                unknown_citations=[],
                malformed_citations=[],
                retrieved_count=0,
                model_id=self.generation_provider.model_id,
                retrieval_model_id=retrieval_res.model_id,
            )

        # 3. Context Construction
        context = self.context_builder.build_context(chunks)

        # 4. Generation
        gen_start = time.perf_counter()
        gen_result = self.generation_provider.generate_answer(
            query=stripped_query,
            evidence=context.evidence_items,
        )
        gen_duration = time.perf_counter() - gen_start

        logger.info(
            "Generation completed in %.3fs: model_id=%s, is_abstained=%s, reported_citations=%d",
            gen_duration,
            gen_result.model_id,
            gen_result.is_abstained,
            len(gen_result.cited_evidence_ids),
        )

        # 5. Handle Generator Model Abstention
        if gen_result.is_abstained:
            logger.info("Generator model abstained explicitly: reason=%s", gen_result.abstention_reason)
            return QAResponse(
                query=stripped_query,
                answer=gen_result.answer,
                is_abstained=True,
                abstention_reason=gen_result.abstention_reason,
                has_citations=False,
                is_valid_citations=True,
                citations=[],
                unknown_citations=[],
                malformed_citations=[],
                retrieved_count=len(chunks),
                model_id=gen_result.model_id,
                retrieval_model_id=retrieval_res.model_id,
            )

        # 6. Citation Validation
        val_result = self.citation_validator.validate(target=gen_result, context=context)

        logger.info(
            "Citation validation complete: is_valid=%s, has_citations=%s, valid_count=%d, unknown_count=%d, malformed_count=%d",
            val_result.is_valid,
            val_result.has_citations,
            len(val_result.citations),
            len(val_result.unknown_citation_ids),
            len(val_result.malformed_citations),
        )

        # 7. Reject Any Answer with Unknown, Fabricated, or Malformed Citations
        if not val_result.is_valid:
            logger.warning(
                "Citation validation failed: unknown_citations=%s, malformed_citations=%s. Rejecting authoritative answer.",
                val_result.unknown_citation_ids,
                val_result.malformed_citations,
            )
            return QAResponse(
                query=stripped_query,
                answer=(
                    "I cannot provide an authoritative answer because the generated response referenced "
                    "invalid or unverified citations."
                ),
                is_abstained=True,
                abstention_reason="citation_validation_failed",
                has_citations=val_result.has_citations,
                is_valid_citations=False,
                citations=[],
                unknown_citations=val_result.unknown_citation_ids,
                malformed_citations=val_result.malformed_citations,
                retrieved_count=len(chunks),
                model_id=gen_result.model_id,
                retrieval_model_id=retrieval_res.model_id,
            )

        # 8. Reject Non-Abstained Responses Lacking Citations
        if not val_result.has_citations:
            logger.warning("Generated answer contained no supporting citations. Rejecting as ungrounded.")
            return QAResponse(
                query=stripped_query,
                answer="I cannot answer this question because the generated response did not cite supporting evidence.",
                is_abstained=True,
                abstention_reason="missing_citations",
                has_citations=False,
                is_valid_citations=True,
                citations=[],
                unknown_citations=[],
                malformed_citations=[],
                retrieved_count=len(chunks),
                model_id=gen_result.model_id,
                retrieval_model_id=retrieval_res.model_id,
            )

        # 9. Deliver Verified, Grounded Answer
        resolved_citations = [
            QACitation(
                marker=c.citation_marker,
                marker_id=c.marker_id,
                chunk_id=c.chunk_id,
                document_id=c.document_id,
                document_filename=c.document_filename,
                chunk_index=c.chunk_index,
                text=c.text,
                score=c.score,
                page_number=c.page_number,
                page_numbers=list(c.page_numbers),
                heading=c.heading,
            )
            for c in val_result.citations
        ]

        return QAResponse(
            query=stripped_query,
            answer=gen_result.answer,
            is_abstained=False,
            abstention_reason=None,
            has_citations=True,
            is_valid_citations=True,
            citations=resolved_citations,
            unknown_citations=[],
            malformed_citations=[],
            retrieved_count=len(chunks),
            model_id=gen_result.model_id,
            retrieval_model_id=retrieval_res.model_id,
        )
