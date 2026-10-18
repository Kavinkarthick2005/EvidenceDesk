"""Question Answering (QA) API route handlers."""

from fastapi import APIRouter, Depends, status
from starlette.concurrency import run_in_threadpool

from app.api.deps import get_qa_service
from app.schemas.qa import QARequest, QAResponse
from app.services.qa import QuestionAnsweringService

router = APIRouter(prefix="/qa", tags=["qa"])


@router.post(
    "/answer",
    response_model=QAResponse,
    status_code=status.HTTP_200_OK,
    summary="Answer a question using grounded document evidence",
    description=(
        "Accepts a natural language question, retrieves relevant passage chunks, "
        "generates an evidence-grounded answer, and validates citation integrity. "
        "Returns an explicit abstention if evidence is missing or citations are invalid."
    ),
)
async def answer_question(
    request: QARequest,
    service: QuestionAnsweringService = Depends(get_qa_service),
) -> QAResponse:
    """Handle question answering request with evidence retrieval and citation verification."""
    return await run_in_threadpool(
        service.answer_question,
        query=request.query,
        top_k=request.top_k,
        document_ids=request.document_ids,
        score_threshold=request.score_threshold,
    )
