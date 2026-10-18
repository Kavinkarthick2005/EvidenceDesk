"""FastAPI route dependencies."""

from collections.abc import Generator

from fastapi import Depends
from sqlalchemy.orm import Session

from app.core.config import Settings, get_settings
from app.db.session import get_db
from app.generation import get_generation_provider
from app.services.document import DocumentService
from app.services.qa import QuestionAnsweringService
from app.services.retrieval import RetrievalService


def get_db_session(
    settings: Settings = Depends(get_settings),
) -> Generator[Session, None, None]:
    """FastAPI dependency yielding an isolated database session with resolved settings."""
    yield from get_db(settings=settings)


def get_document_service(
    db: Session = Depends(get_db_session),
    settings: Settings = Depends(get_settings),
) -> DocumentService:
    """FastAPI dependency providing a DocumentService with request-scoped session."""
    return DocumentService(db=db, settings=settings)


def get_retrieval_service(
    db: Session = Depends(get_db_session),
    settings: Settings = Depends(get_settings),
) -> RetrievalService:
    """FastAPI dependency providing a RetrievalService with request-scoped session."""
    return RetrievalService(db=db, settings=settings)


def get_qa_service(
    retrieval_service: RetrievalService = Depends(get_retrieval_service),
    settings: Settings = Depends(get_settings),
) -> QuestionAnsweringService:
    """FastAPI dependency providing a QuestionAnsweringService."""
    return QuestionAnsweringService(
        retrieval_service=retrieval_service,
        generation_provider=get_generation_provider(settings),
        settings=settings,
    )
