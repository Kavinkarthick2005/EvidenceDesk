"""FastAPI route dependencies."""

from fastapi import Depends
from sqlalchemy.orm import Session

from app.core.config import Settings, get_settings
from app.db.session import get_db
from app.services.document import DocumentService


def get_document_service(
    db: Session = Depends(get_db),
    settings: Settings = Depends(get_settings),
) -> DocumentService:
    """FastAPI dependency providing a DocumentService with request-scoped session."""
    return DocumentService(db=db, settings=settings)
