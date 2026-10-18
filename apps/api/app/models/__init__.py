"""SQLAlchemy persistence models for EvidenceDesk."""

from app.models.document import Document, DocumentStatus

__all__ = [
    "Document",
    "DocumentStatus",
]
