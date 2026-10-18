"""Ingestion pipeline orchestration service.

Coordinates the end-to-end ingestion lifecycle for documents:
1. Storage retrieval (or direct content resolution).
2. Format-aware text extraction (ExtractionService).
3. Structure-aware token windowing and chunking (StructureAwareChunker).
4. Atomic chunk persistence and replacement (ChunkPersistenceService).
5. Document status progression (UPLOADED -> PROCESSING -> READY / FAILED).
"""

from dataclasses import dataclass
import logging
from typing import Any
import uuid

from sqlalchemy.exc import OperationalError
from sqlalchemy.orm import Session

from app.chunking.exceptions import ChunkingError
from app.chunking.models import ChunkingConfig
from app.chunking.service import StructureAwareChunker
from app.core.config import Settings, get_settings
from app.extraction.exceptions import ExtractionError
from app.extraction.service import ExtractionService, get_extraction_service
from app.models.chunk import Chunk as ChunkModel
from app.models.document import Document, DocumentStatus
from app.services.chunk import ChunkPersistenceService
from app.services.exceptions import (
    ChunkPersistenceError,
    ChunkValidationError,
    DocumentNotFoundError,
    EmptyChunkSetError,
    IngestionChunkingError,
    IngestionExtractionError,
    IngestionPersistenceError,
)
from app.storage import get_storage_service
from app.storage.exceptions import StorageError, StorageFileNotFoundError
from app.storage.interface import StorageService

logger = logging.getLogger(__name__)


@dataclass(frozen=True)
class IngestionResult:
    """Summary result returned upon successful document ingestion."""

    document_id: uuid.UUID
    status: DocumentStatus
    chunk_count: int
    total_tokens: int
    chunks: list[ChunkModel]


class IngestionService:
    """Orchestrates end-to-end ingestion from document bytes to persisted chunks."""

    def __init__(
        self,
        db: Session,
        storage: StorageService | None = None,
        extraction_service: ExtractionService | None = None,
        chunker: StructureAwareChunker | None = None,
        chunk_persistence: ChunkPersistenceService | None = None,
        settings: Settings | None = None,
    ) -> None:
        """Initialize IngestionService with dependencies.

        Args:
            db: Scoped SQLAlchemy database session.
            storage: Optional storage service implementation; defaults to configured singleton.
            extraction_service: Optional extraction service; defaults to shared singleton.
            chunker: Optional structure-aware chunker; defaults to standard instance.
            chunk_persistence: Optional chunk persistence service; defaults to instance bound to db.
            settings: Optional application settings.
        """
        self.db = db
        self.settings = settings or get_settings()
        self.storage = storage or get_storage_service(self.settings)
        self.extraction_service = extraction_service or get_extraction_service()
        self.chunker = chunker or StructureAwareChunker()
        self.chunk_persistence = chunk_persistence or ChunkPersistenceService(db)

    def _normalize_uuid(self, document_id: Any) -> uuid.UUID:
        """Validate and convert an input into a uuid.UUID instance."""
        if isinstance(document_id, uuid.UUID):
            return document_id
        try:
            return uuid.UUID(str(document_id))
        except (ValueError, TypeError) as err:
            raise DocumentNotFoundError(f"Invalid document UUID: '{document_id}'") from err

    def _mark_status(
        self,
        document: Document,
        status: DocumentStatus,
        swallow_errors: bool = False,
    ) -> None:
        """Safely transition document lifecycle status in the database.

        Args:
            document: Target Document model instance.
            status: New DocumentStatus to assign.
            swallow_errors: If True, logs database errors without raising. Used during
                error-handling paths (marking FAILED) to avoid masking the primary exception.
                If False (default), raises IngestionPersistenceError on commit failure.

        Raises:
            IngestionPersistenceError: If database commit fails and swallow_errors is False.
        """
        try:
            document.status = status
            self.db.commit()
            self.db.refresh(document)
        except OperationalError as conn_err:
            try:
                self.db.rollback()
            except Exception:
                pass
            logger.error(
                "Database commit failed with ambiguous operational error while setting document '%s' to %s status: %s",
                document.id,
                status.value,
                conn_err,
            )
            if not swallow_errors:
                raise IngestionPersistenceError(
                    f"Database commit failed with ambiguous connection error while setting document "
                    f"'{document.id}' to {status.value} status: {conn_err}. "
                    f"The transaction outcome is ambiguous and changes may have been committed."
                ) from conn_err
        except Exception as err:
            try:
                self.db.rollback()
            except Exception:
                pass
            logger.error(
                "Failed to transition document '%s' to %s status: %s",
                document.id,
                status.value,
                err,
            )
            if not swallow_errors:
                raise IngestionPersistenceError(
                    f"Failed to transition document '{document.id}' to {status.value} status: {err}"
                ) from err

    def ingest_document(
        self,
        document_id: uuid.UUID | str,
        content: bytes | None = None,
        config: ChunkingConfig | None = None,
    ) -> IngestionResult:
        """Run the end-to-end extraction, chunking, and persistence pipeline on a document.

        Args:
            document_id: Target Document UUID.
            content: Optional raw document bytes. If omitted, bytes are read from storage.
            config: Optional ChunkingConfig override for token limits and overlap.

        Returns:
            IngestionResult containing document ID, final status, chunk count, and chunks.

        Raises:
            DocumentNotFoundError: If the document or its storage file does not exist.
            IngestionExtractionError: If text or structure extraction fails.
            IngestionChunkingError: If partitioning into chunks fails.
            IngestionPersistenceError: If database chunk persistence or replacement fails.
        """
        doc_uuid = self._normalize_uuid(document_id)
        document = self.db.get(Document, doc_uuid)
        if document is None:
            raise DocumentNotFoundError(f"Document with ID '{doc_uuid}' not found.")

        # 1. Transition to PROCESSING state
        self._mark_status(document, DocumentStatus.PROCESSING)

        # 2. Resolve content bytes (from parameter or storage)
        if content is None:
            try:
                content = self.storage.read(document.storage_key)
            except StorageFileNotFoundError as err:
                self._mark_status(document, DocumentStatus.FAILED, swallow_errors=True)
                raise DocumentNotFoundError(
                    f"Underlying storage file for document '{doc_uuid}' was not found: {err}"
                ) from err
            except StorageError as err:
                self._mark_status(document, DocumentStatus.FAILED, swallow_errors=True)
                raise IngestionExtractionError(
                    f"Failed to read stored file content for document '{doc_uuid}': {err}"
                ) from err

        if not isinstance(content, (bytes, bytearray)) or len(content) == 0:
            self._mark_status(document, DocumentStatus.FAILED, swallow_errors=True)
            raise IngestionExtractionError(
                f"Document content for '{doc_uuid}' is empty (0 bytes) or invalid."
            )

        # 3. Extraction phase
        try:
            extraction_result = self.extraction_service.extract(
                content=content,
                content_type=document.content_type,
                filename=document.original_filename,
            )
        except ExtractionError as err:
            self._mark_status(document, DocumentStatus.FAILED, swallow_errors=True)
            raise IngestionExtractionError(
                f"Extraction failed for document '{doc_uuid}': {err}"
            ) from err
        except Exception as err:
            self._mark_status(document, DocumentStatus.FAILED, swallow_errors=True)
            raise IngestionExtractionError(
                f"Unexpected extraction failure for document '{doc_uuid}': {err}"
            ) from err

        # 4. Chunking phase
        try:
            chunking_result = self.chunker.chunk(
                extraction_result,
                document_id=doc_uuid,
                config=config,
            )
        except ChunkingError as err:
            self._mark_status(document, DocumentStatus.FAILED, swallow_errors=True)
            raise IngestionChunkingError(
                f"Chunking failed for document '{doc_uuid}': {err}"
            ) from err
        except Exception as err:
            self._mark_status(document, DocumentStatus.FAILED, swallow_errors=True)
            raise IngestionChunkingError(
                f"Unexpected chunking failure for document '{doc_uuid}': {err}"
            ) from err

        # 5. Persistence phase (atomic replacement of prior chunk set)
        try:
            persisted_chunks = self.chunk_persistence.store_chunks(
                document_id=doc_uuid,
                chunks=chunking_result,
                replace=True,
            )
        except (ChunkValidationError, ChunkPersistenceError, EmptyChunkSetError) as err:
            self._mark_status(document, DocumentStatus.FAILED, swallow_errors=True)
            raise IngestionPersistenceError(
                f"Chunk persistence failed for document '{doc_uuid}': {err}"
            ) from err
        except Exception as err:
            self._mark_status(document, DocumentStatus.FAILED, swallow_errors=True)
            raise IngestionPersistenceError(
                f"Unexpected persistence failure for document '{doc_uuid}': {err}"
            ) from err

        # 6. Success: Transition to READY state
        self._mark_status(document, DocumentStatus.READY)

        total_tokens = sum(c.token_count for c in persisted_chunks)
        return IngestionResult(
            document_id=doc_uuid,
            status=DocumentStatus.READY,
            chunk_count=len(persisted_chunks),
            total_tokens=total_tokens,
            chunks=persisted_chunks,
        )
