"""Document service coordinating local file storage and PostgreSQL persistence.

Architecture Note on Dual-Write Atomicity & Consistency:
The local filesystem and PostgreSQL do not share a single distributed or two-phase
transaction. Therefore:
1. On creation: The file is written to storage first. If the subsequent database
   insert/commit fails, the service immediately rolls back the database transaction
   and attempts to clean up the newly created storage object. If storage cleanup also
   fails, the error is logged and reported without masking the root database error.
2. On deletion: The database record is removed and committed first so the catalog
   never falsely advertises a missing document. Then the storage object is unlinked.
   If storage unlinking fails, a DocumentStorageError is raised and the unlinked file
   remains as an unreferenced orphan. Retries will raise DocumentNotFoundError.
   Missing storage files are handled idempotently.
3. Transaction ownership: The service explicitly manages transaction commits and
   rollbacks on the supplied SQLAlchemy session. Callers must supply a dedicated
   request-scoped session without unrelated uncommitted state.
"""

import hashlib
import logging
from pathlib import Path
import uuid

from sqlalchemy.exc import OperationalError, SQLAlchemyError
from sqlalchemy.orm import Session

from app.core.config import Settings, get_settings
from app.models.document import Document, DocumentStatus
from app.services.exceptions import (
    DocumentNotFoundError,
    DocumentPersistenceError,
    DocumentSizeLimitError,
    DocumentStorageError,
    DocumentValidationError,
    EmptyDocumentError,
    UnsupportedDocumentTypeError,
)
from app.storage import get_storage_service
from app.storage.exceptions import (
    StorageError,
    StorageFileNotFoundError,
    StorageSizeLimitError,
    StorageUnsupportedTypeError,
)
from app.storage.interface import StorageService

logger = logging.getLogger(__name__)


class DocumentService:
    """Application service managing document ingestion persistence and lifecycle."""

    ALLOWED_MIME_TYPES: dict[str, str] = {
        "application/pdf": ".pdf",
        "text/markdown": ".md",
        "text/plain": ".txt",
        "text/x-markdown": ".md",
    }

    ALLOWED_EXTENSIONS: dict[str, str] = {
        ".pdf": ".pdf",
        ".md": ".md",
        ".markdown": ".md",
        ".txt": ".txt",
    }

    def __init__(
        self,
        db: Session,
        storage: StorageService | None = None,
        settings: Settings | None = None,
    ) -> None:
        """Initialize DocumentService with database session and storage backend.

        Args:
            db: Scoped SQLAlchemy database session.
            storage: Optional StorageService implementation; defaults to configured singleton.
            settings: Optional application Settings instance.
        """
        self.db = db
        self.settings = settings or get_settings()
        self.storage = storage or get_storage_service(self.settings)
        self.max_size_bytes = self.settings.MAX_UPLOAD_SIZE_MB * 1024 * 1024

    def _validate_and_normalize_format(
        self,
        content_type: str | None,
        filename: str,
    ) -> str:
        """Validate content type and filename against permitted document formats.

        Note on Content Inspection:
            This validation verifies client-provided MIME metadata and filename
            extensions only. It does not inspect the actual byte stream for magic
            numbers, embedded scripts, or valid file structure (which is deferred
            to the document parsing/extraction stage).

        Returns:
            Normalized MIME type string.

        Raises:
            UnsupportedDocumentTypeError: If content type or extension is unsupported.
            DocumentValidationError: If metadata is malformed or exceeds length limits.
        """
        if content_type and len(content_type) > 100:
            raise DocumentValidationError(
                f"Content-type string exceeds maximum length of 100 characters (got {len(content_type)})."
            )

        # 1. Missing or whitespace-only content-type: infer from filename extension
        if not content_type or not content_type.strip():
            ext = Path(filename).suffix.lower()
            if ext in self.ALLOWED_EXTENSIONS:
                for mime, mapped_ext in self.ALLOWED_MIME_TYPES.items():
                    if mapped_ext == ext:
                        return mime
            raise UnsupportedDocumentTypeError(
                f"Missing content type and unsupported or missing file extension '{ext}'. "
                f"Allowed formats: PDF, Markdown, plain text."
            )

        # 2. Provided content-type: normalize parameters and casing
        normalized_mime = content_type.split(";")[0].strip().lower()
        if normalized_mime not in self.ALLOWED_MIME_TYPES:
            raise UnsupportedDocumentTypeError(
                f"Unsupported format: content-type '{content_type}' for filename '{filename}'. "
                f"Allowed formats: PDF, Markdown, plain text."
            )

        # 3. If filename has an extension, verify it is in allowed extensions
        ext = Path(filename).suffix.lower()
        if ext and ext not in self.ALLOWED_EXTENSIONS:
            raise UnsupportedDocumentTypeError(
                f"Unsupported format: file extension '{ext}' for filename '{filename}'. "
                f"Allowed formats: PDF, Markdown, plain text."
            )

        return normalized_mime

    def create_document(
        self,
        content: bytes,
        original_filename: str,
        content_type: str,
    ) -> Document:
        """Validate, store, and record a new document.

        Coordinates local file persistence with database record creation.
        Guarantees that a failed database transaction triggers cleanup of the
        file written to storage for this request.

        Transaction Ownership:
            This method manages the transaction of the provided database session.
            It calls `commit()` on success and `rollback()` on failure. Callers must
            provide a dedicated request-scoped session and must not stage unrelated
            mutations on the session prior to calling this method.

        Args:
            content: Raw document bytes.
            original_filename: Client display filename (max 255 chars).
            content_type: Supplied MIME media type (max 100 chars).

        Returns:
            The persisted Document ORM instance.

        Raises:
            EmptyDocumentError: If content or filename is empty.
            DocumentSizeLimitError: If content exceeds MAX_UPLOAD_SIZE_MB.
            UnsupportedDocumentTypeError: If format is not PDF, Markdown, or plain text.
            DocumentValidationError: If input metadata contains illegal characters or exceeds length limits.
            DocumentPersistenceError: If database commit fails.
            DocumentStorageError: If storage write fails.
        """
        # 1. Validate filename
        if not isinstance(original_filename, str) or not original_filename.strip():
            raise EmptyDocumentError("Original filename cannot be empty or whitespace-only.")

        if "\0" in original_filename:
            raise DocumentValidationError("Original filename contains invalid null characters.")

        cleaned_filename = original_filename.strip()
        if len(cleaned_filename) > 255:
            raise DocumentValidationError(
                f"Original filename exceeds maximum length of 255 characters (got {len(cleaned_filename)})."
            )

        # 2. Validate payload bytes
        if not isinstance(content, (bytes, bytearray)):
            raise DocumentValidationError("Document content must be raw bytes.")

        if len(content) == 0:
            raise EmptyDocumentError("Document content cannot be empty (0 bytes).")

        if len(content) > self.max_size_bytes:
            raise DocumentSizeLimitError(
                f"Document size {len(content)} bytes exceeds configured maximum of "
                f"{self.max_size_bytes} bytes ({self.settings.MAX_UPLOAD_SIZE_MB} MB)."
            )

        # 3. Validate format
        normalized_mime = self._validate_and_normalize_format(content_type, cleaned_filename)

        # 4. Calculate hash and byte count from actual bytes
        sha256_hash = hashlib.sha256(content).hexdigest()
        size_bytes = len(content)

        # 5. Store file in storage service
        try:
            storage_key = self.storage.save(
                content,
                content_type=normalized_mime,
                original_filename=cleaned_filename,
            )
        except StorageSizeLimitError as err:
            raise DocumentSizeLimitError(str(err)) from err
        except StorageUnsupportedTypeError as err:
            raise UnsupportedDocumentTypeError(str(err)) from err
        except StorageError as err:
            raise DocumentStorageError(f"Failed to store file content: {err}") from err

        # 6. Create database record
        doc_id = uuid.uuid4()
        document = Document(
            id=doc_id,
            original_filename=cleaned_filename,
            content_type=normalized_mime,
            size_bytes=size_bytes,
            sha256=sha256_hash,
            storage_key=storage_key,
            status=DocumentStatus.UPLOADED,
        )

        self.db.add(document)

        # 7. Database persistence with two-phase error distinction
        # Phase A: Flush to evaluate schema constraints and database triggers.
        # If flush fails (e.g. constraints, types), the transaction is guaranteed
        # aborted by PostgreSQL and was never committed. Storage cleanup is safe.
        try:
            self.db.flush()
        except Exception as flush_err:
            try:
                self.db.rollback()
            except Exception:
                pass

            try:
                self.storage.delete(storage_key)
            except Exception as cleanup_err:
                logger.error(
                    "Storage cleanup failed for newly created key '%s' following DB flush failure: %s",
                    storage_key,
                    cleanup_err,
                )
                raise DocumentPersistenceError(
                    f"Database persistence failed during flush: {flush_err}. "
                    f"Additionally, subsequent cleanup of storage file '{storage_key}' failed: {cleanup_err}"
                ) from flush_err

            raise DocumentPersistenceError(
                f"Database persistence failed during flush: {flush_err}"
            ) from flush_err

        # Phase B: Issue COMMIT.
        # If commit raises an OperationalError (connection severed, timeout, etc.),
        # the commit outcome is AMBIGUOUS: PostgreSQL may have committed the transaction
        # before the connection dropped. Deleting the storage file in this case risks
        # deleting the file for an active, committed document record.
        # Therefore, we preserve the storage file and fail closed against data loss.
        try:
            self.db.commit()
            self.db.refresh(document)
            return document
        except OperationalError as conn_err:
            try:
                self.db.rollback()
            except Exception:
                pass

            logger.error(
                "Database commit failed with ambiguous operational error for doc '%s', key '%s'. "
                "Preserving storage file to avoid deleting content for a potentially committed row: %s",
                doc_id,
                storage_key,
                conn_err,
            )
            raise DocumentPersistenceError(
                f"Database commit failed with ambiguous connection error: {conn_err}. "
                f"Storage file '{storage_key}' has been preserved to prevent potential data loss."
            ) from conn_err
        except Exception as commit_err:
            try:
                self.db.rollback()
            except Exception:
                pass

            # Non-operational commit failures where connection is intact:
            try:
                self.storage.delete(storage_key)
            except Exception as cleanup_err:
                logger.error(
                    "Storage cleanup failed for newly created key '%s' following DB commit failure: %s",
                    storage_key,
                    cleanup_err,
                )
                raise DocumentPersistenceError(
                    f"Database persistence failed during commit: {commit_err}. "
                    f"Additionally, subsequent cleanup of storage file '{storage_key}' failed: {cleanup_err}"
                ) from commit_err

            raise DocumentPersistenceError(
                f"Database persistence failed during commit: {commit_err}"
            ) from commit_err

    def get_document(self, document_id: uuid.UUID) -> Document:
        """Retrieve a document database record by its UUID.

        Args:
            document_id: UUID of the document.

        Returns:
            The Document ORM instance.

        Raises:
            DocumentNotFoundError: If the document record does not exist.
        """
        if not isinstance(document_id, uuid.UUID):
            try:
                document_id = uuid.UUID(str(document_id))
            except (ValueError, TypeError) as err:
                raise DocumentNotFoundError(f"Invalid document UUID: '{document_id}'") from err

        document = self.db.get(Document, document_id)
        if document is None:
            raise DocumentNotFoundError(f"Document with ID '{document_id}' not found.")
        return document

    def read_document_content(self, document_id: uuid.UUID) -> bytes:
        """Retrieve the raw stored file bytes for a document by its ID.

        Args:
            document_id: UUID of the document.

        Returns:
            The raw byte content of the document.

        Raises:
            DocumentNotFoundError: If the document or underlying file is missing.
            DocumentStorageError: If reading the file fails.
        """
        document = self.get_document(document_id)

        try:
            return self.storage.read(document.storage_key)
        except StorageFileNotFoundError as err:
            raise DocumentNotFoundError(
                f"Underlying storage file for document '{document_id}' was not found."
            ) from err
        except StorageError as err:
            raise DocumentStorageError(
                f"Failed to read file for document '{document_id}': {err}"
            ) from err

    def delete_document(self, document_id: uuid.UUID) -> None:
        """Delete a document record from the database and unlink its underlying file.

        Consistency Strategy & Orphan-File Risk:
            The database record is deleted and committed first to ensure the catalog
            does not falsely advertise an unavailable file. Missing storage files are
            handled idempotently.

            If file deletion subsequently fails (e.g. disk or permission error), a
            DocumentStorageError is raised. In this scenario, the file remains on
            disk as an unreferenced orphan with an unguessable UUID-based storage key.
            If a caller retries delete_document, it will receive DocumentNotFoundError
            because the database record was already committed as deleted.

            This represents the smallest safe strategy for Milestone 4B: catalog truth
            is preserved without introducing background workers or distributed state machines.
            The error is logged with the storage key for administrative/reconciliation auditing.

        Transaction Ownership:
            This method manages the transaction of the provided database session.
            It calls `commit()` on success and `rollback()` on failure. Callers must
            supply a dedicated request-scoped session.

        Args:
            document_id: UUID of the document to delete.

        Raises:
            DocumentNotFoundError: If the document does not exist.
            DocumentPersistenceError: If deleting the database record fails.
            DocumentStorageError: If removing the file from storage fails.
        """
        document = self.get_document(document_id)
        storage_key = document.storage_key

        # 1. Delete database record first
        try:
            self.db.delete(document)
            self.db.commit()
        except Exception as db_err:
            try:
                self.db.rollback()
            except Exception:
                pass
            raise DocumentPersistenceError(
                f"Failed to delete database record for document '{document_id}': {db_err}"
            ) from db_err

        # 2. Delete file from storage
        try:
            self.storage.delete(storage_key)
        except StorageFileNotFoundError:
            # File was already missing; treat deletion as idempotent
            pass
        except StorageError as storage_err:
            logger.error(
                "Document record '%s' was deleted, but storage deletion of key '%s' failed: %s",
                document_id,
                storage_key,
                storage_err,
            )
            raise DocumentStorageError(
                f"Document record '{document_id}' was removed, but failed to delete file from storage: {storage_err}"
            ) from storage_err
