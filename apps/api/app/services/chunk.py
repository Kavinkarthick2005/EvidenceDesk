"""Chunk persistence service coordinating PostgreSQL storage for document chunks.

Architecture and Transaction Ownership:
- Transaction ownership: This service manages transaction commits and rollbacks
  on the supplied SQLAlchemy session. Callers must supply a dedicated request-scoped
  session without unrelated uncommitted state.
- Atomic replacement: When storing chunks with `replace=True` (the default), existing
  chunks for the document are deleted and new chunks are inserted within a single
  database transaction. If an error occurs prior to commit, the transaction is rolled
  back, preserving the previous committed chunk set intact.
- Two-phase commit distinction: Pre-commit flush errors (which guarantee the transaction
  was aborted by the database) are distinguished from commit-time operational errors
  (which represent ambiguous outcomes where the transaction may have been committed
  before connection loss).
- Safety against accidental deletion: Empty chunk inputs are rejected by default with
  EmptyChunkSetError to prevent accidental truncation of previously stored chunks due to
  malformed, incomplete, or truncated parser results.
"""

from collections.abc import Sequence
import logging
from typing import Any
import uuid

import sqlalchemy as sa
from sqlalchemy.exc import OperationalError
from sqlalchemy.orm import Session

from app.chunking.models import Chunk as DomainChunk, ChunkingResult
from app.models.chunk import Chunk as ChunkModel
from app.models.document import Document
from app.services.exceptions import (
    ChunkPersistenceError,
    ChunkValidationError,
    DocumentNotFoundError,
    EmptyChunkSetError,
)

logger = logging.getLogger(__name__)


class ChunkPersistenceService:
    """Application service managing chunk persistence and lifecycle for documents."""

    def __init__(self, db: Session) -> None:
        """Initialize ChunkPersistenceService with a database session.

        Args:
            db: Scoped SQLAlchemy database session.
        """
        self.db = db

    def _normalize_uuid(self, document_id: Any) -> uuid.UUID:
        """Validate and convert an input into a uuid.UUID instance."""
        if isinstance(document_id, uuid.UUID):
            return document_id
        try:
            return uuid.UUID(str(document_id))
        except (ValueError, TypeError) as err:
            raise DocumentNotFoundError(f"Invalid document UUID: '{document_id}'") from err

    def _ensure_document_exists(self, document_id: uuid.UUID) -> Document:
        """Verify the parent document exists in the database.

        Raises:
            DocumentNotFoundError: If the document is not found.
        """
        document = self.db.get(Document, document_id)
        if document is None:
            raise DocumentNotFoundError(f"Document with ID '{document_id}' not found.")
        return document

    def _extract_chunk_list(
        self,
        chunks: Sequence[DomainChunk | ChunkModel | dict[str, Any]] | ChunkingResult,
    ) -> list[DomainChunk | ChunkModel | dict[str, Any]]:
        """Extract a standard list from various input representations."""
        if isinstance(chunks, ChunkingResult):
            return list(chunks.chunks)
        return list(chunks)

    def _validate_chunk_data(
        self,
        target_document_id: uuid.UUID,
        raw_chunks: list[DomainChunk | ChunkModel | dict[str, Any]],
    ) -> list[dict[str, Any]]:
        """Validate and normalize a list of chunk data before database execution.

        Enforces:
        - Non-empty text.
        - Exact character_count matching len(text).
        - Non-negative token_count.
        - Valid offsets: start_char_offset >= 0, end_char_offset >= start_char_offset.
        - Exact offset span: end_char_offset - start_char_offset == character_count.
        - Consecutive zero-based chunk_index sequence (0, 1, ..., N-1).
        - Matching document_id if specified.
        - Positive page_number and consistency with page_numbers list.
        - Heading level constrained to 1 through 6 inclusive if specified.

        Returns:
            List of normalized chunk attribute dictionaries ready for ORM instantiation.

        Raises:
            ChunkValidationError: If any chunk fails validation or ordering checks.
        """
        validated_items: list[dict[str, Any]] = []

        for idx, item in enumerate(raw_chunks):
            if isinstance(item, DomainChunk):
                data = item.model_dump()
            elif isinstance(item, ChunkModel):
                data = {
                    "id": item.id,
                    "document_id": item.document_id,
                    "chunk_index": item.chunk_index,
                    "text": item.text,
                    "token_count": item.token_count,
                    "character_count": item.character_count,
                    "start_char_offset": item.start_char_offset,
                    "end_char_offset": item.end_char_offset,
                    "source_format": item.source_format,
                    "page_number": item.page_number,
                    "page_numbers": list(item.page_numbers or []),
                    "heading": item.heading,
                    "heading_level": item.heading_level,
                    "metadata": dict(item.metadata_ or {}),
                }
            elif isinstance(item, dict):
                data = dict(item)
            else:
                raise ChunkValidationError(
                    f"Chunk item at index {idx} has invalid type '{type(item).__name__}'."
                )

            # 1. Text validation
            text = data.get("text")
            if not isinstance(text, str) or len(text) == 0:
                raise ChunkValidationError(
                    f"Chunk at position {idx} has empty or non-string text."
                )

            # 2. Character count validation
            char_count = data.get("character_count")
            if not isinstance(char_count, int) or char_count < 0:
                raise ChunkValidationError(
                    f"Chunk at position {idx} has invalid character_count '{char_count}'."
                )
            if char_count != len(text):
                raise ChunkValidationError(
                    f"Chunk at position {idx} has character_count ({char_count}) inconsistent "
                    f"with text length ({len(text)})."
                )

            # 3. Token count validation
            token_count = data.get("token_count")
            if not isinstance(token_count, int) or token_count < 0:
                raise ChunkValidationError(
                    f"Chunk at position {idx} has invalid token_count '{token_count}'."
                )

            # 4. Offset validation
            start_offset = data.get("start_char_offset")
            end_offset = data.get("end_char_offset")
            if not isinstance(start_offset, int) or start_offset < 0:
                raise ChunkValidationError(
                    f"Chunk at position {idx} has invalid start_char_offset '{start_offset}'."
                )
            if not isinstance(end_offset, int) or end_offset < start_offset:
                raise ChunkValidationError(
                    f"Chunk at position {idx} has invalid end_char_offset '{end_offset}' "
                    f"(must be >= start_char_offset {start_offset})."
                )
            if (end_offset - start_offset) != char_count:
                raise ChunkValidationError(
                    f"Chunk at position {idx} has offset span ({end_offset} - {start_offset} = "
                    f"{end_offset - start_offset}) inconsistent with character_count ({char_count})."
                )

            # 5. Document ID consistency
            item_doc_id = data.get("document_id")
            if item_doc_id is not None:
                parsed_item_doc_id = self._normalize_uuid(item_doc_id)
                if parsed_item_doc_id != target_document_id:
                    raise ChunkValidationError(
                        f"Chunk at position {idx} has document_id '{parsed_item_doc_id}' "
                        f"which does not match target document_id '{target_document_id}'."
                    )

            # 6. Structural metadata checks (page numbers and headings)
            page_num = data.get("page_number")
            if page_num is not None and (not isinstance(page_num, int) or page_num < 1):
                raise ChunkValidationError(
                    f"Chunk at position {idx} has invalid page_number '{page_num}' (must be >= 1)."
                )

            page_nums = data.get("page_numbers") or []
            if not isinstance(page_nums, list) or any(not isinstance(p, int) or p < 1 for p in page_nums):
                raise ChunkValidationError(
                    f"Chunk at position {idx} has invalid page_numbers list '{page_nums}'."
                )

            if page_num is not None and len(page_nums) > 0 and page_num not in page_nums:
                raise ChunkValidationError(
                    f"Chunk at position {idx} has primary page_number {page_num} not present "
                    f"in page_numbers list {page_nums}."
                )

            heading_lvl = data.get("heading_level")
            if heading_lvl is not None and (not isinstance(heading_lvl, int) or not (1 <= heading_lvl <= 6)):
                raise ChunkValidationError(
                    f"Chunk at position {idx} has invalid heading_level '{heading_lvl}' "
                    f"(must be an integer from 1 to 6 inclusive)."
                )

            validated_items.append(
                {
                    "id": data.get("id") or uuid.uuid4(),
                    "document_id": target_document_id,
                    "chunk_index": data.get("chunk_index"),
                    "text": text,
                    "token_count": token_count,
                    "character_count": char_count,
                    "start_char_offset": start_offset,
                    "end_char_offset": end_offset,
                    "source_format": str(data.get("source_format") or "text/plain"),
                    "page_number": page_num,
                    "page_numbers": page_nums,
                    "heading": data.get("heading"),
                    "heading_level": heading_lvl,
                    "metadata": data.get("metadata") or {},
                }
            )

        # 7. Check index ordering: must be consecutive 0, 1, ..., N-1
        sorted_items = sorted(
            validated_items,
            key=lambda x: x["chunk_index"] if isinstance(x["chunk_index"], int) else -1,
        )
        for expected_idx, item in enumerate(sorted_items):
            actual_idx = item["chunk_index"]
            if actual_idx != expected_idx:
                raise ChunkValidationError(
                    f"Chunks must form a consecutive zero-based index sequence from 0 to {len(sorted_items) - 1}; "
                    f"expected index {expected_idx} but encountered index {actual_idx}."
                )

        return sorted_items

    def store_chunks(
        self,
        document_id: uuid.UUID | str,
        chunks: Sequence[DomainChunk | ChunkModel | dict[str, Any]] | ChunkingResult,
        replace: bool = True,
        allow_empty: bool = False,
        commit: bool = True,
    ) -> list[ChunkModel]:
        """Validate and persist chunks for a document atomically.

        Args:
            document_id: Target Document UUID.
            chunks: Sequence of Chunk models/dicts or a ChunkingResult container.
            replace: If True (default), atomically removes previous chunks for this document.
            allow_empty: If True, allows persisting an empty chunk set (clearing previous chunks).
                         Defaults to False to prevent accidental loss of chunks.
            commit: If True (default), commits transaction and refreshes ORM instances.
                    If False, stages changes and flushes without committing or refreshing,
                    enabling coordinated atomic persistence with embedding records.

        Returns:
            List of persisted Chunk ORM instances in ascending chunk_index order.

        Raises:
            DocumentNotFoundError: If the target document does not exist.
            EmptyChunkSetError: If chunks is empty and allow_empty is False.
            ChunkValidationError: If any chunk fails validation or ordering checks.
            ChunkPersistenceError: If database persistence or commit fails.
        """
        doc_uuid = self._normalize_uuid(document_id)
        self._ensure_document_exists(doc_uuid)

        chunk_list = self._extract_chunk_list(chunks)

        if len(chunk_list) == 0:
            if not allow_empty:
                raise EmptyChunkSetError(
                    f"Chunk set for document '{doc_uuid}' cannot be empty. "
                    f"To explicitly remove all chunks, set allow_empty=True."
                )
            if replace:
                try:
                    self.db.execute(sa.delete(ChunkModel).where(ChunkModel.document_id == doc_uuid))
                    if commit:
                        self.db.commit()
                    else:
                        self.db.flush()
                except OperationalError as conn_err:
                    try:
                        self.db.rollback()
                    except Exception:
                        pass
                    logger.error(
                        "Database commit failed with ambiguous operational error while clearing chunks for doc '%s': %s",
                        doc_uuid,
                        conn_err,
                    )
                    raise ChunkPersistenceError(
                        f"Database commit failed with ambiguous connection error while clearing chunks: {conn_err}."
                    ) from conn_err
                except Exception as err:
                    try:
                        self.db.rollback()
                    except Exception:
                        pass
                    raise ChunkPersistenceError(
                        f"Failed to clear existing chunks for document '{doc_uuid}': {err}"
                    ) from err
            return []

        # Validate chunk contents and ordering before making any database modifications
        validated_chunks = self._validate_chunk_data(doc_uuid, chunk_list)

        # Phase A: Mutate and flush within a single transaction.
        # Deleting old chunk set, staging new chunks, and evaluating constraints/triggers.
        # If any failure occurs during delete, add, or flush, rollback preserves intact prior state.
        try:
            if replace:
                self.db.execute(sa.delete(ChunkModel).where(ChunkModel.document_id == doc_uuid))

            orm_chunks: list[ChunkModel] = []
            for item in validated_chunks:
                chunk_record = ChunkModel(
                    id=item["id"],
                    document_id=item["document_id"],
                    chunk_index=item["chunk_index"],
                    text=item["text"],
                    token_count=item["token_count"],
                    character_count=item["character_count"],
                    start_char_offset=item["start_char_offset"],
                    end_char_offset=item["end_char_offset"],
                    source_format=item["source_format"],
                    page_number=item["page_number"],
                    page_numbers=item["page_numbers"],
                    heading=item["heading"],
                    heading_level=item["heading_level"],
                    metadata_=item["metadata"],
                )
                self.db.add(chunk_record)
                orm_chunks.append(chunk_record)

            self.db.flush()
        except Exception as flush_err:
            try:
                self.db.rollback()
            except Exception:
                pass
            logger.error(
                "Chunk persistence failed during pre-commit execution for document '%s'; transaction rolled back: %s",
                doc_uuid,
                flush_err,
            )
            raise ChunkPersistenceError(
                f"Database persistence failed during flush for document '{doc_uuid}': {flush_err}"
            ) from flush_err

        # If staging mode is requested, return flushed instances without committing or refreshing
        if not commit:
            return orm_chunks

        # Phase B: Issue COMMIT.
        # If commit raises OperationalError, the outcome is ambiguous (connection dropped
        # before confirmation arrived, but PostgreSQL may have committed the transaction).
        try:
            self.db.commit()
        except OperationalError as conn_err:
            try:
                self.db.rollback()
            except Exception:
                pass
            logger.error(
                "Database commit failed with ambiguous operational error for doc '%s': %s",
                doc_uuid,
                conn_err,
            )
            raise ChunkPersistenceError(
                f"Database commit failed with ambiguous connection error for document '{doc_uuid}': {conn_err}. "
                f"The transaction outcome is ambiguous and changes may have been committed."
            ) from conn_err
        except Exception as commit_err:
            try:
                self.db.rollback()
            except Exception:
                pass
            logger.error(
                "Database commit failed for document '%s'; transaction rolled back: %s",
                doc_uuid,
                commit_err,
            )
            raise ChunkPersistenceError(
                f"Database persistence failed during commit for document '{doc_uuid}': {commit_err}"
            ) from commit_err

        # Phase C: Post-commit refresh.
        # The transaction has already committed successfully. Do not fail the operation or
        # roll back if refreshing an ORM attribute fails.
        for chunk_record in orm_chunks:
            try:
                self.db.refresh(chunk_record)
            except Exception as refresh_err:
                logger.warning(
                    "Post-commit refresh failed for chunk '%s' (document '%s'): %s. "
                    "The chunk record was already committed successfully.",
                    chunk_record.id,
                    doc_uuid,
                    refresh_err,
                )

        return orm_chunks

    def get_chunks(self, document_id: uuid.UUID | str) -> list[ChunkModel]:
        """Retrieve all persisted chunks for a document in ascending chunk_index order.

        Args:
            document_id: UUID of the parent document.

        Returns:
            List of Chunk ORM instances ordered by chunk_index ASC.

        Raises:
            DocumentNotFoundError: If the document does not exist.
        """
        doc_uuid = self._normalize_uuid(document_id)
        self._ensure_document_exists(doc_uuid)

        statement = (
            sa.select(ChunkModel)
            .where(ChunkModel.document_id == doc_uuid)
            .order_by(ChunkModel.chunk_index.asc())
        )
        return list(self.db.scalars(statement).all())

    def get_chunk_count(self, document_id: uuid.UUID | str) -> int:
        """Count the total number of persisted chunks for a document.

        Args:
            document_id: UUID of the parent document.

        Returns:
            Total number of chunks.

        Raises:
            DocumentNotFoundError: If the document does not exist.
        """
        doc_uuid = self._normalize_uuid(document_id)
        self._ensure_document_exists(doc_uuid)

        statement = (
            sa.select(sa.func.count())
            .select_from(ChunkModel)
            .where(ChunkModel.document_id == doc_uuid)
        )
        return self.db.scalar(statement) or 0

    def delete_chunks(self, document_id: uuid.UUID | str) -> int:
        """Delete all chunks for a document and commit the change.

        Args:
            document_id: UUID of the parent document.

        Returns:
            Count of chunks deleted.

        Raises:
            DocumentNotFoundError: If the document does not exist.
            ChunkPersistenceError: If database delete/commit fails.
        """
        doc_uuid = self._normalize_uuid(document_id)
        self._ensure_document_exists(doc_uuid)

        try:
            result = self.db.execute(
                sa.delete(ChunkModel).where(ChunkModel.document_id == doc_uuid)
            )
            self.db.commit()
            return result.rowcount  # type: ignore[return-value]
        except OperationalError as conn_err:
            try:
                self.db.rollback()
            except Exception:
                pass
            logger.error(
                "Database commit failed with ambiguous operational error while deleting chunks for doc '%s': %s",
                doc_uuid,
                conn_err,
            )
            raise ChunkPersistenceError(
                f"Database commit failed with ambiguous connection error while deleting chunks: {conn_err}."
            ) from conn_err
        except Exception as err:
            try:
                self.db.rollback()
            except Exception:
                pass
            raise ChunkPersistenceError(
                f"Failed to delete chunks for document '{doc_uuid}': {err}"
            ) from err
