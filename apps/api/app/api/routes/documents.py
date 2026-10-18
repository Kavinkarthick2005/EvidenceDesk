"""Document API route handlers."""

import uuid

from fastapi import APIRouter, Depends, File, Path, UploadFile, status
from starlette.concurrency import run_in_threadpool

from app.api.deps import get_document_service
from app.core.config import Settings, get_settings
from app.schemas.document import DocumentDeleteResponse, DocumentResponse
from app.services.document import DocumentService
from app.services.exceptions import DocumentSizeLimitError

router = APIRouter(prefix="/documents", tags=["documents"])


@router.post(
    "",
    response_model=DocumentResponse,
    status_code=status.HTTP_201_CREATED,
    summary="Upload and register a document",
    description=(
        "Accepts a multipart file upload, validates format and upload-size limits, "
        "persists the raw file in storage, and records document metadata in the database."
    ),
)
async def upload_document(
    file: UploadFile = File(
        ...,
        description="Document file to upload (PDF, Markdown, or plain text)",
    ),
    service: DocumentService = Depends(get_document_service),
    settings: Settings = Depends(get_settings),
) -> DocumentResponse:
    """Handle multipart document upload with size limit enforcement."""
    max_bytes = settings.MAX_UPLOAD_SIZE_MB * 1024 * 1024

    # Stream file into memory bounded by max_bytes to prevent OOM DoS
    content = bytearray()
    while chunk := await file.read(64 * 1024):
        content.extend(chunk)
        if len(content) > max_bytes:
            raise DocumentSizeLimitError(
                f"Document size exceeds configured maximum of {max_bytes} bytes "
                f"({settings.MAX_UPLOAD_SIZE_MB} MB)."
            )

    filename = file.filename or ""
    content_type = file.content_type or ""

    # Execute synchronous storage and database persistence in threadpool
    document = await run_in_threadpool(
        service.create_document,
        content=bytes(content),
        original_filename=filename,
        content_type=content_type,
    )

    return DocumentResponse.model_validate(document)


@router.get(
    "/{document_id}",
    response_model=DocumentResponse,
    status_code=status.HTTP_200_OK,
    summary="Retrieve document metadata",
    description="Retrieves persisted metadata for a document by its UUID.",
)
def get_document(
    document_id: uuid.UUID = Path(
        ...,
        description="Unique UUID identifier of the document",
    ),
    service: DocumentService = Depends(get_document_service),
) -> DocumentResponse:
    """Retrieve document metadata without exposing internal filesystem paths."""
    document = service.get_document(document_id)
    return DocumentResponse.model_validate(document)


@router.delete(
    "/{document_id}",
    response_model=DocumentDeleteResponse,
    status_code=status.HTTP_200_OK,
    summary="Delete a document",
    description="Removes the document database record and unlinks its underlying file from storage.",
)
def delete_document(
    document_id: uuid.UUID = Path(
        ...,
        description="Unique UUID identifier of the document to delete",
    ),
    service: DocumentService = Depends(get_document_service),
) -> DocumentDeleteResponse:
    """Delete document record and associated file storage."""
    service.delete_document(document_id)
    return DocumentDeleteResponse(
        id=document_id,
        status="deleted",
        message="Document deleted successfully",
    )
