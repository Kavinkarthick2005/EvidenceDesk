"""Pydantic schemas for Document API endpoints."""

from datetime import datetime
import uuid

from pydantic import BaseModel, ConfigDict, Field

from app.models.document import DocumentStatus


class DocumentResponse(BaseModel):
    """Document metadata response schema."""

    model_config = ConfigDict(from_attributes=True)

    id: uuid.UUID = Field(description="Unique UUID identifier of the document")
    original_filename: str = Field(description="Original uploaded filename")
    content_type: str = Field(description="Normalized MIME media type")
    size_bytes: int = Field(description="File size in bytes")
    sha256: str = Field(description="SHA-256 checksum digest of the file contents")
    status: DocumentStatus = Field(description="Current processing lifecycle status")
    created_at: datetime = Field(description="Timestamp when the document was uploaded")
    updated_at: datetime = Field(description="Timestamp when the document record was last updated")


class DocumentDeleteResponse(BaseModel):
    """Response schema for successful document deletion."""

    id: uuid.UUID = Field(description="UUID identifier of the deleted document")
    status: str = Field(default="deleted", description="Deletion confirmation status")
    message: str = Field(
        default="Document deleted successfully",
        description="Human-readable status message",
    )
