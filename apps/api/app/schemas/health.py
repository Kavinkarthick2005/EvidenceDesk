"""Health check response schema."""

from pydantic import BaseModel, Field


class HealthResponse(BaseModel):
    """Schema for service health check response."""

    status: str = Field(
        default="ok",
        description="Health status indicator",
        examples=["ok"],
    )
    app: str = Field(
        description="Application name",
        examples=["EvidenceDesk"],
    )
    environment: str = Field(
        description="Deployment environment name",
        examples=["development"],
    )
    version: str = Field(
        description="Application version string",
        examples=["0.1.0"],
    )
