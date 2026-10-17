"""Health check route handler."""

from fastapi import APIRouter, Depends, status

from app.core.config import Settings, get_settings
from app.schemas.health import HealthResponse

router = APIRouter(tags=["health"])


@router.get(
    "/health",
    response_model=HealthResponse,
    status_code=status.HTTP_200_OK,
    summary="Service health check",
    description="Returns service health status and application metadata.",
)
def get_health(settings: Settings = Depends(get_settings)) -> HealthResponse:
    """Return service health status and basic application details."""
    return HealthResponse(
        status="ok",
        app=settings.APP_NAME,
        environment=settings.APP_ENV,
        version=settings.VERSION,
    )
