"""FastAPI application factory and instance."""

from fastapi import FastAPI

from app.api.errors import register_exception_handlers
from app.api.router import api_router
from app.core.config import get_settings


def create_app() -> FastAPI:
    """Create and configure an EvidenceDesk FastAPI application instance."""
    settings = get_settings()

    app = FastAPI(
        title=settings.APP_NAME,
        version=settings.VERSION,
        docs_url="/docs",
        redoc_url="/redoc",
    )

    register_exception_handlers(app)
    app.include_router(api_router)

    return app


app = create_app()
