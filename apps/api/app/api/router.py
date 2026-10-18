"""Main API router combining route modules."""

from fastapi import APIRouter

from app.api.routes import documents, health

api_router = APIRouter()
api_router.include_router(health.router)
api_router.include_router(documents.router)
