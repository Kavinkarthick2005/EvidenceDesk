"""Main API router combining route modules."""

from fastapi import APIRouter

from app.api.routes import documents, health, qa

api_router = APIRouter()
api_router.include_router(health.router)
api_router.include_router(documents.router)
api_router.include_router(qa.router)
