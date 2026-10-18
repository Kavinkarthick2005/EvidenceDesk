"""EvidenceDesk storage package providing file storage abstraction and local implementation."""

from app.core.config import Settings, get_settings
from app.storage.exceptions import (
    StorageCollisionError,
    StorageError,
    StorageFileNotFoundError,
    StorageIOError,
    StorageKeyError,
    StorageSizeLimitError,
    StorageUnsupportedTypeError,
)
from app.storage.interface import StorageService
from app.storage.local import LocalStorageService

_storage_service_instance: StorageService | None = None


def get_storage_service(settings: Settings | None = None) -> StorageService:
    """Return a shared or configured StorageService instance."""
    global _storage_service_instance
    if settings is not None:
        return LocalStorageService(
            root_dir=settings.resolved_document_storage_path,
            max_size_mb=settings.MAX_UPLOAD_SIZE_MB,
            settings=settings,
        )

    if _storage_service_instance is None:
        app_settings = get_settings()
        _storage_service_instance = LocalStorageService(
            root_dir=app_settings.resolved_document_storage_path,
            max_size_mb=app_settings.MAX_UPLOAD_SIZE_MB,
            settings=app_settings,
        )
    return _storage_service_instance


def reset_storage_state() -> None:
    """Reset cached storage service state (primarily for test isolation)."""
    global _storage_service_instance
    _storage_service_instance = None


__all__ = [
    "LocalStorageService",
    "StorageCollisionError",
    "StorageError",
    "StorageFileNotFoundError",
    "StorageIOError",
    "StorageKeyError",
    "StorageService",
    "StorageSizeLimitError",
    "StorageUnsupportedTypeError",
    "get_storage_service",
    "reset_storage_state",
]
