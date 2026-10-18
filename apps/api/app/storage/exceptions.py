"""Exceptions for EvidenceDesk file storage operations."""


class StorageError(Exception):
    """Base exception for all storage service failures."""


class StorageKeyError(StorageError):
    """Raised when a storage key is invalid, malformed, or attempts path traversal."""


class StorageFileNotFoundError(StorageError):
    """Raised when a requested storage key does not exist on disk."""


class StorageSizeLimitError(StorageError):
    """Raised when content exceeds the configured maximum upload size limit."""


class StorageUnsupportedTypeError(StorageError):
    """Raised when file content or MIME type is not among supported document types."""


class StorageCollisionError(StorageError):
    """Raised when an attempt to store a file would overwrite an existing storage key."""


class StorageIOError(StorageError):
    """Raised when an unrecoverable disk or filesystem I/O error occurs."""
