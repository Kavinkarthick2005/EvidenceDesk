"""Local filesystem storage implementation for EvidenceDesk."""

import os
from pathlib import Path
import secrets
from typing import BinaryIO
import uuid

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


class LocalStorageService:
    """Stores documents securely on the local filesystem.

    Adheres to the following security and reliability constraints:
    - Storage root directory is created lazily on first write.
    - Keys are internally generated UUIDs; filesystem paths are never derived
      from untrusted user filenames.
    - Path traversal, absolute paths, and separator injection are strictly rejected.
    - Symlinks that resolve outside the storage root are rejected.
    - Writes are atomic via temporary files with automatic rollback on error.
    - Existing files are never overwritten on collision.
    """

    ALLOWED_MIME_TYPES: dict[str, str] = {
        "application/pdf": ".pdf",
        "text/markdown": ".md",
        "text/plain": ".txt",
        "text/x-markdown": ".md",
    }

    ALLOWED_EXTENSIONS: dict[str, str] = {
        ".pdf": ".pdf",
        ".md": ".md",
        ".markdown": ".md",
        ".txt": ".txt",
    }

    def __init__(
        self,
        root_dir: str | Path | None = None,
        max_size_mb: int | None = None,
        settings: Settings | None = None,
    ) -> None:
        """Initialize LocalStorageService without creating directories on disk.

        Args:
            root_dir: Explicit root directory path for document storage.
            max_size_mb: Maximum allowed upload size in megabytes.
            settings: Optional Settings instance to derive defaults from.
        """
        app_settings = settings or get_settings()
        if root_dir is not None:
            self.storage_root = Path(root_dir).resolve()
        else:
            self.storage_root = app_settings.resolved_document_storage_path

        self.max_size_mb = max_size_mb if max_size_mb is not None else app_settings.MAX_UPLOAD_SIZE_MB
        self.max_size_bytes = self.max_size_mb * 1024 * 1024

    def _validate_key_syntax(self, key: str) -> None:
        """Validate key syntax against path traversal, absolute paths, and null bytes."""
        if not isinstance(key, str) or not key.strip():
            raise StorageKeyError("Storage key cannot be empty or whitespace-only.")

        if "\0" in key:
            raise StorageKeyError("Storage key contains invalid null byte characters.")

        # Reject absolute paths (POSIX and Windows) and drive letters
        if key.startswith(("/", "\\")) or (len(key) >= 2 and key[1] == ":"):
            raise StorageKeyError(f"Storage key '{key}' cannot be an absolute path or drive path.")

        key_path = Path(key)
        if key_path.is_absolute():
            raise StorageKeyError(f"Storage key '{key}' cannot be an absolute path.")

        # Reject traversal components
        for part in key_path.parts:
            if part in ("..", ".") or ".." in part:
                raise StorageKeyError(
                    f"Storage key '{key}' contains invalid traversal component: '{part}'."
                )

    @staticmethod
    def _normalize_resolved_path(path: Path) -> Path:
        """Strip Windows extended-path prefix (\\\\?\\) so resolved paths compare reliably."""
        path_str = str(path)
        if path_str.startswith("\\\\?\\"):
            return Path(path_str[4:])
        return path

    def _resolve_path(self, key: str) -> Path:
        """Resolve a storage key to an absolute Path and verify it remains within storage_root."""
        self._validate_key_syntax(key)

        candidate = self.storage_root / key
        resolved_root = self._normalize_resolved_path(self.storage_root.resolve())
        resolved_candidate = self._normalize_resolved_path(candidate.resolve())

        # Check for symlinks pointing outside storage root
        if candidate.is_symlink():
            target = self._normalize_resolved_path(candidate.resolve())
            if not target.is_relative_to(resolved_root):
                raise StorageKeyError(
                    f"Symlink traversal detected: storage key '{key}' escapes storage root."
                )

        if not resolved_candidate.is_relative_to(resolved_root) or resolved_candidate == resolved_root:
            raise StorageKeyError(
                f"Path traversal detected: storage key '{key}' resolves outside storage root."
            )

        return candidate

    def _resolve_safe_extension(
        self,
        content_type: str | None = None,
        original_filename: str | None = None,
    ) -> str:
        """Derive an allowlisted file extension from validated content type or filename hint."""
        if content_type:
            normalized_mime = content_type.split(";")[0].strip().lower()
            if normalized_mime not in self.ALLOWED_MIME_TYPES:
                raise StorageUnsupportedTypeError(
                    f"Unsupported content type '{content_type}'. Allowed types: "
                    f"{', '.join(sorted(self.ALLOWED_MIME_TYPES.keys()))}."
                )
            return self.ALLOWED_MIME_TYPES[normalized_mime]

        if original_filename:
            raw_ext = Path(original_filename).suffix.lower()
            if raw_ext:
                if raw_ext not in self.ALLOWED_EXTENSIONS:
                    raise StorageUnsupportedTypeError(
                        f"Unsupported file extension '{raw_ext}'. Allowed extensions: "
                        f"{', '.join(sorted(self.ALLOWED_EXTENSIONS.keys()))}."
                    )
                return self.ALLOWED_EXTENSIONS[raw_ext]

        return ".bin"

    def save(
        self,
        content: bytes | BinaryIO,
        *,
        content_type: str | None = None,
        original_filename: str | None = None,
    ) -> str:
        """Store document content securely and return an internally generated storage key."""
        safe_ext = self._resolve_safe_extension(content_type, original_filename)
        key = f"{uuid.uuid4().hex}{safe_ext}"
        dest_path = self._resolve_path(key)

        # Pre-check size limit if content is already in bytes
        if isinstance(content, bytes) and len(content) > self.max_size_bytes:
            raise StorageSizeLimitError(
                f"Content size {len(content)} bytes exceeds limit of "
                f"{self.max_size_bytes} bytes ({self.max_size_mb} MB)."
            )

        # Lazy creation of root directory only upon actual write operation
        self.storage_root.mkdir(parents=True, exist_ok=True)

        temp_path = dest_path.with_name(f".tmp_{secrets.token_hex(16)}")
        bytes_written = 0

        try:
            with open(temp_path, "xb") as f:
                if isinstance(content, bytes):
                    f.write(content)
                    bytes_written = len(content)
                else:
                    while chunk := content.read(64 * 1024):
                        bytes_written += len(chunk)
                        if bytes_written > self.max_size_bytes:
                            raise StorageSizeLimitError(
                                f"Content size exceeded limit of "
                                f"{self.max_size_bytes} bytes ({self.max_size_mb} MB)."
                            )
                        f.write(chunk)
                f.flush()
                os.fsync(f.fileno())

            # Atomic placement with no-overwrite guarantee using os.link
            try:
                os.link(temp_path, dest_path)
                temp_path.unlink(missing_ok=True)
            except FileExistsError as err:
                temp_path.unlink(missing_ok=True)
                raise StorageCollisionError(
                    f"Storage key collision detected for key: '{key}'."
                ) from err
            except OSError as err:
                temp_path.unlink(missing_ok=True)
                raise StorageIOError(
                    f"Atomic no-overwrite placement failed for storage file '{key}': {err}"
                ) from err
        except StorageError:
            if temp_path.exists():
                try:
                    temp_path.unlink()
                except OSError:
                    pass
            raise
        except OSError as err:
            if temp_path.exists():
                try:
                    temp_path.unlink()
                except OSError:
                    pass
            raise StorageIOError(f"Failed to write storage file '{key}': {err}") from err

        return key

    def read(self, storage_key: str) -> bytes:
        """Read and return byte content of the document identified by storage_key."""
        if not self.storage_root.exists():
            self._validate_key_syntax(storage_key)
            raise StorageFileNotFoundError(f"File not found for key: '{storage_key}'.")

        path = self._resolve_path(storage_key)
        if not path.is_file():
            raise StorageFileNotFoundError(f"File not found for key: '{storage_key}'.")

        try:
            return path.read_bytes()
        except OSError as err:
            raise StorageIOError(f"Failed to read storage file '{storage_key}': {err}") from err

    def delete(self, storage_key: str) -> None:
        """Delete the document file identified by storage_key."""
        if not self.storage_root.exists():
            self._validate_key_syntax(storage_key)
            raise StorageFileNotFoundError(f"File not found for key: '{storage_key}'.")

        path = self._resolve_path(storage_key)
        if not path.is_file():
            raise StorageFileNotFoundError(f"File not found for key: '{storage_key}'.")

        try:
            path.unlink()
        except OSError as err:
            raise StorageIOError(f"Failed to delete storage file '{storage_key}': {err}") from err

    def exists(self, storage_key: str) -> bool:
        """Check whether a document exists for the given storage_key."""
        self._validate_key_syntax(storage_key)
        if not self.storage_root.exists():
            return False

        path = self._resolve_path(storage_key)
        return path.is_file()
