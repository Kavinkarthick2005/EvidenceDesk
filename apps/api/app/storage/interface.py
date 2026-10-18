"""Storage service interface for EvidenceDesk."""

from typing import BinaryIO, Protocol


class StorageService(Protocol):
    """Abstract interface defining document storage capabilities."""

    def save(
        self,
        content: bytes | BinaryIO,
        *,
        content_type: str | None = None,
        original_filename: str | None = None,
    ) -> str:
        """Store document content securely and return an internally generated storage key.

        Args:
            content: Raw byte content or readable binary stream.
            content_type: Optional MIME media type to validate.
            original_filename: Optional display filename (used only for extension hint;
                never used as a filesystem path).

        Returns:
            An internally generated, unpredictable storage key.

        Raises:
            StorageSizeLimitError: If content exceeds maximum allowed size.
            StorageUnsupportedTypeError: If provided type or extension is not permitted.
            StorageCollisionError: If generated key collides with an existing file.
            StorageIOError: If an I/O failure occurs during writing.
        """
        ...

    def read(self, storage_key: str) -> bytes:
        """Read and return byte content of the document identified by storage_key.

        Args:
            storage_key: The storage key returned by save().

        Returns:
            The raw file content as bytes.

        Raises:
            StorageKeyError: If the key is malformed or invalid.
            StorageFileNotFoundError: If the file does not exist.
            StorageIOError: If an I/O failure occurs during reading.
        """
        ...

    def delete(self, storage_key: str) -> None:
        """Delete the document file identified by storage_key.

        Args:
            storage_key: The storage key returned by save().

        Raises:
            StorageKeyError: If the key is malformed or invalid.
            StorageFileNotFoundError: If the file does not exist.
            StorageIOError: If an I/O failure occurs during deletion.
        """
        ...

    def exists(self, storage_key: str) -> bool:
        """Check whether a document exists for the given storage_key.

        Args:
            storage_key: The storage key to check.

        Returns:
            True if the file exists on storage, False otherwise.

        Raises:
            StorageKeyError: If the key is malformed or attempts path traversal.
        """
        ...
