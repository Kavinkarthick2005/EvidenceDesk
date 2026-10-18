"""Tests for EvidenceDesk storage service, path containment, and reliability."""

import io
import os
from pathlib import Path
import unittest.mock as mock
import uuid

import pytest

from app.core.config import Settings
from app.storage import (
    LocalStorageService,
    StorageCollisionError,
    StorageFileNotFoundError,
    StorageIOError,
    StorageKeyError,
    StorageSizeLimitError,
    StorageUnsupportedTypeError,
    get_storage_service,
    reset_storage_state,
)


@pytest.fixture
def storage_dir(tmp_path: Path) -> Path:
    """Return a path to a non-existent directory within tmp_path for storage testing."""
    return tmp_path / "test_docs_storage"


@pytest.fixture
def storage_service(storage_dir: Path) -> LocalStorageService:
    """Return an isolated LocalStorageService instance with a 2MB max upload limit."""
    return LocalStorageService(root_dir=storage_dir, max_size_mb=2)


def test_lazy_directory_creation(storage_dir: Path) -> None:
    """Verify storage root directory is not created upon instantiation or existence checks."""
    assert not storage_dir.exists()

    service = LocalStorageService(root_dir=storage_dir, max_size_mb=1)
    assert not storage_dir.exists()

    # exists() on nonexistent directory should safely return False without creating directory
    assert service.exists("some_file.pdf") is False
    assert not storage_dir.exists()

    # read() and delete() should raise StorageFileNotFoundError without creating directory
    with pytest.raises(StorageFileNotFoundError):
        service.read("missing.pdf")
    assert not storage_dir.exists()

    with pytest.raises(StorageFileNotFoundError):
        service.delete("missing.pdf")
    assert not storage_dir.exists()

    # save() creates storage root directory lazily
    key = service.save(b"lazy document", content_type="text/plain")
    assert storage_dir.exists()
    assert service.exists(key) is True


def test_save_and_read_byte_integrity(storage_service: LocalStorageService) -> None:
    """Verify saved file can be read back with exact byte-for-byte integrity."""
    raw_payload = b"\x00\xff\xfe\x01\x02\x03\r\n\tEvidenceDesk Document Content"
    key = storage_service.save(raw_payload, content_type="application/pdf")

    assert key.endswith(".pdf")
    read_payload = storage_service.read(key)
    assert read_payload == raw_payload


def test_save_binary_stream(storage_service: LocalStorageService) -> None:
    """Verify save accepts binary streams in addition to bytes."""
    stream_content = b"streamed markdown content"
    stream = io.BytesIO(stream_content)

    key = storage_service.save(stream, content_type="text/markdown")
    assert key.endswith(".md")
    assert storage_service.read(key) == stream_content


def test_unique_storage_keys_for_separate_writes(storage_service: LocalStorageService) -> None:
    """Verify distinct random storage keys are generated even for identical content."""
    payload = b"Identical repeated file content"
    key1 = storage_service.save(payload, content_type="text/plain")
    key2 = storage_service.save(payload, content_type="text/plain")

    assert key1 != key2
    assert storage_service.exists(key1) is True
    assert storage_service.exists(key2) is True
    assert storage_service.read(key1) == payload
    assert storage_service.read(key2) == payload


def test_delete_and_missing_file_handling(storage_service: LocalStorageService) -> None:
    """Verify file deletion and subsequent access errors."""
    key = storage_service.save(b"temporary text", content_type="text/plain")
    assert storage_service.exists(key) is True

    storage_service.delete(key)
    assert storage_service.exists(key) is False

    with pytest.raises(StorageFileNotFoundError):
        storage_service.read(key)

    with pytest.raises(StorageFileNotFoundError):
        storage_service.delete(key)


@pytest.mark.parametrize(
    "hostile_key",
    [
        "../secret.txt",
        "..\\secret.txt",
        "../../etc/passwd",
        "..\\..\\Windows\\System32\\cmd.exe",
        "nested/../../secret.txt",
        "....//secret.txt",
        "foo/..bar/secret.txt",
    ],
)
def test_reject_path_traversal_attempts(
    storage_service: LocalStorageService,
    hostile_key: str,
) -> None:
    """Verify path traversal sequences are rejected across read, delete, and exists."""
    with pytest.raises(StorageKeyError):
        storage_service.read(hostile_key)

    with pytest.raises(StorageKeyError):
        storage_service.delete(hostile_key)

    with pytest.raises(StorageKeyError):
        storage_service.exists(hostile_key)


@pytest.mark.parametrize(
    "absolute_key",
    [
        "/etc/passwd",
        "/root/data.txt",
        "\\Windows\\System32\\calc.exe",
        "C:\\secret.txt",
        "C:relative_drive.txt",
        "D:/documents/doc.pdf",
    ],
)
def test_reject_absolute_paths_and_drive_specifiers(
    storage_service: LocalStorageService,
    absolute_key: str,
) -> None:
    """Verify absolute paths and Windows drive letters are rejected."""
    with pytest.raises(StorageKeyError):
        storage_service.read(absolute_key)

    with pytest.raises(StorageKeyError):
        storage_service.delete(absolute_key)

    with pytest.raises(StorageKeyError):
        storage_service.exists(absolute_key)


@pytest.mark.parametrize(
    "malformed_key",
    [
        "",
        "   ",
        "\t\n",
        "doc\0with_null.pdf",
    ],
)
def test_reject_malformed_keys(
    storage_service: LocalStorageService,
    malformed_key: str,
) -> None:
    """Verify empty keys, whitespace-only keys, and keys with null bytes are rejected."""
    with pytest.raises(StorageKeyError):
        storage_service.read(malformed_key)

    with pytest.raises(StorageKeyError):
        storage_service.delete(malformed_key)

    with pytest.raises(StorageKeyError):
        storage_service.exists(malformed_key)


def test_size_limit_enforcement_bytes(storage_dir: Path) -> None:
    """Verify byte payloads exceeding max_size_mb are rejected and leave no partial files."""
    service = LocalStorageService(root_dir=storage_dir, max_size_mb=1)  # 1MB limit
    oversized = b"X" * (1024 * 1024 + 1)

    with pytest.raises(StorageSizeLimitError, match="exceeds limit"):
        service.save(oversized, content_type="text/plain")

    # Assert no files or temporary files remained
    if storage_dir.exists():
        assert list(storage_dir.iterdir()) == []


def test_size_limit_enforcement_stream(storage_dir: Path) -> None:
    """Verify streamed payloads exceeding max_size_mb are rejected and leave no partial files."""
    service = LocalStorageService(root_dir=storage_dir, max_size_mb=1)
    oversized_stream = io.BytesIO(b"Y" * (1024 * 1024 + 50))

    with pytest.raises(StorageSizeLimitError, match="exceeded limit"):
        service.save(oversized_stream, content_type="text/plain")

    if storage_dir.exists():
        assert list(storage_dir.iterdir()) == []


def test_write_failure_cleans_up_temporary_file(storage_service: LocalStorageService, storage_dir: Path) -> None:
    """Verify unhandled disk/write errors do not leave orphaned temporary files."""
    with mock.patch("os.fsync", side_effect=OSError("Simulated disk error")):
        with pytest.raises(StorageIOError, match="Failed to write storage file"):
            storage_service.save(b"test document", content_type="text/plain")

    # Ensure no lingering files or temp files exist in storage_dir
    assert list(storage_dir.glob(".tmp_*")) == []
    assert list(storage_dir.glob("*.txt")) == []


def test_external_symlink_cannot_escape_storage_root(
    storage_service: LocalStorageService,
    tmp_path: Path,
    storage_dir: Path,
) -> None:
    """Verify external symlinks cannot be traversed to read or delete files outside the root where supported."""
    storage_dir.mkdir(parents=True, exist_ok=True)
    external_secret = tmp_path / "host_secrets.txt"
    external_secret.write_bytes(b"SUPER_SECRET_HOST_DATA")

    symlink_file = storage_dir / "symlink_escape.txt"
    try:
        os.symlink(external_secret, symlink_file)
    except (OSError, NotImplementedError):
        pytest.skip("Symlink creation not supported in this test environment")

    # Attempting to read via the symlink key must be rejected
    with pytest.raises(StorageKeyError, match="escapes storage root"):
        storage_service.read("symlink_escape.txt")

    # Attempting to delete via the symlink key must be rejected
    with pytest.raises(StorageKeyError, match="escapes storage root"):
        storage_service.delete("symlink_escape.txt")

    # Ensure external target was not altered or deleted
    assert external_secret.exists()
    assert external_secret.read_bytes() == b"SUPER_SECRET_HOST_DATA"


def test_symlink_traversal_detection_simulated(
    storage_service: LocalStorageService,
    tmp_path: Path,
) -> None:
    """Verify symlink escape is rejected when candidate resolves outside storage_root."""
    outside_target = (tmp_path / "outside" / "target.txt").resolve()
    orig_resolve = Path.resolve

    def custom_resolve(self: Path) -> Path:
        if self.name == "fake_symlink.txt":
            return outside_target
        return orig_resolve(self)

    with mock.patch("pathlib.Path.is_symlink", return_value=True), \
         mock.patch("pathlib.Path.resolve", side_effect=custom_resolve, autospec=True):
        with pytest.raises(StorageKeyError, match="escapes storage root"):
            storage_service._resolve_path("fake_symlink.txt")


def test_allowed_and_unsupported_file_types(storage_service: LocalStorageService) -> None:
    """Verify supported formats (PDF, Markdown, plain text) are accepted and unsupported are rejected."""
    # Supported MIME types
    pdf_key = storage_service.save(b"%PDF-1.4", content_type="application/pdf")
    assert pdf_key.endswith(".pdf")

    md_key = storage_service.save(b"# Title", content_type="text/markdown")
    assert md_key.endswith(".md")

    txt_key = storage_service.save(b"plain text", content_type="text/plain")
    assert txt_key.endswith(".txt")

    # Supported via original_filename hint
    hint_pdf = storage_service.save(b"%PDF-1.4", original_filename="manual.pdf")
    assert hint_pdf.endswith(".pdf")

    # Unsupported MIME types
    with pytest.raises(StorageUnsupportedTypeError, match="Unsupported content type"):
        storage_service.save(b"payload", content_type="application/x-sh")

    with pytest.raises(StorageUnsupportedTypeError, match="Unsupported content type"):
        storage_service.save(b"image bytes", content_type="image/png")

    # Unsupported extension
    with pytest.raises(StorageUnsupportedTypeError, match="Unsupported file extension"):
        storage_service.save(b"binary", original_filename="script.sh")

    with pytest.raises(StorageUnsupportedTypeError, match="Unsupported file extension"):
        storage_service.save(b"binary", original_filename="program.exe")


def test_never_derive_storage_path_from_untrusted_original_filename(
    storage_service: LocalStorageService,
) -> None:
    """Verify original_filename is strictly ignored when determining storage key and path."""
    hostile_filename = "../../../etc/cron.d/malicious.pdf"
    key = storage_service.save(b"content", original_filename=hostile_filename)

    assert "etc" not in key
    assert "cron" not in key
    assert "malicious" not in key
    assert ".." not in key
    assert key.endswith(".pdf")

    # File on disk must reside directly within storage_root
    dest_path = storage_service.storage_root / key
    assert dest_path.is_file()
    assert dest_path.parent == storage_service.storage_root


def test_collision_detection_prevents_overwrite(
    storage_service: LocalStorageService,
    monkeypatch: pytest.MonkeyPatch,
) -> None:
    """Verify collision detection raises StorageCollisionError and does not overwrite existing file."""
    fixed_uuid = uuid.UUID("12345678-1234-5678-1234-567812345678")
    monkeypatch.setattr(uuid, "uuid4", lambda: fixed_uuid)

    key1 = storage_service.save(b"Original File", content_type="text/plain")
    assert storage_service.read(key1) == b"Original File"

    # Attempting to save again with the colliding UUID must fail
    with pytest.raises(StorageCollisionError, match="collision detected"):
        storage_service.save(b"Overwritten File", content_type="text/plain")

    # Ensure original content was not overwritten
    assert storage_service.read(key1) == b"Original File"


def test_concurrent_save_race_condition_collision_safety(
    storage_service: LocalStorageService,
    storage_dir: Path,
    monkeypatch: pytest.MonkeyPatch,
) -> None:
    """Verify that concurrent saves racing on the same key atomically prevent overwrite."""
    import threading

    fixed_uuid = uuid.UUID("aabbccdd-1234-5678-1234-567812345678")
    monkeypatch.setattr(uuid, "uuid4", lambda: fixed_uuid)

    successes = []
    collisions = []

    def concurrent_writer(thread_id: int) -> None:
        try:
            storage_service.save(f"Payload from thread {thread_id}".encode(), content_type="text/plain")
            successes.append(thread_id)
        except StorageCollisionError:
            collisions.append(thread_id)

    threads = [threading.Thread(target=concurrent_writer, args=(i,)) for i in range(8)]
    for t in threads:
        t.start()
    for t in threads:
        t.join()

    # Exactly one thread must have succeeded and all others must have been rejected
    assert len(successes) == 1
    assert len(collisions) == 7

    # No leftover temporary files must exist in storage_dir
    assert list(storage_dir.glob(".tmp_*")) == []

    # File on disk must exist and contain the winning thread's exact payload
    winning_id = successes[0]
    key = f"{fixed_uuid.hex}.txt"
    assert storage_service.read(key) == f"Payload from thread {winning_id}".encode()


def test_atomic_link_failure_fails_closed_without_overwriting(
    storage_service: LocalStorageService,
    storage_dir: Path,
    monkeypatch: pytest.MonkeyPatch,
) -> None:
    """Verify that when atomic os.link fails with OSError, the service fails closed and never overwrites."""
    fixed_uuid = uuid.UUID("ccccdddd-1234-5678-1234-567812345678")
    monkeypatch.setattr(uuid, "uuid4", lambda: fixed_uuid)

    # Initial write succeeds
    key = storage_service.save(b"Initial Safe Content", content_type="text/plain")
    assert storage_service.read(key) == b"Initial Safe Content"

    # Simulate filesystem where os.link fails with OSError (e.g. cross-device or unsupported hardlinks)
    def broken_link(src: Path, dst: Path) -> None:
        raise OSError(38, "Function not implemented")

    monkeypatch.setattr(os, "link", broken_link)

    # A subsequent save targeting this destination must fail closed with StorageIOError
    with pytest.raises(StorageIOError, match="Atomic no-overwrite placement failed"):
        storage_service.save(b"Hostile Overwrite Payload", content_type="text/plain")

    # Destination file must remain unchanged
    assert storage_service.read(key) == b"Initial Safe Content"

    # All temporary files must be cleaned up
    assert list(storage_dir.glob(".tmp_*")) == []


def test_storage_service_factory_and_reset(tmp_path: Path) -> None:
    """Verify get_storage_service returns a singleton and reset_storage_state clears it."""
    test_settings = Settings(
        DOCUMENT_STORAGE_PATH=str(tmp_path / "factory_docs"),
        MAX_UPLOAD_SIZE_MB=5,
        _env_file=None,
    )
    reset_storage_state()

    svc1 = get_storage_service(test_settings)
    svc2 = get_storage_service(test_settings)
    assert svc1 is not svc2  # custom settings produce dedicated instance

    default_svc1 = get_storage_service()
    default_svc2 = get_storage_service()
    assert default_svc1 is default_svc2  # default singleton reused

    reset_storage_state()
    default_svc3 = get_storage_service()
    assert default_svc3 is not default_svc1
