"""Tokenizer implementation using tiktoken for deterministic subword tokenization.

Architectural note:
    `cl100k_base` is used exclusively as a reproducible, structure-aware chunking
    token-count baseline. It does NOT guarantee exact token counts for the future
    dense embedding model (which will use its own model-specific tokenizer).
    All vocabulary assets are packaged locally and verified via SHA-256 to ensure
    100% offline initialization with network access disabled and an empty cache.
"""

import base64
import hashlib
from pathlib import Path
from typing import Any

import tiktoken

from app.chunking.exceptions import ChunkingError
from app.chunking.interface import Tokenizer

CL100K_BASE_SHA256 = "223921b76ee99bde995b7ff738513eef100fb51d18c93597a113bcffe865b2a7"

CL100K_BASE_PAT_STR = (
    r"""'(?i:[sdmt]|ll|ve|re)|[^\r\n\p{L}\p{N}]?+\p{L}++|\p{N}{1,3}+| ?[^\s\p{L}\p{N}]++[\r\n]*+|\s++$|\s*[\r\n]|\s+(?!\S)|\s"""
)

CL100K_BASE_SPECIAL_TOKENS = {
    "<|endoftext|>": 100257,
    "<|fim_prefix|>": 100258,
    "<|fim_middle|>": 100259,
    "<|fim_suffix|>": 100260,
    "<|endofprompt|>": 100276,
}

DEFAULT_ASSET_PATH = Path(__file__).resolve().parent / "assets" / "cl100k_base.tiktoken"

_cached_encoding: tiktoken.Encoding | None = None


def _parse_tiktoken_bpe(data: bytes) -> dict[bytes, int]:
    """Parse BPE vocabulary ranks from base64-encoded lines."""
    ranks: dict[bytes, int] = {}
    for line in data.splitlines():
        if not line:
            continue
        token, rank = line.split()
        ranks[base64.b64decode(token)] = int(rank)
    return ranks


def load_cl100k_base_encoding(asset_path: Path | str | None = None) -> tiktoken.Encoding:
    """Load and verify the local cl100k_base tokenizer vocabulary asset offline.

    Guarantees that no network access occurs and that the user cache is not required.
    Validates the SHA-256 checksum to prevent corruption or tampering.
    """
    global _cached_encoding

    resolved_path = Path(asset_path) if asset_path is not None else DEFAULT_ASSET_PATH

    # Return cached encoding if using default asset and already loaded
    if asset_path is None and _cached_encoding is not None:
        return _cached_encoding

    if not resolved_path.is_file():
        raise ChunkingError(
            f"Tokenizer asset not found at '{resolved_path}'. "
            "Offline tokenizer initialization requires the locally packaged vocabulary asset."
        )

    try:
        data = resolved_path.read_bytes()
    except OSError as err:
        raise ChunkingError(f"Failed to read tokenizer asset from '{resolved_path}': {err}") from err

    actual_hash = hashlib.sha256(data).hexdigest()
    if actual_hash != CL100K_BASE_SHA256:
        raise ChunkingError(
            f"Integrity check failed for tokenizer asset '{resolved_path}'. "
            f"Expected SHA-256 {CL100K_BASE_SHA256}, got {actual_hash}."
        )

    ranks = _parse_tiktoken_bpe(data)

    encoding = tiktoken.Encoding(
        name="cl100k_base",
        pat_str=CL100K_BASE_PAT_STR,
        mergeable_ranks=ranks,
        special_tokens=CL100K_BASE_SPECIAL_TOKENS,
    )

    if asset_path is None:
        _cached_encoding = encoding

    return encoding


class TiktokenTokenizer:
    """Subword BPE tokenizer using tiktoken's cl100k_base encoding.

    Provides deterministic, fast, offline token counting and encoding.
    Disables special token errors so raw untrusted document content
    (e.g. containing '<|endoftext|>') is safely encoded without crashing.
    """

    def __init__(
        self,
        encoding_name: str = "cl100k_base",
        asset_path: Path | str | None = None,
    ) -> None:
        self.encoding_name = encoding_name
        if encoding_name == "cl100k_base":
            self._encoding = load_cl100k_base_encoding(asset_path=asset_path)
        else:
            self._encoding = tiktoken.get_encoding(encoding_name)

    def encode(self, text: str) -> list[int]:
        """Encode text to token IDs, treating all special token strings as literal text."""
        return self._encoding.encode(text, disallowed_special=())

    def decode(self, tokens: list[int]) -> str:
        """Decode token IDs back into text."""
        return self._encoding.decode(tokens)

    def count_tokens(self, text: str) -> int:
        """Count tokens in text deterministically."""
        return len(self.encode(text))


_default_tokenizer: Tokenizer | None = None


def get_default_tokenizer() -> Tokenizer:
    """Return a shared default TiktokenTokenizer instance."""
    global _default_tokenizer
    if _default_tokenizer is None:
        _default_tokenizer = TiktokenTokenizer()
    return _default_tokenizer
