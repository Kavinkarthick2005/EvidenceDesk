"""EvidenceDesk structure-aware document chunking package."""

from app.chunking.exceptions import (
    ChunkingConfigError,
    ChunkingEmptyError,
    ChunkingError,
)
from app.chunking.interface import Chunker, Tokenizer
from app.chunking.models import Chunk, ChunkingConfig, ChunkingResult
from app.chunking.service import (
    StructureAwareChunker,
    chunk_document,
    get_chunker,
)
from app.chunking.tokenizer import TiktokenTokenizer, get_default_tokenizer

__all__ = [
    "Chunk",
    "Chunker",
    "ChunkingConfig",
    "ChunkingConfigError",
    "ChunkingEmptyError",
    "ChunkingError",
    "ChunkingResult",
    "StructureAwareChunker",
    "TiktokenTokenizer",
    "Tokenizer",
    "chunk_document",
    "get_chunker",
    "get_default_tokenizer",
]
