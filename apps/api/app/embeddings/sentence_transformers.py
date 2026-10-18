"""Local Sentence Transformers embedding provider for EvidenceDesk."""

from collections.abc import Sequence
import logging
from typing import Any

import numpy as np

from app.embeddings.exceptions import (
    EmbeddingDimensionMismatchError,
    EmbeddingInferenceError,
    EmbeddingModelLoadError,
    EmbeddingNonFiniteError,
)

logger = logging.getLogger(__name__)


class SentenceTransformersEmbeddingProvider:
    """Local embedding provider using Sentence Transformers and PyTorch on CPU/GPU."""

    BGE_QUERY_INSTRUCTION_PREFIX: str = "Represent this sentence for searching relevant passages: "

    def __init__(
        self,
        model_name_or_path: str = "BAAI/bge-small-en-v1.5",
        expected_dimension: int = 384,
        device: str = "cpu",
        batch_size: int = 32,
        num_threads: int | None = None,
        query_instruction_prefix: str | None = None,
    ) -> None:
        """Initialize SentenceTransformers model and introspect dimensions and context limits.

        Args:
            model_name_or_path: Hugging Face hub model identifier or local path.
            expected_dimension: Enforced schema dimensionality (defaults to 384).
            device: Compute device ('cpu', 'cuda', etc.).
            batch_size: Inference batch size.
            num_threads: Optional CPU thread throttle for PyTorch.
            query_instruction_prefix: Optional instruction prefix prepended to search queries.
                If None, defaults to BGE retrieval instruction for BGE models and empty for others.

        Raises:
            EmbeddingModelLoadError: If model cannot be loaded.
            EmbeddingDimensionMismatchError: If model dimension does not match expected_dimension.
        """
        self._model_id = model_name_or_path
        self._batch_size = batch_size
        self._truncated_chunks_count = 0

        if query_instruction_prefix is not None:
            self._query_instruction_prefix = query_instruction_prefix
        elif "bge" in model_name_or_path.lower():
            self._query_instruction_prefix = self.BGE_QUERY_INSTRUCTION_PREFIX
        else:
            self._query_instruction_prefix = ""

        # Optional PyTorch thread throttling to prevent CPU starvation
        if num_threads is not None and num_threads > 0:
            try:
                import torch

                torch.set_num_threads(num_threads)
            except Exception as thread_err:
                logger.warning("Failed to set PyTorch thread limit to %d: %s", num_threads, thread_err)

        try:
            from sentence_transformers import SentenceTransformer

            self._model = SentenceTransformer(model_name_or_path, device=device)
        except Exception as load_err:
            raise EmbeddingModelLoadError(
                f"Failed to load SentenceTransformer model '{model_name_or_path}' on device '{device}': {load_err}"
            ) from load_err

        # Introspect dimension and max context window from the loaded model
        try:
            if hasattr(self._model, "get_embedding_dimension"):
                self._dimension = int(self._model.get_embedding_dimension())
            else:
                self._dimension = int(self._model.get_sentence_embedding_dimension())
        except Exception as dim_err:
            raise EmbeddingModelLoadError(
                f"Failed to determine embedding dimension for model '{model_name_or_path}': {dim_err}"
            ) from dim_err

        if self._dimension != expected_dimension:
            raise EmbeddingDimensionMismatchError(
                f"Model '{model_name_or_path}' produces {self._dimension} dimensions, "
                f"which violates the required {expected_dimension} dimensional schema."
            )

        self._max_sequence_length = int(getattr(self._model, "max_seq_length", 512))
        logger.info(
            "Initialized SentenceTransformers provider (model='%s', dimension=%d, max_sequence_length=%d, device='%s').",
            self._model_id,
            self._dimension,
            self._max_sequence_length,
            device,
        )

    @property
    def model_id(self) -> str:
        """Canonical identifier of the active embedding model."""
        return self._model_id

    @property
    def dimension(self) -> int:
        """Vector dimensionality derived dynamically from the active model."""
        return self._dimension

    @property
    def max_sequence_length(self) -> int:
        """Maximum token context window before truncation."""
        return self._max_sequence_length

    @property
    def truncated_chunks_count(self) -> int:
        """Number of text passages that exceeded max_sequence_length and were truncated."""
        return self._truncated_chunks_count

    @property
    def query_instruction_prefix(self) -> str:
        """Instruction prefix prepended to search queries prior to embedding."""
        return self._query_instruction_prefix

    def _detect_truncation(self, text: str) -> None:
        """Measure token count using the model's actual tokenizer (including special tokens)."""
        try:
            tokenizer = getattr(self._model, "tokenizer", None)
            if tokenizer is not None and hasattr(tokenizer, "encode"):
                tokens = tokenizer.encode(text, add_special_tokens=True)
                token_count = len(tokens)
                if token_count > self._max_sequence_length:
                    excess = token_count - self._max_sequence_length
                    self._truncated_chunks_count += 1
                    logger.warning(
                        "Chunk text exceeded model context window (%d > %d tokens, including special tokens). "
                        "Truncated %d trailing tokens for model '%s'.",
                        token_count,
                        self._max_sequence_length,
                        excess,
                        self._model_id,
                    )
        except Exception as token_err:
            logger.debug("Tokenizer measurement failed for truncation check: %s", token_err)

    def embed_texts(self, texts: Sequence[str]) -> list[list[float]]:
        """Batch-embed a sequence of texts using SentenceTransformers."""
        if not texts:
            return []

        # Detect and report tokenizer truncation for each input passage
        for text in texts:
            self._detect_truncation(text)

        try:
            embeddings_np = self._model.encode(
                list(texts),
                batch_size=self._batch_size,
                show_progress_bar=False,
                normalize_embeddings=True,
                convert_to_numpy=True,
            )
        except Exception as inf_err:
            raise EmbeddingInferenceError(
                f"Embedding inference failed for model '{self._model_id}': {inf_err}"
            ) from inf_err

        # Validate finiteness
        if not np.isfinite(embeddings_np).all():
            raise EmbeddingNonFiniteError(
                f"Embedding inference for model '{self._model_id}' produced non-finite values (NaN/Inf)."
            )

        # Validate dimensionality
        if embeddings_np.shape[1] != self._dimension:
            raise EmbeddingDimensionMismatchError(
                f"Output embeddings have dimension {embeddings_np.shape[1]}, expected {self._dimension}."
            )

        return embeddings_np.tolist()

    def embed_query(self, query: str) -> list[float]:
        """Embed a single query string, applying query instruction prefix if configured."""
        prefixed_query = f"{self._query_instruction_prefix}{query}" if self._query_instruction_prefix else query
        result = self.embed_texts([prefixed_query])
        if not result:
            raise EmbeddingInferenceError("Empty query embedding output.")
        return result[0]
