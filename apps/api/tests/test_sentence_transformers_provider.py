"""Integration tests for local Sentence Transformers embedding provider.

These tests download and execute real model weights and are gated behind
the RUN_MODEL_TESTS environment variable so ordinary CI runs offline and fast.
"""

import math
import os

import pytest

from app.embeddings.exceptions import EmbeddingDimensionMismatchError
from app.embeddings.sentence_transformers import SentenceTransformersEmbeddingProvider

# Gate model-downloading integration tests behind environment variable
RUN_MODEL_TESTS = os.getenv("RUN_MODEL_TESTS", "").lower() in ("1", "true", "yes")

pytestmark = [
    pytest.mark.integration,
    pytest.mark.skipif(
        not RUN_MODEL_TESTS,
        reason="Model tests require downloading weights and are skipped during ordinary CI. Set RUN_MODEL_TESTS=1 to run.",
    ),
]


def test_real_bge_small_model_introspection() -> None:
    """Verify BAAI/bge-small-en-v1.5 dynamic dimension and sequence length discovery."""
    provider = SentenceTransformersEmbeddingProvider(
        model_name_or_path="BAAI/bge-small-en-v1.5",
        expected_dimension=384,
        device="cpu",
    )
    assert provider.model_id == "BAAI/bge-small-en-v1.5"
    assert provider.dimension == 384
    assert provider.max_sequence_length == 512


def test_real_bge_small_model_normalization_and_inference() -> None:
    """Verify output vectors are 384-dimensional, finite, and unit-normalized."""
    provider = SentenceTransformersEmbeddingProvider(
        model_name_or_path="BAAI/bge-small-en-v1.5",
        expected_dimension=384,
        device="cpu",
    )
    texts = [
        "EvidenceDesk is a local-first evidence-grounded document question answering platform.",
        "Retrieval augmented generation requires grounded citations.",
    ]
    vectors = provider.embed_texts(texts)

    assert len(vectors) == 2
    for vec in vectors:
        assert len(vec) == 384
        l2_norm = math.sqrt(sum(x * x for x in vec))
        assert pytest.approx(l2_norm, abs=1e-4) == 1.0


def test_real_model_dimension_mismatch_raises_error() -> None:
    """Verify provider raises EmbeddingDimensionMismatchError if configured dimension violates model output."""
    with pytest.raises(EmbeddingDimensionMismatchError, match="violates the required 768 dimensional schema"):
        SentenceTransformersEmbeddingProvider(
            model_name_or_path="BAAI/bge-small-en-v1.5",
            expected_dimension=768,
            device="cpu",
        )
