"""Unit tests for embedding infrastructure, mock provider, dimensions, and model persistence.

Runs fast and completely in-memory using SQLite with zero external network or model downloads.
"""

import math
import uuid

import pytest
import sqlalchemy as sa
from sqlalchemy.exc import IntegrityError
from sqlalchemy.orm import Session

from app.core.config import Settings
from app.db.base import Base
from app.embeddings import (
    DeterministicMockEmbeddingProvider,
    EmbeddingConfigurationError,
    EmbeddingDimensionMismatchError,
    EmbeddingInferenceError,
    EmbeddingNonFiniteError,
    get_embedding_provider,
    reset_embedding_state,
)
from app.models.chunk import Chunk
from app.models.document import Document, DocumentStatus
from app.models.embedding import ChunkEmbedding


@pytest.fixture
def sqlite_engine() -> sa.Engine:
    """Provide an in-memory SQLite engine with PRAGMA foreign_keys enabled and tables created."""
    engine = sa.create_engine("sqlite:///:memory:")

    @sa.event.listens_for(engine, "connect")
    def set_sqlite_pragma(dbapi_connection, connection_record):  # type: ignore[no-untyped-def]
        cursor = dbapi_connection.cursor()
        cursor.execute("PRAGMA foreign_keys=ON")
        cursor.close()

    Base.metadata.create_all(engine)
    return engine


@pytest.fixture
def db_session(sqlite_engine: sa.Engine) -> Session:
    """Provide a scoped database session."""
    with Session(sqlite_engine) as session:
        yield session


def _create_sample_document_and_chunk(db_session: Session) -> tuple[Document, Chunk]:
    """Helper creating a parent Document and Chunk."""
    doc = Document(
        original_filename="sample.txt",
        content_type="text/plain",
        size_bytes=100,
        sha256="a" * 64,
        storage_key="sample.txt",
        status=DocumentStatus.READY,
    )
    db_session.add(doc)
    db_session.flush()

    chunk = Chunk(
        document_id=doc.id,
        chunk_index=0,
        text="Sample chunk text for embedding tests.",
        token_count=8,
        character_count=39,
        start_char_offset=0,
        end_char_offset=39,
        source_format="text/plain",
    )
    db_session.add(chunk)
    db_session.commit()
    db_session.refresh(doc)
    db_session.refresh(chunk)
    return doc, chunk


def test_mock_provider_introspects_dimensions_and_context_length() -> None:
    """Verify mock provider exposes canonical model ID, dimension, and context length."""
    provider = DeterministicMockEmbeddingProvider(
        model_id="test-model",
        dimension=384,
        max_sequence_length=512,
    )
    assert provider.model_id == "test-model"
    assert provider.dimension == 384
    assert provider.max_sequence_length == 512
    assert provider.truncated_chunks_count == 0


def test_mock_provider_generates_unit_normalized_vectors() -> None:
    """Verify mock provider generates 384-dimensional L2 unit-normalized vectors."""
    provider = DeterministicMockEmbeddingProvider()
    texts = ["First passage text", "Second distinct passage"]
    vectors = provider.embed_texts(texts)

    assert len(vectors) == 2
    for vec in vectors:
        assert len(vec) == 384
        # Calculate L2 norm: sqrt(sum(x^2)) must equal 1.0 within floating point precision
        l2_norm = math.sqrt(sum(x * x for x in vec))
        assert pytest.approx(l2_norm, abs=1e-5) == 1.0


def test_mock_provider_is_deterministic() -> None:
    """Verify identical text produces identical vectors, while distinct texts produce different vectors."""
    provider = DeterministicMockEmbeddingProvider()
    vec1 = provider.embed_query("Identical search query")
    vec2 = provider.embed_query("Identical search query")
    vec3 = provider.embed_query("Completely different query")

    assert vec1 == vec2
    assert vec1 != vec3


def test_mock_provider_simulates_inference_failure() -> None:
    """Verify should_fail raises EmbeddingInferenceError."""
    provider = DeterministicMockEmbeddingProvider(should_fail=True)
    with pytest.raises(EmbeddingInferenceError, match="Mock inference failure"):
        provider.embed_texts(["Test text"])

    with pytest.raises(EmbeddingInferenceError, match="Mock query inference failure"):
        provider.embed_query("Query")


def test_mock_provider_detects_and_counts_truncation(caplog: pytest.LogCaptureFixture) -> None:
    """Verify text exceeding max_sequence_length increments truncation count and logs structured warning."""
    provider = DeterministicMockEmbeddingProvider(max_sequence_length=10)
    # 20 words + 2 special tokens = 22 tokens, which exceeds 10 tokens
    long_text = " ".join([f"word{i}" for i in range(20)])

    with caplog.at_level("WARNING"):
        vectors = provider.embed_texts([long_text])

    assert len(vectors) == 1
    assert provider.truncated_chunks_count == 1
    assert "Chunk text exceeded model context window" in caplog.text
    assert "Truncated 12 trailing tokens" in caplog.text


def test_get_embedding_provider_factory() -> None:
    """Verify provider factory returns configured mock provider."""
    settings = Settings(EMBEDDING_PROVIDER="mock", EMBEDDING_DIMENSION=384, _env_file=None)
    provider = get_embedding_provider(settings)
    assert isinstance(provider, DeterministicMockEmbeddingProvider)
    assert provider.dimension == 384


def test_get_embedding_provider_rejects_unsupported_provider() -> None:
    """Verify factory raises EmbeddingConfigurationError for unknown providers."""
    settings = Settings(EMBEDDING_PROVIDER="unknown_ai_provider", _env_file=None)
    with pytest.raises(EmbeddingConfigurationError, match="Unsupported EMBEDDING_PROVIDER"):
        get_embedding_provider(settings)


def test_chunk_embedding_model_persistence(db_session: Session) -> None:
    """Verify ChunkEmbedding record persistence and attributes in database."""
    doc, chunk = _create_sample_document_and_chunk(db_session)
    dummy_vec = [0.05] * 384

    embedding = ChunkEmbedding(
        chunk_id=chunk.id,
        document_id=doc.id,
        model_id="BAAI/bge-small-en-v1.5",
        dimension=384,
        embedding=dummy_vec,
    )
    db_session.add(embedding)
    db_session.commit()
    db_session.refresh(embedding)

    assert embedding.id is not None
    assert embedding.chunk_id == chunk.id
    assert embedding.document_id == doc.id
    assert embedding.model_id == "BAAI/bge-small-en-v1.5"
    assert embedding.dimension == 384
    assert len(embedding.embedding) == 384


def test_chunk_embedding_model_uniqueness_constraint(db_session: Session) -> None:
    """Verify unique constraint on (chunk_id, model_id) prevents duplicate embeddings for same model."""
    doc, chunk = _create_sample_document_and_chunk(db_session)
    dummy_vec = [0.1] * 384

    emb1 = ChunkEmbedding(
        chunk_id=chunk.id,
        document_id=doc.id,
        model_id="BAAI/bge-small-en-v1.5",
        dimension=384,
        embedding=dummy_vec,
    )
    db_session.add(emb1)
    db_session.commit()

    # Second embedding with identical chunk_id and model_id must violate unique constraint
    emb2 = ChunkEmbedding(
        chunk_id=chunk.id,
        document_id=doc.id,
        model_id="BAAI/bge-small-en-v1.5",
        dimension=384,
        embedding=dummy_vec,
    )
    db_session.add(emb2)
    with pytest.raises(IntegrityError):
        db_session.commit()
    db_session.rollback()


def test_chunk_embedding_multi_model_coexistence(db_session: Session) -> None:
    """Verify multiple models can co-exist for the same chunk when satisfying the 384-dimensional schema."""
    doc, chunk = _create_sample_document_and_chunk(db_session)
    vec_a = [0.1] * 384
    vec_b = [0.2] * 384

    emb_a = ChunkEmbedding(
        chunk_id=chunk.id,
        document_id=doc.id,
        model_id="BAAI/bge-small-en-v1.5",
        dimension=384,
        embedding=vec_a,
    )
    emb_b = ChunkEmbedding(
        chunk_id=chunk.id,
        document_id=doc.id,
        model_id="sentence-transformers/all-MiniLM-L6-v2",
        dimension=384,
        embedding=vec_b,
    )
    db_session.add_all([emb_a, emb_b])
    db_session.commit()

    # Query both embeddings for chunk
    records = db_session.scalars(
        sa.select(ChunkEmbedding).where(ChunkEmbedding.chunk_id == chunk.id)
    ).all()
    assert len(records) == 2
    model_ids = {r.model_id for r in records}
    assert model_ids == {"BAAI/bge-small-en-v1.5", "sentence-transformers/all-MiniLM-L6-v2"}


def test_chunk_embedding_cascades_on_chunk_deletion(db_session: Session) -> None:
    """Verify deleting a chunk cascades to delete associated embeddings."""
    doc, chunk = _create_sample_document_and_chunk(db_session)
    embedding = ChunkEmbedding(
        chunk_id=chunk.id,
        document_id=doc.id,
        model_id="BAAI/bge-small-en-v1.5",
        dimension=384,
        embedding=[0.05] * 384,
    )
    db_session.add(embedding)
    db_session.commit()

    # Delete the chunk
    db_session.delete(chunk)
    db_session.commit()

    # Embedding record must have been deleted by cascade
    remaining = db_session.scalars(
        sa.select(ChunkEmbedding).where(ChunkEmbedding.id == embedding.id)
    ).all()
    assert len(remaining) == 0


def test_chunk_embedding_dimension_constraint(db_session: Session) -> None:
    """Verify dimension constraint check_chunk_embedding_dimension_384 rejects dimensions != 384."""
    doc, chunk = _create_sample_document_and_chunk(db_session)
    invalid_embedding = ChunkEmbedding(
        chunk_id=chunk.id,
        document_id=doc.id,
        model_id="custom-768-model",
        dimension=768,  # Violates check_chunk_embedding_dimension_384
        embedding=[0.01] * 768,
    )
    db_session.add(invalid_embedding)
    with pytest.raises(IntegrityError):
        db_session.commit()
    db_session.rollback()


def test_sentence_transformers_query_instruction_prefixing_isolated() -> None:
    """Verify SentenceTransformersEmbeddingProvider prefixes only query embeddings and leaves document texts unchanged."""
    from unittest.mock import MagicMock, patch
    import numpy as np
    from app.embeddings.sentence_transformers import SentenceTransformersEmbeddingProvider

    with patch("sentence_transformers.SentenceTransformer") as mock_st_cls:
        mock_model = MagicMock()
        mock_model.get_embedding_dimension.return_value = 384
        mock_model.max_seq_length = 512
        mock_st_cls.return_value = mock_model

        # Unit-normalized mock output
        dummy_vec = np.ones((1, 384), dtype=np.float32) / math.sqrt(384)
        mock_model.encode.side_effect = lambda texts, **kwargs: np.repeat(dummy_vec, len(texts), axis=0)

        # 1. BGE model defaults to BGE query instruction prefix
        bge_provider = SentenceTransformersEmbeddingProvider(
            model_name_or_path="BAAI/bge-small-en-v1.5",
            expected_dimension=384,
            device="cpu",
        )
        assert bge_provider.query_instruction_prefix == "Represent this sentence for searching relevant passages: "

        # 2. embed_texts must pass chunk texts unchanged (no prefix)
        bge_provider.embed_texts(["Chunk passage one", "Chunk passage two"])
        assert mock_model.encode.call_args[0][0] == ["Chunk passage one", "Chunk passage two"]

        # 3. embed_query must prepend query instruction prefix
        bge_provider.embed_query("User search question")
        assert mock_model.encode.call_args[0][0] == [
            "Represent this sentence for searching relevant passages: User search question"
        ]

        # 4. Non-BGE model defaults to empty prefix (safe compatibility)
        minilm_provider = SentenceTransformersEmbeddingProvider(
            model_name_or_path="sentence-transformers/all-MiniLM-L6-v2",
            expected_dimension=384,
            device="cpu",
        )
        assert minilm_provider.query_instruction_prefix == ""
        minilm_provider.embed_query("MiniLM query")
        assert mock_model.encode.call_args[0][0] == ["MiniLM query"]

        # 5. Explicitly disabling prefix on BGE model passes raw query
        unprefixed_bge = SentenceTransformersEmbeddingProvider(
            model_name_or_path="BAAI/bge-small-en-v1.5",
            expected_dimension=384,
            query_instruction_prefix="",
        )
        assert unprefixed_bge.query_instruction_prefix == ""
        unprefixed_bge.embed_query("Raw search query")
        assert mock_model.encode.call_args[0][0] == ["Raw search query"]

        # 6. Truncation detection accounts for prefix tokens
        mock_tokenizer = MagicMock()
        mock_model.tokenizer = mock_tokenizer
        # Return 515 tokens for prefixed text, triggering truncation count increment
        mock_tokenizer.encode.return_value = list(range(515))

        assert bge_provider.truncated_chunks_count == 0
        bge_provider.embed_query("Very long query exceeding window")
        assert bge_provider.truncated_chunks_count == 1
        # Tokenizer encode was called with prefixed text
        assert mock_tokenizer.encode.call_args[0][0].startswith("Represent this sentence for searching relevant passages: ")
