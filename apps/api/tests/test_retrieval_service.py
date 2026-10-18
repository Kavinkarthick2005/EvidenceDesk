"""Unit and deterministic integration tests for RetrievalService.

Tests query validation, top_k clamping, score thresholds, document safety (READY status),
exact model ID isolation, deterministic tie-breaking, and PostgreSQL native pgvector search.
"""

import json
import sqlite3
import uuid

import pytest
import sqlalchemy as sa
from sqlalchemy.orm import Session

from app.core.config import Settings
from app.db.base import Base
from app.embeddings.exceptions import EmbeddingInferenceError
from app.embeddings.mock import DeterministicMockEmbeddingProvider
from app.models.chunk import Chunk
from app.models.document import Document, DocumentStatus
from app.models.embedding import ChunkEmbedding
from app.services.exceptions import (
    InvalidQueryError,
    InvalidScoreThresholdError,
    InvalidTopKError,
    RetrievalError,
)
from app.services.retrieval import RetrievalService

# Register sqlite adapter for list so in-memory SQLite tests can handle vector JSON
sqlite3.register_adapter(list, json.dumps)


@pytest.fixture
def sqlite_engine() -> sa.Engine:
    """Provide an in-memory SQLite engine with PRAGMA foreign_keys and cosine_distance user function."""
    engine = sa.create_engine("sqlite:///:memory:")

    @sa.event.listens_for(engine, "connect")
    def setup_sqlite_connection(conn: sqlite3.Connection, _record: object) -> None:
        cursor = conn.cursor()
        cursor.execute("PRAGMA foreign_keys=ON")
        cursor.close()

        def sqlite_cosine_distance(vec_a: str | list[float], vec_b: str | list[float]) -> float:
            if isinstance(vec_a, str):
                vec_a = json.loads(vec_a)
            if isinstance(vec_b, str):
                vec_b = json.loads(vec_b)
            # Dot product of unit-normalized vectors yields cosine similarity; distance = 1.0 - dot
            dot = sum(x * y for x, y in zip(vec_a, vec_b, strict=True))
            return 1.0 - dot

        conn.create_function("cosine_distance", 2, sqlite_cosine_distance)

    Base.metadata.create_all(engine)
    return engine


@pytest.fixture
def db_session(sqlite_engine: sa.Engine) -> Session:
    """Provide a scoped SQLite database session."""
    with Session(sqlite_engine) as session:
        yield session


@pytest.fixture
def mock_provider() -> DeterministicMockEmbeddingProvider:
    """Provide standard deterministic 384-dimensional mock embedding provider."""
    return DeterministicMockEmbeddingProvider(model_id="mock-bge-small-384", dimension=384)


@pytest.fixture
def retrieval_service(
    db_session: Session,
    mock_provider: DeterministicMockEmbeddingProvider,
) -> RetrievalService:
    """Provide RetrievalService instance backed by SQLite and mock provider."""
    settings = Settings(EMBEDDING_PROVIDER="mock", EMBEDDING_MODEL="mock-bge-small-384", EMBEDDING_DIMENSION=384)
    return RetrievalService(db=db_session, provider=mock_provider, settings=settings)


def _seed_document_with_chunks(
    db: Session,
    filename: str = "doc.txt",
    status: DocumentStatus = DocumentStatus.READY,
    chunk_texts: list[str] | None = None,
    model_id: str = "mock-bge-small-384",
    explicit_vectors: list[list[float]] | None = None,
) -> tuple[Document, list[Chunk], list[ChunkEmbedding]]:
    """Helper to populate document, chunks, and embeddings for testing."""
    doc = Document(
        original_filename=filename,
        content_type="text/plain",
        size_bytes=500,
        sha256=f"hash_{uuid.uuid4().hex[:16]}",
        storage_key=f"storage_{uuid.uuid4().hex[:8]}.txt",
        status=status,
    )
    db.add(doc)
    db.flush()

    texts = chunk_texts or ["Sample passage one.", "Sample passage two."]
    chunks: list[Chunk] = []
    embeddings: list[ChunkEmbedding] = []

    provider = DeterministicMockEmbeddingProvider(model_id=model_id, dimension=384)
    vectors = explicit_vectors or provider.embed_texts(texts)

    char_offset = 0
    for idx, (text, vec) in enumerate(zip(texts, vectors, strict=True)):
        chunk = Chunk(
            document_id=doc.id,
            chunk_index=idx,
            text=text,
            token_count=len(text.split()),
            character_count=len(text),
            start_char_offset=char_offset,
            end_char_offset=char_offset + len(text),
            source_format="text/plain",
            page_number=1 if idx == 0 else 2,
            page_numbers=[1] if idx == 0 else [2],
            heading=f"Section {idx + 1}",
            heading_level=2,
            metadata_={"tag": f"chunk-{idx}"},
        )
        db.add(chunk)
        db.flush()
        chunks.append(chunk)

        emb = ChunkEmbedding(
            chunk_id=chunk.id,
            document_id=doc.id,
            model_id=model_id,
            dimension=384,
            embedding=vec,
        )
        db.add(emb)
        embeddings.append(emb)
        char_offset += len(text) + 1

    db.commit()
    return doc, chunks, embeddings


# --- 1. Query and Input Validation Tests ---


def test_query_validation_empty_or_whitespace(retrieval_service: RetrievalService) -> None:
    """Verify empty or whitespace-only queries raise InvalidQueryError."""
    with pytest.raises(InvalidQueryError, match="empty or whitespace-only"):
        retrieval_service.retrieve("")

    with pytest.raises(InvalidQueryError, match="empty or whitespace-only"):
        retrieval_service.retrieve("   \n\t  ")


def test_query_validation_non_string(retrieval_service: RetrievalService) -> None:
    """Verify non-string query raises InvalidQueryError."""
    with pytest.raises(InvalidQueryError, match="must be a string"):
        retrieval_service.retrieve(12345)  # type: ignore[arg-type]


def test_query_validation_length_limit(retrieval_service: RetrievalService) -> None:
    """Verify queries exceeding 2,000 characters raise InvalidQueryError."""
    oversized_query = "a" * 2001
    with pytest.raises(InvalidQueryError, match="exceeds maximum permitted limit of 2,000 characters"):
        retrieval_service.retrieve(oversized_query)


def test_query_whitespace_is_trimmed(
    retrieval_service: RetrievalService,
    db_session: Session,
) -> None:
    """Verify query whitespace is trimmed and reported cleaned in response."""
    _seed_document_with_chunks(db_session)
    response = retrieval_service.retrieve("   hello world query   ")
    assert response.query == "hello world query"


def test_top_k_bounds_validation(retrieval_service: RetrievalService) -> None:
    """Verify top_k must be an integer between 1 and 50."""
    with pytest.raises(InvalidTopKError, match="between 1 and 50"):
        retrieval_service.retrieve("valid query", top_k=0)

    with pytest.raises(InvalidTopKError, match="between 1 and 50"):
        retrieval_service.retrieve("valid query", top_k=-5)

    with pytest.raises(InvalidTopKError, match="between 1 and 50"):
        retrieval_service.retrieve("valid query", top_k=51)

    with pytest.raises(InvalidTopKError, match="must be an integer"):
        retrieval_service.retrieve("valid query", top_k=True)  # type: ignore[arg-type]

    with pytest.raises(InvalidTopKError, match="must be an integer"):
        retrieval_service.retrieve("valid query", top_k="5")  # type: ignore[arg-type]


def test_score_threshold_bounds_validation(retrieval_service: RetrievalService) -> None:
    """Verify score_threshold must be a finite float in [-1.0, 1.0]."""
    with pytest.raises(InvalidScoreThresholdError, match="between -1.0 and 1.0"):
        retrieval_service.retrieve("valid query", score_threshold=-1.1)

    with pytest.raises(InvalidScoreThresholdError, match="between -1.0 and 1.0"):
        retrieval_service.retrieve("valid query", score_threshold=1.1)

    with pytest.raises(InvalidScoreThresholdError, match="between -1.0 and 1.0"):
        retrieval_service.retrieve("valid query", score_threshold=float("nan"))

    with pytest.raises(InvalidScoreThresholdError, match="must be a float"):
        retrieval_service.retrieve("valid query", score_threshold=True)  # type: ignore[arg-type]


def test_document_ids_validation_invalid_uuid(retrieval_service: RetrievalService) -> None:
    """Verify invalid document ID strings raise ValueError."""
    with pytest.raises(ValueError, match="Invalid document UUID"):
        retrieval_service.retrieve("valid query", document_ids=["not-a-uuid"])


# --- 2. Document Safety and Lifecycle Filtering Tests ---


def test_retrieval_excludes_non_ready_documents(
    retrieval_service: RetrievalService,
    db_session: Session,
) -> None:
    """Verify chunks belonging to UPLOADED, PROCESSING, or FAILED documents are never retrieved."""
    # Seed documents in various non-ready lifecycle states
    _seed_document_with_chunks(db_session, filename="uploaded.txt", status=DocumentStatus.UPLOADED)
    _seed_document_with_chunks(db_session, filename="processing.txt", status=DocumentStatus.PROCESSING)
    _seed_document_with_chunks(db_session, filename="failed.txt", status=DocumentStatus.FAILED)

    # Seed one READY document
    ready_doc, ready_chunks, _ = _seed_document_with_chunks(
        db_session, filename="ready.txt", status=DocumentStatus.READY
    )

    response = retrieval_service.retrieve("sample passage", top_k=10)

    # Only chunks from the READY document must appear in the result
    assert response.returned_count > 0
    for chunk in response.results:
        assert chunk.document_id == ready_doc.id
        assert chunk.document_filename == "ready.txt"


# --- 3. Model ID and Vector Space Isolation Tests ---


def test_retrieval_strictly_isolates_by_model_id(
    retrieval_service: RetrievalService,
    db_session: Session,
) -> None:
    """Verify chunks embedded under an incompatible or different model_id are excluded."""
    # Seed chunks with a foreign model_id
    _seed_document_with_chunks(
        db_session,
        filename="foreign_model.txt",
        status=DocumentStatus.READY,
        model_id="other-vendor-model-384",
    )

    # Seed chunks with active mock model_id
    active_doc, _, _ = _seed_document_with_chunks(
        db_session,
        filename="active_model.txt",
        status=DocumentStatus.READY,
        model_id="mock-bge-small-384",
    )

    response = retrieval_service.retrieve("sample query", top_k=10)

    assert response.returned_count > 0
    for chunk in response.results:
        assert chunk.model_id == "mock-bge-small-384"
        assert chunk.document_id == active_doc.id


# --- 4. Empty Results and No-Match Handling Tests ---


def test_empty_database_returns_empty_response(retrieval_service: RetrievalService) -> None:
    """Verify query against empty database returns empty results without error."""
    response = retrieval_service.retrieve("empty database query", top_k=5)
    assert response.returned_count == 0
    assert response.results == []
    assert response.model_id == "mock-bge-small-384"


def test_score_threshold_filters_distant_chunks(
    retrieval_service: RetrievalService,
    db_session: Session,
) -> None:
    """Verify score_threshold drops candidate chunks whose similarity is below the cutoff."""
    _seed_document_with_chunks(db_session, filename="doc.txt", chunk_texts=["Target text passage."])

    # With a high threshold (e.g. 0.999), non-identical vectors will be excluded
    response = retrieval_service.retrieve("Unrelated distinct query string", score_threshold=0.999)
    assert response.returned_count == 0
    assert response.results == []


# --- 5. Deterministic Tie-Breaking and Contract Tests ---


def test_deterministic_tie_breaking_order(
    retrieval_service: RetrievalService,
    db_session: Session,
) -> None:
    """Verify that equidistant chunks are ordered deterministically by document_id ASC, chunk_index ASC."""
    # Create two identical vectors
    unit_vec = [1.0] + [0.0] * 383
    doc1, _, _ = _seed_document_with_chunks(
        db_session,
        filename="doc1.txt",
        chunk_texts=["Text A"],
        explicit_vectors=[unit_vec],
    )
    doc2, _, _ = _seed_document_with_chunks(
        db_session,
        filename="doc2.txt",
        chunk_texts=["Text B"],
        explicit_vectors=[unit_vec],
    )

    # The mock query for "test" might produce any vector, but relative to unit_vec both have identical distance
    # Let's ensure tie-breaking happens deterministically
    response = retrieval_service.retrieve("query", top_k=5)
    assert response.returned_count >= 2

    # Verify results are sorted by cosine_distance first
    for i in range(len(response.results) - 1):
        curr = response.results[i]
        nxt = response.results[i + 1]
        assert curr.cosine_distance <= nxt.cosine_distance
        if curr.cosine_distance == nxt.cosine_distance:
            # Deterministic tie-breaker check
            assert (str(curr.document_id), curr.chunk_index) <= (str(nxt.document_id), nxt.chunk_index)


def test_retrieval_result_attributes_and_score_contract(
    retrieval_service: RetrievalService,
    db_session: Session,
) -> None:
    """Verify all chunk attributes, structural metadata, and score = 1.0 - distance contract."""
    doc, chunks, _ = _seed_document_with_chunks(
        db_session,
        filename="spec.pdf",
        chunk_texts=["First chunk with heading and metadata."],
    )

    response = retrieval_service.retrieve("First chunk", top_k=5)
    assert response.returned_count == 1
    result = response.results[0]

    assert result.chunk_id == chunks[0].id
    assert result.document_id == doc.id
    assert result.document_filename == "spec.pdf"
    assert result.chunk_index == 0
    assert result.text == "First chunk with heading and metadata."
    assert result.page_number == 1
    assert result.page_numbers == [1]
    assert result.heading == "Section 1"
    assert result.heading_level == 2
    assert result.metadata == {"tag": "chunk-0"}
    assert result.model_id == "mock-bge-small-384"

    # Score contract: score == 1.0 - cosine_distance exactly
    assert pytest.approx(result.score, rel=1e-6) == 1.0 - result.cosine_distance
    assert response.returned_count == len(response.results)


def test_document_ids_scoping_filter(
    retrieval_service: RetrievalService,
    db_session: Session,
) -> None:
    """Verify document_ids parameter restricts search to the specified documents."""
    doc1, _, _ = _seed_document_with_chunks(db_session, filename="target.txt")
    doc2, _, _ = _seed_document_with_chunks(db_session, filename="excluded.txt")

    response = retrieval_service.retrieve("query", top_k=10, document_ids=[doc1.id])
    assert response.returned_count > 0
    for chunk in response.results:
        assert chunk.document_id == doc1.id
        assert chunk.document_filename == "target.txt"


def test_provider_error_raises_retrieval_error(
    db_session: Session,
) -> None:
    """Verify provider embedding failure is safely wrapped in RetrievalError."""
    failing_provider = DeterministicMockEmbeddingProvider(should_fail=True)
    service = RetrievalService(db=db_session, provider=failing_provider)

    with pytest.raises(RetrievalError, match="Failed to generate query embedding"):
        service.retrieve("query text")


# --- 6. Live PostgreSQL pgvector Native Search Integration Test ---


def test_postgresql_native_pgvector_search() -> None:
    """Verify native PostgreSQL pgvector <=> operator with real transactional execution and rollback."""
    pg_url = "postgresql+psycopg://evidencedesk:replace_with_a_local_secret@127.0.0.1:5432/evidencedesk"
    try:
        engine = sa.create_engine(pg_url)
        with engine.connect() as conn:
            conn.execute(sa.text("SELECT 1"))
    except Exception:
        pytest.skip("Local PostgreSQL pgvector container is not reachable.")

    with Session(engine) as session:
        # Run inside a nested savepoint or roll back everything at the end
        try:
            doc = Document(
                original_filename="pg_test.txt",
                content_type="text/plain",
                size_bytes=100,
                sha256="pg_test_hash_" + uuid.uuid4().hex[:16],
                storage_key="pg_test_key_" + uuid.uuid4().hex[:8] + ".txt",
                status=DocumentStatus.READY,
            )
            session.add(doc)
            session.flush()

            # Orthogonal vectors: vector A along axis 0, vector B along axis 1
            vec_a = [1.0] + [0.0] * 383
            vec_b = [0.0, 1.0] + [0.0] * 382

            chunk_a = Chunk(
                document_id=doc.id,
                chunk_index=0,
                text="Axis 0 text",
                token_count=3,
                character_count=11,
                start_char_offset=0,
                end_char_offset=11,
            )
            chunk_b = Chunk(
                document_id=doc.id,
                chunk_index=1,
                text="Axis 1 text",
                token_count=3,
                character_count=11,
                start_char_offset=12,
                end_char_offset=23,
            )
            session.add_all([chunk_a, chunk_b])
            session.flush()

            emb_a = ChunkEmbedding(
                chunk_id=chunk_a.id,
                document_id=doc.id,
                model_id="mock-bge-small-384",
                dimension=384,
                embedding=vec_a,
            )
            emb_b = ChunkEmbedding(
                chunk_id=chunk_b.id,
                document_id=doc.id,
                model_id="mock-bge-small-384",
                dimension=384,
                embedding=vec_b,
            )
            session.add_all([emb_a, emb_b])
            session.flush()

            # Mock provider configured to output vec_a for query "query axis 0"
            provider = DeterministicMockEmbeddingProvider(model_id="mock-bge-small-384", dimension=384)
            # Patch embed_query to return vec_a exactly
            provider.embed_query = lambda q: vec_a  # type: ignore[assignment]

            service = RetrievalService(db=session, provider=provider)
            response = service.retrieve("query axis 0", top_k=2)

            assert response.returned_count == 2
            # vec_a <=> vec_a distance is 0.0, score is 1.0
            assert response.results[0].chunk_id == chunk_a.id
            assert pytest.approx(response.results[0].cosine_distance, abs=1e-5) == 0.0
            assert pytest.approx(response.results[0].score, abs=1e-5) == 1.0

            # vec_b <=> vec_a distance is 1.0 (orthogonal), score is 0.0
            assert response.results[1].chunk_id == chunk_b.id
            assert pytest.approx(response.results[1].cosine_distance, abs=1e-5) == 1.0
            assert pytest.approx(response.results[1].score, abs=1e-5) == 0.0

        finally:
            # Strictly rollback to avoid polluting the development database volume
            session.rollback()
