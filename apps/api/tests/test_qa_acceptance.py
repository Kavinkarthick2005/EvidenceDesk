"""End-to-End QA Acceptance Tests for EvidenceDesk.

Exercises the complete, production pipeline:
1. Multipart file upload via POST /documents.
2. Structure extraction, windowed chunking, vector embedding, and atomic persistence
   via production IngestionService.
3. Metadata verification via GET /documents/{id}.
4. Semantic vector retrieval, evidence framing, answer generation, and citation validation
   via POST /qa/answer.
5. Grounded citation traceability to real document UUIDs, chunk UUIDs, headings, and scores.
6. Honest abstention on unanswerable queries and out-of-domain topics.
7. Scoped retrieval by document_ids and score_threshold bounds.
8. Cascading deletion via DELETE /documents/{id} and storage file cleanup.
9. Guaranteed two-layer savepoint database isolation ensuring 0 persistent records remain.
"""

from collections.abc import Generator, Sequence
from pathlib import Path
import uuid

from fastapi import Depends
from fastapi.testclient import TestClient
import pytest
import sqlalchemy as sa
from sqlalchemy.orm import Session

from app.api.deps import get_db, get_db_session, get_retrieval_service
from app.core.config import Settings, get_settings
from app.embeddings.interface import EmbeddingProvider
from app.main import create_app
from app.models.chunk import Chunk
from app.models.document import Document
from app.models.embedding import ChunkEmbedding
from app.services.embedding import EmbeddingService
from app.services.ingestion import IngestionResult, IngestionService
from app.services.retrieval import RetrievalService
from app.storage.local import LocalStorageService


class AcceptanceEmbeddingProvider:
    """Deterministic topic-axis embedding provider for acceptance testing.

    Maps semantic concepts to orthogonal 384-dimensional unit vector axes:
    - Axis 0: Revenue, growth, financials
    - Axis 1: Capital deployment, share repurchases
    - Axis 2: Governance, debt maturity, audit
    - Axis 3: HR, personnel, employment policies
    - Axis 100: Out-of-domain / unanswerable topics (e.g. cafeteria soup)

    Produces unit-normalized vectors (||v||_2 == 1.0) with zero external network
    calls or model downloads, enabling exact, deterministic pgvector cosine similarity.
    """

    def __init__(
        self,
        model_id: str = "acceptance-bge-small-384",
        dimension: int = 384,
    ) -> None:
        self._model_id = model_id
        self._dimension = dimension
        self._truncated_chunks_count = 0

    @property
    def model_id(self) -> str:
        return self._model_id

    @property
    def dimension(self) -> int:
        return self._dimension

    @property
    def max_sequence_length(self) -> int:
        return 512

    @property
    def truncated_chunks_count(self) -> int:
        return self._truncated_chunks_count

    def _text_to_vector(self, text: str) -> list[float]:
        lower = text.lower()
        vec = [0.0] * self._dimension

        if "revenue" in lower or "3.42 billion" in lower or "subscription" in lower:
            vec[0] = 1.0
        elif "repurchase" in lower or "500 million" in lower or "capital" in lower:
            vec[1] = 1.0
        elif "debt" in lower or "2033" in lower or "governance" in lower:
            vec[2] = 1.0
        elif "hr" in lower or "employee" in lower or "policy" in lower or "leave" in lower:
            vec[3] = 1.0
        else:
            # Orthogonal axis for unknown or unanswerable queries
            vec[100] = 1.0

        return vec

    def embed_texts(self, texts: Sequence[str]) -> list[list[float]]:
        return [self._text_to_vector(t) for t in texts]

    def embed_query(self, query: str) -> list[float]:
        return self._text_to_vector(query)


# Synthetic acceptance fixtures
FIXTURE_FINANCIAL_REPORT = (
    b"# Apex Technologies Annual Report 2026\n\n"
    b"## Revenue and Growth\n"
    b"In fiscal year 2026, Apex Technologies reported total consolidated revenue of $3.42 billion.\n"
    b"Enterprise cloud subscription revenue increased by 31.5% year-over-year.\n\n"
    b"## Capital Deployment\n"
    b"The Board of Directors approved an aggregate share repurchase authorization of $500 million.\n"
    b"Capital expenditures were strictly allocated at $320 million for infrastructure upgrades.\n\n"
    b"## Corporate Governance\n"
    b"Apex Technologies maintained 100% independent oversight on the Audit Committee throughout 2026.\n"
    b"No long-term debt instruments are scheduled to mature prior to September 2033.\n"
)

FIXTURE_HR_HANDBOOK = (
    b"# Apex Technologies Employee Handbook 2026\n\n"
    b"## Paid Time Off Policy\n"
    b"All full-time employees accrue 25 business days of paid time off annually.\n"
    b"Carryover is capped at 5 unused days into the subsequent calendar year.\n\n"
    b"## Remote Work Guidelines\n"
    b"Employees may work remotely up to two days per week with departmental director approval.\n"
)


@pytest.fixture
def acceptance_context(
    tmp_path: Path,
) -> Generator[tuple[TestClient, IngestionService, Session, LocalStorageService], None, None]:
    """Provide fully isolated acceptance test environment bound to a two-layer savepoint transaction.

    Guarantees that:
    1. All HTTP route handlers (upload, get, delete, QA) use the isolated test database session.
    2. IngestionService uses the same isolated session and storage.
    3. Storage operations write into a dedicated temporary directory.
    4. Upon test completion, outer transaction rollback ensures 0 records persist in PostgreSQL.
    """
    settings = Settings(
        DOCUMENT_STORAGE_PATH=str(tmp_path / "acceptance_storage"),
        _env_file=None,
    )

    try:
        engine = sa.create_engine(settings.DATABASE_URL)
        with engine.connect() as probe_conn:
            probe_conn.execute(sa.text("SELECT 1"))
    except Exception:
        pytest.skip("Local PostgreSQL pgvector container is not reachable for acceptance tests.")

    conn = engine.connect()
    outer_trans = conn.begin()
    session = Session(bind=conn, join_transaction_mode="create_savepoint")

    emb_provider = AcceptanceEmbeddingProvider()
    storage = LocalStorageService(root_dir=tmp_path / "acceptance_storage", max_size_mb=25)

    app_instance = create_app()

    def override_db() -> Generator[Session, None, None]:
        yield session

    def override_settings() -> Settings:
        return settings

    def override_retrieval(
        db: Session = Depends(override_db),
        s: Settings = Depends(override_settings),
    ) -> RetrievalService:
        return RetrievalService(db=db, provider=emb_provider, settings=s)

    app_instance.dependency_overrides[get_db_session] = override_db
    app_instance.dependency_overrides[get_db] = override_db
    app_instance.dependency_overrides[get_settings] = override_settings
    app_instance.dependency_overrides[get_retrieval_service] = override_retrieval

    ingestion_service = IngestionService(
        db=session,
        storage=storage,
        embedding_service=EmbeddingService(db=session, provider=emb_provider, settings=settings),
        settings=settings,
    )

    with TestClient(app_instance) as client:
        try:
            yield client, ingestion_service, session, storage
        finally:
            session.close()
            outer_trans.rollback()
            conn.close()

            # Verify zero persistent database leakage
            with engine.connect() as check_conn:
                rem_docs = check_conn.execute(sa.select(sa.func.count(Document.id))).scalar()
                rem_chunks = check_conn.execute(sa.select(sa.func.count(Chunk.id))).scalar()
                rem_embs = check_conn.execute(sa.select(sa.func.count(ChunkEmbedding.id))).scalar()
                assert rem_docs == 0, f"PostgreSQL rollback leak: {rem_docs} documents remained."
                assert rem_chunks == 0, f"PostgreSQL rollback leak: {rem_chunks} chunks remained."
                assert rem_embs == 0, f"PostgreSQL rollback leak: {rem_embs} embeddings remained."


def test_qa_e2e_full_lifecycle_grounded_answer(
    acceptance_context: tuple[TestClient, IngestionService, Session, LocalStorageService],
) -> None:
    """Verify full grounded lifecycle: upload -> ingest -> verify ready -> QA -> grounded citation."""
    client, ingestion_service, session, storage = acceptance_context

    # 1. Upload via POST /documents
    upload_resp = client.post(
        "/documents",
        files={"file": ("apex_annual_report_2026.md", FIXTURE_FINANCIAL_REPORT, "text/markdown")},
    )
    assert upload_resp.status_code == 201
    doc_data = upload_resp.json()
    doc_id = uuid.UUID(doc_data["id"])
    assert doc_data["status"] == "uploaded"
    assert doc_data["original_filename"] == "apex_annual_report_2026.md"

    # 2. Ingest via production IngestionService
    ingest_result = ingestion_service.ingest_document(doc_id)
    assert ingest_result.status.value == "ready"
    assert ingest_result.chunk_count >= 3

    # 3. Verify document readiness via GET /documents/{id}
    get_resp = client.get(f"/documents/{doc_id}")
    assert get_resp.status_code == 200
    assert get_resp.json()["status"] == "ready"

    # 4. Ask grounded question via POST /qa/answer
    qa_resp = client.post(
        "/qa/answer",
        json={"query": "What was the consolidated revenue for fiscal year 2026?", "top_k": 3},
    )
    assert qa_resp.status_code == 200
    qa_data = qa_resp.json()

    # Assert honest, grounded answer
    assert qa_data["is_abstained"] is False
    assert qa_data["abstention_reason"] is None
    assert qa_data["has_citations"] is True
    assert qa_data["is_valid_citations"] is True
    assert qa_data["retrieved_count"] >= 1
    assert len(qa_data["citations"]) >= 1

    # Verify citation links to actual document and chunk
    citation = qa_data["citations"][0]
    assert citation["marker"] == "[1]"
    assert citation["marker_id"] == "1"
    assert citation["document_id"] == str(doc_id)
    assert citation["document_filename"] == "apex_annual_report_2026.md"
    assert "$3.42 billion" in citation["text"]
    assert citation["score"] == 1.0  # Exact match on Axis 0
    assert citation["heading"] == "Revenue and Growth"


def test_qa_e2e_unanswerable_query_honest_abstention(
    acceptance_context: tuple[TestClient, IngestionService, Session, LocalStorageService],
) -> None:
    """Verify honest abstention on an out-of-domain question without supporting evidence."""
    client, ingestion_service, _, _ = acceptance_context

    # Ingest document
    upload_resp = client.post(
        "/documents",
        files={"file": ("apex_annual_report_2026.md", FIXTURE_FINANCIAL_REPORT, "text/markdown")},
    )
    doc_id = uuid.UUID(upload_resp.json()["id"])
    ingestion_service.ingest_document(doc_id)

    # Ask unanswerable question with a standard similarity threshold
    qa_resp = client.post(
        "/qa/answer",
        json={
            "query": "What is the secret recipe for cafeteria soup?",
            "score_threshold": 0.5,
        },
    )
    assert qa_resp.status_code == 200
    qa_data = qa_resp.json()

    assert qa_data["is_abstained"] is True
    assert qa_data["abstention_reason"] == "no_evidence_found"
    assert qa_data["has_citations"] is False
    assert qa_data["citations"] == []
    assert "no supporting evidence was found" in qa_data["answer"]


def test_qa_e2e_document_id_scoping(
    acceptance_context: tuple[TestClient, IngestionService, Session, LocalStorageService],
) -> None:
    """Verify document_ids filtering scopes retrieval exclusively to authorized documents."""
    client, ingestion_service, _, _ = acceptance_context

    # Upload Doc A (Financial Report)
    resp_a = client.post(
        "/documents",
        files={"file": ("apex_financials.md", FIXTURE_FINANCIAL_REPORT, "text/markdown")},
    )
    doc_a_id = uuid.UUID(resp_a.json()["id"])
    ingestion_service.ingest_document(doc_a_id)

    # Upload Doc B (HR Handbook)
    resp_b = client.post(
        "/documents",
        files={"file": ("apex_hr.md", FIXTURE_HR_HANDBOOK, "text/markdown")},
    )
    doc_b_id = uuid.UUID(resp_b.json()["id"])
    ingestion_service.ingest_document(doc_b_id)

    # Query targeting financials but scoped to Doc B (HR) with threshold
    scoped_resp_mismatch = client.post(
        "/qa/answer",
        json={
            "query": "What was the total revenue?",
            "document_ids": [str(doc_b_id)],
            "score_threshold": 0.5,
        },
    )
    assert scoped_resp_mismatch.status_code == 200
    mismatch_data = scoped_resp_mismatch.json()
    assert mismatch_data["is_abstained"] is True
    assert mismatch_data["abstention_reason"] == "no_evidence_found"

    # Query targeting financials scoped to Doc A (Financials)
    scoped_resp_match = client.post(
        "/qa/answer",
        json={
            "query": "What was the total revenue?",
            "document_ids": [str(doc_a_id)],
        },
    )
    assert scoped_resp_match.status_code == 200
    match_data = scoped_resp_match.json()
    assert match_data["is_abstained"] is False
    assert all(c["document_id"] == str(doc_a_id) for c in match_data["citations"])


def test_qa_e2e_score_threshold_filtering(
    acceptance_context: tuple[TestClient, IngestionService, Session, LocalStorageService],
) -> None:
    """Verify score_threshold suppresses weakly-matching candidate passages."""
    client, ingestion_service, _, _ = acceptance_context

    upload_resp = client.post(
        "/documents",
        files={"file": ("apex_annual_report_2026.md", FIXTURE_FINANCIAL_REPORT, "text/markdown")},
    )
    doc_id = uuid.UUID(upload_resp.json()["id"])
    ingestion_service.ingest_document(doc_id)

    # Unattainable score threshold
    strict_resp = client.post(
        "/qa/answer",
        json={
            "query": "What is the policy on employee vacation?",
            "score_threshold": 0.999,
        },
    )
    assert strict_resp.status_code == 200
    strict_data = strict_resp.json()
    assert strict_data["is_abstained"] is True
    assert strict_data["abstention_reason"] == "no_evidence_found"
    assert strict_data["retrieved_count"] == 0

    # Permissive threshold retrieves available passages
    permissive_resp = client.post(
        "/qa/answer",
        json={
            "query": "Tell me about revenue",
            "score_threshold": 0.1,
        },
    )
    assert permissive_resp.status_code == 200
    permissive_data = permissive_resp.json()
    assert permissive_data["is_abstained"] is False
    assert permissive_data["retrieved_count"] >= 1


def test_qa_e2e_cascading_deletion_and_cleanup(
    acceptance_context: tuple[TestClient, IngestionService, Session, LocalStorageService],
) -> None:
    """Verify DELETE /documents unlinks storage and cascades to chunks, embeddings, and QA search."""
    client, ingestion_service, session, storage = acceptance_context

    # 1. Ingest document
    upload_resp = client.post(
        "/documents",
        files={"file": ("apex_annual_report_2026.md", FIXTURE_FINANCIAL_REPORT, "text/markdown")},
    )
    doc_id = uuid.UUID(upload_resp.json()["id"])
    doc = session.get(Document, doc_id)
    assert doc is not None
    storage_key = doc.storage_key
    ingestion_service.ingest_document(doc_id)

    # Storage file exists
    assert storage.exists(storage_key) is True

    # 2. Delete document via DELETE /documents/{id}
    del_resp = client.delete(f"/documents/{doc_id}")
    assert del_resp.status_code == 200
    assert del_resp.json()["status"] == "deleted"

    # 3. Verify GET returns 404
    get_resp = client.get(f"/documents/{doc_id}")
    assert get_resp.status_code == 404

    # 4. Verify storage file was unlinked
    assert storage.exists(storage_key) is False

    # 5. Verify database chunks and embeddings were cascaded
    remaining_chunks = session.query(Chunk).filter(Chunk.document_id == doc_id).count()
    assert remaining_chunks == 0

    # 6. Verify QA search no longer retrieves the deleted document
    qa_resp = client.post(
        "/qa/answer",
        json={"query": "What was the consolidated revenue for fiscal year 2026?"},
    )
    assert qa_resp.status_code == 200
    qa_data = qa_resp.json()
    assert qa_data["is_abstained"] is True
    assert qa_data["abstention_reason"] == "no_evidence_found"


def test_qa_e2e_database_rollback_isolation_proof() -> None:
    """Explicitly prove that savepoint rollback guarantees zero persistent database mutation."""
    settings = Settings(_env_file=None)
    try:
        engine = sa.create_engine(settings.DATABASE_URL)
        with engine.connect() as probe_conn:
            probe_conn.execute(sa.text("SELECT 1"))
    except Exception:
        pytest.skip("Local PostgreSQL pgvector container is not reachable.")

    with engine.connect() as check_conn:
        before_docs = check_conn.execute(sa.select(sa.func.count(Document.id))).scalar()
        before_chunks = check_conn.execute(sa.select(sa.func.count(Chunk.id))).scalar()
        before_embs = check_conn.execute(sa.select(sa.func.count(ChunkEmbedding.id))).scalar()

    assert before_docs == 0
    assert before_chunks == 0
    assert before_embs == 0

    conn = engine.connect()
    outer_trans = conn.begin()
    session = Session(bind=conn, join_transaction_mode="create_savepoint")
    try:
        doc = Document(
            id=uuid.uuid4(),
            original_filename="isolation_proof.md",
            content_type="text/markdown",
            size_bytes=42,
            sha256="isolation_proof_sha256",
            storage_key="isolation_proof_storage_key",
            status="ready",
        )
        session.add(doc)
        session.commit()

        chunk = Chunk(
            id=uuid.uuid4(),
            document_id=doc.id,
            chunk_index=0,
            text="Isolation text",
            token_count=2,
            character_count=14,
            start_char_offset=0,
            end_char_offset=14,
        )
        session.add(chunk)
        session.commit()

        emb = ChunkEmbedding(
            chunk_id=chunk.id,
            document_id=doc.id,
            model_id="test-model",
            dimension=384,
            embedding=[0.0] * 384,
        )
        session.add(emb)
        session.commit()

        # Visible inside savepoint session
        assert session.query(Document).filter(Document.id == doc.id).count() == 1
    finally:
        session.close()
        outer_trans.rollback()
        conn.close()

    # Verify that in a separate connection, count is strictly 0
    with engine.connect() as check_conn:
        after_docs = check_conn.execute(sa.select(sa.func.count(Document.id))).scalar()
        after_chunks = check_conn.execute(sa.select(sa.func.count(Chunk.id))).scalar()
        after_embs = check_conn.execute(sa.select(sa.func.count(ChunkEmbedding.id))).scalar()

    assert after_docs == 0
    assert after_chunks == 0
    assert after_embs == 0
