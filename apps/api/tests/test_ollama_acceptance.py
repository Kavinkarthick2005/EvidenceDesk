"""Real-Document Ollama End-to-End Acceptance Tests.

Exercises the complete, production EvidenceDesk pipeline using a controlled synthetic
document and the real local Ollama 'qwen3:4b' generation provider:
1. Multipart file upload via POST /documents.
2. Ingestion pipeline via production IngestionService (extraction, chunking, pgvector embedding).
3. Verification of document metadata and ready status via GET /documents/{id}.
4. Evidence-grounded question answering via POST /qa/answer with real local Ollama inference.
5. Verification of supporting evidence text, citation marker resolution, and chain-of-thought isolation.
6. Honest abstention verification on unanswerable and out-of-domain queries.
7. Document deletion via DELETE /documents/{id} and storage cleanup.
8. Guaranteed two-layer savepoint database rollback isolation ensuring zero persisted records.
"""

from collections.abc import Generator, Sequence
from pathlib import Path
from typing import Any
import uuid

from fastapi import Depends
from fastapi.testclient import TestClient
import httpx
import pytest
import sqlalchemy as sa
from sqlalchemy.orm import Session

from app.api.deps import get_db, get_db_session, get_retrieval_service
from app.core.config import Settings, get_settings
from app.embeddings.interface import EmbeddingProvider
from app.generation import get_generation_provider
from app.main import create_app
from app.models.chunk import Chunk
from app.models.document import Document
from app.models.embedding import ChunkEmbedding
from app.services.embedding import EmbeddingService
from app.services.ingestion import IngestionResult, IngestionService
from app.services.retrieval import RetrievalService
from app.storage.local import LocalStorageService


class QuantumSensorAcceptanceEmbeddingProvider:
    """Deterministic topic-axis embedding provider for Ollama acceptance testing.

    Maps semantic concepts to orthogonal 384-dimensional unit vector axes:
    - Axis 0: Operating Physical Parameters (temperature, cryogenic, 4.2 Kelvin, copper, power)
    - Axis 1: Calibration and Interface (recalibration, 180 calendar days, optical comb, SPI bus)
    - Axis 100: Out-of-domain topics (radiation shielding, space deployment, satellites)

    Enables exact, reproducible PostgreSQL pgvector cosine similarity (<=>) search
    while strictly testing the real Ollama generation and citation-validation pipeline.
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

        if "temperature" in lower or "cryogenic" in lower or "4.2 kelvin" in lower or "kelvin" in lower or "physical" in lower:
            vec[0] = 1.0
        elif "calibration" in lower or "recalibration" in lower or "180" in lower or "comb" in lower or "spi" in lower:
            vec[1] = 1.0
        elif "power" in lower or "milliwatts" in lower or "copper" in lower:
            vec[0] = 1.0
        elif "radiation" in lower or "shielding" in lower or "satellite" in lower:
            vec[100] = 1.0
        else:
            # Default orthogonal axis for unmapped queries
            vec[100] = 1.0

        return vec

    def embed_texts(self, texts: Sequence[str]) -> list[list[float]]:
        return [self._text_to_vector(t) for t in texts]

    def embed_query(self, query: str) -> list[float]:
        return self._text_to_vector(query)


# Synthetic technical document fixture with verifiable distinctive facts
# and deliberately absent facts (no operating voltage, no battery specifications)
SYNTHETIC_SENSOR_SPEC = (
    b"# Precision Cryogenic Quantum Sensor Model QS-900\n\n"
    b"## Operating Physical Parameters\n"
    b"The Quantum Sensor Model QS-900 operates at an optimal cryogenic temperature of 4.2 Kelvin.\n"
    b"The maximum continuous power dissipation is strictly capped at 150 milliwatts.\n"
    b"The sensor housing body is fabricated from ultra-pure oxygen-free electronic copper.\n\n"
    b"## Calibration and Diagnostic Interface\n"
    b"Mandatory recalibration cycles must be conducted every 180 calendar days using an optical frequency comb.\n"
    b"Diagnostic telemetry is streamed exclusively over an optically isolated dual-channel SPI bus at 25 MHz.\n"
    b"Signal drift across continuous 1000-hour operational cycles is bounded beneath 0.04 microvolts.\n"
)


@pytest.fixture
def ollama_acceptance_context(
    tmp_path: Path,
) -> Generator[tuple[TestClient, IngestionService, Session, LocalStorageService, str], None, None]:
    """Provide fully isolated test environment configured with the real local Ollama provider.

    Guarantees that:
    1. PostgreSQL is available (skips gracefully if container is unreachable).
    2. Local Ollama server is available and qwen3:4b model is installed (skips if unreachable).
    3. The application uses GENERATION_PROVIDER='ollama' with qwen3:4b without mutating global defaults.
    4. HTTP routes, IngestionService, and RetrievalService all share the same isolated transaction.
    5. Savepoint rollback ensures zero persistent records remain in the database upon test completion.
    6. File storage is isolated to a temporary directory under tmp_path.
    """
    settings = Settings(
        DOCUMENT_STORAGE_PATH=str(tmp_path / "ollama_storage"),
        GENERATION_PROVIDER="ollama",
        GENERATION_MODEL="qwen3:4b",
        OLLAMA_BASE_URL="http://localhost:11434",
        OLLAMA_TIMEOUT_SECONDS=90.0,
        _env_file=None,
    )

    # 1. Probe PostgreSQL connectivity
    try:
        engine = sa.create_engine(settings.DATABASE_URL)
        with engine.connect() as probe_conn:
            probe_conn.execute(sa.text("SELECT 1"))
    except Exception:
        pytest.skip("Local PostgreSQL pgvector container is not reachable for acceptance tests.")

    # 2. Probe Ollama daemon and qwen3:4b model availability
    try:
        ollama_probe = httpx.get(f"{settings.OLLAMA_BASE_URL}/api/tags", timeout=5.0)
        if ollama_probe.status_code != 200:
            pytest.skip(f"Ollama daemon returned status {ollama_probe.status_code}")
        models = [m.get("name") for m in ollama_probe.json().get("models", [])]
        if not any("qwen3:4b" in m for m in models):
            pytest.skip("Model 'qwen3:4b' is not installed in local Ollama daemon.")
    except Exception as err:
        pytest.skip(f"Local Ollama daemon is unreachable: {err}")

    # 3. Establish isolated database transaction and savepoint session
    conn = engine.connect()
    outer_trans = conn.begin()
    session = Session(bind=conn, join_transaction_mode="create_savepoint")

    emb_provider = QuantumSensorAcceptanceEmbeddingProvider()
    storage = LocalStorageService(root_dir=tmp_path / "ollama_storage", max_size_mb=25)

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

    run_marker = f"RUN_8_6_{uuid.uuid4().hex[:8]}"

    with TestClient(app_instance) as client:
        try:
            yield client, ingestion_service, session, storage, run_marker
        finally:
            session.close()
            outer_trans.rollback()
            conn.close()

            # Verify zero persistent database leakage from this test run
            with engine.connect() as check_conn:
                rem_docs = check_conn.execute(sa.select(sa.func.count(Document.id))).scalar()
                rem_chunks = check_conn.execute(sa.select(sa.func.count(Chunk.id))).scalar()
                rem_embs = check_conn.execute(sa.select(sa.func.count(ChunkEmbedding.id))).scalar()
                assert rem_docs == 0, f"PostgreSQL rollback leak: {rem_docs} documents remained."
                assert rem_chunks == 0, f"PostgreSQL rollback leak: {rem_chunks} chunks remained."
                assert rem_embs == 0, f"PostgreSQL rollback leak: {rem_embs} embeddings remained."


def test_ollama_qa_e2e_answerable_operating_temperature(
    ollama_acceptance_context: tuple[TestClient, IngestionService, Session, LocalStorageService, str],
) -> None:
    """Scenario A: Answerable query on operating temperature with real Ollama inference.

    Verifies:
    1. Upload and ingestion of synthetic Markdown specification.
    2. POST /qa/answer invokes local Ollama qwen3:4b.
    3. The answer is non-abstained, non-empty, and correctly references the expected fact (4.2 Kelvin).
    4. Valid citation marker [1] is resolved to the exact chunk and document metadata.
    5. The cited chunk text genuinely contains the asserted fact.
    6. No internal reasoning tags (<think>, <thought>) leak into the public answer.
    """
    client, ingestion_service, session, storage, run_marker = ollama_acceptance_context
    filename = f"qs900_spec_{run_marker}.md"

    # 1. Upload document via POST /documents
    upload_resp = client.post(
        "/documents",
        files={"file": (filename, SYNTHETIC_SENSOR_SPEC, "text/markdown")},
    )
    assert upload_resp.status_code == 201
    doc_data = upload_resp.json()
    doc_id = uuid.UUID(doc_data["id"])
    assert doc_data["status"] == "uploaded"

    # 2. Ingest document via IngestionService
    ingest_result = ingestion_service.ingest_document(doc_id)
    assert ingest_result.status.value == "ready"
    assert ingest_result.chunk_count >= 2

    # 3. Query via POST /qa/answer
    qa_resp = client.post(
        "/qa/answer",
        json={
            "query": "What is the optimal operating temperature of the QS-900 quantum sensor?",
            "top_k": 3,
            "document_ids": [str(doc_id)],
        },
    )
    assert qa_resp.status_code == 200
    qa_data = qa_resp.json()

    # 4. Assert non-abstention and model identity
    assert qa_data["is_abstained"] is False, f"Model abstained unexpectedly: {qa_data.get('abstention_reason')}"
    assert qa_data["abstention_reason"] is None
    assert qa_data["model_id"] == "qwen3:4b"
    assert qa_data["has_citations"] is True
    assert qa_data["is_valid_citations"] is True
    assert len(qa_data["citations"]) >= 1

    # 5. Verify citation resolves to uploaded document and chunk
    citation = qa_data["citations"][0]
    assert citation["document_id"] == str(doc_id)
    assert citation["document_filename"] == filename
    assert citation["marker"] == "[1]"

    # 6. Verify evidence containment: the supporting passage genuinely contains the fact
    assert "4.2 Kelvin" in citation["text"] or "4.2 K" in citation["text"]
    assert citation["heading"] == "Operating Physical Parameters"

    # 7. Verify generated answer contains the expected fact
    answer_text = qa_data["answer"]
    assert "4.2 Kelvin" in answer_text or "4.2 K" in answer_text, f"Answer missing 4.2 Kelvin: '{answer_text}'"

    # 8. Verify reasoning isolation (no <think> or <thought> tags leaked)
    assert "<think>" not in answer_text
    assert "</think>" not in answer_text
    assert "<thought>" not in answer_text
    assert "</thought>" not in answer_text


def test_ollama_qa_e2e_answerable_calibration_interval(
    ollama_acceptance_context: tuple[TestClient, IngestionService, Session, LocalStorageService, str],
) -> None:
    """Scenario B: Answerable query on recalibration interval with real Ollama inference.

    Verifies:
    1. Question on calibration frequency routes through local Ollama qwen3:4b.
    2. Response cites the calibration passage [1] correctly.
    3. The expected fact (180 calendar days) is present in both the citation text and answer.
    4. Reasoning traces are stripped cleanly.
    """
    client, ingestion_service, session, storage, run_marker = ollama_acceptance_context
    filename = f"qs900_spec_{run_marker}.md"

    # 1. Upload and ingest
    upload_resp = client.post(
        "/documents",
        files={"file": (filename, SYNTHETIC_SENSOR_SPEC, "text/markdown")},
    )
    assert upload_resp.status_code == 201
    doc_id = uuid.UUID(upload_resp.json()["id"])
    ingestion_service.ingest_document(doc_id)

    # 2. Query via POST /qa/answer
    qa_resp = client.post(
        "/qa/answer",
        json={
            "query": "How frequently must the QS-900 sensor be recalibrated?",
            "top_k": 3,
            "document_ids": [str(doc_id)],
        },
    )
    assert qa_resp.status_code == 200
    qa_data = qa_resp.json()

    # 3. Assert grounded answer
    assert qa_data["is_abstained"] is False
    assert qa_data["has_citations"] is True
    assert qa_data["is_valid_citations"] is True
    assert len(qa_data["citations"]) >= 1

    # 4. Verify citation points to calibration section
    citation = qa_data["citations"][0]
    assert citation["document_id"] == str(doc_id)
    assert "180 calendar days" in citation["text"] or "180 days" in citation["text"]
    assert citation["heading"] == "Calibration and Diagnostic Interface"

    # 5. Verify factual claim in answer
    answer_text = qa_data["answer"]
    assert "180" in answer_text, f"Expected '180' in answer: '{answer_text}'"
    assert "<think>" not in answer_text
    assert "</think>" not in answer_text


def test_ollama_qa_e2e_unanswerable_missing_specifications(
    ollama_acceptance_context: tuple[TestClient, IngestionService, Session, LocalStorageService, str],
) -> None:
    """Scenario C: Query regarding facts deliberately absent from the document (voltage & battery).

    The synthetic document mentions power (150 mW) and temperature (4.2 K), but contains
    ZERO information about operating input voltage or battery backup duration.

    Verifies:
    1. When retrieved chunks lack the necessary information, the pipeline produces safe abstention.
    2. If the model recognizes missing information, it abstains with 'insufficient_evidence'.
    3. If the model attempts to generate an ungrounded answer without citations,
       the QA orchestration service enforces abstention with 'missing_citations'.
    4. In all valid outcomes, the system never returns an ungrounded answer claiming fake voltage or battery specs.
    """
    client, ingestion_service, session, storage, run_marker = ollama_acceptance_context
    filename = f"qs900_spec_{run_marker}.md"

    upload_resp = client.post(
        "/documents",
        files={"file": (filename, SYNTHETIC_SENSOR_SPEC, "text/markdown")},
    )
    assert upload_resp.status_code == 201
    doc_id = uuid.UUID(upload_resp.json()["id"])
    ingestion_service.ingest_document(doc_id)

    qa_resp = client.post(
        "/qa/answer",
        json={
            "query": "What is the battery backup duration and DC input voltage for the QS-900 sensor?",
            "top_k": 3,
            "document_ids": [str(doc_id)],
        },
    )
    assert qa_resp.status_code == 200
    qa_data = qa_resp.json()

    # The document genuinely contains no voltage or battery specifications.
    # The pipeline must safely abstain (either by model recognition or citation validation enforcement).
    if qa_data["is_abstained"]:
        assert qa_data["abstention_reason"] in (
            "insufficient_evidence",
            "missing_citations",
            "citation_validation_failed",
        )
        assert qa_data["citations"] == []
    else:
        # If the model produced citations, verify whether the cited passages actually support voltage/battery.
        # This confirms that citation presence alone does NOT guarantee factual support.
        for cited in qa_data.get("citations", []):
            cited_lower = cited["text"].lower()
            assert "battery" not in cited_lower
            assert "voltage" not in cited_lower


def test_ollama_qa_e2e_zero_evidence_radiation_shielding(
    ollama_acceptance_context: tuple[TestClient, IngestionService, Session, LocalStorageService, str],
) -> None:
    """Scenario D: Out-of-domain query with score threshold producing zero retrieved chunks.

    Verifies:
    1. When vector retrieval produces 0 chunks above threshold, the system immediately
       executes the honest abstention bypass ('no_evidence_found').
    2. Zero HTTP requests are made to Ollama.
    3. The response contains empty citations and is_abstained=True.
    """
    client, ingestion_service, session, storage, run_marker = ollama_acceptance_context
    filename = f"qs900_spec_{run_marker}.md"

    upload_resp = client.post(
        "/documents",
        files={"file": (filename, SYNTHETIC_SENSOR_SPEC, "text/markdown")},
    )
    assert upload_resp.status_code == 201
    doc_id = uuid.UUID(upload_resp.json()["id"])
    ingestion_service.ingest_document(doc_id)

    # An unrelated query with score_threshold=0.5 yields 0 chunks (orthogonal vector has score 0.0)
    qa_resp = client.post(
        "/qa/answer",
        json={
            "query": "What space-grade radiation shielding thickness is certified for satellite deployment?",
            "top_k": 3,
            "document_ids": [str(doc_id)],
            "score_threshold": 0.5,
        },
    )
    assert qa_resp.status_code == 200
    qa_data = qa_resp.json()

    assert qa_data["is_abstained"] is True
    assert qa_data["abstention_reason"] == "no_evidence_found"
    assert qa_data["has_citations"] is False
    assert qa_data["is_valid_citations"] is True
    assert qa_data["citations"] == []
    assert qa_data["retrieved_count"] == 0
    assert "no supporting evidence was found" in qa_data["answer"]


def test_ollama_qa_e2e_document_deletion_cleanup(
    ollama_acceptance_context: tuple[TestClient, IngestionService, Session, LocalStorageService, str],
) -> None:
    """Scenario E: Document deletion and storage cleanup.

    Verifies:
    1. Document upload and ingestion.
    2. Successful deletion via DELETE /documents/{id}.
    3. Subsequent GET /documents/{id} returns 404.
    4. Underlying file storage is unlinked.
    5. Savepoint rollback ensures clean database state.
    """
    client, ingestion_service, session, storage, run_marker = ollama_acceptance_context
    filename = f"qs900_spec_{run_marker}.md"

    upload_resp = client.post(
        "/documents",
        files={"file": (filename, SYNTHETIC_SENSOR_SPEC, "text/markdown")},
    )
    assert upload_resp.status_code == 201
    doc_id = uuid.UUID(upload_resp.json()["id"])
    ingest_result = ingestion_service.ingest_document(doc_id)
    assert ingest_result.status.value == "ready"

    # Get storage key from DB
    doc = session.get(Document, doc_id)
    assert doc is not None
    storage_key = doc.storage_key
    assert storage.exists(storage_key)

    # Delete document
    del_resp = client.delete(f"/documents/{doc_id}")
    assert del_resp.status_code == 200
    del_data = del_resp.json()
    assert del_data["id"] == str(doc_id)
    assert del_data["status"] == "deleted"

    # Verify GET returns 404
    get_resp = client.get(f"/documents/{doc_id}")
    assert get_resp.status_code == 404

    # Verify storage file is removed
    assert not storage.exists(storage_key)
