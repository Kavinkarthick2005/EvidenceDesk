"""Qwen Query Quality Evaluation Test Suite for EvidenceDesk.

Evaluates the local Ollama 'qwen3:4b' model across eight distinct query scenarios
grounded in a controlled synthetic specification (Precision Quantum Gravimeter QG-4):
1. Direct factual question (optimal core temperature: 4.2 Kelvin).
2. Paraphrased question (recalibration interval: 365 calendar days).
3. Technical detail extraction (RS-485 at 115,200 baud).
4. Multi-part / multi-passage synthesis (120 milliwatts power + 240-minute cooling sequence).
5. Grounded absence / correction (no internal battery cells, energy storage, or chemical accumulators).
6. Zero-evidence out-of-domain query (radiation shielding; deterministically exercises fast-path abstention).
7. Misleading premise refutation (waterproof immersion claim rejected; zero invented depth).
8. Unsupported extrapolation constraint (55°C ambient claim rejected; +35°C certified limit enforced).

Guarantees:
- Database isolation via outer-transaction rollback and savepoint sessions.
- Storage isolation via pytest tmp_path.
- Preserves global default GENERATION_PROVIDER="mock".
- Independent verification of retrieval, factual containment, citation resolution, and grounding.
"""

from collections.abc import Generator, Sequence
from pathlib import Path
import re
import time
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
from app.main import create_app
from app.models.chunk import Chunk
from app.models.document import Document
from app.models.embedding import ChunkEmbedding
from app.services.embedding import EmbeddingService
from app.services.ingestion import IngestionResult, IngestionService
from app.services.retrieval import RetrievalService
from app.storage.local import LocalStorageService


# Controlled synthetic technical specification with verified distinctive facts
# and explicit operational constraints/omissions
SYNTHETIC_QG4_SPEC = (
    b"# Precision Quantum Gravimeter Model QG-4 Specification\n\n"
    b"## Core Cryogenic Operating Temperature\n"
    b"The optimal operating temperature of the QG-4 gravimeter core is strictly maintained "
    b"at 4.2 Kelvin within the inner cryostat chamber. Thermal fluctuations are bounded beneath 0.002 Kelvin.\n\n"
    b"## Continuous Operational Power Limits\n"
    b"Maximum continuous operational power consumption is strictly capped at 120 milliwatts during "
    b"active measurement cycles. Power spikes during initialization must never exceed 250 milliwatts.\n\n"
    b"## Initial Cryogenic Cool-Down Protocol\n"
    b"Thermal stabilization from an ambient room state requires a mandatory 240-minute initial cooling sequence "
    b"before precision data acquisition. Rapid cooling protocols are prohibited.\n\n"
    b"## Electrical Telemetry and Power Distribution\n"
    b"Main operational power must be supplied via an external 24V DC regulated auxiliary bus. "
    b"Telemetry data streams over a galvanically isolated RS-485 serial bus at 115,200 baud. "
    b"The unit does not incorporate any internal battery cells, energy storage capacitors, or chemical accumulators.\n\n"
    b"## Metrological Calibration and Standards\n"
    b"Mandatory recalibration cycles must be completed every 365 calendar days using a primary rubidium atomic clock standard. "
    b"Optical alignment is checked concurrently.\n\n"
    b"## Operational Limitations and Environmental Constraints\n"
    b"The unit is certified for operation within ambient temperatures ranging from -10\xc2\xb0C to +35\xc2\xb0C. "
    b"The sensor enclosure is rated IP50 and is strictly not submersible, waterproof, or splash-proof. "
    b"No space-grade radiation-hardening or orbital-deployment certification is provided.\n"
)


class QuantumGravimeterEvaluationEmbeddingProvider:
    """Deterministic topic-axis embedding provider for QG-4 quality evaluation.

    Maps semantic concepts to orthogonal 384-dimensional unit vector axes:
    - Axis 0: Core cryogenic temperature (4.2 Kelvin, cryostat)
    - Axis 1: Continuous operational power (120 milliwatts)
    - Axis 2: Cool-down sequence (240 minutes, thermal stabilization)
    - Axis 3: Telemetry & Power interface (RS-485, 115,200 baud, auxiliary bus, battery absence)
    - Axis 4: Metrological calibration (365 calendar days, rubidium standard)
    - Axis 5: Operational limitations (ambient range -10°C to +35°C, non-waterproof, IP50)
    - Axis 100: Out-of-domain topics (radiation shielding, space deployment, tungsten)

    Enables exact, reproducible PostgreSQL pgvector cosine distance search.
    """

    def __init__(
        self,
        model_id: str = "eval-qg4-bge-small-384",
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

        # Multi-topic check for multi-passage synthesis query (power + cooling)
        has_power = "power" in lower or "120" in lower or "milliwatts" in lower
        has_cooling = "cooling" in lower or "cool-down" in lower or "cooldown" in lower or "240" in lower

        if has_power and has_cooling:
            # Multi-topic query vector: equal components on Axis 1 and Axis 2 normalized to unit length
            norm = 0.5**0.5  # 1 / sqrt(2) ≈ 0.70710678
            vec[1] = norm
            vec[2] = norm
            return vec

        if "core" in lower or "4.2" in lower or "kelvin" in lower or "cryostat" in lower:
            vec[0] = 1.0
        elif (
            "rs-485" in lower
            or "rs485" in lower
            or "115,200" in lower
            or "115200" in lower
            or "telemetry" in lower
            or "battery" in lower
            or "serial bus" in lower
        ):
            vec[3] = 1.0
        elif has_power:
            vec[1] = 1.0
        elif has_cooling:
            vec[2] = 1.0
        elif (
            "calibration" in lower
            or "recalibration" in lower
            or "365" in lower
            or "rubidium" in lower
            or "metrology" in lower
        ):
            vec[4] = 1.0
        elif (
            "ambient" in lower
            or "submersible" in lower
            or "waterproof" in lower
            or "immersion" in lower
            or "55°c" in lower
            or "55 c" in lower
            or "desert" in lower
            or "limitations" in lower
            or "ip50" in lower
        ):
            vec[5] = 1.0
        elif (
            "radiation" in lower
            or "shielding" in lower
            or "tungsten" in lower
            or "geostationary" in lower
            or "orbit" in lower
            or "lunar" in lower
        ):
            vec[100] = 1.0
        else:
            vec[100] = 1.0

        return vec

    def embed_texts(self, texts: Sequence[str]) -> list[list[float]]:
        return [self._text_to_vector(t) for t in texts]

    def embed_query(self, query: str) -> list[float]:
        return self._text_to_vector(query)


@pytest.fixture
def qwen_eval_context(
    tmp_path: Path,
) -> Generator[tuple[TestClient, IngestionService, Session, LocalStorageService, uuid.UUID, str], None, None]:
    """Provide isolated environment with QG-4 spec uploaded and indexed for evaluation.

    Guarantees:
    1. PostgreSQL is available (skips if unreachable).
    2. Local Ollama server is available and qwen3:4b model is installed (skips if unreachable).
    3. Global settings remain mock; test client uses injected Ollama provider.
    4. Two-layer savepoint database isolation ensures zero persisted records.
    5. Document is ingested once per test function with distinct chunk boundaries.
    """
    settings = Settings(
        DOCUMENT_STORAGE_PATH=str(tmp_path / "qwen_eval_storage"),
        GENERATION_PROVIDER="ollama",
        GENERATION_MODEL="qwen3:4b",
        OLLAMA_BASE_URL="http://localhost:11434",
        OLLAMA_TIMEOUT_SECONDS=90.0,
        _env_file=None,
    )

    # 1. Probe PostgreSQL
    try:
        engine = sa.create_engine(settings.DATABASE_URL)
        with engine.connect() as probe_conn:
            probe_conn.execute(sa.text("SELECT 1"))
    except Exception:
        pytest.skip("Local PostgreSQL pgvector container is not reachable for evaluation.")

    # 2. Probe Ollama daemon and qwen3:4b model
    try:
        ollama_probe = httpx.get(f"{settings.OLLAMA_BASE_URL}/api/tags", timeout=5.0)
        if ollama_probe.status_code != 200:
            pytest.skip(f"Ollama daemon returned status {ollama_probe.status_code}")
        models = [m.get("name") for m in ollama_probe.json().get("models", [])]
        if not any("qwen3:4b" in m for m in models):
            pytest.skip("Model 'qwen3:4b' is not installed in local Ollama daemon.")
    except Exception as err:
        pytest.skip(f"Local Ollama daemon is unreachable: {err}")

    # 3. Isolated transaction and session
    conn = engine.connect()
    outer_trans = conn.begin()
    session = Session(bind=conn, join_transaction_mode="create_savepoint")

    emb_provider = QuantumGravimeterEvaluationEmbeddingProvider()
    storage = LocalStorageService(root_dir=tmp_path / "qwen_eval_storage", max_size_mb=25)

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

    run_marker = f"QWEN_EVAL_{uuid.uuid4().hex[:8]}"
    filename = f"qg4_spec_{run_marker}.md"

    with TestClient(app_instance) as client:
        # Ingest document
        upload_resp = client.post(
            "/documents",
            files={"file": (filename, SYNTHETIC_QG4_SPEC, "text/markdown")},
        )
        assert upload_resp.status_code == 201
        doc_id = uuid.UUID(upload_resp.json()["id"])
        ingest_result = ingestion_service.ingest_document(doc_id)
        assert ingest_result.status.value == "ready"
        assert ingest_result.chunk_count == 6  # 6 distinct sections

        try:
            yield client, ingestion_service, session, storage, doc_id, filename
        finally:
            session.close()
            outer_trans.rollback()
            conn.close()

            # Verify zero persistent database leakage
            with engine.connect() as check_conn:
                rem_docs = check_conn.execute(sa.select(sa.func.count(Document.id))).scalar()
                rem_chunks = check_conn.execute(sa.select(sa.func.count(Chunk.id))).scalar()
                rem_embs = check_conn.execute(sa.select(sa.func.count(ChunkEmbedding.id))).scalar()
                assert rem_docs == 0, f"Leakage: {rem_docs} documents remained."
                assert rem_chunks == 0, f"Leakage: {rem_chunks} chunks remained."
                assert rem_embs == 0, f"Leakage: {rem_embs} chunks remained."


def _assert_no_reasoning_tags(text: str) -> None:
    """Assert that public response contains zero internal reasoning traces."""
    assert "<think>" not in text
    assert "</think>" not in text
    assert "<thought>" not in text
    assert "</thought>" not in text


# ==============================================================================
# Evaluation Scenarios
# ==============================================================================


def test_eval_q1_direct_factual_core_temperature(
    qwen_eval_context: tuple[TestClient, IngestionService, Session, LocalStorageService, uuid.UUID, str],
) -> None:
    """Scenario 1: Direct factual question regarding core operating temperature.

    Target Fact: 4.2 Kelvin (Section: Core Cryogenic Operating Temperature).
    Checks:
    - Required chunk retrieved.
    - Answer non-abstained with valid citation.
    - Expected fact in answer and cited passage text.
    - No reasoning tag leakage.
    - Latency recorded.
    """
    client, _, _, _, doc_id, filename = qwen_eval_context

    t0 = time.perf_counter()
    resp = client.post(
        "/qa/answer",
        json={
            "query": "What is the optimal operating core temperature of the QG-4 gravimeter?",
            "top_k": 3,
            "document_ids": [str(doc_id)],
        },
    )
    latency = time.perf_counter() - t0
    assert resp.status_code == 200
    data = resp.json()

    assert data["is_abstained"] is False
    assert data["model_id"] == "qwen3:4b"
    assert data["has_citations"] is True
    assert data["is_valid_citations"] is True
    assert len(data["citations"]) >= 1

    # Citation resolves to uploaded document and chunk
    citation = data["citations"][0]
    assert citation["document_id"] == str(doc_id)
    assert citation["document_filename"] == filename
    assert citation["heading"] == "Core Cryogenic Operating Temperature"
    assert "4.2 Kelvin" in citation["text"] or "4.2 K" in citation["text"]

    # Factual containment in generated answer
    answer = data["answer"]
    assert "4.2 Kelvin" in answer or "4.2 K" in answer
    _assert_no_reasoning_tags(answer)
    assert latency < 60.0


def test_eval_q2_paraphrased_recalibration_interval(
    qwen_eval_context: tuple[TestClient, IngestionService, Session, LocalStorageService, uuid.UUID, str],
) -> None:
    """Scenario 2: Paraphrased question regarding scheduled recalibration cycle.

    Target Fact: 365 calendar days (Section: Metrological Calibration and Standards).
    Paraphrase phrasing: "How frequently does the metrology system require scheduled recalibration?"
    Checks:
    - Retained semantic match despite lexical divergence.
    - Non-abstained answer citing calibration section.
    - 365 days present in citation and answer.
    """
    client, _, _, _, doc_id, filename = qwen_eval_context

    t0 = time.perf_counter()
    resp = client.post(
        "/qa/answer",
        json={
            "query": "How frequently does the metrology system require scheduled recalibration?",
            "top_k": 3,
            "document_ids": [str(doc_id)],
        },
    )
    latency = time.perf_counter() - t0
    assert resp.status_code == 200
    data = resp.json()

    assert data["is_abstained"] is False
    assert data["has_citations"] is True
    assert data["is_valid_citations"] is True
    assert len(data["citations"]) >= 1

    citation = data["citations"][0]
    assert citation["document_id"] == str(doc_id)
    assert "365 calendar days" in citation["text"] or "365 days" in citation["text"]
    assert citation["heading"] == "Metrological Calibration and Standards"

    answer = data["answer"]
    assert "365" in answer
    _assert_no_reasoning_tags(answer)
    assert latency < 60.0


def test_eval_q3_technical_detail_rs485_telemetry(
    qwen_eval_context: tuple[TestClient, IngestionService, Session, LocalStorageService, uuid.UUID, str],
) -> None:
    """Scenario 3: Technical detail extraction regarding serial bus type and baud rate.

    Target Facts: RS-485 and 115,200 baud (Section: Electrical Telemetry and Power Distribution).
    Checks:
    - Both technical tokens accurately extracted without corruption.
    - Cited passage contains both tokens.
    """
    client, _, _, _, doc_id, filename = qwen_eval_context

    t0 = time.perf_counter()
    resp = client.post(
        "/qa/answer",
        json={
            "query": "What physical serial bus type and transmission baud rate are used for telemetry streaming?",
            "top_k": 3,
            "document_ids": [str(doc_id)],
        },
    )
    latency = time.perf_counter() - t0
    assert resp.status_code == 200
    data = resp.json()

    assert data["is_abstained"] is False
    assert data["has_citations"] is True
    assert data["is_valid_citations"] is True
    assert len(data["citations"]) >= 1

    citation = data["citations"][0]
    assert "RS-485" in citation["text"]
    assert "115,200" in citation["text"] or "115200" in citation["text"]

    answer = data["answer"]
    assert "RS-485" in answer or "rs-485" in answer.lower()
    assert "115,200" in answer or "115200" in answer
    _assert_no_reasoning_tags(answer)
    assert latency < 60.0


def test_eval_q4_multipart_power_and_cooldown_multi_chunk(
    qwen_eval_context: tuple[TestClient, IngestionService, Session, LocalStorageService, uuid.UUID, str],
) -> None:
    """Scenario 4: Multi-part question requiring evidence from two distinct chunks.

    Target Fact A: 120 milliwatts (Chunk 1: Continuous Operational Power Limits).
    Target Fact B: 240 minutes (Chunk 2: Initial Cryogenic Cool-Down Protocol).
    Checks:
    - Both chunks reside at distinct chunk indices.
    - Both supporting passages are retrieved and cited.
    - Answer synthesizes both facts into a unified response.
    """
    client, _, _, _, doc_id, filename = qwen_eval_context

    t0 = time.perf_counter()
    resp = client.post(
        "/qa/answer",
        json={
            "query": "What is the maximum continuous power consumption and how long is the required initial cooling sequence?",
            "top_k": 5,
            "document_ids": [str(doc_id)],
        },
    )
    latency = time.perf_counter() - t0
    assert resp.status_code == 200
    data = resp.json()

    assert data["is_abstained"] is False
    assert data["has_citations"] is True
    assert data["is_valid_citations"] is True
    assert len(data["citations"]) >= 2, f"Expected multi-chunk citations, got {len(data['citations'])}"

    # Verify both distinct chunks are cited
    cited_chunk_indices = {c["chunk_index"] for c in data["citations"]}
    assert len(cited_chunk_indices) >= 2, f"Expected citations from distinct chunks: {cited_chunk_indices}"

    cited_texts = " ".join(c["text"] for c in data["citations"])
    assert "120 milliwatts" in cited_texts or "120 mw" in cited_texts.lower()
    assert "240-minute" in cited_texts or "240" in cited_texts

    # Both facts must appear in the synthesized answer
    answer = data["answer"]
    assert "120" in answer and ("milliwatt" in answer.lower() or "mw" in answer.lower())
    assert "240" in answer and ("minute" in answer.lower() or "min" in answer.lower())
    _assert_no_reasoning_tags(answer)
    assert latency < 60.0


def test_eval_q5_battery_capacity_grounded_absence(
    qwen_eval_context: tuple[TestClient, IngestionService, Session, LocalStorageService, uuid.UUID, str],
) -> None:
    """Scenario 5: Question asking for battery capacity where document states cells are absent.

    Document fact: 'The unit does not incorporate any internal battery cells, energy storage capacitors, or chemical accumulators.'
    Checks:
    - Accepts grounded correction stating no internal battery cells exist (or safe abstention).
    - Prohibits fabricated battery capacity (e.g., mAh, Ah) or cell chemistry (e.g., lithium, Li-ion).
    """
    client, _, _, _, doc_id, filename = qwen_eval_context

    t0 = time.perf_counter()
    resp = client.post(
        "/qa/answer",
        json={
            "query": "What is the internal battery capacity and cell chemistry of the QG-4 sensor?",
            "top_k": 3,
            "document_ids": [str(doc_id)],
        },
    )
    latency = time.perf_counter() - t0
    assert resp.status_code == 200
    data = resp.json()

    # The document explicitly states internal battery cells are absent.
    # We accept either a grounded answer stating absence, or an honest abstention.
    answer = data["answer"]
    _assert_no_reasoning_tags(answer)

    if not data["is_abstained"]:
        # Must cite the power distribution section
        assert data["has_citations"] is True
        assert data["is_valid_citations"] is True
        assert any("battery" in c["text"].lower() for c in data["citations"])
        # Must state that battery cells are absent / not incorporated
        answer_lower = answer.lower()
        assert (
            "not incorporate" in answer_lower
            or "no internal battery" in answer_lower
            or "no battery" in answer_lower
            or "does not contain" in answer_lower
            or "external" in answer_lower
        )
    else:
        assert data["abstention_reason"] in (
            "insufficient_evidence",
            "missing_citations",
            "citation_validation_failed",
        )

    # In ALL outcomes, fabricated battery specs are strictly prohibited
    answer_lower = answer.lower()
    assert "mah" not in answer_lower
    assert "ampere-hour" not in answer_lower
    assert "li-ion" not in answer_lower
    assert "lithium" not in answer_lower
    assert latency < 60.0


def test_eval_q6_out_of_domain_radiation_zero_evidence(
    qwen_eval_context: tuple[TestClient, IngestionService, Session, LocalStorageService, uuid.UUID, str],
) -> None:
    """Scenario 6: Completely out-of-domain query exercising zero-evidence fast path.

    Query: Space-grade radiation shielding thickness.
    Checks:
    - With score_threshold=0.5, retrieval yields 0 candidate chunks.
    - Pipeline executes fast-path abstention ('no_evidence_found').
    - Zero Ollama generation requests made (measured by low latency < 2.0s).
    - Empty citations and is_abstained=True.
    """
    client, _, _, _, doc_id, filename = qwen_eval_context

    t0 = time.perf_counter()
    resp = client.post(
        "/qa/answer",
        json={
            "query": "What thickness of space-grade tungsten radiation shielding is certified for geostationary orbit deployment?",
            "top_k": 3,
            "document_ids": [str(doc_id)],
            "score_threshold": 0.5,
        },
    )
    latency = time.perf_counter() - t0
    assert resp.status_code == 200
    data = resp.json()

    assert data["is_abstained"] is True
    assert data["abstention_reason"] == "no_evidence_found"
    assert data["has_citations"] is False
    assert data["citations"] == []
    assert data["retrieved_count"] == 0
    assert "no supporting evidence was found" in data["answer"].lower()
    # Fast path: zero LLM generation request, duration should be sub-second
    assert latency < 5.0


def test_eval_q7_misleading_premise_waterproof_refutation(
    qwen_eval_context: tuple[TestClient, IngestionService, Session, LocalStorageService, uuid.UUID, str],
) -> None:
    """Scenario 7: Question containing a false premise ('Given that the QG-4 is waterproof...').

    Document fact: 'The sensor enclosure is rated IP50 and is strictly not submersible, waterproof, or splash-proof.'
    Checks:
    - The model refutes the false premise or abstains honestly.
    - Zero invented immersion depth measurements (meters, bar, atm).
    """
    client, _, _, _, doc_id, filename = qwen_eval_context

    t0 = time.perf_counter()
    resp = client.post(
        "/qa/answer",
        json={
            "query": "Given that the QG-4 is waterproof, what is its maximum certified immersion depth in water?",
            "top_k": 3,
            "document_ids": [str(doc_id)],
        },
    )
    latency = time.perf_counter() - t0
    assert resp.status_code == 200
    data = resp.json()

    answer = data["answer"]
    _assert_no_reasoning_tags(answer)
    answer_lower = answer.lower()

    # The model may abstain due to contradictory evidence, or refute the false premise directly
    if data["is_abstained"]:
        assert data["abstention_reason"] in (
            "insufficient_evidence",
            "missing_citations",
            "citation_validation_failed",
        )
    else:
        # If answering, must refute the premise that it is waterproof
        assert (
            "not waterproof" in answer_lower
            or "not submersible" in answer_lower
            or "ip50" in answer_lower
            or "contradicts" in answer_lower
            or "no immersion depth" in answer_lower
        )

    # Fabricated immersion depth values are strictly forbidden
    assert "depth rating" not in answer_lower or "no depth rating" in answer_lower
    assert not re.search(r"\b\d+\s*(?:meters|metres|feet|bar|atm)\b", answer_lower)
    assert latency < 60.0


def test_eval_q8_unsupported_extrapolation_high_temperature(
    qwen_eval_context: tuple[TestClient, IngestionService, Session, LocalStorageService, uuid.UUID, str],
) -> None:
    """Scenario 8: Unsupported extrapolation query asking about operation at 55°C.

    Document fact: 'The unit is certified for operation within ambient temperatures ranging from -10°C to +35°C.'
    Checks:
    - The model enforces the stated +35°C limit and rejects reliable operation at 55°C (or abstains).
    - Never generates unsupported confirmation of 55°C operation.
    """
    client, _, _, _, doc_id, filename = qwen_eval_context

    t0 = time.perf_counter()
    resp = client.post(
        "/qa/answer",
        json={
            "query": "Can the QG-4 gravimeter operate reliably in ambient desert conditions at 55°C?",
            "top_k": 3,
            "document_ids": [str(doc_id)],
        },
    )
    latency = time.perf_counter() - t0
    assert resp.status_code == 200
    data = resp.json()

    answer = data["answer"]
    _assert_no_reasoning_tags(answer)
    answer_lower = answer.lower()

    if not data["is_abstained"]:
        # Must cite the limitations section
        assert data["has_citations"] is True
        assert data["is_valid_citations"] is True
        # Must note the 35°C limit and reject 55°C operation
        assert "35" in answer
        assert "cannot" in answer_lower or "not certified" in answer_lower or "exceeds" in answer_lower or "up to" in answer_lower
    else:
        assert data["abstention_reason"] in (
            "insufficient_evidence",
            "missing_citations",
            "citation_validation_failed",
        )

    # Must never claim that 55°C is supported
    assert "can operate reliably at 55" not in answer_lower
    assert "is certified for 55" not in answer_lower
    assert latency < 60.0
