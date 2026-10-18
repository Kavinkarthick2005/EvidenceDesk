"""Semantic retrieval evaluation harness for EvidenceDesk.

Measures Recall@1, Recall@3, Recall@5, and Mean Reciprocal Rank (MRR)
over answerable queries, and evaluates score distributions and threshold trade-offs
(False Acceptance Rate vs False Rejection Rate) for in-domain unanswerable queries.

Gated behind the RUN_MODEL_TESTS=1 environment variable to keep standard CI
fast, offline, and deterministic.
"""

import json
import os
from pathlib import Path
from typing import Any
import uuid

import pytest
import sqlalchemy as sa
from sqlalchemy.orm import Session

from app.core.config import Settings
from app.db.base import Base
from app.embeddings.sentence_transformers import SentenceTransformersEmbeddingProvider
from app.models.chunk import Chunk
from app.models.document import Document, DocumentStatus
from app.models.embedding import ChunkEmbedding
from app.schemas.retrieval import RetrievedChunk
from app.services.chunk import ChunkPersistenceService
from app.services.embedding import EmbeddingService
from app.services.ingestion import IngestionService
from app.services.retrieval import RetrievalService
from app.storage.local import LocalStorageService

FIXTURES_DIR = Path(__file__).parent / "evaluation" / "fixtures"
CASES_FILE = Path(__file__).parent / "evaluation" / "baseline_retrieval_cases.json"


def is_chunk_relevant(chunk: RetrievedChunk, case: dict[str, Any]) -> bool:
    """Determine ground-truth relevance using document filename and evidence text containment.

    A retrieved chunk is relevant if it belongs to the target document AND
    encloses the labeled evidence text.
    Unanswerable cases (where answerable is False or evidence_text is empty) return False.
    """
    if not case.get("answerable", True):
        return False

    if chunk.document_filename != case["target_document_filename"]:
        return False

    evidence_text = case.get("evidence_text", "").strip().lower()
    return bool(evidence_text and evidence_text in chunk.text.lower())


@pytest.mark.skipif(
    os.getenv("RUN_MODEL_TESTS") != "1",
    reason="Semantic retrieval evaluation requires real model (RUN_MODEL_TESTS=1).",
)
def test_bge_small_semantic_retrieval_evaluation(tmp_path: Path) -> None:
    """Evaluate retrieval quality across an expanded multi-document corpus using IngestionService.

    Measures:
    - Recall@1, Recall@3, Recall@5, and MRR exclusively on answerable queries.
    - Empirical score distribution for answerable vs unanswerable queries.
    - False Acceptance Rate (FAR) and False Rejection Rate (FRR) across thresholds [0.55 .. 0.75].
    - Per-query near misses and failure diagnostics.
    """
    # 1. Load evaluation dataset
    assert CASES_FILE.is_file(), f"Evaluation dataset not found at {CASES_FILE}"
    with open(CASES_FILE, encoding="utf-8") as f:
        cases = json.load(f)

    answerable_cases = [c for c in cases if c.get("answerable", True)]
    unanswerable_cases = [c for c in cases if not c.get("answerable", True)]

    assert len(answerable_cases) >= 30, f"Expected >= 30 answerable cases, got {len(answerable_cases)}"
    assert len(unanswerable_cases) >= 5, f"Expected >= 5 unanswerable cases, got {len(unanswerable_cases)}"

    # 2. Database connection & two-layer savepoint transaction isolation
    pg_url = "postgresql+psycopg://evidencedesk:replace_with_a_local_secret@127.0.0.1:5432/evidencedesk"
    engine = sa.create_engine(pg_url)
    conn = engine.connect()
    outer_trans = conn.begin()
    session = Session(bind=conn, join_transaction_mode="create_savepoint")

    # 3. Setup temporary local storage for document files
    storage_root = tmp_path / "eval_storage"
    storage_root.mkdir(parents=True, exist_ok=True)
    storage = LocalStorageService(root_dir=storage_root, max_size_mb=25)

    # 4. Initialize real BGE-small provider
    provider = SentenceTransformersEmbeddingProvider(
        model_name_or_path="BAAI/bge-small-en-v1.5",
        expected_dimension=384,
        device="cpu",
    )

    settings = Settings(
        EMBEDDING_PROVIDER="sentence_transformers",
        EMBEDDING_MODEL="BAAI/bge-small-en-v1.5",
        EMBEDDING_DIMENSION=384,
    )

    ingestion_service = IngestionService(db=session, storage=storage, settings=settings)
    # Inject real SentenceTransformers provider
    ingestion_service.embedding_service.provider = provider

    ingested_doc_ids: list[uuid.UUID] = []

    try:
        # 5. Ingest all 4 fixture documents using production IngestionService
        fixture_files = sorted(FIXTURES_DIR.glob("*.md"))
        assert len(fixture_files) >= 4, f"Expected at least 4 fixture files in {FIXTURES_DIR}, found {len(fixture_files)}"

        total_ingested_chunks = 0
        for fixture_path in fixture_files:
            filename = fixture_path.name
            file_bytes = fixture_path.read_bytes()
            storage_key = storage.save(
                file_bytes,
                content_type="text/markdown",
                original_filename=filename,
            )
            doc = Document(
                original_filename=filename,
                content_type="text/markdown",
                size_bytes=len(file_bytes),
                sha256="eval_fixture_" + uuid.uuid4().hex[:16],
                storage_key=storage_key,
                status=DocumentStatus.UPLOADED,
            )
            session.add(doc)
            session.flush()
            ingested_doc_ids.append(doc.id)

            res = ingestion_service.ingest_document(doc.id)
            assert res.status == DocumentStatus.READY
            total_ingested_chunks += res.chunk_count

        assert total_ingested_chunks >= 30, f"Expected >= 30 chunks ingested across fixtures, got {total_ingested_chunks}"

        # 6. Initialize RetrievalService
        retrieval_service = RetrievalService(db=session, provider=provider, settings=settings)

        # 7. Evaluate Answerable Queries
        hit_at_1 = 0
        hit_at_3 = 0
        hit_at_5 = 0
        reciprocal_ranks: list[float] = []
        answerable_top1_scores: list[float] = []
        answerable_first_rel_scores: list[float] = []
        near_misses: list[tuple[dict[str, Any], int, float, str]] = []
        failures: list[dict[str, Any]] = []

        print("\n" + "=" * 80)
        print("EVIDENCEDESK — HARDENED RETRIEVAL EVALUATION REPORT (Milestone 7C)")
        print(f"Model: BAAI/bge-small-en-v1.5 | Indexed: {len(fixture_files)} docs, {total_ingested_chunks} chunks")
        print("=" * 80)

        for case in answerable_cases:
            resp = retrieval_service.retrieve(case["question"], top_k=5)
            first_rel_rank: int | None = None
            first_rel_score: float | None = None

            if resp.results:
                answerable_top1_scores.append(resp.results[0].score)

            for rank_idx, candidate in enumerate(resp.results, start=1):
                if is_chunk_relevant(candidate, case):
                    first_rel_rank = rank_idx
                    first_rel_score = candidate.score
                    break

            if first_rel_rank is not None and first_rel_score is not None:
                if first_rel_rank <= 1:
                    hit_at_1 += 1
                if first_rel_rank <= 3:
                    hit_at_3 += 1
                if first_rel_rank <= 5:
                    hit_at_5 += 1
                reciprocal_ranks.append(1.0 / first_rel_rank)
                answerable_first_rel_scores.append(first_rel_score)

                if first_rel_rank > 1:
                    top_cand_text = resp.results[0].text[:80].replace("\n", " ")
                    near_misses.append((case, first_rel_rank, first_rel_score, top_cand_text))
            else:
                reciprocal_ranks.append(0.0)
                answerable_first_rel_scores.append(-1.0)
                failures.append(case)

        total_ans = len(answerable_cases)
        recall_1 = hit_at_1 / total_ans
        recall_3 = hit_at_3 / total_ans
        recall_5 = hit_at_5 / total_ans
        mrr = sum(reciprocal_ranks) / total_ans

        # 8. Evaluate Unanswerable Queries
        unanswerable_max_scores: list[float] = []
        unans_details: list[tuple[dict[str, Any], float, str]] = []

        for unans_case in unanswerable_cases:
            resp = retrieval_service.retrieve(unans_case["question"], top_k=5)
            if resp.results:
                max_score = resp.results[0].score
                top_text = resp.results[0].text[:80].replace("\n", " ")
            else:
                max_score = -1.0
                top_text = "(empty)"
            unanswerable_max_scores.append(max_score)
            unans_details.append((unans_case, max_score, top_text))

        # 9. Compute Threshold Sweep (FAR vs FRR) across thresholds 0.55 .. 0.75
        threshold_levels = [0.55, 0.60, 0.65, 0.70, 0.75]
        far_values: dict[float, float] = {}
        frr_values: dict[float, float] = {}

        total_unans = len(unanswerable_cases)
        for tau in threshold_levels:
            # FAR: unanswerable query whose max score >= tau
            fa_count = sum(1 for s in unanswerable_max_scores if s >= tau)
            far_values[tau] = fa_count / total_unans if total_unans > 0 else 0.0

            # FRR: answerable query for which no relevant retrieved chunk meets tau
            # (i.e. first relevant score < tau or no relevant chunk found)
            fr_count = sum(1 for s in answerable_first_rel_scores if s < tau)
            frr_values[tau] = fr_count / total_ans

        # Print Evaluation Output
        print("\n--- 1. INFORMATION RETRIEVAL METRICS (Answerable Cases: %d) ---" % total_ans)
        print(f"Recall@1:        {recall_1:.4f} ({hit_at_1}/{total_ans})")
        print(f"Recall@3:        {recall_3:.4f} ({hit_at_3}/{total_ans})")
        print(f"Recall@5:        {recall_5:.4f} ({hit_at_5}/{total_ans})")
        print(f"MRR:             {mrr:.4f}")

        print("\n--- 2. SCORE DISTRIBUTIONS ---")
        if answerable_top1_scores:
            ans_avg = sum(answerable_top1_scores) / len(answerable_top1_scores)
            print(f"Answerable Top-1 Score:     Avg: {ans_avg:.3f} | Min: {min(answerable_top1_scores):.3f} | Max: {max(answerable_top1_scores):.3f}")
        if unanswerable_max_scores:
            unans_avg = sum(unanswerable_max_scores) / len(unanswerable_max_scores)
            print(f"Unanswerable Max Score:     Avg: {unans_avg:.3f} | Min: {min(unanswerable_max_scores):.3f} | Max: {max(unanswerable_max_scores):.3f}")

        print("\n--- 3. THRESHOLD TRADE-OFF SWEEP (FAR vs FRR) ---")
        for tau in threshold_levels:
            print(f"  tau = {tau:.2f} -> FAR: {far_values[tau] * 100:5.1f}% | FRR: {frr_values[tau] * 100:5.1f}%")

        if near_misses:
            print("\n--- 4. NEAR MISSES (Relevant Evidence at Rank 2-5) ---")
            for c, r, sc, top_txt in near_misses:
                print(f"  [{c['case_id']}] Rank {r} (Score: {sc:.3f}) | Q: {c['question'][:60]}...")
                print(f"     Top Distractor: {top_txt}...")

        if failures:
            print("\n--- 5. RETRIEVAL FAILURES (Evidence Not in Top 5) ---")
            for c in failures:
                print(f"  [{c['case_id']}] FAIL | Q: {c['question']}")

        print("\n--- 6. UNANSWERABLE QUERY RESPONSES ---")
        for c, sc, top_txt in unans_details:
            print(f"  [{c['case_id']}] Max Score: {sc:.3f} | Q: {c['question'][:60]}...")
            print(f"     Top Nearest Chunk: {top_txt}...")

        print("=" * 80 + "\n")

        # Basic mathematical sanity checks
        assert 0.0 <= recall_1 <= recall_3 <= recall_5 <= 1.0
        assert 0.0 <= mrr <= 1.0
        assert len(answerable_cases) == total_ans
        assert len(unanswerable_cases) == total_unans

    finally:
        # Layer 2 defensive cleanup: ensure all created documents are explicitly deleted
        try:
            if ingested_doc_ids:
                session.execute(sa.delete(Document).where(Document.id.in_(ingested_doc_ids)))
                session.commit()
        except Exception:
            pass

        # Layer 1 rollback: abort connection transaction
        outer_trans.rollback()
        conn.close()

    # Post-test verification: assert that the persistent database has 0 records
    with engine.connect() as check_conn:
        rem_docs = check_conn.execute(sa.select(sa.func.count(Document.id))).scalar()
        rem_chunks = check_conn.execute(sa.select(sa.func.count(Chunk.id))).scalar()
        rem_embs = check_conn.execute(sa.select(sa.func.count(ChunkEmbedding.id))).scalar()
        assert rem_docs == 0, f"Expected 0 persistent documents, found {rem_docs}"
        assert rem_chunks == 0, f"Expected 0 persistent chunks, found {rem_chunks}"
        assert rem_embs == 0, f"Expected 0 persistent embeddings, found {rem_embs}"
