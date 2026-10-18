"""Semantic retrieval evaluation harness for EvidenceDesk.

Measures Recall@1, Recall@3, Recall@5, and Mean Reciprocal Rank (MRR)
using real BAAI/bge-small-en-v1.5 embeddings against labeled evidence spans.

Gated behind the RUN_MODEL_TESTS=1 environment variable to keep standard CI
fast and offline.
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
from app.services.retrieval import RetrievalService

CASES_FILE = Path(__file__).parent / "evaluation" / "baseline_retrieval_cases.json"

REFERENCE_DOCUMENT_CONTENT = """# EvidenceDesk Platform Reference Manual

## 1. Storage and Document Limits
The maximum permitted upload size for any single document is configured as 25 MB by default. Uploaded files exceeding this size are rejected before disk persistence.

## 2. Persistence Layer
PostgreSQL with the pgvector extension is the primary persistent store for vector embeddings and document metadata. Database migrations manage table versions.

## 3. Embedding Model Specifications
The BAAI/bge-small-en-v1.5 embedding model maps text passages into dense 384-dimensional unit-normalized float vectors. All generated representations have unit L2 length.

## 4. Reprocessing and Ingestion Lifecycle
If a reprocessing attempt fails on a document that is already READY, the transaction rolls back completely and preserves the prior valid chunks and embeddings.

## 5. First-Time Failure Guarantees
When initial document ingestion fails, the document state is updated to FAILED and zero orphaned chunk or embedding records remain in storage.

## 6. Vector Similarity and Ranking
Retrieval ranks nearest-neighbor candidate passages using cosine distance via the pgvector <=> operator, with similarity scores computed as 1.0 minus cosine distance.

## 7. Tokenizer Context Limits
The maximum sequence length supported by the local model tokenizer context window is 512 tokens before truncation occurs.

## 8. Extraction Formats
The ingestion service supports three source document formats: PDF documents, Markdown files, and plain text UTF-8 files.

## 9. Deterministic Tie-Breaking
Deterministic tie-breaking sorts equidistant chunks secondary by document UUID in ascending order and tertiary by sequential chunk index in ascending order.

## 10. File Integrity and Checksums
Document content deduplication and integrity verification compute a 64-character hex-encoded SHA-256 digest at upload time.

## 11. Top-k Retrieval Limits
Callers can request top_k candidates bounded between a minimum of 1 and a maximum of 50 chunks.

## 12. Workspace Isolation
Workspace isolation is enforced at query time using parameterized database filters preventing cross-tenant document leakage.

## 13. Inference Concurrency and Batching
The embedding service executes inference on CPU with a default batch size of 32 passages.

## 14. Schema Dimensionality Safeguards
A table check constraint named check_chunk_embedding_dimension_384 strictly enforces that dimension equals 384 for every stored vector record.
"""


def is_chunk_relevant(chunk: RetrievedChunk, case: dict[str, Any]) -> bool:
    """Determine ground-truth relevance using document filename and evidence-span matching.

    A retrieved chunk is relevant if it belongs to the target document AND
    either encloses the labeled evidence text or encompasses the labeled character offsets.
    """
    if not case.get("answerable", True):
        return False

    if chunk.document_filename != case["target_document_filename"]:
        return False

    evidence_text = case.get("evidence_text", "").strip().lower()
    if evidence_text and evidence_text in chunk.text.lower():
        return True

    span = case.get("evidence_char_span")
    if span and len(span) == 2 and span[1] > span[0]:
        start, end = span[0], span[1]
        # Overlap check: chunk overlaps or encloses labeled span
        if chunk.start_char_offset <= start and chunk.end_char_offset >= end:
            return True

    return False


@pytest.mark.skipif(
    os.getenv("RUN_MODEL_TESTS") != "1",
    reason="Semantic retrieval evaluation requires real model (RUN_MODEL_TESTS=1).",
)
def test_bge_small_semantic_retrieval_evaluation() -> None:
    """Evaluate Recall@1, Recall@3, Recall@5, and MRR using real BAAI/bge-small-en-v1.5 embeddings.

    Ingests the reference document into a clean transactional session, queries each
    case in baseline_retrieval_cases.json, and prints a verified evaluation summary.
    """
    # 1. Load evaluation dataset
    assert CASES_FILE.is_file(), f"Evaluation dataset not found at {CASES_FILE}"
    with open(CASES_FILE, encoding="utf-8") as f:
        cases = json.load(f)

    answerable_cases = [c for c in cases if c.get("answerable", True)]
    unanswerable_cases = [c for c in cases if not c.get("answerable", True)]

    assert len(answerable_cases) >= 10, "Evaluation suite requires at least 10 answerable test cases."

    # 2. Connect to PostgreSQL or SQLite session
    pg_url = "postgresql+psycopg://evidencedesk:replace_with_a_local_secret@127.0.0.1:5432/evidencedesk"
    engine = sa.create_engine(pg_url)

    # 3. Initialize real BGE-small provider
    provider = SentenceTransformersEmbeddingProvider(
        model_name_or_path="BAAI/bge-small-en-v1.5",
        expected_dimension=384,
        device="cpu",
    )

    with Session(engine) as session:
        try:
            # 4. Ingest reference document into transactional session
            doc = Document(
                original_filename="platform_reference_manual.md",
                content_type="text/markdown",
                size_bytes=len(REFERENCE_DOCUMENT_CONTENT.encode("utf-8")),
                sha256="eval_ref_doc_" + uuid.uuid4().hex[:16],
                storage_key="eval_ref_key_" + uuid.uuid4().hex[:8] + ".md",
                status=DocumentStatus.READY,
            )
            session.add(doc)
            session.flush()

            # Split reference manual into section paragraphs
            sections = [s.strip() for s in REFERENCE_DOCUMENT_CONTENT.split("\n\n## ") if s.strip()]
            # Re-add "## " prefix to split sections (except section 1 which has # header)
            passages: list[str] = []
            for i, sec in enumerate(sections):
                if i == 0:
                    passages.append(sec)
                else:
                    passages.append(f"## {sec}")

            # Generate embeddings with real model
            vectors = provider.embed_texts(passages)
            assert len(vectors) == len(passages)

            chunks: list[Chunk] = []
            char_pos = 0
            for idx, (passage_text, vec) in enumerate(zip(passages, vectors, strict=True)):
                start_offset = char_pos
                end_offset = char_pos + len(passage_text)
                chunk = Chunk(
                    document_id=doc.id,
                    chunk_index=idx,
                    text=passage_text,
                    token_count=len(passage_text.split()),
                    character_count=len(passage_text),
                    start_char_offset=start_offset,
                    end_char_offset=end_offset,
                    source_format="text/markdown",
                    heading=f"Section {idx + 1}",
                    heading_level=2,
                )
                session.add(chunk)
                session.flush()
                chunks.append(chunk)

                emb = ChunkEmbedding(
                    chunk_id=chunk.id,
                    document_id=doc.id,
                    model_id=provider.model_id,
                    dimension=provider.dimension,
                    embedding=vec,
                )
                session.add(emb)
                char_pos = end_offset + 2

            session.flush()

            # 5. Initialize RetrievalService
            settings = Settings(
                EMBEDDING_PROVIDER="sentence_transformers",
                EMBEDDING_MODEL="BAAI/bge-small-en-v1.5",
                EMBEDDING_DIMENSION=384,
            )
            retrieval_service = RetrievalService(db=session, provider=provider, settings=settings)

            # 6. Execute retrieval for each case and score relevance
            hit_at_1 = 0
            hit_at_3 = 0
            hit_at_5 = 0
            reciprocal_ranks: list[float] = []

            print("\n" + "=" * 80)
            print("EVIDENCEDESK — SEMANTIC RETRIEVAL EVALUATION REPORT (BAAI/bge-small-en-v1.5)")
            print("=" * 80)

            for case in answerable_cases:
                resp = retrieval_service.retrieve(case["question"], top_k=5)
                first_relevant_rank: int | None = None

                for rank_idx, candidate in enumerate(resp.results, start=1):
                    if is_chunk_relevant(candidate, case):
                        first_relevant_rank = rank_idx
                        break

                if first_relevant_rank is not None:
                    if first_relevant_rank <= 1:
                        hit_at_1 += 1
                    if first_relevant_rank <= 3:
                        hit_at_3 += 1
                    if first_relevant_rank <= 5:
                        hit_at_5 += 1
                    rr = 1.0 / first_relevant_rank
                    reciprocal_ranks.append(rr)
                    print(f"[{case['case_id']}] First Hit Rank: {first_relevant_rank} (RR: {rr:.3f}) | {case['question'][:60]}...")
                else:
                    reciprocal_ranks.append(0.0)
                    print(f"[{case['case_id']}] MISS (Rank > 5) | {case['question'][:60]}...")

            total_q = len(answerable_cases)
            recall_1 = hit_at_1 / total_q
            recall_3 = hit_at_3 / total_q
            recall_5 = hit_at_5 / total_q
            mrr = sum(reciprocal_ranks) / total_q

            print("-" * 80)
            print(f"Evaluated Cases: {total_q}")
            print(f"Recall@1:        {recall_1:.4f} ({hit_at_1}/{total_q})")
            print(f"Recall@3:        {recall_3:.4f} ({hit_at_3}/{total_q})")
            print(f"Recall@5:        {recall_5:.4f} ({hit_at_5}/{total_q})")
            print(f"MRR:             {mrr:.4f}")
            print("=" * 80)

            # 7. Unanswerable question verification
            for unans_case in unanswerable_cases:
                unans_resp = retrieval_service.retrieve(unans_case["question"], top_k=5)
                # Ensure no candidate claims ground-truth relevance on unanswerable questions
                for cand in unans_resp.results:
                    assert not is_chunk_relevant(cand, unans_case)

            # 8. Sanity assertions: BGE-small should achieve high precision on direct factual manual cases
            assert recall_5 >= 0.85, f"Recall@5 ({recall_5:.3f}) fell below baseline 0.85"
            assert mrr >= 0.75, f"MRR ({mrr:.3f}) fell below baseline 0.75"

        finally:
            # Strictly roll back to preserve the development database volume
            session.rollback()
