# EvidenceDesk Platform Reference Manual

## 1. Storage and Document Limits
The maximum permitted upload size for any single document is configured as 25 MB by default. Uploaded files exceeding this size are rejected before disk persistence. The default local storage path is configured as ./data/documents on the host filesystem.

## 2. Persistence Layer and Vector Schema
PostgreSQL with the pgvector extension is the primary persistent store for vector embeddings and document metadata. All vector columns use the pgvector data type. Database migrations manage table versions through Alembic revision files.

## 3. Embedding Model Specifications
The BAAI/bge-small-en-v1.5 embedding model maps text passages into dense 384-dimensional unit-normalized float vectors. All generated representations have unit L2 length. Model inference runs locally on CPU using SentenceTransformers and PyTorch.

## 4. Reprocessing and Ingestion Lifecycle
If a reprocessing attempt fails on a document that is already READY, the transaction rolls back completely and preserves the prior valid chunks and embeddings. The document remains in READY status with its previously indexed data fully intact.

## 5. First-Time Failure Guarantees
When initial document ingestion fails, the document state is updated to FAILED and zero orphaned chunk or embedding records remain in storage. All staged database entities are reverted to maintain a clean database state.

## 6. Vector Similarity and Ranking Semantics
Retrieval ranks nearest-neighbor candidate passages using cosine distance via the pgvector <=> operator, with similarity scores computed as 1.0 minus cosine distance. Scores range across -1.0 to 1.0 without artificial non-negativity clamping.

## 7. Tokenizer Context Limits and Windowing
The maximum sequence length supported by the local model tokenizer context window is 512 tokens before truncation occurs. Default document chunking splits text into 500 tokens per chunk with 75 tokens of overlap between adjacent passages.

## 8. Extraction Formats and Structural Hierarchy
The ingestion service supports three source document formats: PDF documents, Markdown files, and plain text UTF-8 files. Structure-aware chunking preserves Markdown headings and PDF page numbers across extracted passages.

## 9. Deterministic Tie-Breaking
Deterministic tie-breaking sorts equidistant chunks secondary by document UUID in ascending order and tertiary by sequential chunk index in ascending order. This guarantees completely reproducible search results across queries.

## 10. File Integrity and Deduplication
Document content deduplication and integrity verification compute a 64-character hex-encoded SHA-256 digest at upload time. Identical files with matching hashes are detected before processing.

## 11. Top-k Retrieval Limits and Query Bounds
Callers can request top_k candidates bounded between a minimum of 1 and a maximum of 50 chunks. Natural language query strings must contain between 1 and 2,000 characters after whitespace trimming.

## 12. Workspace Isolation and Query Security
Workspace isolation is enforced at query time using parameterized database filters preventing cross-tenant document leakage. Chunks from unprocessed or failed documents are excluded by enforcing DocumentStatus.READY in SQL.

## 13. Inference Concurrency and Batching
The embedding service executes inference on CPU with a default batch size of 32 passages. PyTorch thread counts can be throttled to prevent CPU starvation on constrained machines.

## 14. Schema Dimensionality Safeguards
A table check constraint named check_chunk_embedding_dimension_384 strictly enforces that dimension equals 384 for every stored vector record. Incompatible dimensions are rejected by the database engine.
