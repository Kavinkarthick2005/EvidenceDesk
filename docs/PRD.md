# EvidenceDesk — Product Requirements Document

**Status:** Initial baseline
**Version:** 0.1.0
**Product type:** Local-first, evidence-grounded document Q&A platform

## 1. Product vision

EvidenceDesk helps users ask questions about their own documents and inspect the evidence supporting each answer. It also helps developers measure how ingestion, chunking, retrieval, reranking, and generation choices affect reliability.

The product must prioritize traceability, reproducibility, privacy, and honest handling of insufficient evidence over producing an answer to every question.

## 2. Problem statement

A document chatbot can produce fluent answers that are unsupported, cite irrelevant passages, or behave differently after a configuration change. A useful knowledge platform must make source evidence inspectable and provide a repeatable way to test retrieval and answer quality.

## 3. Target users

- Students and researchers working with collections of reference documents.
- Engineers and technical teams searching internal documentation.
- Developers evaluating retrieval-augmented generation (RAG) systems.

These are initial target groups, not a claim that the product has validated market demand.

## 4. Product principles

1. Evidence before fluency.
2. Cite source passages for substantive factual claims.
3. Abstain when available evidence is insufficient.
4. Keep evaluation reproducible and separate from normal answer generation.
5. Prefer local processing and local storage by default.
6. Make important configuration choices explicit and inspectable.
7. Build incrementally, with tests and clear module boundaries.

## 5. Initial scope

### 5.1 Document ingestion

- Accept PDF, Markdown, and plain-text files in the first release.
- Validate file type, file size, and content before processing.
- Extract text while preserving useful structural information where possible.
- Split content into configurable chunks with source metadata.
- Generate embeddings and store searchable chunk records.
- Track ingestion status and report failures clearly.
- Support document removal, including associated chunks and embeddings.

Scanned PDFs, OCR, DOCX, spreadsheets, websites, and database connectors are outside the initial release.

### 5.2 Question answering

- Accept a natural-language question over the indexed document collection.
- Retrieve relevant passages before generating an answer.
- Display citations that identify the source document and supporting passage.
- Make source passages inspectable by the user.
- Abstain or clearly state uncertainty when evidence is insufficient.
- Never invent a source, quotation, page number, or citation.

A citation's existence does not by itself prove that the cited passage supports the associated claim.

### 5.3 Retrieval and evaluation

- Provide a repeatable evaluation workflow using a versioned question set.
- Include answerable questions, unanswerable questions, and expected supporting evidence.
- Record retrieval configuration and evaluation results.
- Establish a dense-retrieval baseline before adding more complex retrieval methods.
- Later compare dense retrieval with lexical-plus-vector retrieval, rank fusion, and optional reranking.
- Report relevant retrieval metrics, citation support, abstention behavior, latency, and cost where measurable.
- Keep evaluation runs comparable and document changes to datasets or configuration.

An initial development target is a manually reviewed set of approximately 20 questions. Expand it as time and reliable labels permit. This is a starting target, not a claim of statistical representativeness.

### 5.4 Developer experience

- Provide documented local setup and configuration.
- Support Docker Compose for local infrastructure.
- Provide health checks and automated tests.
- Keep model and embedding providers replaceable behind explicit interfaces.
- Make failures diagnosable without exposing secrets or private document contents.

## 6. Reliability requirements

- Answers must be grounded in retrieved evidence rather than model memory alone.
- Retrieved text must be treated as untrusted input.
- Citation references must resolve to real stored source passages.
- The system must handle empty documents, extraction failures, no-result searches, and provider errors safely.
- Retrieval and generation settings must be recorded for evaluation runs.
- Evaluation must distinguish retrieval quality from answer quality.
- Automated or LLM-based judging must not be treated as unquestionable ground truth.
- Quality thresholds must be calibrated on evaluation data rather than assumed to be universally valid.

## 7. Security and privacy requirements

- Do not commit secrets, API keys, private documents, or local datasets.
- Keep provider credentials on the server side.
- Validate uploaded files and enforce configured size limits.
- Prevent path traversal and unsafe file handling.
- Use parameterized database access.
- Design workspace isolation before introducing multi-user access.
- Ensure document deletion removes derived searchable data according to documented retention behavior.
- Avoid logging full document text, credentials, or unnecessary personal data.
- Document trust boundaries, threats, and mitigations in the threat model.

## 8. Local-first and deployment model

- The initial development and evaluation environment runs locally.
- PostgreSQL with pgvector is the planned persistence and vector-search layer.
- Docker Compose is the planned local infrastructure workflow.
- Local embeddings and compatible local generation runtimes should be supported where practical.
- Hosted model providers may be optional and must use user-supplied credentials.
- No paid cloud service is required as a product design assumption.
- Cloud deployment is a later, separately reviewed stage requiring appropriate authentication, access controls, TLS, secret management, storage isolation, and operational safeguards.

## 9. Initial technology direction

- Frontend: Next.js, React, and TypeScript.
- API: Python and FastAPI.
- Validation: Pydantic.
- Persistence: PostgreSQL and pgvector.
- Database access and migrations: SQLAlchemy and Alembic.
- Initial document extraction: PyMuPDF for PDFs and suitable parsers for Markdown and text.
- Initial embeddings: a configurable sentence-transformers implementation.
- Testing: pytest plus suitable frontend tests.
- Local services: Docker Compose.

These are planned choices. Add dependencies only when their need is justified, and confirm compatibility during implementation.

## 10. Non-functional requirements

- Maintain clear module boundaries and a modular-monolith architecture.
- Make ingestion retries bounded and safe to repeat where possible.
- Return clear, non-sensitive error messages.
- Provide automated tests for core behavior and security-sensitive operations.
- Support reproducible evaluation configurations.
- Measure performance before making optimization claims.
- Document backup, restore, and deletion behavior before treating the product as production-ready.

Numeric latency, throughput, and availability targets will be set after a working baseline is measured.

## 11. Initial release acceptance criteria

The first meaningful end-to-end release should demonstrate that a user can:

1. Start the documented local environment.
2. Ingest supported documents and see their processing status.
3. Ask a question against successfully indexed documents.
4. Inspect source passages associated with the answer.
5. Receive an honest insufficient-evidence response for a question the collection cannot answer.
6. Run the versioned evaluation set and inspect the recorded results.
7. Delete a document and verify that it is no longer retrievable.
8. Run automated tests using documented commands.
9. Follow security and configuration documentation without needing committed secrets.

These criteria define the intended release outcome, not the current implementation status.

## 12. Out of scope for the initial release

- Autonomous agents or multi-agent orchestration.
- Microservices and Kubernetes.
- Public multi-tenant hosting.
- OCR and every possible document format.
- Automatic web crawling.
- Fine-tuning a foundation model.
- Claims of novel research without a supporting literature review.
- Claims of production readiness before security, recovery, and operational requirements are tested.

## 13. Open questions

- Which local generation runtime and models provide acceptable quality on supported hardware?
- What are the practical limits for document size and collection size?
- Which citation-support checks provide useful signal without excessive false positives?
- Which retrieval configuration performs best on the versioned evaluation set?
- What authentication and workspace model is required before any hosted deployment?

Resolve these questions through small experiments and documented decisions rather than assumptions.

## 14. Change control

Changes to initial scope, security boundaries, or core architecture must be explained and reviewed before implementation. Update this document when an approved product requirement changes.
