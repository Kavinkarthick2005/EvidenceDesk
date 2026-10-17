# EvidenceDesk

**An evidence-grounded document Q&A platform with reproducible RAG evaluation.**

EvidenceDesk is a local-first project designed to help users ask questions about their documents, inspect supporting source passages, and measure the reliability of retrieval-augmented generation (RAG) configurations.

> **Development status:** Project foundation and specification. Application features are not implemented yet.

## Why EvidenceDesk?

A fluent AI answer is not necessarily a correct or well-supported answer. EvidenceDesk is designed around three principles:

- **Traceability:** Inspect the source passages associated with an answer.
- **Honest uncertainty:** Abstain when the available evidence is insufficient.
- **Measurable reliability:** Compare retrieval and answer quality using repeatable evaluation runs.

These are product goals, not claims about features already delivered.

## Planned capabilities

- Ingest PDF, Markdown, and plain-text documents.
- Extract text and create searchable, metadata-rich chunks.
- Retrieve relevant evidence using dense search, with hybrid retrieval and reranking explored later.
- Generate answers with source-linked citations.
- Validate citation references and evaluate whether evidence supports answer claims.
- Evaluate answerability, abstention behavior, retrieval quality, latency, and cost where measurable.
- Run locally with documented configuration and containerized infrastructure.

## Planned technology stack

| Area | Planned technology |
|---|---|
| Web UI | Next.js, React, TypeScript |
| API | Python, FastAPI, Pydantic |
| Database | PostgreSQL, pgvector |
| ORM and migrations | SQLAlchemy, Alembic |
| PDF extraction | PyMuPDF |
| Embeddings | Configurable sentence-transformers implementation |
| Testing | pytest and suitable frontend tests |
| Local infrastructure | Docker Compose |

The stack is a planned direction. Components will be introduced incrementally and verified before being treated as implemented requirements.

## Architecture

The intended architecture is a modular monolith with distinct responsibilities for:

1. Document ingestion and extraction.
2. Chunking and embeddings.
3. Retrieval and ranking.
4. Answer generation and citation validation.
5. Evaluation and experiment tracking.
6. API, persistence, and user interface.

See [the architecture document](docs/ARCHITECTURE.md) for the planned data flow and module boundaries.

## Repository structure

```text
EvidenceTask/
├── .env.example
├── .gitignore
├── AGENTS.md
├── README.md
├── SECURITY.md
└── docs/
    ├── ARCHITECTURE.md
    ├── PRD.md
    ├── research/
    │   └── RESEARCH_PLAN.md
    └── security/
        └── THREAT_MODEL.md
```

Application source code, automated tests, and deployment configuration will be added in later stages.

## Getting started

### Prerequisites

The planned local development environment uses:

- Git
- Docker Desktop with Docker Compose
- Python for backend development
- Node.js and npm for frontend development

Installations and version requirements will be finalized as the application foundation is implemented.

### Current repository setup

Clone the repository after it has been published to your Git hosting service:

```powershell
git clone <repository-url>
cd EvidenceTask
```

The repository currently contains project documentation and configuration placeholders. There is no application startup command yet.

Do not copy real credentials into tracked files. `.env.example` is intended to contain safe placeholder values only.

## Development workflow

EvidenceDesk is being developed in small, verifiable stages:

1. Establish repository rules and product documentation.
2. Build the backend foundation and health checks.
3. Introduce database models and migrations.
4. Implement and test document ingestion.
5. Build the first dense-retrieval Q&A workflow.
6. Add citation validation and insufficient-evidence handling.
7. Create the evaluation harness.
8. Compare retrieval strategies and calibrate abstention.
9. Build the frontend.
10. Review security, reliability, and release readiness.

Each stage should have explicit acceptance criteria and verified test results before the next stage begins.

## Research direction

The initial research direction is to investigate how retrieval configuration, evidence support, and abstention behavior interact.

The project will not claim research novelty without a review of related literature and existing evaluation frameworks. See [the research plan](docs/research/RESEARCH_PLAN.md).

## Security and privacy

- Treat uploaded documents and retrieved passages as untrusted input.
- Never commit real API keys, passwords, or private documents.
- Keep hosted-provider credentials on the server.
- Validate files and restrict processing resources.
- Test citation validity separately from semantic evidence support.
- Implement and verify access controls before untrusted multi-user hosting.

Read [SECURITY.md](SECURITY.md) and the [threat model](docs/security/THREAT_MODEL.md).

## Documentation

- [Product requirements](docs/PRD.md)
- [System architecture](docs/ARCHITECTURE.md)
- [Research plan](docs/research/RESEARCH_PLAN.md)
- [Threat model](docs/security/THREAT_MODEL.md)
- [Agent instructions](AGENTS.md)

## Current limitations

- The application is not implemented.
- No document ingestion or question-answering endpoint is available.
- No database schema or migrations have been created.
- No evaluation results have been measured.
- No production-readiness or security-audit claim is made.

## Contributing

Changes should be focused, tested, and consistent with the product requirements and architecture. Review the repository instructions in `AGENTS.md` before proposing implementation changes.

Do not commit secrets, generated datasets, private documents, or local model artifacts.

## License

No license has been selected yet. Until one is added, do not assume the repository grants permission to redistribute or reuse its contents.
