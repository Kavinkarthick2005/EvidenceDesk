# EvidenceDesk — Agent Instructions

## 1. Project mission
Build EvidenceDesk, a local-first, evidence-grounded document question-answering platform. Users should be able to ingest documents, ask questions, inspect supporting evidence, and evaluate retrieval and answer reliability.

## 2. Execution discipline
- Work on one explicitly defined task at a time.
- Inspect the existing repository before editing anything.
- Do not generate the entire application or skip planned stages.
- Before editing, state the task, intended files, approach, and acceptance criteria.
- Make the smallest complete change that satisfies the task.
- Run relevant tests and checks after each change.
- Report files changed, commands run, results, and known limitations.
- Stop and request review when acceptance criteria are met.
- Never claim a test passed unless it was actually run and passed.
- Do not silently change product requirements or architectural decisions.

## 3. Architecture
- Prefer a modular monolith with clear module boundaries.
- Keep business logic separate from API routes, infrastructure, and presentation.
- Use explicit interfaces and dependency injection where they materially improve testability.
- Avoid circular dependencies, duplicated logic, oversized modules, and speculative abstractions.
- Do not introduce microservices, Kubernetes, agents, or extra infrastructure without an approved requirement.
- Document significant architectural decisions and their trade-offs.

## 4. Planned technology direction
- Backend: Python, FastAPI, Pydantic, SQLAlchemy, Alembic.
- Database: PostgreSQL with pgvector.
- Frontend: Next.js, React, and TypeScript.
- Testing: pytest for backend tests and suitable frontend tests.
- Local development: Docker Compose.
- Ingestion and background processing should be introduced only when their stage is reached.
- Keep model providers and embedding implementations behind replaceable interfaces.
- Prefer local-first components and reproducible configurations.
- Treat this list as the approved direction, not permission to implement every component immediately.

## 5. Evidence and evaluation
- Every substantive answer claim should be traceable to retrieved source evidence where applicable.
- Never fabricate document contents, quotations, citations, or evaluation results.
- Support explicit abstention when evidence is insufficient.
- Keep retrieval, generation, citation validation, and evaluation responsibilities distinct.
- Use labeled evaluation data and reproducible configurations.
- Treat automated or LLM-based judging as an imperfect signal, not ground truth.

## 6. Security and privacy
- Never hardcode credentials, API keys, passwords, or tokens.
- Never commit real secrets or local user documents.
- Validate file types, sizes, paths, and uploaded content.
- Treat document contents and retrieved text as untrusted input.
- Prevent cross-workspace data access when workspace isolation is introduced.
- Apply parameterized database access and safe error handling.
- Avoid logging secrets, full document contents, or unnecessary personal data.
- Do not disable security checks to make tests pass.
- Follow SECURITY.md and docs/security/THREAT_MODEL.md as they are developed.

## 7. Code quality
- Use type hints for Python and strict TypeScript where practical.
- Keep functions focused and modules cohesive.
- Prefer clear, conventional code over clever shortcuts.
- Handle failures explicitly and avoid silent exception swallowing.
- Add tests for behavior, edge cases, and security-sensitive paths.
- Pin or constrain dependencies appropriately and explain new dependencies.
- Do not add a dependency when the standard library or existing stack is sufficient.

## 8. Configuration and operations
- Validate configuration at startup and fail clearly when required settings are missing.
- Keep secrets in ignored environment files or an approved secret manager.
- Provide safe example configuration without real credentials.
- Make ingestion retries bounded and operations idempotent where appropriate.
- Introduce structured logs, health checks, backups, and observability in their planned stages.

## 9. Documentation
- Keep README.md focused on setup, usage, testing, and troubleshooting.
- Record system boundaries and data flow in docs/ARCHITECTURE.md.
- Treat docs/PRD.md as the product requirements baseline.
- Record evaluation research and unresolved hypotheses in docs/research/RESEARCH_PLAN.md.
- Record security assumptions and mitigations in docs/security/THREAT_MODEL.md.
- Update documentation when approved behavior or architecture changes.

## 10. Git discipline
- Never expose or commit secrets, credentials, private documents, local datasets, or generated model artifacts.
- Do not rewrite Git history, force-push, or discard user changes without explicit approval.
- Review the diff before proposing a commit.
- Do not commit or push unless explicitly requested.
- Keep changes focused and explain any unexpected modified files.

## 11. Required task report
At the end of each task, report:
1. What changed.
2. Files created or modified.
3. Tests and checks actually run, with results.
4. Known limitations or follow-up work.
5. Whether the stated acceptance criteria were met.
