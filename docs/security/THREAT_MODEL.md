# EvidenceDesk — Security Threat Model

**Status:** Initial threat model
**Version:** 0.1.0
**Scope:** Initial local-first document Q&A platform

## 1. Purpose

Identify the main security and privacy risks in document ingestion, retrieval, answer generation, evaluation, storage, and local operation.

This is an initial design document. It is not a completed security audit or evidence that mitigations have already been implemented.

## 2. Assets to protect

- Original uploaded documents.
- Extracted text, chunks, and embeddings.
- User questions and generated answers.
- Document identifiers, metadata, and evaluation records.
- Provider API keys, database credentials, and session secrets.
- Workspace membership and access-control information when introduced.
- Application integrity and the reproducibility of evaluation results.

## 3. Trust boundaries

Treat the following as untrusted:
- Uploaded files and their filenames.
- Extracted text and document metadata.
- Instructions embedded in documents.
- User-supplied questions and request parameters.
- Retrieved passages and third-party model responses.
- External model providers and other integrations.

Trusted application code must validate inputs, enforce authorization, and control which tools and resources may be accessed.

A document's contents are evidence to analyze, not instructions that override application policy.

## 4. Threats and required mitigations

### T1 — Indirect prompt injection

**Scenario:** A document contains instructions telling the model to reveal secrets, ignore policy, or retrieve unrelated private information.

**Impact:** Unsupported answers, policy bypass, or attempted disclosure of sensitive information.

**Required mitigations:**
- Treat document text as untrusted evidence.
- Separate system instructions from retrieved content.
- Instruct the generation layer not to execute instructions found in source documents.
- Do not give the model unrestricted access to tools, credentials, or arbitrary resources.
- Validate source references and evidence sufficiency outside the model where possible.
- Test with adversarial documents and questions.

Prompt wording alone is not a sufficient security boundary.

### T2 — Malicious or malformed file uploads

**Scenario:** A user uploads a malformed PDF, a misleading extension, a very large file, or content designed to exhaust parser resources.

**Impact:** Resource exhaustion, parser failures, or possible exploitation of a vulnerable dependency.

**Required mitigations:**
- Enforce file-size and request-size limits.
- Validate extensions and inspect content using appropriate parsers.
- Handle extraction failures safely.
- Use maintained parsing dependencies.
- Apply time and resource limits to expensive processing.
- Keep file handling isolated from executable application paths.
- Consider malware scanning and stronger processing isolation before hosted deployment.

Do not rely on the filename or MIME type alone.

### T3 — Path traversal and unsafe filenames

**Scenario:** A filename contains path separators or traversal sequences.

**Impact:** Unauthorized file reads, overwrites, or writes outside the intended storage directory.

**Required mitigations:**
- Generate internal storage identifiers.
- Never use an uploaded filename directly as a filesystem path.
- Resolve and validate storage paths against the intended root.
- Keep display filenames separate from storage paths.
- Test traversal attempts on supported platforms.

### T4 — Cross-document or cross-workspace data exposure

**Scenario:** A query retrieves chunks belonging to a different user's workspace or an unauthorized document.

**Impact:** Confidential information disclosure.

**Required mitigations:**
- Design ownership and workspace associations before multi-user access.
- Apply authorization in backend queries and retrieval filters.
- Do not depend on frontend filtering.
- Test unauthorized document IDs, search filters, citations, and deletion requests.
- Consider database-level safeguards when the access model warrants them.

Until isolation is implemented and tested, do not represent the application as safe for untrusted multi-user hosting.

### T5 — Secret exposure

**Scenario:** A provider key or database password is committed, returned to the browser, or written to logs.

**Impact:** Unauthorized resource access, financial loss, or data exposure.

**Required mitigations:**
- Keep real secrets in ignored environment files or an approved secret manager.
- Commit only placeholder configuration in `.env.example`.
- Keep hosted-provider calls on the server.
- Redact credentials from logs and error responses.
- Rotate exposed credentials immediately.
- Review staged changes before every commit.

### T6 — SQL injection and unsafe database access

**Scenario:** User-controlled text is incorporated into database commands unsafely.

**Impact:** Unauthorized reads, modification, or deletion.

**Required mitigations:**
- Use parameterized database operations and safe ORM patterns.
- Avoid constructing SQL through untrusted string concatenation.
- Validate dynamic identifiers and sort fields against allowlists.
- Test hostile input in search and filtering paths.

### T7 — Citation fabrication or evidence mismatch

**Scenario:** The model invents a citation, references a nonexistent passage, or cites a passage that does not support its claim.

**Impact:** Misleading answers and loss of trust.

**Required mitigations:**
- Build citations from actual retrieved records.
- Validate citation identifiers against the current retrieval result.
- Keep source metadata attached to chunks.
- Evaluate semantic support separately from identifier validity.
- Support abstention when evidence is insufficient.
- Measure false support judgments and manually review a sample.

A structurally valid citation is not proof of a factually supported claim.

### T8 — Denial of service and resource exhaustion

**Scenario:** Repeated requests, huge documents, expensive model calls, or repeated retries exhaust CPU, memory, storage, or provider quotas.

**Impact:** Unavailability, high costs, or corrupted processing workflows.

**Required mitigations:**
- Set request, upload, chunk-count, and context-size limits.
- Bound retries and processing time.
- Avoid unbounded concurrency and queues.
- Apply rate limits and authentication before exposing a service beyond trusted local use.
- Track resource usage and provider costs where practical.
- Make failed ingestion states visible and recoverable.

### T9 — Unsafe logging and diagnostics

**Scenario:** Logs contain full documents, private questions, credentials, or model payloads.

**Impact:** Sensitive information persists in log files or diagnostic exports.

**Required mitigations:**
- Log request and job identifiers, stages, durations, and safe error codes.
- Avoid logging full document contents by default.
- Redact credentials and sensitive headers.
- Define retention and access controls for logs.
- Make verbose payload logging an explicit development-only choice, if needed.

### T10 — Incomplete deletion or stale derived data

**Scenario:** A deleted document remains searchable through chunks, embeddings, cached results, or evaluation artifacts.

**Impact:** Continued access to content the user expected to remove.

**Required mitigations:**
- Define deletion behavior for originals and all derived searchable records.
- Make deletion operations safe to retry.
- Invalidate relevant caches and indexes.
- Record deletion status without retaining unnecessary content.
- Test that deleted content is no longer retrievable.
- Document any retention exceptions explicitly.

### T11 — Dependency or supply-chain compromise

**Scenario:** A vulnerable or malicious package is introduced.

**Impact:** Data theft, code execution, or compromised builds.

**Required mitigations:**
- Prefer established dependencies and minimize dependency count.
- Pin or constrain versions appropriately.
- Review dependency changes.
- Use automated vulnerability checks where practical.
- Keep supported dependencies updated.
- Do not run unreviewed installation scripts or downloaded code blindly.

### T12 — Local service exposure

**Scenario:** A development API, database, or monitoring endpoint is unintentionally exposed to a network.

**Impact:** Unauthorized access to local data or services.

**Required mitigations:**
- Bind development services to loopback where practical.
- Review Docker port mappings and firewall exposure.
- Use credentials for services that require them.
- Do not assume a local application is protected from every process or user on the machine.
- Reassess network exposure before deployment.

## 5. Initial risk priorities

| Threat | Initial priority | Reason |
|---|---|---|
| Prompt injection | High | Retrieved text directly influences model behavior |
| Cross-workspace access | High | Potential disclosure of private documents |
| Secret exposure | High | Can compromise external services and data |
| Malicious uploads | High | Untrusted parsers process arbitrary files |
| Citation mismatch | High | Core product trust depends on evidence |
| Unsafe deletion | High | Deleted content must not remain searchable |
| SQL injection | High | Database integrity and confidentiality |
| Resource exhaustion | Medium–High | Parsing and model calls can be expensive |
| Logging exposure | Medium–High | Diagnostics may contain sensitive content |
| Dependency compromise | Medium–High | External code runs within the application |
| Local service exposure | Medium–High | Depends on host and network configuration |

These are preliminary qualitative priorities. Reassess likelihood and impact against the actual implementation and deployment model.

## 6. Security testing plan

Add tests incrementally as each relevant feature is implemented.

- Upload rejection for oversized, malformed, and unsupported files.
- Filename traversal and unsafe path handling.
- Prompt-injection documents that attempt to override instructions.
- Invalid and fabricated citation identifiers.
- Unanswerable questions and insufficient-evidence cases.
- Authorization checks for documents, retrieval, and deletion.
- SQL injection attempts in query and filter parameters.
- Secret redaction in logs and error responses.
- Bounded retries and resource limits.
- Verification that deleted documents are no longer retrievable.

Use synthetic documents and fake credentials in automated tests. Security tests must not require real provider secrets.

## 7. Operational response

If a secret is exposed:
1. Revoke or rotate it.
2. Identify affected services and data.
3. Remove it from active configuration and future commits.
4. Assess whether Git history or logs require remediation.
5. Document the incident and prevention steps.

If unauthorized document access is suspected:
1. Restrict access to the affected service.
2. Preserve relevant non-sensitive diagnostics.
3. Determine the scope of exposed records.
4. Fix and test the authorization boundary.
5. Review whether additional notification or remediation is required.

## 8. Deployment gate

Before any hosted or untrusted multi-user deployment, review and verify:
- Authentication and authorization.
- Workspace isolation.
- HTTPS and secret management.
- File-processing resource limits.
- Database and storage access controls.
- Rate limits and abuse handling.
- Backups and restore procedures.
- Deletion and retention behavior.
- Logging and monitoring.
- Dependency and vulnerability management.
- Incident response procedures.

## 9. Known limitations

This document describes intended controls, not verified implementation. Each mitigation must be checked against actual code and tests. A completed checklist alone does not establish security, and a successful test suite does not replace a threat-informed review.
