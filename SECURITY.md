# Security Policy

## Project status

EvidenceDesk is under active development. Security controls and mitigations described in project documentation are planned until implemented and verified.

Do not use an unfinished development instance to store sensitive documents or expose it to untrusted users.

## Reporting a vulnerability

Please do not publish exploitable details, credentials, private documents, or proof-of-concept material containing real sensitive data in a public issue.

Until a dedicated private reporting channel is established, repository owners should arrange a private disclosure method before accepting sensitive vulnerability reports.

When reporting a suspected issue, include:
- A concise description of the potential impact.
- Affected component and version or commit, if known.
- Reproduction steps using synthetic data where possible.
- Relevant logs with credentials and personal information removed.
- Any suggested mitigation, if known.

Do not include real API keys, passwords, access tokens, or private document contents.

## Security expectations

Contributors must:
- Never commit real secrets or credentials.
- Keep local environment files and private datasets out of version control.
- Validate uploads and enforce resource limits.
- Treat uploaded and retrieved document text as untrusted.
- Use safe database access patterns.
- Avoid exposing sensitive content through logs or error messages.
- Add tests for security-sensitive behavior.
- Review changes to authentication, authorization, storage, and data deletion carefully.

## Supported versions

No stable release has been published yet. Security support expectations for released versions will be documented when a release policy is established.

## Responsible disclosure

Please allow maintainers reasonable time to investigate and prepare a fix before publicly disclosing an exploitable vulnerability. Avoid accessing, modifying, or deleting data that you do not own or have explicit permission to test.
