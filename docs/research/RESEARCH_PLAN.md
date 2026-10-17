# EvidenceDesk — Research Plan

**Status:** Initial investigation plan
**Version:** 0.1.0
**Research theme:** Evidence-grounded reliability in retrieval-augmented generation (RAG)

## 1. Research objective

Investigate how document chunking, retrieval configuration, ranking, and evidence validation affect the reliability of document-grounded question answering.

The goal is to build a reproducible evaluation workflow that identifies failure modes and supports defensible comparisons between configurations.

This is a research hypothesis, not an established novel contribution.

## 2. Research questions

### RQ1 — Retrieval configuration
How do chunk size, chunk overlap, and top-k selection affect retrieval quality on a fixed document-question dataset?

### RQ2 — Ranking strategies
How do dense retrieval, lexical-plus-vector retrieval, rank fusion, and optional reranking compare on the same evaluation cases?

### RQ3 — Evidence support
How often do generated answer claims have support in the cited source passages, and which types of failures remain after citation-reference validation?

### RQ4 — Abstention
How does an abstention policy affect answer quality, unsupported answers, and the rate of answerable questions incorrectly rejected?

### RQ5 — Reproducibility
Can the system reproduce meaningful comparisons when the dataset, software version, model identifiers, and retrieval settings are held constant?

## 3. Initial hypotheses

These are hypotheses to test, not conclusions:

- H1: Retrieval settings produce measurable differences in evidence recall.
- H2: Additional ranking stages may improve relevance but can increase latency and resource use.
- H3: Valid citation identifiers alone are insufficient to establish semantic evidence support.
- H4: Abstention can reduce unsupported answers but may also reject answerable questions.
- H5: Recording configurations and dataset versions improves the interpretability of regression results.

A hypothesis may be rejected or refined based on the results.

## 4. Literature and existing-system review

Before claiming novelty, investigate:

- Foundational RAG research.
- Dense passage retrieval and neural information retrieval.
- Hybrid lexical and vector retrieval.
- Rank fusion and reranking.
- Faithfulness, groundedness, attribution, and citation evaluation.
- Selective prediction, confidence calibration, and abstention.
- Existing RAG evaluation frameworks and benchmark datasets.
- Security threats such as indirect prompt injection through retrieved documents.

For each relevant paper or system, record:
- Full title, authors, publication date, and venue where available.
- DOI or stable URL.
- Problem addressed.
- Method and evaluation dataset.
- Reported metrics and limitations.
- Relationship to EvidenceDesk.
- What remains unverified or potentially overlapping.

Prefer primary papers, official documentation, and reproducible implementations. Distinguish peer-reviewed results from preprints, vendor claims, and informal commentary.

## 5. Baseline experiment

### Dataset

Start with a small, manually reviewed collection of approximately 20 question cases.

Include:
- Direct factual questions answerable from one passage.
- Questions requiring evidence from multiple passages.
- Questions with relevant-looking but insufficient evidence.
- Questions that cannot be answered from the indexed collection.
- Cases involving conflicting or outdated source information where feasible.

Each case should record the question, answerability label, expected answer criteria, and expected supporting passages.

Do not use model-generated labels as unquestioned ground truth. Review the evidence and labels manually.

### Baseline configuration

- Use a fixed document collection and dataset version.
- Use structure-aware chunking.
- Start experiments around 500 tokens per chunk and 75 tokens of overlap.
- Use dense vector retrieval.
- Record the embedding model, generation model, top-k, context limit, and other relevant settings.
- Keep software versions and configuration consistent within each comparison.

These settings are starting points, not presumed optimal values.

### Comparison sequence

1. Establish a dense-retrieval baseline.
2. Test a small, predefined set of chunking configurations.
3. Test retrieval top-k and context-size alternatives.
4. Add lexical retrieval and rank fusion.
5. Evaluate reranking only if the earlier results justify it.
6. Evaluate evidence-support checks and abstention policies separately.
7. Repeat important runs to identify variability where model behavior is nondeterministic.

Change one major variable at a time when practical.

## 6. Metrics

### Retrieval quality

- Recall@k using reviewed expected evidence.
- Mean Reciprocal Rank when the first relevant result matters.
- nDCG when graded relevance judgments are available.

### Answer quality and grounding

- Answer correctness against reviewed answer criteria.
- Citation-reference validity.
- Citation support rate on a manually reviewed sample.
- Unsupported-claim rate on that sample.
- Completeness when multiple pieces of evidence are required.

Define the unit of analysis and scoring rules before comparing systems.

### Abstention

- Abstention precision and recall.
- False abstention rate on answerable cases.
- Unsupported-answer rate on unanswerable cases.
- Coverage versus answer-quality trade-offs.

### Operational performance

- Ingestion duration.
- Retrieval and generation latency.
- Total query latency.
- Token usage and estimated cost where measurable.
- Failures, retries, and resource consumption.

Report denominators and sample sizes. A small pilot can reveal engineering issues but cannot justify broad statistical claims.

## 7. Evidence-support review

Citation validation should have distinct stages:

1. Confirm that each citation identifies an actual stored passage.
2. Confirm that the cited passage was available to the answer-generation step.
3. Review whether the passage supports the associated answer claim.
4. Record partial support, contradiction, irrelevant citation, and unsupported claims separately.

Use manual review for an initial subset and document reviewer instructions. If an automated or LLM-based evaluator is added, compare its judgments against reviewed examples and report disagreement.

Do not equate a fluent answer, a high similarity score, or a valid citation identifier with correctness.

## 8. Experiment record

For every run, record:
- Run identifier and timestamp.
- Dataset and document-collection version.
- Code revision.
- Chunking settings.
- Embedding model and version.
- Retrieval method, top-k, and ranking configuration.
- Reranker and generation model identifiers where applicable.
- Prompt or policy version.
- Evaluation metric definitions.
- Results, failures, and known limitations.

Do not store secrets or unnecessary private document content in experiment records.

## 9. Threats to validity

Potential threats include:
- A small or unrepresentative dataset.
- Subjective relevance and answer labels.
- Leakage between configuration development and final evaluation.
- Changes in model versions or provider behavior.
- Nondeterministic generation.
- Evaluator bias or poor evaluator agreement.
- Unfair comparisons caused by different compute budgets or context limits.
- Overfitting thresholds to the initial dataset.
- Selectively reporting favorable results.

Keep a held-out set when the dataset becomes large enough. Report negative results and avoid claiming generality beyond the evidence.

## 10. Candidate contribution

A potential contribution is a practical, reproducible workflow for comparing retrieval configurations using both retrieval metrics and reviewed claim-level evidence support, including abstention behavior and operational cost.

This contribution is not yet established as novel. The literature and existing-system review must determine whether the proposed combination is sufficiently distinct and useful.

## 11. Research milestones

1. Review related papers and existing evaluation frameworks.
2. Define annotation guidelines and the initial dataset schema.
3. Build and verify the dense-retrieval baseline.
4. Establish repeatable metric calculations.
5. Run controlled chunking and retrieval comparisons.
6. Add evidence-support and abstention evaluations.
7. Validate important automated judgments against human review.
8. Document results, limitations, and reproducibility instructions.
9. Reassess the contribution after reviewing related work.

## 12. Reporting standard

For every reported finding, include:
- The question investigated.
- The experimental setup.
- The dataset and sample size.
- The metric definition.
- The measured result.
- Relevant uncertainty and limitations.
- Whether the result supports the original hypothesis.

Do not invent results, citations, papers, benchmarks, or novelty claims.
