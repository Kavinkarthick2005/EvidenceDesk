"""Citation parser and validator for EvidenceDesk question-answering generation.

Parses bracketed citation markers from generated answer text, resolves them against
retrieved passage metadata, detects unknown or malformed references, and tracks
citation frequency and duplicates.

EVALUATION & SECURITY NOTE:
Reference validation confirms syntactic citation presence, valid numeric marker syntax,
and exact mapping to retrieved source passages. It does NOT prove or imply that the
generated claim factually or logically entails the source text. Factual entailment
requires downstream natural language inference (NLI) or evaluation judging.
"""

from collections.abc import Mapping, Sequence
import logging
import re
from typing import Any
import uuid

from pydantic import BaseModel, ConfigDict, Field

from app.generation.context import EvidenceContext, EvidenceSourceMapping
from app.generation.interface import GenerationResult

logger = logging.getLogger(__name__)

# Pattern to locate all bracketed tokens [...] within text
_BRACKET_REGEX = re.compile(r"\[([^\[\]]*)\]")

# Pattern for comma-separated positive integers (e.g., '1', '1, 2', '1, 3, 5')
_NUMERIC_CITATION_REGEX = re.compile(r"^\s*\d+(?:\s*,\s*\d+)*\s*$")

# Pattern for bare citation keywords that represent incomplete citation syntax
_BARE_CITATION_KEYWORDS = frozenset({"ref", "citation", "source", "evidence"})

# Patterns to mask Markdown fenced code blocks (```...``` or ~~~...~~~)
# and inline code spans (`...`) so bracketed expressions in code (e.g., array
# indexing `arr[0]` or code literals `[1, 2]`) are not interpreted as citations.
_FENCED_CODE_BLOCK_REGEX = re.compile(r"```[\s\S]*?(?:```|$)|~~~[\s\S]*?(?:~~~|$)")
_INLINE_CODE_SPAN_REGEX = re.compile(r"`+[^\n`]*?`+")


def strip_markdown_code(text: str) -> str:
    """Mask Markdown fenced code blocks and inline code spans with whitespace.

    Replaces fenced code blocks and inline backtick spans with spaces so that
    bracketed code syntax (e.g., `items[0]` or `[1, 2]`) is not evaluated as
    citations or malformed citation markers, while preserving surrounding prose.
    """
    text_without_blocks = _FENCED_CODE_BLOCK_REGEX.sub(" ", text)
    return _INLINE_CODE_SPAN_REGEX.sub(" ", text_without_blocks)


class ResolvedCitation(BaseModel):
    """Resolved citation binding a citation marker to its source document and chunk evidence.

    NOTE: Resolving a citation to a retrieved source chunk confirms syntactic
    traceability and reference existence. It does NOT guarantee that the generated
    claim factually or logically entails the passage content.
    """

    model_config = ConfigDict(frozen=True)

    marker_id: str = Field(
        description="Citation identifier string (e.g., '1', '2').",
    )
    citation_marker: str = Field(
        description="Bracketed marker string (e.g., '[1]', '[2]').",
    )
    chunk_id: uuid.UUID = Field(
        description="UUID of the cited chunk.",
    )
    document_id: uuid.UUID = Field(
        description="UUID of the parent source document.",
    )
    document_filename: str = Field(
        description="Original filename of the parent document.",
    )
    chunk_index: int = Field(
        description="Sequential zero-based index of the chunk within the document.",
    )
    text: str = Field(
        description="Exact textual passage of the cited chunk.",
    )
    score: float = Field(
        description="Retrieval cosine similarity score.",
    )
    cosine_distance: float = Field(
        description="Raw cosine distance from similarity search.",
    )
    page_number: int | None = Field(
        default=None,
        description="Primary page number if available.",
    )
    page_numbers: list[int] = Field(
        default_factory=list,
        description="List of page numbers spanned by the chunk.",
    )
    heading: str | None = Field(
        default=None,
        description="Enclosing heading if available.",
    )
    heading_level: int | None = Field(
        default=None,
        description="Depth of enclosing heading if available.",
    )
    metadata: dict[str, Any] = Field(
        default_factory=dict,
        description="Preserved structural metadata.",
    )
    model_id: str = Field(
        description="Model identifier that produced the chunk embedding.",
    )


class CitationValidationResult(BaseModel):
    """Result of validating citations within a generated answer against the evidence context.

    Compatible with the GenerationResult contract: cited_evidence_ids contains the
    ordered, deduplicated list of valid cited evidence IDs.
    """

    model_config = ConfigDict(frozen=True)

    is_valid: bool = Field(
        description="True if all parsed and reported citations exist in the evidence context "
        "and no malformed citation markers were detected.",
    )
    has_citations: bool = Field(
        description="True if at least one valid citation was found in the answer.",
    )
    answer: str = Field(
        description="The validated answer text.",
    )
    cited_evidence_ids: list[str] = Field(
        default_factory=list,
        description="Ordered, deduplicated list of valid cited evidence IDs. "
        "Matches GenerationResult.cited_evidence_ids schema.",
    )
    citations: list[ResolvedCitation] = Field(
        default_factory=list,
        description="List of resolved citations with complete source metadata.",
    )
    unknown_citation_ids: list[str] = Field(
        default_factory=list,
        description="Citation markers or IDs found in the answer or reported by generator "
        "that were not present in the evidence context.",
    )
    malformed_citations: list[str] = Field(
        default_factory=list,
        description="Bracketed strings identified as malformed citation attempts.",
    )
    duplicate_citation_ids: list[str] = Field(
        default_factory=list,
        description="Citation IDs that appeared more than once in the answer text.",
    )
    citation_counts: dict[str, int] = Field(
        default_factory=dict,
        description="Frequency count of each citation ID appearing in the answer.",
    )


class CitationValidator:
    """Validates citations in generated answers against an EvidenceContext.

    Parses supported bracketed numeric citation markers (e.g., [1], [2], [1, 2]),
    resolves them to real chunk and document metadata, flags unknown and malformed markers,
    and preserves non-citation bracketed prose without false positive classification.
    """

    def parse_bracketed_tokens(
        self,
        text: str,
        reported_ids: Sequence[str] | None = None,
    ) -> tuple[list[str], list[str]]:
        """Parse text for candidate citation IDs and malformed citation markers.

        Distinguishes numeric citation markers from malformed syntax and ordinary prose.

        Args:
            text: Text to scan for citations.
            reported_ids: Optional list of citation IDs reported by the generator.

        Returns:
            Tuple of:
            - candidate_ids: List of parsed citation IDs in order of appearance (may contain duplicates).
            - malformed_markers: List of full bracketed strings identified as malformed citation syntax.
        """
        candidate_ids: list[str] = []
        malformed_markers: list[str] = []
        reported_set = set(reported_ids or [])

        # Mask Markdown code blocks and inline spans so code syntax
        # (e.g., `items[0]` or `[1, 2]`) is not parsed as citations
        clean_text = strip_markdown_code(text)

        for match in _BRACKET_REGEX.finditer(clean_text):
            raw_bracket = match.group(0)
            inner = match.group(1).strip()

            # 1. Empty brackets: [] or [   ]
            if not inner:
                malformed_markers.append(raw_bracket)
                continue

            # 2. Comma-separated positive integers: e.g., '1', '1, 2', '10'
            if _NUMERIC_CITATION_REGEX.match(inner):
                tokens = [t.strip() for t in inner.split(",") if t.strip()]
                has_zero = any(t == "0" or (t.startswith("0") and len(t) > 1) for t in tokens)
                if has_zero:
                    malformed_markers.append(raw_bracket)
                else:
                    candidate_ids.extend(tokens)
                continue

            # 3. Zero or negative integers: e.g., '0', '-1', '-2'
            if re.fullmatch(r"^-?\d+$", inner) and int(inner) <= 0:
                malformed_markers.append(raw_bracket)
                continue

            # 4. Broken comma syntax with digits: e.g., '1,', ',1', '1,,2', ','
            if re.search(r",\s*,|^\s*,|,\s*$", inner) and any(c.isdigit() for c in inner):
                malformed_markers.append(raw_bracket)
                continue

            # 5. Hash-prefixed numbers: e.g., '#1', '#2', '# 1'
            if re.fullmatch(r"^#\s*\d+$", inner):
                malformed_markers.append(raw_bracket)
                continue

            # 6. Bare citation keywords: e.g., '[ref]', '[citation]', '[source]'
            if inner.lower() in _BARE_CITATION_KEYWORDS:
                malformed_markers.append(raw_bracket)
                continue

            # 7. Explicit non-numeric citation ID matching reported IDs or hallucination pattern
            # (e.g., 'unsupplied-evidence-999' injected by mock generator)
            if inner in reported_set or inner.startswith(("unsupplied-", "chunk-", "evidence-")):
                candidate_ids.append(inner)
                continue

            # 8. All other bracketed text is treated as ordinary prose
            # e.g., '[Mock Answer]', '[Note: ...]', '[Table 1]', '[sic]'
            logger.debug("Treated bracketed content '%s' as ordinary prose.", raw_bracket)

        return candidate_ids, malformed_markers

    def validate(
        self,
        target: str | GenerationResult,
        context: EvidenceContext | Mapping[str, EvidenceSourceMapping],
        reported_cited_ids: Sequence[str] | None = None,
    ) -> CitationValidationResult:
        """Validate citations within an answer string or GenerationResult against evidence context.

        Args:
            target: Answer string or GenerationResult instance.
            context: EvidenceContext or mapping of marker_id -> EvidenceSourceMapping.
            reported_cited_ids: Optional explicit sequence of cited evidence IDs.
                Defaults to target.cited_evidence_ids if target is GenerationResult.

        Returns:
            CitationValidationResult detailing resolved citations, unknown markers,
            malformed syntax, and overall validity.
        """
        # Extract answer string and reported IDs
        if isinstance(target, GenerationResult):
            answer_text = target.answer
            generator_reported_ids = list(target.cited_evidence_ids)
        elif isinstance(target, str):
            answer_text = target
            generator_reported_ids = []
        else:
            raise TypeError(
                f"Validation target must be a str or GenerationResult, got {type(target).__name__}."
            )

        combined_reported_ids = (
            list(reported_cited_ids) if reported_cited_ids is not None else generator_reported_ids
        )

        # Normalize context lookup
        if isinstance(context, EvidenceContext):
            marker_map = context.marker_map
        elif isinstance(context, Mapping):
            marker_map = context
        else:
            raise TypeError(
                f"Context must be an EvidenceContext or Mapping, got {type(context).__name__}."
            )

        # Parse citation tokens from answer text
        parsed_ids, malformed_citations = self.parse_bracketed_tokens(
            answer_text,
            reported_ids=combined_reported_ids,
        )

        # Track counts and occurrences
        citation_counts: dict[str, int] = {}
        duplicate_citation_ids: list[str] = []
        valid_citations: list[ResolvedCitation] = []
        valid_marker_ids: list[str] = []
        unknown_citation_ids: list[str] = []

        seen_valid_ids: set[str] = set()
        seen_unknown_ids: set[str] = set()

        for marker_id in parsed_ids:
            citation_counts[marker_id] = citation_counts.get(marker_id, 0) + 1
            if citation_counts[marker_id] == 2:
                duplicate_citation_ids.append(marker_id)

            if marker_id in marker_map:
                if marker_id not in seen_valid_ids:
                    seen_valid_ids.add(marker_id)
                    valid_marker_ids.append(marker_id)
                    mapping = marker_map[marker_id]
                    valid_citations.append(
                        ResolvedCitation(
                            marker_id=mapping.marker_id,
                            citation_marker=mapping.citation_marker,
                            chunk_id=mapping.chunk_id,
                            document_id=mapping.document_id,
                            document_filename=mapping.document_filename,
                            chunk_index=mapping.chunk_index,
                            text=mapping.text,
                            score=mapping.score,
                            cosine_distance=mapping.cosine_distance,
                            page_number=mapping.page_number,
                            page_numbers=list(mapping.page_numbers),
                            heading=mapping.heading,
                            heading_level=mapping.heading_level,
                            metadata=dict(mapping.metadata),
                            model_id=mapping.model_id,
                        )
                    )
            else:
                if marker_id not in seen_unknown_ids:
                    seen_unknown_ids.add(marker_id)
                    unknown_citation_ids.append(marker_id)

        # Also inspect reported IDs that might not have appeared in text
        for reported_id in combined_reported_ids:
            if reported_id not in marker_map:
                if reported_id not in seen_unknown_ids:
                    seen_unknown_ids.add(reported_id)
                    unknown_citation_ids.append(reported_id)
            else:
                # If reported by generator and valid, but not in answer text,
                # record in counts if not present
                if reported_id not in seen_valid_ids:
                    seen_valid_ids.add(reported_id)
                    valid_marker_ids.append(reported_id)
                    mapping = marker_map[reported_id]
                    valid_citations.append(
                        ResolvedCitation(
                            marker_id=mapping.marker_id,
                            citation_marker=mapping.citation_marker,
                            chunk_id=mapping.chunk_id,
                            document_id=mapping.document_id,
                            document_filename=mapping.document_filename,
                            chunk_index=mapping.chunk_index,
                            text=mapping.text,
                            score=mapping.score,
                            cosine_distance=mapping.cosine_distance,
                            page_number=mapping.page_number,
                            page_numbers=list(mapping.page_numbers),
                            heading=mapping.heading,
                            heading_level=mapping.heading_level,
                            metadata=dict(mapping.metadata),
                            model_id=mapping.model_id,
                        )
                    )

        # Validity requires no unknown citation IDs and no malformed markers
        is_valid = len(unknown_citation_ids) == 0 and len(malformed_citations) == 0
        has_citations = len(valid_citations) > 0

        return CitationValidationResult(
            is_valid=is_valid,
            has_citations=has_citations,
            answer=answer_text,
            cited_evidence_ids=valid_marker_ids,
            citations=valid_citations,
            unknown_citation_ids=unknown_citation_ids,
            malformed_citations=malformed_citations,
            duplicate_citation_ids=duplicate_citation_ids,
            citation_counts=citation_counts,
        )
