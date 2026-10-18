"""Structure-aware document chunker with deterministic token windowing."""

import re
from typing import Any
import uuid

from app.chunking.exceptions import ChunkingEmptyError
from app.chunking.interface import Chunker, Tokenizer
from app.chunking.models import Chunk, ChunkingConfig, ChunkingResult
from app.chunking.tokenizer import get_default_tokenizer
from app.extraction.models import ExtractedElement, ExtractionResult


class StructureAwareChunker:
    """Chunks extracted documents while preserving structural hierarchy and source traceability.

    Key behaviors:
    - Preserves structural boundaries (Markdown headings, PDF pages).
    - Splits oversized sections at paragraph and sentence boundaries before token boundaries.
    - Applies bounded, deterministic overlap (default 75 tokens for 500-token chunks).
    - Tracks exact character offsets (start_char_offset, end_char_offset) in the source text.
    - Guarantees source_text[chunk.start_char_offset:chunk.end_char_offset] == chunk.text.
    - Strictly avoids infinite loops, empty chunks, and duplicate chunk indices.
    """

    SENTENCE_SPLIT_REGEX = re.compile(r"(?<=[.!?;\n])\s+")

    def __init__(
        self,
        tokenizer: Tokenizer | None = None,
        default_config: ChunkingConfig | None = None,
    ) -> None:
        self.tokenizer = tokenizer or get_default_tokenizer()
        self.default_config = default_config or ChunkingConfig()

    def chunk(
        self,
        extraction_result: ExtractionResult,
        document_id: uuid.UUID | None = None,
        config: ChunkingConfig | None = None,
    ) -> ChunkingResult:
        """Partition an ExtractionResult into structured, deterministic chunks."""
        full_text = extraction_result.text
        if not full_text or not full_text.strip():
            raise ChunkingEmptyError("Document content is empty or contains only whitespace.")

        active_config = config or self.default_config

        elements = [el for el in extraction_result.elements if el.text.strip()]
        if not elements:
            # Fall back to entire document text if elements are missing
            elements = [ExtractedElement(text=full_text)]

        chunks: list[Chunk] = []
        global_chunk_idx = 0
        search_pos = 0

        for el in elements:
            el_text = el.text
            # Locate element in the full document text sequentially
            el_start = full_text.find(el_text, search_pos)
            if el_start == -1:
                el_start = full_text.find(el_text)
            if el_start != -1:
                base_char_offset = el_start
                search_pos = el_start + len(el_text)
            else:
                base_char_offset = search_pos

            element_chunks = self._chunk_element_text(
                text=el_text,
                base_offset=base_char_offset,
                config=active_config,
            )

            for chunk_text, tok_count, c_start, c_end in element_chunks:
                page_nums: list[int] = [el.page_number] if el.page_number is not None else []
                chunk_meta: dict[str, Any] = {}
                if el.metadata:
                    chunk_meta.update(el.metadata)

                chunk = Chunk(
                    chunk_index=global_chunk_idx,
                    text=chunk_text,
                    token_count=tok_count,
                    character_count=len(chunk_text),
                    start_char_offset=c_start,
                    end_char_offset=c_end,
                    document_id=document_id,
                    source_format=extraction_result.content_type,
                    page_number=el.page_number,
                    page_numbers=page_nums,
                    heading=el.heading,
                    heading_level=el.heading_level,
                    metadata=chunk_meta,
                )
                chunks.append(chunk)
                global_chunk_idx += 1

        if not chunks:
            raise ChunkingEmptyError("No valid chunks could be produced from document content.")

        return ChunkingResult(
            chunks=chunks,
            total_chunks=len(chunks),
            total_tokens=sum(c.token_count for c in chunks),
            document_id=document_id,
            config=active_config,
        )

    @staticmethod
    def _trim_span(text: str, start: int, end: int) -> tuple[str, int, int]:
        """Trim leading and trailing whitespace from text[start:end] and adjust offsets.

        Guarantees that text[adj_start:adj_end] == trimmed_text.
        """
        raw_slice = text[start:end]
        l_strip = raw_slice.lstrip()
        leading_ws = len(raw_slice) - len(l_strip)
        trailing_ws = len(l_strip) - len(l_strip.rstrip())

        adj_start = start + leading_ws
        adj_end = end - trailing_ws
        trimmed = text[adj_start:adj_end]
        return trimmed, adj_start, adj_end

    def _safe_prefix_char_len(self, text: str, tokens: list[int], tok_idx: int) -> int:
        """Determine exact character boundary corresponding to tokens[:tok_idx].

        If a token boundary splits a multi-byte UTF-8 character, steps backwards
        until self.tokenizer.decode(tokens[:idx]) is a clean prefix of text.
        """
        if tok_idx <= 0:
            return 0
        if tok_idx >= len(tokens):
            return len(text)
        idx = tok_idx
        while idx > 0:
            dec = self.tokenizer.decode(tokens[:idx])
            if text.startswith(dec):
                return len(dec)
            idx -= 1
        return 0

    def _chunk_element_text(
        self,
        text: str,
        base_offset: int,
        config: ChunkingConfig,
    ) -> list[tuple[str, int, int, int]]:
        """Split a single structural element's text into bounded chunks.

        Returns:
            List of (chunk_text, token_count, start_char_offset, end_char_offset)
        """
        tokens = self.tokenizer.encode(text)
        total_tokens = len(tokens)

        # Fast path: element fits comfortably in a single chunk
        if total_tokens <= config.chunk_size_tokens:
            c_text, adj_s, adj_e = self._trim_span(text, 0, len(text))
            if c_text:
                c_toks = self.tokenizer.count_tokens(c_text)
                return [(c_text, c_toks, base_offset + adj_s, base_offset + adj_e)]
            return []

        # Break text into atomic segments (sentences/paragraphs or sub-token windows)
        segments = self._segment_text(text, config)
        if not segments:
            c_text, adj_s, adj_e = self._trim_span(text, 0, len(text))
            if c_text:
                c_toks = self.tokenizer.count_tokens(c_text)
                return [(c_text, c_toks, base_offset + adj_s, base_offset + adj_e)]
            return []

        chunks: list[tuple[str, int, int, int]] = []
        current_segs: list[tuple[str, int, int, int]] = []
        current_tokens = 0

        for seg in segments:
            _, _, _, n_toks = seg

            if current_tokens + n_toks <= config.chunk_size_tokens:
                current_segs.append(seg)
                current_tokens += n_toks
            else:
                if current_segs:
                    # Emit chunk from current segments
                    c_start = current_segs[0][1]
                    c_end = current_segs[-1][2]
                    c_text, adj_s, adj_e = self._trim_span(text, c_start, c_end)
                    if c_text:
                        c_toks = self.tokenizer.count_tokens(c_text)
                        chunks.append((c_text, c_toks, base_offset + adj_s, base_offset + adj_e))

                    # Calculate overlap segments (look backwards from current_segs)
                    overlap_segs: list[tuple[str, int, int, int]] = []
                    overlap_count = 0
                    # Exclude the very first segment to guarantee strict forward advancement
                    for s in reversed(current_segs[1:]):
                        if overlap_count + s[3] <= config.chunk_overlap_tokens:
                            overlap_segs.insert(0, s)
                            overlap_count += s[3]
                        else:
                            break

                    current_segs = list(overlap_segs) + [seg]
                    current_tokens = sum(s[3] for s in current_segs)
                else:
                    # Single segment exceeds chunk size even after sub-splitting
                    current_segs = [seg]
                    current_tokens = n_toks

        if current_segs:
            c_start = current_segs[0][1]
            c_end = current_segs[-1][2]
            c_text, adj_s, adj_e = self._trim_span(text, c_start, c_end)
            if c_text:
                c_toks = self.tokenizer.count_tokens(c_text)
                chunks.append((c_text, c_toks, base_offset + adj_s, base_offset + adj_e))

        return chunks

    def _segment_text(
        self,
        text: str,
        config: ChunkingConfig,
    ) -> list[tuple[str, int, int, int]]:
        """Divide text into sentence/paragraph segments and sub-split oversized items.

        Returns:
            List of (seg_text, start_char, end_char, token_count)
            where text[start_char:end_char] == seg_text.
        """
        raw_spans: list[tuple[str, int, int]] = []
        start = 0

        for match in self.SENTENCE_SPLIT_REGEX.finditer(text):
            end = match.start()
            seg, adj_s, adj_e = self._trim_span(text, start, end)
            if seg:
                raw_spans.append((seg, adj_s, adj_e))
            start = match.end()

        trailing = text[start:]
        if trailing.strip():
            seg, adj_s, adj_e = self._trim_span(text, start, len(text))
            if seg:
                raw_spans.append((seg, adj_s, adj_e))

        if not raw_spans:
            seg, adj_s, adj_e = self._trim_span(text, 0, len(text))
            if seg:
                raw_spans = [(seg, adj_s, adj_e)]

        segments: list[tuple[str, int, int, int]] = []

        for seg_text, s_start, s_end in raw_spans:
            toks = self.tokenizer.encode(seg_text)
            tok_count = len(toks)

            if tok_count <= config.chunk_size_tokens:
                segments.append((seg_text, s_start, s_end, tok_count))
            else:
                # Sub-split oversized sentence using strict token windowing
                sub_segments = self._sub_split_tokens(
                    seg_text,
                    s_start,
                    toks,
                    config.chunk_size_tokens,
                    config.chunk_overlap_tokens,
                )
                segments.extend(sub_segments)

        return segments

    def _sub_split_tokens(
        self,
        seg_text: str,
        base_start: int,
        tokens: list[int],
        chunk_size: int,
        overlap: int,
    ) -> list[tuple[str, int, int, int]]:
        """Slice an oversized sequence of tokens into bounded, overlapping units.

        Guarantees that for every returned tuple (chunk_text, start_offset, end_offset, tok_count):
            seg_text[start_offset - base_start : end_offset - base_start] == chunk_text
            tok_count <= chunk_size
        """
        step = max(1, chunk_size - overlap)
        results: list[tuple[str, int, int, int]] = []
        i = 0

        while i < len(tokens):
            end_tok = min(len(tokens), i + chunk_size)
            c_start = self._safe_prefix_char_len(seg_text, tokens, i)
            c_end = self._safe_prefix_char_len(seg_text, tokens, end_tok)

            trimmed, a_s, a_e = self._trim_span(seg_text, c_start, c_end)
            c_toks = self.tokenizer.count_tokens(trimmed) if trimmed else 0

            # If standalone token count exceeds chunk_size, contract end boundary until it fits
            while c_toks > chunk_size and end_tok > i + 1:
                end_tok -= 1
                c_end = self._safe_prefix_char_len(seg_text, tokens, end_tok)
                trimmed, a_s, a_e = self._trim_span(seg_text, c_start, c_end)
                c_toks = self.tokenizer.count_tokens(trimmed) if trimmed else 0

            if trimmed:
                results.append((
                    trimmed,
                    base_start + a_s,
                    base_start + a_e,
                    c_toks,
                ))

            if end_tok >= len(tokens):
                break

            next_i = i + step
            while next_i < len(tokens) and self._safe_prefix_char_len(seg_text, tokens, next_i) <= a_s:
                next_i += 1
            if next_i <= i:
                next_i = i + 1
            i = next_i

        return results


_default_chunker: StructureAwareChunker | None = None


def get_chunker(
    tokenizer: Tokenizer | None = None,
    default_config: ChunkingConfig | None = None,
) -> StructureAwareChunker:
    """Return a shared or newly configured StructureAwareChunker instance."""
    global _default_chunker
    if tokenizer is not None or default_config is not None:
        return StructureAwareChunker(tokenizer=tokenizer, default_config=default_config)
    if _default_chunker is None:
        _default_chunker = StructureAwareChunker()
    return _default_chunker


def chunk_document(
    extraction_result: ExtractionResult,
    document_id: uuid.UUID | None = None,
    config: ChunkingConfig | None = None,
    tokenizer: Tokenizer | None = None,
) -> ChunkingResult:
    """Convenience helper to chunk an ExtractionResult into structured passages."""
    chunker = get_chunker(tokenizer=tokenizer)
    return chunker.chunk(
        extraction_result=extraction_result,
        document_id=document_id,
        config=config,
    )
