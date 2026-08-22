"""Hierarchical semantic chunking engine."""

from __future__ import annotations

import hashlib
import re

from app.kb.chunking.config import ChunkingConfig
from app.kb.chunking.duplicates import suppress_same_context_duplicates
from app.kb.chunking.models import ChunkProvenance, ChunkRecord, SplitMethod
from app.kb.chunking.noise import classify_chunk_noise, should_suppress_noise
from app.kb.chunking.semantic_units import SemanticUnit, build_semantic_units
from app.kb.chunking.tokenizer import (
    CharacterEstimateTokenizer,
    Tokenizer,
    smart_hard_token_split,
    split_sentences,
)
from app.kb.retrieval.models import RetrievalDocument

PIPELINE_HASH = re.compile(r"sha256:[a-f0-9]{64}", re.I)


def _chunk_id(
    *,
    kb_dataset_version: str,
    document_id: str,
    document_version: int,
    chunk_index: int,
    content: str,
) -> str:
    return hashlib.sha256(
        f"{kb_dataset_version}|{document_id}|v{document_version}|{chunk_index}|{content}".encode("utf-8")
    ).hexdigest()


def _render_content(section_path: list[str], body: str) -> str:
    cleaned = body.strip()
    if not cleaned:
        return ""
    path = [part.strip() for part in section_path if part.strip()]
    if len(path) <= 1:
        return cleaned
    context_lines = path[-2:] if len(path) >= 2 else path
    context = " > ".join(context_lines)
    if cleaned.startswith(context):
        return cleaned
    return f"{context}\n\n{cleaned}"


def _split_method_for_unit(unit_type: str) -> SplitMethod:
    mapping = {
        "section": SplitMethod.SUBSECTION,
        "heading": SplitMethod.HEADING,
        "paragraph": SplitMethod.PARAGRAPH,
        "clause": SplitMethod.CLAUSE,
        "list": SplitMethod.LIST,
        "table": SplitMethod.TABLE,
        "faq_pair": SplitMethod.FAQ_PAIR,
        "page": SplitMethod.SEMANTIC,
    }
    return mapping.get(unit_type, SplitMethod.SEMANTIC)


class ChunkingEngine:
    """Document-aware hierarchical chunker operating on retrieval documents."""

    def __init__(self, config: ChunkingConfig | None = None, tokenizer: Tokenizer | None = None) -> None:
        self.config = config or ChunkingConfig()
        self.tokenizer = tokenizer or CharacterEstimateTokenizer(chars_per_token=self.config.chars_per_token)

    def chunk_document(self, retrieval: RetrievalDocument) -> list[ChunkRecord]:
        records, _ = self._chunk_and_filter(retrieval)
        return records

    def chunk_document_with_suppressed(
        self, retrieval: RetrievalDocument
    ) -> tuple[list[ChunkRecord], list[dict]]:
        """Chunk document returning suppressed chunk metadata for dry-run reporting."""
        return self._chunk_and_filter(retrieval)

    def _chunk_and_filter(self, retrieval: RetrievalDocument) -> tuple[list[ChunkRecord], list[dict]]:
        units = build_semantic_units(
            structured_content=retrieval.structured_content,
            document_type=retrieval.document_type,
            title=retrieval.title,
            canonical_url=retrieval.canonical_url,
        )
        chunk_counter = [0]
        records: list[ChunkRecord] = []
        for unit in units:
            if unit.unit_type in {"page", "section"} and unit.children:
                records.extend(self._materialize_siblings(unit.children, retrieval=retrieval, chunk_counter=chunk_counter))
            else:
                records.extend(self._materialize_unit(unit, retrieval=retrieval, chunk_counter=chunk_counter))
        records = self._enforce_chunk_token_limits(records, retrieval=retrieval)
        records = self._merge_orphan_hard_tails(records)
        records, suppressed = self._apply_quality_filters_with_report(
            records, retrieval=retrieval, chunk_counter=chunk_counter
        )
        for index, record in enumerate(records):
            record.chunk_index = index
            record.chunk_id = _chunk_id(
                kb_dataset_version=record.kb_dataset_version,
                document_id=record.document_id,
                document_version=record.document_version,
                chunk_index=index,
                content=record.content,
            )
        return records, suppressed

    def _materialize_siblings(
        self,
        units: list[SemanticUnit],
        *,
        retrieval: RetrievalDocument,
        chunk_counter: list[int],
    ) -> list[ChunkRecord]:
        chunks: list[ChunkRecord] = []
        for unit in units:
            chunks.extend(self._materialize_unit(unit, retrieval=retrieval, chunk_counter=chunk_counter))
        return chunks

    def _materialize_unit(
        self,
        unit: SemanticUnit,
        *,
        retrieval: RetrievalDocument,
        chunk_counter: list[int],
        inherited_split: SplitMethod | None = None,
    ) -> list[ChunkRecord]:
        if unit.children:
            child_chunks: list[ChunkRecord] = []
            for child in unit.children:
                child_chunks.extend(
                    self._materialize_unit(
                        child,
                        retrieval=retrieval,
                        chunk_counter=chunk_counter,
                        inherited_split=_split_method_for_unit(child.unit_type),
                    )
                )
            if child_chunks:
                return child_chunks

        body = PIPELINE_HASH.sub("", unit.content).strip()
        if not body:
            return []

        if unit.unit_type == "heading" and unit.section_path and body == unit.section_path[-1].strip():
            return []

        rendered = _render_content(unit.section_path, body)
        token_count = self.tokenizer.count(rendered)
        split_method = inherited_split or _split_method_for_unit(unit.unit_type)

        if token_count <= self.config.max_chunk_tokens:
            return [
                self._make_chunk(
                    retrieval=retrieval,
                    content=rendered,
                    token_count=token_count,
                    section_path=unit.section_path,
                    page_number=unit.page_number,
                    split_method=split_method,
                    source_block_path=unit.source_block_path,
                    chunk_counter=chunk_counter,
                )
            ]

        # Context prefix may push an otherwise valid semantic unit slightly over the limit.
        if token_count <= int(self.config.max_chunk_tokens * 1.05) and split_method in {
            SplitMethod.PARAGRAPH,
            SplitMethod.CLAUSE,
            SplitMethod.FAQ_PAIR,
            SplitMethod.LIST,
        }:
            return [
                self._make_chunk(
                    retrieval=retrieval,
                    content=rendered,
                    token_count=token_count,
                    section_path=unit.section_path,
                    page_number=unit.page_number,
                    split_method=split_method,
                    source_block_path=unit.source_block_path,
                    chunk_counter=chunk_counter,
                )
            ]

        if unit.unit_type == "table":
            return self._split_table(unit, retrieval=retrieval, chunk_counter=chunk_counter)

        sentences = split_sentences(body)
        if len(sentences) > 1:
            return self._split_sentences(sentences, unit=unit, retrieval=retrieval, chunk_counter=chunk_counter)

        parts = smart_hard_token_split(
            body,
            max_tokens=self.config.max_chunk_tokens,
            min_meaningful_tokens=self.config.min_meaningful_chunk_tokens,
            tokenizer=self.tokenizer,
        )
        chunks: list[ChunkRecord] = []
        overlap = self.config.emergency_overlap_tokens
        previous_tail = ""
        for part in parts:
            merged = f"{previous_tail} {part}".strip() if previous_tail else part
            rendered_part = _render_content(unit.section_path, merged)
            overlap_used = self.tokenizer.count(previous_tail) if previous_tail else 0
            chunks.append(
                self._make_chunk(
                    retrieval=retrieval,
                    content=rendered_part,
                    token_count=self.tokenizer.count(rendered_part),
                    section_path=unit.section_path,
                    page_number=unit.page_number,
                    split_method=SplitMethod.HARD_TOKEN_FALLBACK,
                    source_block_path=unit.source_block_path,
                    chunk_counter=chunk_counter,
                    overlap_tokens=overlap_used,
                )
            )
            if overlap > 0:
                tail_words = part.split()
                tail_len = max(1, int(overlap / max(1, self.config.chars_per_token)))
                previous_tail = " ".join(tail_words[-tail_len:])
            else:
                previous_tail = ""
        return chunks

    def _enforce_chunk_token_limits(
        self,
        records: list[ChunkRecord],
        *,
        retrieval: RetrievalDocument,
    ) -> list[ChunkRecord]:
        """Last-resort normalization so no chunk exceeds the configured ceiling."""
        normalized: list[ChunkRecord] = []
        for record in records:
            if record.token_count <= self.config.max_chunk_tokens:
                normalized.append(record)
                continue
            parts = smart_hard_token_split(
                record.content,
                max_tokens=self.config.max_chunk_tokens,
                min_meaningful_tokens=self.config.min_meaningful_chunk_tokens,
                tokenizer=self.tokenizer,
            )
            for part in parts:
                token_count = self.tokenizer.count(part)
                normalized.append(
                    ChunkRecord(
                        chunk_id="pending",
                        document_id=record.document_id,
                        document_version=record.document_version,
                        kb_dataset_version=record.kb_dataset_version,
                        chunk_index=0,
                        website=record.website,
                        document_type=record.document_type,
                        title=record.title,
                        section_path=record.section_path,
                        content=part,
                        token_count=token_count,
                        split_method=SplitMethod.HARD_TOKEN_FALLBACK,
                        source_url=record.source_url,
                        canonical_url=record.canonical_url,
                        page_number=record.page_number,
                        source_type=record.source_type,
                        extraction_method=record.extraction_method,
                        overlap_tokens=0,
                        provenance=record.provenance,
                        language=record.language,
                    )
                )
        return normalized

    def _merge_orphan_hard_tails(self, records: list[ChunkRecord]) -> list[ChunkRecord]:
        """Merge consecutive hard-fallback orphan tails into prior chunks when possible."""
        if not records:
            return records

        merged: list[ChunkRecord] = []
        for record in records:
            if (
                merged
                and record.split_method == SplitMethod.HARD_TOKEN_FALLBACK
                and record.token_count < self.config.min_meaningful_chunk_tokens
                and merged[-1].split_method == SplitMethod.HARD_TOKEN_FALLBACK
                and merged[-1].section_path == record.section_path
            ):
                combined = f"{merged[-1].content}\n\n{record.content}".strip()
                combined_tokens = self.tokenizer.count(combined)
                if combined_tokens <= self.config.max_chunk_tokens:
                    merged[-1] = merged[-1].model_copy(
                        update={"content": combined, "token_count": combined_tokens}
                    )
                    continue
            merged.append(record)
        return merged

    def _apply_quality_filters(
        self,
        records: list[ChunkRecord],
        *,
        retrieval: RetrievalDocument,
        chunk_counter: list[int],
    ) -> list[ChunkRecord]:
        kept, _ = self._apply_quality_filters_with_report(records, retrieval=retrieval, chunk_counter=chunk_counter)
        return kept

    def _apply_quality_filters_with_report(
        self,
        records: list[ChunkRecord],
        *,
        retrieval: RetrievalDocument,
        chunk_counter: list[int],
    ) -> tuple[list[ChunkRecord], list[dict]]:
        suppressed: list[dict] = []
        kept: list[ChunkRecord] = []

        for record in records:
            noise = classify_chunk_noise(
                record.content,
                split_method=record.split_method,
                token_count=record.token_count,
            )
            if should_suppress_noise(noise):
                suppressed.append(
                    {
                        "document_id": record.document_id,
                        "section_path": record.section_path,
                        "token_count": record.token_count,
                        "split_method": record.split_method.value,
                        "suppression_reason": noise.category,
                        "confidence": noise.confidence,
                        "detail": noise.reason,
                        "content_preview": record.content[:200],
                        "page_number": record.page_number,
                        "provenance": record.provenance.model_dump(),
                    }
                )
                continue
            kept.append(record)

        kept, duplicate_suppressed = suppress_same_context_duplicates(kept)
        for entry in duplicate_suppressed:
            chunk = entry.chunk
            suppressed.append(
                {
                    "document_id": chunk.document_id,
                    "section_path": chunk.section_path,
                    "token_count": chunk.token_count,
                    "split_method": chunk.split_method.value,
                    "suppression_reason": entry.classification.value.lower(),
                    "confidence": 1.0,
                    "detail": f"duplicate_of={entry.duplicate_of_chunk_id}",
                    "content_preview": chunk.content[:200],
                    "page_number": chunk.page_number,
                    "provenance": chunk.provenance.model_dump(),
                }
            )

        return kept, suppressed

    def _split_sentences(
        self,
        sentences: list[str],
        *,
        unit: SemanticUnit,
        retrieval: RetrievalDocument,
        chunk_counter: list[int],
    ) -> list[ChunkRecord]:
        chunks: list[ChunkRecord] = []
        group: list[str] = []

        def flush(group_sentences: list[str]) -> None:
            if not group_sentences:
                return
            body = " ".join(group_sentences).strip()
            rendered = _render_content(unit.section_path, body)
            chunks.append(
                self._make_chunk(
                    retrieval=retrieval,
                    content=rendered,
                    token_count=self.tokenizer.count(rendered),
                    section_path=unit.section_path,
                    page_number=unit.page_number,
                    split_method=SplitMethod.SENTENCE,
                    source_block_path=unit.source_block_path,
                    chunk_counter=chunk_counter,
                )
            )

        for sentence in sentences:
            candidate = " ".join([*group, sentence]).strip()
            rendered = _render_content(unit.section_path, candidate)
            if group and self.tokenizer.count(rendered) > self.config.max_chunk_tokens:
                flush(group)
                group = [sentence]
            else:
                group.append(sentence)
        if group:
            flush(group)

        if len(chunks) <= 1:
            return chunks

        overlap = self.config.emergency_overlap_tokens
        if overlap > 0:
            for idx in range(1, len(chunks)):
                prev_words = chunks[idx - 1].content.split()
                tail = " ".join(prev_words[-2:])
                merged = f"{tail}\n\n{chunks[idx].content}".strip()
                chunks[idx].content = merged
                chunks[idx].token_count = self.tokenizer.count(merged)
                chunks[idx].overlap_tokens = self.tokenizer.count(tail)
        return chunks

    def _split_table(
        self,
        unit: SemanticUnit,
        *,
        retrieval: RetrievalDocument,
        chunk_counter: list[int],
    ) -> list[ChunkRecord]:
        lines = [line for line in unit.content.splitlines() if line.strip()]
        if len(lines) <= 1:
            return self._materialize_unit(
                SemanticUnit(
                    content=unit.content,
                    section_path=unit.section_path,
                    page_number=unit.page_number,
                    unit_type="paragraph",
                    source_block_path=unit.source_block_path,
                ),
                retrieval=retrieval,
                chunk_counter=chunk_counter,
                inherited_split=SplitMethod.HARD_TOKEN_FALLBACK,
            )
        header = lines[0]
        rows = lines[1:]
        chunks: list[ChunkRecord] = []
        group: list[str] = []
        for row in rows:
            candidate_rows = [*group, row]
            body = "\n".join([header, *candidate_rows])
            rendered = _render_content(unit.section_path, body)
            if group and self.tokenizer.count(rendered) > self.config.max_chunk_tokens:
                body = "\n".join([header, *group])
                rendered = _render_content(unit.section_path, body)
                chunks.append(
                    self._make_chunk(
                        retrieval=retrieval,
                        content=rendered,
                        token_count=self.tokenizer.count(rendered),
                        section_path=unit.section_path,
                        page_number=unit.page_number,
                        split_method=SplitMethod.TABLE,
                        source_block_path=unit.source_block_path,
                        chunk_counter=chunk_counter,
                    )
                )
                group = [row]
            else:
                group.append(row)
        if group:
            body = "\n".join([header, *group])
            rendered = _render_content(unit.section_path, body)
            chunks.append(
                self._make_chunk(
                    retrieval=retrieval,
                    content=rendered,
                    token_count=self.tokenizer.count(rendered),
                    section_path=unit.section_path,
                    page_number=unit.page_number,
                    split_method=SplitMethod.TABLE,
                    source_block_path=unit.source_block_path,
                    chunk_counter=chunk_counter,
                )
            )
        return chunks

    def _make_chunk(
        self,
        *,
        retrieval: RetrievalDocument,
        content: str,
        token_count: int,
        section_path: list[str],
        page_number: int | None,
        split_method: SplitMethod,
        source_block_path: list[str],
        chunk_counter: list[int],
        overlap_tokens: int = 0,
    ) -> ChunkRecord:
        index = chunk_counter[0]
        chunk_counter[0] += 1
        ocr = retrieval.provenance.ocr
        page_confidence = None
        if page_number is not None and retrieval.structured_content.pages:
            page_confidence = next(
                (page.ocr_confidence for page in retrieval.structured_content.pages if page.page_number == page_number),
                None,
            )
        return ChunkRecord(
            chunk_id="pending",
            document_id=retrieval.document_id,
            document_version=retrieval.document_version,
            kb_dataset_version=retrieval.kb_dataset_version,
            chunk_index=index,
            website=retrieval.website,
            document_type=retrieval.document_type,
            title=retrieval.title,
            section_path=section_path,
            content=content,
            token_count=token_count,
            split_method=split_method,
            source_url=retrieval.source_url,
            canonical_url=retrieval.canonical_url,
            page_number=page_number,
            source_type=retrieval.source_type,
            extraction_method=retrieval.extraction_method,
            overlap_tokens=overlap_tokens,
            provenance=ChunkProvenance(
                content_hash=retrieval.content_hash,
                source_block_path=source_block_path,
                ocr_engine=ocr.engine if ocr else None,
                ocr_dpi=ocr.dpi if ocr else None,
                ocr_language=ocr.language if ocr else None,
                ocr_page_confidence=page_confidence,
            ),
            language=retrieval.language,
        )
