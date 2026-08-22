"""Phase 16 V3 chunk quality reporting."""

from __future__ import annotations

import statistics
from typing import Any

from app.config import settings
from app.kb.ingestion.models import Phase12ChunkRecord
from app.kb.ingestion.phase16_corpus import Phase16Corpus


def _is_heading_only(chunk: Phase12ChunkRecord) -> bool:
    text = chunk.content.strip()
    if chunk.token_count <= 12 and not text.endswith((".", "?", "!")):
        return True
    return False


def build_phase16_quality_report(corpus: Phase16Corpus) -> dict[str, Any]:
    chunks = corpus.all_chunks
    token_counts = [chunk.token_count for chunk in chunks]
    heading_only = [chunk.chunk_id for chunk in chunks if _is_heading_only(chunk)]
    over_max = [chunk.chunk_id for chunk in chunks if chunk.token_count > settings.chunk_max_tokens]
    below_target = [chunk.chunk_id for chunk in chunks if chunk.token_count < settings.phase16_target_min_tokens]
    in_target = [
        chunk.chunk_id
        for chunk in chunks
        if settings.phase16_target_min_tokens <= chunk.token_count <= settings.phase16_target_max_tokens
    ]

    unit_types: dict[str, int] = {}
    for chunk in chunks:
        unit_types[chunk.content_type] = unit_types.get(chunk.content_type, 0) + 1

    missing_hierarchy = [
        chunk.chunk_id for chunk in chunks if not chunk.section_path or not chunk.parent_section
    ]

    critical = heading_only + over_max
    gate = "PASS" if not critical else "NEEDS_REVIEW"

    return {
        "phase16_chunking": gate,
        "chunking_algorithm_version": settings.phase16_chunking_algorithm_version,
        "kb_dataset_version": settings.phase16_kb_dataset_version,
        "embedding_input_manifest": settings.phase16_embedding_input_manifest,
        "read_only": True,
        "pgvector_writes": 0,
        "source_pdfs": corpus.source_pdfs,
        "summary": {
            "documents": len(corpus.documents),
            "total_chunks": len(chunks),
            "suppressed_duplicates": len(corpus.suppressed_duplicates),
            "max_token_count": max(token_counts) if token_counts else 0,
            "median_token_count": round(statistics.median(token_counts), 1) if token_counts else 0,
            "mean_token_count": round(statistics.mean(token_counts), 1) if token_counts else 0,
            "in_target_band_30_150": len(in_target),
            "below_target_min": len(below_target),
            "above_target_max": len([c for c in chunks if c.token_count > settings.phase16_target_max_tokens]),
            "heading_only_chunks": len(heading_only),
            "over_hard_max_512": len(over_max),
        },
        "unit_type_counts": unit_types,
        "issues": {
            "heading_only": heading_only,
            "over_hard_max": over_max,
            "missing_hierarchy": missing_hierarchy[:20],
        },
        "suppressed_duplicates": corpus.suppressed_duplicates,
        "documents": [
            {
                "label": doc.label,
                "filename": doc.filename,
                "document_id": doc.record.document_id,
                "chunk_count": len(doc.chunks),
            }
            for doc in corpus.documents
        ],
    }


def format_phase16_quality_text(report: dict[str, Any]) -> str:
    summary = report["summary"]
    lines = [
        f"PHASE16_CHUNKING: {report['phase16_chunking']}",
        f"Algorithm: {report['chunking_algorithm_version']}",
        f"Dataset: {report['kb_dataset_version']}",
        "",
        f"Documents: {summary['documents']}",
        f"Total chunks: {summary['total_chunks']}",
        f"Suppressed duplicates: {summary['suppressed_duplicates']}",
        f"Max tokens: {summary['max_token_count']}",
        f"Median tokens: {summary['median_token_count']}",
        f"Mean tokens: {summary['mean_token_count']}",
        f"In target band (30–150): {summary['in_target_band_30_150']}",
        f"Below target min (<30): {summary['below_target_min']}",
        f"Above target max (>150): {summary['above_target_max']}",
        f"Heading-only chunks: {summary['heading_only_chunks']}",
        f"Over hard max (>512): {summary['over_hard_max_512']}",
        "",
        "Unit types:",
    ]
    for unit_type, count in sorted(report["unit_type_counts"].items()):
        lines.append(f"  {unit_type}: {count}")
    if report["suppressed_duplicates"]:
        lines.extend(["", "Suppressed duplicates:"])
        for item in report["suppressed_duplicates"][:10]:
            lines.append(
                f"  {item['document']}:{item['chunk_id'][:12]} -> kept {item['kept_document']}"
            )
    return "\n".join(lines)
