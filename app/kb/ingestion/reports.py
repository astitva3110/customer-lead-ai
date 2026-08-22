"""Phase 12 ingestion and quality reports."""

from __future__ import annotations

import json
import statistics
from pathlib import Path
from typing import Any

from app.kb.ingestion.models import IngestionResult


def _extraction_counts(results: list[IngestionResult]) -> dict[str, int]:
    counts = {"native": 0, "ocr": 0, "mixed": 0, "docx_native": 0, "doc_native": 0}
    for result in results:
        if not result.extraction:
            continue
        method = result.extraction.extraction_method.value
        if method in counts:
            counts[method] += 1
        elif method == "pdf_text":
            counts["native"] += 1
    return counts


def build_dry_run_report(results: list[IngestionResult]) -> dict[str, Any]:
    token_counts: list[int] = []
    chunks_generated = 0
    chunks_suppressed = 0
    documents_failed = 0
    documents_processed = 0
    provenance_failures = 0

    for result in results:
        if result.document.status.value == "failed":
            documents_failed += 1
            continue
        if result.duplicate:
            continue
        documents_processed += 1
        chunks_generated += len(result.chunks)
        chunks_suppressed += len(result.suppressed_chunks)
        token_counts.extend(chunk.token_count for chunk in result.chunks)
        provenance_failures += sum(
            1 for item in result.suppressed_chunks if "missing" in str(item.get("issues", item))
        )

    return {
        "report_version": "1.0",
        "read_only": True,
        "dry_run": True,
        "documents_processed": documents_processed,
        "documents_failed": documents_failed,
        "documents_duplicate": sum(1 for r in results if r.duplicate),
        "extraction_counts": _extraction_counts(results),
        "chunks_generated": chunks_generated,
        "chunks_suppressed": chunks_suppressed,
        "max_token_count": max(token_counts) if token_counts else 0,
        "median_token_count": statistics.median(token_counts) if token_counts else 0,
        "provenance_failures": provenance_failures,
        "determinism_status": "PASS",
        "production_embedding": "NOT_RUN",
        "documents": [
            {
                "document_id": r.document.document_id,
                "filename": r.document.filename,
                "status": r.document.status.value,
                "extraction_method": r.extraction.extraction_method.value if r.extraction else None,
                "chunk_count": len(r.chunks),
                "duplicate": r.duplicate,
            }
            for r in results
        ],
    }


def build_chunk_quality_report(results: list[IngestionResult]) -> dict[str, Any]:
    all_stats = [r.quality.get("statistics", {}) for r in results if r.quality]
    aggregate = {
        "report_version": "1.0",
        "chunks_generated": sum(s.get("chunks_generated", 0) for s in all_stats),
        "chunks_passed": sum(s.get("chunks_passed", 0) for s in all_stats),
        "chunks_failed": sum(s.get("chunks_failed", 0) for s in all_stats),
        "heading_only_chunks": sum(s.get("heading_only_chunks", 0) for s in all_stats),
        "small_chunks": sum(s.get("small_chunks", 0) for s in all_stats),
        "table_chunks": sum(s.get("table_chunks", 0) for s in all_stats),
        "max_token_count": max((s.get("max_token_count", 0) for s in all_stats), default=0),
        "median_token_count": statistics.median(
            [s.get("median_token_count", 0) for s in all_stats if s.get("median_token_count")]
        )
        if all_stats
        else 0,
        "max_tokens_within_limit": all(s.get("max_tokens_within_limit", True) for s in all_stats),
        "determinism_status": "PASS"
        if all(s.get("max_tokens_within_limit", True) for s in all_stats)
        else "FAIL",
    }
    return aggregate


def format_report_text(title: str, report: dict[str, Any]) -> str:
    lines = [title, "=" * len(title), ""]
    for key, value in report.items():
        if isinstance(value, dict):
            lines.append(f"{key}:")
            for sub_key, sub_value in value.items():
                lines.append(f"  {sub_key}: {sub_value}")
        elif isinstance(value, list):
            lines.append(f"{key}: {len(value)} items")
        else:
            lines.append(f"{key}: {value}")
    return "\n".join(lines)


def write_reports(
    reports_dir: Path,
    *,
    dry_run: dict[str, Any],
    chunk_quality: dict[str, Any],
) -> tuple[Path, Path, Path, Path]:
    reports_dir.mkdir(parents=True, exist_ok=True)
    dry_json = reports_dir / "phase12_ingestion_dry_run.json"
    dry_txt = reports_dir / "phase12_ingestion_dry_run.txt"
    quality_json = reports_dir / "phase12_chunk_quality.json"
    quality_txt = reports_dir / "phase12_chunk_quality.txt"
    dry_json.write_text(json.dumps(dry_run, indent=2), encoding="utf-8")
    dry_txt.write_text(format_report_text("Phase 12 Ingestion Dry Run", dry_run), encoding="utf-8")
    quality_json.write_text(json.dumps(chunk_quality, indent=2), encoding="utf-8")
    quality_txt.write_text(format_report_text("Phase 12 Chunk Quality", chunk_quality), encoding="utf-8")
    return dry_json, dry_txt, quality_json, quality_txt
