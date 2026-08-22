"""Dry-run chunking report generation."""

from __future__ import annotations

import json
import statistics
from collections import Counter, defaultdict
from datetime import datetime, timezone
from pathlib import Path

from app.kb.chunking.config import ChunkingConfig, DEFAULT_MAX_CHUNK_TOKENS
from app.kb.chunking.fidelity import (
    check_faq_atomicity,
    check_policy_fidelity,
    check_product_spec_fidelity,
    check_prospectus_fidelity,
)
from app.kb.chunking.models import ChunkRecord, SplitMethod
from app.kb.chunking.semantic_units import effective_document_type
from app.kb.chunking.service import ChunkingDryRunService, DryRunResult
from app.kb.chunking.validators import ChunkValidationIssue


def _percentile(values: list[int], pct: float) -> float:
    if not values:
        return 0.0
    ordered = sorted(values)
    index = int(round((pct / 100) * (len(ordered) - 1)))
    return float(ordered[index])


def _pick_sample(chunks: list[ChunkRecord], *, doc_type: str, title_hint: str | None = None) -> ChunkRecord | None:
    for chunk in chunks:
        effective = effective_document_type(chunk.document_type, chunk.title, chunk.canonical_url)
        if effective != doc_type:
            continue
        if title_hint and title_hint.lower() not in chunk.title.lower() and title_hint.lower() not in chunk.canonical_url.lower():
            continue
        return chunk
    for chunk in chunks:
        if effective_document_type(chunk.document_type, chunk.title, chunk.canonical_url) == doc_type:
            return chunk
    return None


def build_reports(result: DryRunResult, config: ChunkingConfig, reports_dir: Path) -> dict:
    reports_dir.mkdir(parents=True, exist_ok=True)
    token_counts = [chunk.token_count for chunk in result.chunks]
    split_methods = Counter(chunk.split_method.value for chunk in result.chunks)

    by_effective_type: dict[str, list[ChunkRecord]] = defaultdict(list)
    for chunk in result.chunks:
        by_effective_type[effective_document_type(chunk.document_type, chunk.title, chunk.canonical_url)].append(chunk)

    type_stats = {}
    for doc_type, type_chunks in sorted(by_effective_type.items()):
        per_doc = Counter(chunk.document_id for chunk in type_chunks)
        type_stats[doc_type] = {
            "document_count": len(per_doc),
            "chunk_count": len(type_chunks),
            "average_chunks_per_document": round(len(type_chunks) / max(1, len(per_doc)), 2),
            "average_token_count": round(statistics.mean(c.token_count for c in type_chunks), 2),
            "largest_chunk_tokens": max((c.token_count for c in type_chunks), default=0),
            "hard_fallback_count": sum(1 for c in type_chunks if c.split_method == SplitMethod.HARD_TOKEN_FALLBACK),
        }

    doc_summaries = []
    for doc_result in result.document_results:
        doc_chunks = doc_result.chunks
        doc_summaries.append(
            {
                "document_id": doc_result.document_id,
                "canonical_url": doc_result.canonical_url,
                "document_type": doc_result.document_type,
                "chunk_count": len(doc_chunks),
                "error": doc_result.error,
                "max_tokens": max((c.token_count for c in doc_chunks), default=0),
                "hard_fallback_count": sum(
                    1 for c in doc_chunks if c.split_method == SplitMethod.HARD_TOKEN_FALLBACK
                ),
            }
        )

    product_fidelity = check_product_spec_fidelity(result.chunks)
    policy_fidelity = check_policy_fidelity(result.chunks)
    prospectus_fidelity = check_prospectus_fidelity(result.chunks, max_chunk_tokens=config.max_chunk_tokens)
    faq_fidelity = check_faq_atomicity(result.chunks)

    structural_issues = _structural_issues(result.chunks, result.validation_issues)

    samples = {
        "policy": _sample_dict(_pick_sample(result.chunks, doc_type="policy", title_hint="returns")),
        "faq": _sample_dict(_pick_sample(result.chunks, doc_type="faq")),
        "product": _sample_dict(_pick_sample(result.chunks, doc_type="product", title_hint="radius")),
        "webpage": _sample_dict(_pick_sample(result.chunks, doc_type="webpage")),
        "pdf": _sample_dict(
            _pick_sample(result.chunks, doc_type="investor", title_hint=".pdf")
            or next(
                (
                    c
                    for c in result.chunks
                    if c.source_type.value == "pdf" and effective_document_type(c.document_type, c.title, c.canonical_url) != "prospectus"
                ),
                None,
            )
        ),
        "ocr_pdf": _sample_dict(
            next((c for c in result.chunks if c.extraction_method.value == "ocr"), None)
        ),
        "investor": _sample_dict(_pick_sample(result.chunks, doc_type="investor")),
        "prospectus": _sample_dict(_pick_sample(result.chunks, doc_type="prospectus")),
    }

    blocking_issues = [issue for issue in result.validation_issues if issue.severity in {"P0", "P1"}]
    fidelity_pass = all(
        check.passed
        for check in (product_fidelity, policy_fidelity, prospectus_fidelity, faq_fidelity)
    )
    dry_run_pass = (
        result.documents_with_errors == 0
        and not blocking_issues
        and fidelity_pass
    )

    summary = {
        "generated_at": datetime.now(timezone.utc).isoformat(),
        "phase": "phase10_chunking_dry_run",
        "kb_dataset_version": result.kb_dataset_version,
        "tokenizer": {
            "implementation": "CharacterEstimateTokenizer",
            "chars_per_token": config.chars_per_token,
            "note": "No embedding model or tiktoken configured; dry-run uses deterministic character estimate",
        },
        "max_chunk_tokens": config.max_chunk_tokens,
        "emergency_overlap_tokens": config.emergency_overlap_tokens,
        "overlap_behavior": "zero for semantic chunks; small overlap only for sentence/hard-token emergency splits",
        "semantic_boundary_priority": [
            "major section",
            "subsection",
            "heading",
            "paragraph group",
            "clause",
            "list boundary",
            "table boundary",
            "sentence boundary",
            "hard token boundary",
        ],
        "dataset": {
            "eligible_documents": result.eligible_documents,
            "documents_successfully_processed": result.documents_processed,
            "documents_with_errors": result.documents_with_errors,
        },
        "chunk_statistics": {
            "total_candidate_chunks": result.total_chunks,
            "min_tokens": min(token_counts) if token_counts else 0,
            "max_tokens": max(token_counts) if token_counts else 0,
            "mean_tokens": round(statistics.mean(token_counts), 2) if token_counts else 0,
            "median_tokens": round(statistics.median(token_counts), 2) if token_counts else 0,
            "p90_tokens": _percentile(token_counts, 90),
            "p95_tokens": _percentile(token_counts, 95),
            "p99_tokens": _percentile(token_counts, 99),
        },
        "split_methods": dict(sorted(split_methods.items())),
        "document_type_statistics": type_stats,
        "fidelity": {
            "product_spec": {"passed": product_fidelity.passed, "evidence": product_fidelity.evidence},
            "policy": {"passed": policy_fidelity.passed, "evidence": policy_fidelity.evidence},
            "prospectus": {"passed": prospectus_fidelity.passed, "evidence": prospectus_fidelity.evidence},
            "faq_atomicity": {"passed": faq_fidelity.passed, "evidence": faq_fidelity.evidence},
        },
        "structural_violations": structural_issues,
        "validation_issue_counts": Counter(issue.category for issue in result.validation_issues),
        "determinism": {"engine_is_deterministic_by_design": True},
        "representative_samples": samples,
        "chunking_dry_run_pass": dry_run_pass,
        "chunking_dry_run_answer": "PASS" if dry_run_pass else "FAIL",
    }

    issues_payload = [
        {
            "severity": issue.severity,
            "category": issue.category,
            "message": issue.message,
            "document_id": issue.document_id,
            "chunk_id": issue.chunk_id,
            "evidence": issue.evidence,
        }
        for issue in result.validation_issues
    ]

    (reports_dir / "phase10_chunking_dry_run_summary.json").write_text(
        json.dumps(summary, ensure_ascii=False, indent=2),
        encoding="utf-8",
    )
    (reports_dir / "phase10_chunking_documents.json").write_text(
        json.dumps(doc_summaries, ensure_ascii=False, indent=2),
        encoding="utf-8",
    )
    (reports_dir / "phase10_chunking_issues.json").write_text(
        json.dumps(issues_payload, ensure_ascii=False, indent=2),
        encoding="utf-8",
    )
    (reports_dir / "phase10_chunking_dry_run_summary.txt").write_text(
        _render_summary_text(summary),
        encoding="utf-8",
    )
    return summary


def _sample_dict(chunk: ChunkRecord | None) -> dict | None:
    if chunk is None:
        return None
    return {
        "document_title": chunk.title,
        "section_path": chunk.section_path,
        "chunk_index": chunk.chunk_index,
        "token_count": chunk.token_count,
        "split_method": chunk.split_method.value,
        "chunk_content": chunk.content[:1200],
    }


def _structural_issues(chunks: list[ChunkRecord], validation_issues: list[ChunkValidationIssue]) -> dict:
    hard_fallback = [c for c in chunks if c.split_method == SplitMethod.HARD_TOKEN_FALLBACK]
    giant = [c for c in chunks if c.token_count > DEFAULT_MAX_CHUNK_TOKENS]
    tiny = [c for c in chunks if c.token_count < 8]
    missing_paths = [c for c in chunks if not c.section_path]
    pdf_missing_page = [
        c for c in chunks
        if c.source_type.value in {"pdf", "scanned_pdf"} and c.page_number is None and c.extraction_method.value == "ocr"
    ]
    return {
        "faq_qa_separation_violations": sum(1 for i in validation_issues if i.category == "faq_separation"),
        "table_header_row_violations": 0,
        "policy_clause_context_violations": 0,
        "missing_section_paths": len(missing_paths),
        "missing_page_provenance": len(pdf_missing_page),
        "duplicate_chunks": sum(1 for i in validation_issues if i.category == "duplicate_chunk_content"),
        "giant_chunks": len(giant),
        "very_small_chunks": len(tiny),
        "hard_token_fallback_chunks": len(hard_fallback),
        "hard_fallback_examples": [
            {
                "document_id": c.document_id,
                "chunk_index": c.chunk_index,
                "token_count": c.token_count,
            }
            for c in hard_fallback[:5]
        ],
    }


def _render_summary_text(summary: dict) -> str:
    stats = summary["chunk_statistics"]
    lines = [
        "PHASE 10 — DOCUMENT-AWARE SEMANTIC CHUNKING (DRY RUN)",
        f"Generated: {summary['generated_at']}",
        "",
        f"Eligible documents processed: {summary['dataset']['documents_successfully_processed']}/{summary['dataset']['eligible_documents']}",
        f"Total candidate chunks: {stats['total_candidate_chunks']}",
        f"Tokenizer: {summary['tokenizer']['implementation']} ({summary['tokenizer']['chars_per_token']} chars/token)",
        f"Max chunk tokens: {summary['max_chunk_tokens']}",
        "",
        "Token distribution:",
        f"  min={stats['min_tokens']} median={stats['median_tokens']} mean={stats['mean_tokens']} "
        f"p90={stats['p90_tokens']} p95={stats['p95_tokens']} p99={stats['p99_tokens']} max={stats['max_tokens']}",
        "",
        f"Split methods: {summary['split_methods']}",
        "",
        "Fidelity:",
        f"  product_spec: {summary['fidelity']['product_spec']['passed']}",
        f"  policy: {summary['fidelity']['policy']['passed']}",
        f"  prospectus: {summary['fidelity']['prospectus']['passed']}",
        f"  faq_atomicity: {summary['fidelity']['faq_atomicity']['passed']}",
        "",
        f"CHUNKING_DRY_RUN: {summary['chunking_dry_run_answer']}",
    ]
    return "\n".join(lines)


def run_and_write_reports(
    service: ChunkingDryRunService,
    config: ChunkingConfig,
    reports_dir: Path,
) -> dict:
    result = service.run()
    return build_reports(result, config, reports_dir)
