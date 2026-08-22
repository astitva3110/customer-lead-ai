"""Phase 10.6 chunk quality dry-run report with Phase 10.5 comparison."""

from __future__ import annotations

import json
import re
import statistics
from collections import Counter, defaultdict
from datetime import datetime, timezone
from pathlib import Path

from app.config import settings
from app.kb.chunking.config import ChunkingConfig
from app.kb.chunking.duplicates import DuplicateClassification, analyze_document_duplicates
from app.kb.chunking.fidelity import (
    check_faq_atomicity,
    check_policy_fidelity,
    check_product_spec_fidelity,
    check_prospectus_fidelity,
)
from app.kb.chunking.models import SplitMethod
from app.kb.chunking.noise import classify_chunk_noise
from app.kb.chunking.semantic_units import effective_document_type
from app.kb.chunking.service import ChunkingDryRunService
from app.kb.chunking.validators import validate_chunks
from app.kb.integrity.version import KB_DATASET_VERSION
from app.kb.retrieval.storage import RetrievalStore

VERY_SMALL_THRESHOLD = 8
RETURNS_DOC_ID = "5f692faa-9f76-5643-9c78-b749258d06b4"
RADIUS_DOC_ID = "5f2ec8ed-c57d-5511-8d23-df8adf1b548d"
PROSPECTUS_IDS = (
    "01e76cb2-317f-509a-b8b5-f473f5bd5400",
    "8bd45b19-cb3d-5e56-ad5a-b67f1af7d2f9",
)
ALL_CAPS_HEADING = re.compile(r"^[A-Z0-9][A-Z0-9\s&,\-./():'\"]{2,60}$")
SHOPIFY_UI_STANDALONE = re.compile(r"^open media \d+ in modal$", re.I)


def _percentile(values: list[int], pct: float) -> float:
    if not values:
        return 0.0
    ordered = sorted(values)
    index = int(round((pct / 100) * (len(ordered) - 1)))
    return float(ordered[index])


def _body(content: str) -> str:
    return content.split("\n\n")[-1].strip() if content else ""


def _is_heading_only(chunk) -> bool:
    body = _body(chunk.content)
    path_tail = chunk.section_path[-1] if chunk.section_path else ""
    if body == path_tail and len(body) < 50:
        return True
    return len(body) < 40 and bool(ALL_CAPS_HEADING.match(body)) and chunk.token_count < VERY_SMALL_THRESHOLD


def _orphan_tail_count(chunks, config: ChunkingConfig) -> int:
    count = 0
    for chunk in chunks:
        if (
            chunk.split_method == SplitMethod.HARD_TOKEN_FALLBACK
            and chunk.token_count < config.min_meaningful_chunk_tokens
            and len(chunk.content.split()) < 20
        ):
            count += 1
    return count


def _type_stats(chunks) -> dict:
    by_type: dict[str, list] = defaultdict(list)
    for chunk in chunks:
        by_type[effective_document_type(chunk.document_type, chunk.title, chunk.canonical_url)].append(chunk)
    stats = {}
    for doc_type, type_chunks in sorted(by_type.items()):
        tokens = [c.token_count for c in type_chunks]
        docs = {c.document_id for c in type_chunks}
        stats[doc_type] = {
            "document_count": len(docs),
            "chunk_count": len(type_chunks),
            "min_tokens": min(tokens),
            "median_tokens": statistics.median(tokens),
            "mean_tokens": round(statistics.mean(tokens), 2),
            "p90_tokens": _percentile(tokens, 90),
            "p95_tokens": _percentile(tokens, 95),
            "max_tokens": max(tokens),
            "small_chunk_count": sum(1 for t in type_chunks if t.token_count < VERY_SMALL_THRESHOLD),
            "hard_fallback_count": sum(1 for t in type_chunks if t.split_method == SplitMethod.HARD_TOKEN_FALLBACK),
        }
    return stats


def _source_stats(chunks) -> dict:
    by_source: dict[str, list] = defaultdict(list)
    for chunk in chunks:
        by_source[chunk.source_type.value].append(chunk)
    stats = {}
    for source_type, source_chunks in sorted(by_source.items()):
        tokens = [c.token_count for c in source_chunks]
        docs = {c.document_id for c in source_chunks}
        stats[source_type] = {
            "document_count": len(docs),
            "chunk_count": len(source_chunks),
            "min_tokens": min(tokens),
            "median_tokens": statistics.median(tokens),
            "mean_tokens": round(statistics.mean(tokens), 2),
            "p90_tokens": _percentile(tokens, 90),
            "p95_tokens": _percentile(tokens, 95),
            "max_tokens": max(tokens),
            "small_chunk_count": sum(1 for t in source_chunks if t.token_count < VERY_SMALL_THRESHOLD),
            "hard_fallback_count": sum(1 for t in source_chunks if t.split_method == SplitMethod.HARD_TOKEN_FALLBACK),
        }
    return stats


def _inspect_doc(chunks, document_id: str) -> dict:
    doc_chunks = [c for c in chunks if c.document_id == document_id]
    if not doc_chunks:
        return {"found": False, "document_id": document_id}
    tokens = [c.token_count for c in doc_chunks]
    return {
        "found": True,
        "document_id": document_id,
        "title": doc_chunks[0].title,
        "chunk_count": len(doc_chunks),
        "token_median": statistics.median(tokens),
        "token_max": max(tokens),
        "hard_fallback_count": sum(1 for c in doc_chunks if c.split_method == SplitMethod.HARD_TOKEN_FALLBACK),
        "orphan_tails": _orphan_tail_count(doc_chunks, ChunkingConfig()),
        "heading_only_count": sum(1 for c in doc_chunks if _is_heading_only(c)),
        "page_numbers_present": sum(1 for c in doc_chunks if c.page_number is not None),
    }


def _load_phase10_5_baseline() -> dict | None:
    path = settings.reports_dir / "phase10_5_chunk_quality.json"
    if not path.exists():
        return None
    return json.loads(path.read_text(encoding="utf-8"))


def main() -> None:
    config = ChunkingConfig()
    service = ChunkingDryRunService(RetrievalStore(settings.retrieval_dir), config=config)
    result = service.run()
    chunks = result.chunks
    tokens = [c.token_count for c in chunks]

    validation = validate_chunks(chunks, config)
    dup_by_doc: Counter = Counter()
    for doc_id in {c.document_id for c in chunks}:
        doc_chunks = [c for c in chunks if c.document_id == doc_id]
        for entry in analyze_document_duplicates(doc_chunks):
            dup_by_doc[entry.classification.value] += 1

    heading_only = sum(1 for c in chunks if _is_heading_only(c))
    context_noise = sum(1 for s in result.suppressed_chunks if s.get("suppression_reason") == "context_noise")
    standalone_low = sum(
        1 for s in result.suppressed_chunks if s.get("suppression_reason") == "standalone_low_value"
    )
    orphan_tails = _orphan_tail_count(chunks, config)
    max_violations = sum(1 for c in chunks if c.token_count > config.max_chunk_tokens)
    ui_chunks = sum(
        1
        for c in chunks
        if SHOPIFY_UI_STANDALONE.match(_body(c.content))
    )
    ui_fragments_suppressed = sum(
        1
        for s in result.suppressed_chunks
        if s.get("detail") == "shopify_ui_fragment"
    )

    product_fidelity = check_product_spec_fidelity(chunks)
    policy_fidelity = check_policy_fidelity(chunks)
    prospectus_fidelity = check_prospectus_fidelity(chunks, max_chunk_tokens=config.max_chunk_tokens)
    faq_fidelity = check_faq_atomicity(chunks)

    baseline = _load_phase10_5_baseline()
    comparison = None
    if baseline:
        comparison = {
            "phase10_5_total_chunks": baseline.get("total_chunks"),
            "phase10_6_total_chunks": len(chunks),
            "phase10_5_very_small": baseline.get("very_small_chunks", {}).get("total"),
            "phase10_6_very_small": sum(1 for t in tokens if t < VERY_SMALL_THRESHOLD),
            "phase10_5_heading_only": baseline.get("very_small_chunks", {}).get("classification_counts", {}).get(
                "HEADING_ONLY"
            ),
            "phase10_6_heading_only": heading_only,
            "phase10_5_unsafe_tails": baseline.get("hard_token_fallback_chunks", {}).get("unsafe_tail_count"),
            "phase10_6_orphan_tails": orphan_tails,
            "phase10_5_hard_fallback": baseline.get("hard_token_fallback_chunks", {}).get("total"),
            "phase10_6_hard_fallback": sum(1 for c in chunks if c.split_method == SplitMethod.HARD_TOKEN_FALLBACK),
            "phase10_5_duplicate_issues": baseline.get("duplicate_content_warnings", {}).get(
                "validator_reported_issues"
            ),
            "phase10_6_suppressed_duplicates": sum(
                1
                for s in result.suppressed_chunks
                if "duplicate" in s.get("suppression_reason", "") or s.get("suppression_reason") == "boilerplate"
            ),
        }

    focal = {
        "return_replacement_policy": _inspect_doc(chunks, RETURNS_DOC_ID),
        "radius_m16_bte": _inspect_doc(chunks, RADIUS_DOC_ID),
        "draft_prospectus": _inspect_doc(chunks, PROSPECTUS_IDS[0]),
        "final_prospectus": _inspect_doc(chunks, PROSPECTUS_IDS[1]),
    }

    pass_verdict = (
        orphan_tails == 0
        and max_violations == 0
        and product_fidelity.passed
        and policy_fidelity.passed
        and prospectus_fidelity.passed
        and faq_fidelity.passed
        and ui_chunks == 0
    )
    verdict = "PASS" if pass_verdict else "NEEDS_REVIEW"

    report = {
        "generated_at": datetime.now(timezone.utc).isoformat(),
        "phase": "phase10_6_chunk_quality",
        "kb_dataset_version": KB_DATASET_VERSION,
        "total_documents": result.documents_processed,
        "total_chunks": len(chunks),
        "token_distribution": {
            "min": min(tokens) if tokens else 0,
            "median": statistics.median(tokens) if tokens else 0,
            "mean": round(statistics.mean(tokens), 2) if tokens else 0,
            "p90": _percentile(tokens, 90),
            "p95": _percentile(tokens, 95),
            "max": max(tokens) if tokens else 0,
        },
        "quality_metrics": {
            "very_small_chunks": sum(1 for t in tokens if t < VERY_SMALL_THRESHOLD),
            "heading_only_chunks": heading_only,
            "context_noise_suppressed": context_noise,
            "standalone_low_value_suppressed": standalone_low,
            "total_suppressed": len(result.suppressed_chunks),
            "hard_fallback_chunks": sum(1 for c in chunks if c.split_method == SplitMethod.HARD_TOKEN_FALLBACK),
            "orphan_tails": orphan_tails,
            "max_token_violations": max_violations,
            "ui_navigation_chunks": ui_chunks,
            "ui_fragments_suppressed": ui_fragments_suppressed,
        },
        "duplicate_classifications": dict(dup_by_doc),
        "statistics_by_document_type": _type_stats(chunks),
        "statistics_by_source_type": _source_stats(chunks),
        "regression_documents": focal,
        "fidelity": {
            "product_spec": product_fidelity.passed,
            "policy": policy_fidelity.passed,
            "prospectus": prospectus_fidelity.passed,
            "faq_atomicity": faq_fidelity.passed,
        },
        "comparison_phase10_5_to_10_6": comparison,
        "suppressed_sample": result.suppressed_chunks[:20],
        "verdict": verdict,
        "chunk_quality_review_answer": verdict,
    }

    reports_dir = settings.reports_dir
    reports_dir.mkdir(parents=True, exist_ok=True)
    json_path = reports_dir / "phase10_6_chunk_quality.json"
    txt_path = reports_dir / "phase10_6_chunk_quality.txt"
    json_path.write_text(json.dumps(report, ensure_ascii=False, indent=2), encoding="utf-8")

    lines = [
        "PHASE 10.6 — CHUNK QUALITY FIXES (DRY RUN)",
        f"Generated: {report['generated_at']}",
        "",
        f"Total chunks: {len(chunks)}",
        f"Token distribution: {report['token_distribution']}",
        "",
        "Quality metrics:",
    ]
    for key, value in report["quality_metrics"].items():
        lines.append(f"  {key}: {value}")
    lines.extend(["", "Duplicate classifications:", f"  {dict(dup_by_doc)}"])
    if comparison:
        lines.extend(["", "Phase 10.5 → 10.6 comparison:", f"  {comparison}"])
    lines.extend(["", "Fidelity:", f"  {report['fidelity']}", "", f"CHUNK_QUALITY_REVIEW: {verdict}"])
    txt_path.write_text("\n".join(lines), encoding="utf-8")

    print(json.dumps({"verdict": verdict, "chunks": len(chunks), "orphan_tails": orphan_tails}, indent=2))


if __name__ == "__main__":
    main()
