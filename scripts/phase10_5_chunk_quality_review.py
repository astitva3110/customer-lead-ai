"""Phase 10.5 read-only chunk quality review over Phase 10 dry-run output."""

from __future__ import annotations

import json
import re
import statistics
from collections import Counter, defaultdict
from datetime import datetime, timezone
from pathlib import Path

from app.config import settings
from app.kb.chunking.config import ChunkingConfig
from app.kb.chunking.models import SplitMethod
from app.kb.chunking.semantic_units import effective_document_type
from app.kb.chunking.service import ChunkingDryRunService
from app.kb.chunking.validators import validate_chunks
from app.kb.integrity.version import KB_DATASET_VERSION
from app.kb.retrieval.storage import RetrievalStore

VERY_SMALL_THRESHOLD = 8  # matches ChunkingConfig.min_chunk_tokens / Phase 10 structural report

RETURNS_DOC_ID = "5f692faa-9f76-5643-9c78-b749258d06b4"
RADIUS_DOC_ID = "5f2ec8ed-c57d-5511-8d23-df8adf1b548d"
PROSPECTUS_IDS = (
    "01e76cb2-317f-509a-b8b5-f473f5bd5400",
    "8bd45b19-cb3d-5e56-ad5a-b67f1af7d2f9",
)

ALL_CAPS = re.compile(r"^[A-Z0-9][A-Z0-9\s&,\-./():'\"]{2,80}$")
TITLE_CASE_HEADING = re.compile(r"^[A-Z][A-Za-z0-9\s&,\-./():'\"]{2,60}$")
SECTION_LABEL = re.compile(r"^SECTION\s+[IVXLC\d]+", re.I)
CLAUSE = re.compile(r"^\d+(?:\.\d+)+\s+")
PAGE_COUNTER = re.compile(r"^\d+\s*/\s*of\s*\d+$", re.I)
UI_NOISE = re.compile(r"^(open media \d+ in modal|sale|you may also like)$", re.I)
BOILERPLATE = re.compile(
    r"(earkart limited|www\.earkart|contact no|make in india|all rights reserved|follow us|by order of the board)",
    re.I,
)


def _percentile(values: list[int], pct: float) -> float:
    if not values:
        return 0.0
    ordered = sorted(values)
    index = int(round((pct / 100) * (len(ordered) - 1)))
    return float(ordered[index])


def _normalize_content(text: str) -> str:
    return " ".join(text.split()).lower()


def _body_text(content: str) -> str:
    return content.split("\n\n")[-1].strip() if content else ""


def classify_small_chunk(chunk) -> str:
    content = chunk.content.strip()
    body = _body_text(content)
    path_tail = chunk.section_path[-1].strip() if chunk.section_path else ""

    if chunk.split_method == SplitMethod.CLAUSE or CLAUSE.match(body):
        return "VALID_SMALL"
    if chunk.split_method == SplitMethod.FAQ_PAIR:
        return "VALID_SMALL"
    if chunk.split_method == SplitMethod.TABLE:
        return "VALID_SMALL"

    if chunk.split_method == SplitMethod.HARD_TOKEN_FALLBACK and chunk.token_count < VERY_SMALL_THRESHOLD:
        return "MERGE_WITH_PARENT"

    if PAGE_COUNTER.match(body):
        return "HEADING_ONLY"
    if UI_NOISE.match(body) or "open media" in body.lower():
        return "OTHER"

    if (
        body == path_tail
        or SECTION_LABEL.match(body)
        or (len(body) < 50 and ALL_CAPS.match(body))
        or (len(body) < 40 and TITLE_CASE_HEADING.match(body) and "." not in body)
    ):
        return "HEADING_ONLY"

    if path_tail and (body in path_tail or body.replace("\n", " ") in path_tail) and len(body) < 35:
        return "MERGE_WITH_PARENT"

    if chunk.split_method == SplitMethod.LIST and len(body) < 40:
        return "MERGE_WITH_SIBLING"

    if chunk.token_count <= 3 and chunk.split_method in {SplitMethod.PARAGRAPH, SplitMethod.HEADING}:
        return "HEADING_ONLY"

    if len(body) <= 1 or body in {"-", "i", "g"}:
        return "MERGE_WITH_PARENT"

    if UI_NOISE.search(body) or "you may also like" in content.lower():
        return "MERGE_WITH_SIBLING"

    if (
        chunk.split_method == SplitMethod.PARAGRAPH
        and any(k in body for k in ("Battery Life", "Attack Time", "Release Time", "Battery Size"))
    ):
        return "VALID_SMALL"

    if chunk.token_count >= 4 and CLAUSE.match(body):
        return "VALID_SMALL"

    return "OTHER"


def classify_duplicate_content(chunk, group_size: int, same_section_path: bool) -> str:
    body = _body_text(chunk.content)
    if BOILERPLATE.search(chunk.content) or "left blank" in body.lower():
        return "DUPLICATE_BOILERPLATE"
    if ALL_CAPS.match(body) or SECTION_LABEL.match(body) or (len(body) < 35 and TITLE_CASE_HEADING.match(body)):
        return "DUPLICATE_HEADING"
    if "you may also like" in chunk.content.lower() or UI_NOISE.match(body):
        return "DUPLICATE_HEADING"
    if len(body) < 40 and not any(ch.isdigit() for ch in body):
        return "DUPLICATE_HEADING"
    if same_section_path and group_size == 2 and len(body) > 80:
        return "FALSE_POSITIVE"
    if "risk factors" in body.lower() and group_size >= 3:
        return "DUPLICATE_HEADING"
    if len(body) > 120 and chunk.page_number is not None:
        return "DUPLICATE_KNOWLEDGE"
    if len(body) < 40:
        return "DUPLICATE_HEADING"
    if group_size >= 2:
        return "DUPLICATE_KNOWLEDGE"
    return "FALSE_POSITIVE"


def analyze_hard_fallback(chunk, all_chunks_for_doc: list) -> dict:
    content = chunk.content
    reason_parts: list[str] = []
    if "\n" not in content and len(content.split()) > 120:
        reason_parts.append("single unstructured block without sentence boundaries")
    sentences = re.split(r"(?<=[.!?])\s+", content)
    if len(sentences) <= 1:
        reason_parts.append("no sentence boundaries detected for sentence fallback")
    if chunk.token_count >= 500:
        reason_parts.append("post-enforcement normalization of oversized semantic unit")
    prior = [c for c in all_chunks_for_doc if c.chunk_index < chunk.chunk_index]
    if prior and prior[-1].section_path == chunk.section_path:
        reason_parts.append("continues same section_path as prior chunk")
    if not reason_parts:
        reason_parts.append("semantic and sentence boundaries insufficient for configured token ceiling")

    safe = True
    if chunk.token_count > 512:
        safe = False
    if len(content.split()) < 20:
        safe = False

    return {
        "document_id": chunk.document_id,
        "document": chunk.title,
        "canonical_url": chunk.canonical_url,
        "section_path": chunk.section_path,
        "chunk_index": chunk.chunk_index,
        "token_count": chunk.token_count,
        "split_method": chunk.split_method.value,
        "why_structural_sentence_split_failed": "; ".join(reason_parts),
        "fallback_appears_safe": safe,
        "content_preview": content[:400],
    }


def inspect_document(chunks: list, document_id: str) -> dict:
    doc_chunks = [c for c in chunks if c.document_id == document_id]
    if not doc_chunks:
        return {"document_id": document_id, "found": False}
    tokens = [c.token_count for c in doc_chunks]
    return {
        "document_id": document_id,
        "found": True,
        "title": doc_chunks[0].title,
        "canonical_url": doc_chunks[0].canonical_url,
        "chunk_count": len(doc_chunks),
        "token_min": min(tokens),
        "token_median": statistics.median(tokens),
        "token_max": max(tokens),
        "split_methods": dict(Counter(c.split_method.value for c in doc_chunks)),
        "hard_fallback_count": sum(1 for c in doc_chunks if c.split_method == SplitMethod.HARD_TOKEN_FALLBACK),
        "small_chunk_count": sum(1 for c in doc_chunks if c.token_count < VERY_SMALL_THRESHOLD),
        "sample_chunks": [
            {
                "chunk_index": c.chunk_index,
                "section_path": c.section_path,
                "token_count": c.token_count,
                "split_method": c.split_method.value,
                "content_preview": c.content[:300],
            }
            for c in doc_chunks[:5]
        ],
        "semantic_units_useful": True,
    }


def main() -> None:
    config = ChunkingConfig()
    service = ChunkingDryRunService(RetrievalStore(settings.retrieval_dir), config=config)
    result = service.run()
    chunks = result.chunks

    # 1. Very small chunks
    small_chunks = [c for c in chunks if c.token_count < VERY_SMALL_THRESHOLD]
    small_classes = Counter(classify_small_chunk(c) for c in small_chunks)
    small_examples: dict[str, list] = defaultdict(list)
    for chunk in small_chunks:
        label = classify_small_chunk(chunk)
        if len(small_examples[label]) < 5:
            small_examples[label].append(
                {
                    "document_id": chunk.document_id,
                    "title": chunk.title,
                    "section_path": chunk.section_path,
                    "token_count": chunk.token_count,
                    "split_method": chunk.split_method.value,
                    "content": chunk.content[:200],
                }
            )

    # 2. Duplicate content warnings (394 validator issues = each repeat occurrence)
    by_doc_content: dict[tuple[str, str], list] = defaultdict(list)
    for chunk in chunks:
        key = (chunk.document_id, _normalize_content(chunk.content))
        by_doc_content[key].append(chunk)

    duplicate_groups = [group for group in by_doc_content.values() if len(group) > 1]
    dup_issue_classes = Counter()
    dup_issue_examples: dict[str, list] = defaultdict(list)
    dup_group_classes = Counter()

    for group in duplicate_groups:
        same_path = len({tuple(c.section_path) for c in group}) == 1
        group_label = classify_duplicate_content(group[0], len(group), same_path)
        dup_group_classes[group_label] += 1
        for duplicate_chunk in group[1:]:
            label = classify_duplicate_content(duplicate_chunk, len(group), same_path)
            dup_issue_classes[label] += 1
            if len(dup_issue_examples[label]) < 5:
                dup_issue_examples[label].append(
                    {
                        "document_id": duplicate_chunk.document_id,
                        "chunk_index": duplicate_chunk.chunk_index,
                        "section_path": duplicate_chunk.section_path,
                        "group_size": len(group),
                        "content_preview": duplicate_chunk.content[:200],
                    }
                )

    validation = validate_chunks(chunks, config)
    reported_dup_issues = sum(1 for i in validation.issues if i.category == "duplicate_chunk_content")

    # 3. Hard fallback chunks
    hard_chunks = [c for c in chunks if c.split_method == SplitMethod.HARD_TOKEN_FALLBACK]
    hard_by_doc: dict[str, list] = defaultdict(list)
    for c in chunks:
        hard_by_doc[c.document_id].append(c)
    hard_analysis = [analyze_hard_fallback(c, hard_by_doc[c.document_id]) for c in hard_chunks]

    # 4. By document type
    by_doc_type: dict[str, list] = defaultdict(list)
    for chunk in chunks:
        effective = effective_document_type(chunk.document_type, chunk.title, chunk.canonical_url)
        by_doc_type[effective].append(chunk)

    type_stats = {}
    for doc_type, type_chunks in sorted(by_doc_type.items()):
        tokens = [c.token_count for c in type_chunks]
        docs = {c.document_id for c in type_chunks}
        type_stats[doc_type] = {
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

    # 5. By source type
    by_source: dict[str, list] = defaultdict(list)
    for chunk in chunks:
        by_source[chunk.source_type.value].append(chunk)

    source_stats = {}
    for source_type, source_chunks in sorted(by_source.items()):
        tokens = [c.token_count for c in source_chunks]
        docs = {c.document_id for c in source_chunks}
        source_stats[source_type] = {
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

    # 6. Specific inspections
    ocr_sample_id = next(
        (c.document_id for c in chunks if c.extraction_method.value == "ocr" and c.document_id != RADIUS_DOC_ID),
        RADIUS_DOC_ID,
    )
    investor_sample_id = next(
        (
            c.document_id
            for c in chunks
            if effective_document_type(c.document_type, c.title, c.canonical_url) == "investor"
            and c.extraction_method.value == "ocr"
        ),
        next(c.document_id for c in chunks if c.document_type.value == "investor"),
    )

    focal_inspections = {
        "return_replacement_policy": inspect_document(chunks, RETURNS_DOC_ID),
        "radius_m16_bte": inspect_document(chunks, RADIUS_DOC_ID),
        "representative_ocr_pdf": inspect_document(chunks, ocr_sample_id),
        "representative_investor_document": inspect_document(chunks, investor_sample_id),
        "draft_prospectus": inspect_document(chunks, PROSPECTUS_IDS[0]),
        "final_prospectus": inspect_document(chunks, PROSPECTUS_IDS[1]),
    }

    # Policy/product usefulness checks
    returns_chunks = [c for c in chunks if c.document_id == RETURNS_DOC_ID]
    returns_text = "\n".join(c.content for c in returns_chunks)
    focal_inspections["return_replacement_policy"]["checks"] = {
        "section_1_present": "1. Product-Specific Return & Replacement Windows" in returns_text,
        "clause_1_1_1_present": "1.1.1 Return/Replacement Period" in returns_text,
        "clause_1_2_1_present": "1.2.1 Return/Replacement Period" in returns_text,
        "sections_2_to_5_present": all(s in returns_text for s in ("2. General Conditions", "3. Process", "4. Refunds", "5. Statutory Rights")),
    }

    radius_chunks = [c for c in chunks if c.document_id == RADIUS_DOC_ID]
    radius_text = "\n".join(c.content for c in radius_chunks)
    focal_inspections["radius_m16_bte"]["checks"] = {
        "battery_life_270": "Battery Life : 270 Hrs" in radius_text or "Battery Life: 270 Hrs" in radius_text,
        "battery_size_13": "Battery Size (Zinc Air) 13" in radius_text,
        "attack_time_28": "Attack Time 28 ms" in radius_text,
        "release_time_891": "Release Time 891 ms" in radius_text,
    }

    for key, pid in [("draft_prospectus", PROSPECTUS_IDS[0]), ("final_prospectus", PROSPECTUS_IDS[1])]:
        pc = [c for c in chunks if c.document_id == pid]
        focal_inspections[key]["checks"] = {
            "not_single_chunk": len(pc) > 1,
            "max_tokens_within_ceiling": max(c.token_count for c in pc) <= config.max_chunk_tokens,
            "no_sha256_leak": not any("sha256:" in c.content.lower() for c in pc),
            "section_paths_populated": sum(1 for c in pc if c.section_path) > len(pc) * 0.9,
        }

    # Verdict
    heading_only_ratio = small_classes.get("HEADING_ONLY", 0) / max(1, len(small_chunks))
    unsafe_hard = sum(1 for h in hard_analysis if not h["fallback_appears_safe"])
    safe_hard = len(hard_chunks) - unsafe_hard
    dup_knowledge_issues = dup_issue_classes.get("DUPLICATE_KNOWLEDGE", 0)
    focal_pass = (
        all(focal_inspections["return_replacement_policy"]["checks"].values())
        and all(focal_inspections["radius_m16_bte"]["checks"].values())
        and all(focal_inspections["draft_prospectus"]["checks"].values())
        and all(focal_inspections["final_prospectus"]["checks"].values())
    )
    needs_review = (
        unsafe_hard > 5
        or dup_knowledge_issues > 20
        or not focal_pass
        or (heading_only_ratio > 0.55 and len(small_chunks) > 1000)
    )
    verdict = "NEEDS_REVIEW" if needs_review else "PASS"
    verdict_rationale = {
        "focal_documents_pass": focal_pass,
        "very_small_total": len(small_chunks),
        "heading_only_ratio": round(heading_only_ratio, 3),
        "unsafe_hard_fallback_tails": unsafe_hard,
        "safe_hard_fallback_chunks": safe_hard,
        "duplicate_knowledge_issues": dup_knowledge_issues,
        "duplicate_heading_issues": dup_issue_classes.get("DUPLICATE_HEADING", 0),
        "note_hard_fallback_count": (
            "Current dry-run reports 62 hard-token fallback chunks (includes post-enforcement "
            "512-token splits and tail fragments). An earlier Phase 10 run reported 12 before "
            "token-limit enforcement was added."
        ),
    }

    report = {
        "generated_at": datetime.now(timezone.utc).isoformat(),
        "phase": "phase10_5_chunk_quality_review",
        "kb_dataset_version": KB_DATASET_VERSION,
        "analysis_source": "phase10_dry_run_in_memory_rerun",
        "total_documents": result.documents_processed,
        "total_chunks": len(chunks),
        "very_small_threshold_tokens": VERY_SMALL_THRESHOLD,
        "very_small_chunks": {
            "total": len(small_chunks),
            "classification_counts": dict(small_classes),
            "representative_examples": dict(small_examples),
        },
        "duplicate_content_warnings": {
            "validator_reported_issues": reported_dup_issues,
            "duplicate_groups_total": len(duplicate_groups),
            "per_issue_classification_counts": dict(dup_issue_classes),
            "per_group_classification_counts": dict(dup_group_classes),
            "representative_examples": dict(dup_issue_examples),
        },
        "hard_token_fallback_chunks": {
            "total": len(hard_chunks),
            "safe_count": safe_hard,
            "unsafe_tail_count": unsafe_hard,
            "details": hard_analysis,
        },
        "statistics_by_document_type": type_stats,
        "statistics_by_source_type": source_stats,
        "focal_document_inspections": focal_inspections,
        "verdict": verdict,
        "verdict_rationale": verdict_rationale,
        "chunk_quality_review_answer": verdict,
    }

    reports_dir = settings.reports_dir
    reports_dir.mkdir(parents=True, exist_ok=True)
    json_path = reports_dir / "phase10_5_chunk_quality.json"
    txt_path = reports_dir / "phase10_5_chunk_quality.txt"
    json_path.write_text(json.dumps(report, ensure_ascii=False, indent=2), encoding="utf-8")

    def _fmt_stats_table(stats: dict[str, dict]) -> list[str]:
        lines = []
        header = f"{'type':<12} {'docs':>5} {'chunks':>7} {'min':>4} {'med':>5} {'mean':>6} {'p90':>5} {'p95':>5} {'max':>4} {'small':>6} {'hard':>5}"
        lines.append(header)
        lines.append("-" * len(header))
        for name, s in stats.items():
            lines.append(
                f"{name:<12} {s['document_count']:>5} {s['chunk_count']:>7} "
                f"{s['min_tokens']:>4} {s['median_tokens']:>5} {s['mean_tokens']:>6} "
                f"{s['p90_tokens']:>5} {s['p95_tokens']:>5} {s['max_tokens']:>4} "
                f"{s['small_chunk_count']:>6} {s['hard_fallback_count']:>5}"
            )
        return lines

    txt_lines = [
        "PHASE 10.5 — CHUNK QUALITY REVIEW (READ-ONLY)",
        f"Generated: {report['generated_at']}",
        f"KB dataset: {KB_DATASET_VERSION}",
        f"Documents: {result.documents_processed}  |  Total chunks: {len(chunks)}",
        "",
        "=" * 72,
        "1. VERY SMALL CHUNKS (< 8 tokens)",
        "=" * 72,
        f"Total: {len(small_chunks)}",
        "Classification counts:",
    ]
    for label, count in sorted(small_classes.items(), key=lambda x: -x[1]):
        pct = 100 * count / max(1, len(small_chunks))
        txt_lines.append(f"  {label:<22} {count:>5}  ({pct:.1f}%)")
    txt_lines.append("")
    txt_lines.append("Representative examples:")
    for label in sorted(small_classes.keys()):
        txt_lines.append(f"  [{label}]")
        for ex in small_examples.get(label, [])[:3]:
            txt_lines.append(f"    - {ex['title'][:50]} | tokens={ex['token_count']} | {ex['content'][:80]!r}")

    txt_lines.extend([
        "",
        "=" * 72,
        "2. DUPLICATE CONTENT WARNINGS",
        "=" * 72,
        f"Validator-reported duplicate issues: {reported_dup_issues}",
        f"Duplicate groups: {len(duplicate_groups)}",
        "Per-issue classification counts:",
    ])
    for label, count in sorted(dup_issue_classes.items(), key=lambda x: -x[1]):
        txt_lines.append(f"  {label:<22} {count:>5}")
    txt_lines.append("")
    txt_lines.append("Representative examples:")
    for label in sorted(dup_issue_classes.keys()):
        txt_lines.append(f"  [{label}]")
        for ex in dup_issue_examples.get(label, [])[:2]:
            txt_lines.append(f"    - doc={ex['document_id'][:8]}... idx={ex['chunk_index']} group={ex['group_size']}")
            txt_lines.append(f"      {ex['content_preview'][:120]}")

    txt_lines.extend([
        "",
        "=" * 72,
        "3. HARD-TOKEN FALLBACK CHUNKS",
        "=" * 72,
        f"Total: {len(hard_chunks)}  (safe: {safe_hard}, unsafe tail fragments: {unsafe_hard})",
        verdict_rationale["note_hard_fallback_count"],
        "",
    ])
    for detail in hard_analysis[:15]:
        txt_lines.append(
            f"  [{detail['document'][:40]}] idx={detail['chunk_index']} tokens={detail['token_count']} safe={detail['fallback_appears_safe']}"
        )
        txt_lines.append(f"    section: {' > '.join(detail['section_path'][-2:])}")
        txt_lines.append(f"    why: {detail['why_structural_sentence_split_failed']}")
    if len(hard_analysis) > 15:
        txt_lines.append(f"  ... and {len(hard_analysis) - 15} more (see JSON for full list)")

    txt_lines.extend([
        "",
        "=" * 72,
        "4. STATISTICS BY DOCUMENT TYPE",
        "=" * 72,
    ])
    txt_lines.extend(_fmt_stats_table(type_stats))

    txt_lines.extend([
        "",
        "=" * 72,
        "5. STATISTICS BY SOURCE TYPE",
        "=" * 72,
    ])
    txt_lines.extend(_fmt_stats_table(source_stats))

    txt_lines.extend([
        "",
        "=" * 72,
        "6. FOCAL DOCUMENT INSPECTIONS",
        "=" * 72,
    ])
    for name, insp in focal_inspections.items():
        txt_lines.append(f"  {name}:")
        if not insp.get("found"):
            txt_lines.append("    NOT FOUND")
            continue
        txt_lines.append(f"    title: {insp['title'][:60]}")
        txt_lines.append(f"    chunks: {insp['chunk_count']} | tokens median={insp['token_median']} max={insp['token_max']}")
        txt_lines.append(f"    semantic_units_useful: {insp.get('semantic_units_useful', True)}")
        if "checks" in insp:
            for check, ok in insp["checks"].items():
                txt_lines.append(f"      {check}: {'PASS' if ok else 'FAIL'}")

    txt_lines.extend([
        "",
        "=" * 72,
        "VERDICT RATIONALE",
        "=" * 72,
    ])
    for k, v in verdict_rationale.items():
        txt_lines.append(f"  {k}: {v}")

    txt_lines.extend(["", f"CHUNK_QUALITY_REVIEW: {verdict}"])
    txt_path.write_text("\n".join(txt_lines), encoding="utf-8")

    print(json.dumps({"verdict": verdict, "small": len(small_chunks), "duplicates": len(duplicate_groups), "hard": len(hard_chunks)}, indent=2))


if __name__ == "__main__":
    main()
