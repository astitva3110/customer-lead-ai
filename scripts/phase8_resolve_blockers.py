"""Phase 8: resolve Phase 7 blockers and regenerate retrieval artifacts."""

from __future__ import annotations

import hashlib
import json
from datetime import datetime, timezone
from pathlib import Path

from app.kb.integrity.contract import downstream_eligible
from app.kb.integrity.version import KB_DATASET_VERSION
from app.kb.retrieval.low_value_decisions import (
    LowValueDecisionRecord,
    classify_low_value,
    write_low_value_decisions,
)
from app.kb.retrieval.recovery import _count_giant_paragraphs, recover_structure
from app.kb.retrieval.review_decisions import build_review_decisions, write_review_decisions
from app.kb.retrieval.service import RetrievalPreparationService
from app.kb.retrieval.storage import RetrievalStore
from app.kb.storage.canonical_store import CanonicalStore

ROOT = Path(__file__).resolve().parents[1]
REPORTS = ROOT / "reports"
CANONICAL_DIR = ROOT / "data" / "canonical"
RAW_DIR = ROOT / "data" / "raw"
RETRIEVAL_DIR = ROOT / "data" / "retrieval"


def _canonical_fingerprint() -> dict:
    ids: list[str] = []
    hashes: list[str] = []
    for website in ("earkart.in", "earkart.com"):
        site_dir = CANONICAL_DIR / website
        if not site_dir.exists():
            continue
        for path in sorted(site_dir.glob("*/current.json")):
            payload = json.loads(path.read_text(encoding="utf-8"))
            if payload.get("processing_status") in {"ACTIVE", "VALIDATED", "EXCLUDED"}:
                ids.append(payload["document_id"])
                hashes.append(payload["content_hash"])
    return {
        "production_count": len(ids),
        "document_ids_hash": hashlib.sha256("\n".join(sorted(ids)).encode()).hexdigest(),
        "content_hashes_hash": hashlib.sha256("\n".join(sorted(hashes)).encode()).hexdigest(),
    }


def _raw_fingerprint() -> dict:
    raw_files = sorted(RAW_DIR.rglob("sha256_*.json"))
    digests = []
    for path in raw_files:
        digests.append(hashlib.sha256(path.read_bytes()).hexdigest())
    return {
        "artifact_count": len(raw_files),
        "artifacts_hash": hashlib.sha256("\n".join(digests).encode()).hexdigest(),
    }


def _count_giant_blocks(retrieval_store: RetrievalStore) -> int:
    total = 0
    dataset = retrieval_store.dataset_dir(KB_DATASET_VERSION)
    for path in dataset.rglob("*.json"):
        if path.name in {"manifest.json", "eligibility.json"}:
            continue
        doc = json.loads(path.read_text(encoding="utf-8"))
        content = doc.get("structured_content", {})
        if content.get("pages"):
            from app.kb.models.structured_content import DocumentContent

            parsed = DocumentContent.model_validate(content)
            total += sum(_count_giant_paragraphs(page.blocks) for page in parsed.pages or [])
        else:
            from app.kb.models.structured_content import DocumentContent

            parsed = DocumentContent.model_validate(content)
            total += _count_giant_paragraphs(parsed.children)
    return total


def _nav_heavy_count(retrieval_store: RetrievalStore) -> int:
    count = 0
    dataset = retrieval_store.dataset_dir(KB_DATASET_VERSION)
    for path in dataset.rglob("*.json"):
        if path.name in {"manifest.json", "eligibility.json"}:
            continue
        doc = json.loads(path.read_text(encoding="utf-8"))
        if doc.get("eligibility_status") != "eligible":
            continue
        text = doc.get("retrieval_text", "")
        if text.count("](http") >= 10:
            count += 1
    return count


def _orphan_excluded_count(retrieval_store: RetrievalStore, eligibility: dict) -> int:
    excluded_ids = {
        item["document_id"]
        for item in eligibility.get("documents", [])
        if item.get("eligibility_status") == "excluded"
    }
    orphans = 0
    dataset = retrieval_store.dataset_dir(KB_DATASET_VERSION)
    for path in dataset.rglob("*.json"):
        if path.name in {"manifest.json", "eligibility.json"}:
            continue
        if path.stem in excluded_ids:
            orphans += 1
    return orphans


def main() -> None:
    canonical_store = CanonicalStore(CANONICAL_DIR)
    retrieval_store = RetrievalStore(RETRIEVAL_DIR)

    canonical_before = _canonical_fingerprint()
    raw_before = _raw_fingerprint()

    production = RetrievalPreparationService.load_production_canonicals(CANONICAL_DIR)
    review_records = build_review_decisions(production)
    write_review_decisions(REPORTS / "phase8_review_decisions.json", review_records)

    low_value_records: list[LowValueDecisionRecord] = []
    for canonical in production:
        from app.kb.retrieval.eligibility import assess_retrieval_eligibility

        base = assess_retrieval_eligibility(canonical)
        recovered, _ = recover_structure(canonical.structured_content)
        from app.kb.retrieval.renderer import render_retrieval_text

        preview = render_retrieval_text(
            title=canonical.title,
            canonical_url=canonical.canonical_url,
            document_type=base.document_type,
            source_type=canonical.source_type,
            structured_content=recovered,
        )
        record = classify_low_value(canonical, preview)
        if record is not None:
            low_value_records.append(record)
    write_low_value_decisions(REPORTS / "phase8_low_value_decisions.json", low_value_records)

    giant_before = _count_giant_blocks(retrieval_store)
    nav_before = _nav_heavy_count(retrieval_store)

    service = RetrievalPreparationService(canonical_store, retrieval_store)
    report = service.prepare_all_production()
    summary_path = REPORTS / "retrieval_preparation_summary.json"
    summary_path.write_text(json.dumps(report.to_dict(), ensure_ascii=False, indent=2), encoding="utf-8")

    eligibility = json.loads(retrieval_store.eligibility_manifest_path(KB_DATASET_VERSION).read_text(encoding="utf-8"))
    giant_after = _count_giant_blocks(retrieval_store)
    nav_after = _nav_heavy_count(retrieval_store)
    orphans_after = _orphan_excluded_count(retrieval_store, eligibility)

    canonical_after = _canonical_fingerprint()
    raw_after = _raw_fingerprint()

    phase8_summary = {
        "generated_at": datetime.now(timezone.utc).isoformat(),
        "phase": "phase8_resolve_blockers",
        "review_decisions": {
            "total": len(review_records),
            "approve": sum(1 for r in review_records if r.decision.value == "APPROVE"),
            "reject": sum(1 for r in review_records if r.decision.value == "REJECT"),
            "keep_review": sum(1 for r in review_records if r.decision.value == "KEEP_REVIEW"),
        },
        "remaining_review_documents": eligibility.get("review_count"),
        "low_value_decisions": {
            "total": len(low_value_records),
            "useful": sum(1 for r in low_value_records if r.classification.value == "USEFUL"),
            "not_useful": sum(1 for r in low_value_records if r.classification.value == "NOT_USEFUL"),
            "review": sum(1 for r in low_value_records if r.classification.value == "REVIEW"),
        },
        "nav_heavy_eligible_before": nav_before,
        "nav_heavy_eligible_after": nav_after,
        "giant_blocks_before": giant_before,
        "giant_blocks_after": giant_after,
        "orphan_retrieval_files_after": orphans_after,
        "retrieval_counts": {
            "eligible": report.documents_eligible,
            "review": report.documents_review,
            "excluded": report.documents_excluded,
        },
        "canonical_integrity": {
            "before": canonical_before,
            "after": canonical_after,
            "unchanged": canonical_before == canonical_after,
        },
        "raw_integrity": {
            "before": raw_before,
            "after": raw_after,
            "unchanged": raw_before == raw_after,
        },
        "structure_recovery": {
            "giant_paragraphs_before": report.giant_paragraphs_before,
            "giant_paragraphs_after": report.giant_paragraphs_after,
            "reduced": report.giant_paragraphs_before - report.giant_paragraphs_after,
        },
    }

    ready = (
        orphans_after == 0
        and nav_after == 0
        and canonical_before == canonical_after
        and raw_before == raw_after
    )
    phase8_summary["ready_for_chunking"] = ready
    phase8_summary["ready_for_chunking_answer"] = "YES" if ready else "NO"

    (REPORTS / "phase8_summary.json").write_text(
        json.dumps(phase8_summary, ensure_ascii=False, indent=2),
        encoding="utf-8",
    )

    txt_lines = [
        "PHASE 8 — RESOLVE PHASE 7 BLOCKERS",
        f"Generated: {phase8_summary['generated_at']}",
        "",
        f"Review decisions: {phase8_summary['review_decisions']}",
        f"Remaining REVIEW: {phase8_summary['remaining_review_documents']}",
        f"Nav-heavy eligible: {nav_before} → {nav_after}",
        f"Giant blocks: {giant_before} → {giant_after}",
        f"Orphan excluded files: {orphans_after}",
        f"Retrieval eligible/review/excluded: {report.documents_eligible}/{report.documents_review}/{report.documents_excluded}",
        f"Canonical integrity unchanged: {canonical_before == canonical_after}",
        f"RAW integrity unchanged: {raw_before == raw_after}",
        "",
        f"READY_FOR_CHUNKING: {phase8_summary['ready_for_chunking_answer']}",
    ]
    (REPORTS / "phase8_summary.txt").write_text("\n".join(txt_lines), encoding="utf-8")
    print(json.dumps(phase8_summary, indent=2))


if __name__ == "__main__":
    main()
