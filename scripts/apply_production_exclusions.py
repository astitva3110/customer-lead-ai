"""Apply production exclusions, reprocess pipeline, and verify KB state."""

from __future__ import annotations

import json
import sys
from collections import Counter
from pathlib import Path

ROOT = Path(__file__).resolve().parents[1]
sys.path.insert(0, str(ROOT))

from dotenv import load_dotenv

load_dotenv(ROOT / ".env")

PRODUCTION_EXCLUDED_URLS = {
    "https://earkart.in/investor/notices/FE-Delhi-April-02--2026-Earkart.pdf",
    "https://earkart.in/investor/bp/WhistleBlower-Policy.pdf",
    "https://earkart.in/investor/gm/FE-Delhi-June-25-2026.pdf",
    "https://earkart.in/investor/gm/JS-Delhi-25-June-2026.pdf",
    "https://earkart.in/investor/bp/Code-of-Conduct-Policy.pdf",
    "https://earkart.in/ZB.pdf",
}
TARGET_SITES = ("earkart.in", "earkart.com")


def _load_canonical_statuses(canonical_dir: Path) -> dict[str, str]:
    statuses: dict[str, str] = {}
    for website in TARGET_SITES:
        site_dir = canonical_dir / website
        if not site_dir.exists():
            continue
        for path in site_dir.glob("*/current.json"):
            doc = json.loads(path.read_text(encoding="utf-8"))
            url = doc.get("canonical_url") or doc.get("source_url", "")
            statuses[url] = doc.get("processing_status", "")
    return statuses


def main() -> int:
    from app.config import settings
    from app.kb.audit.runner import KnowledgeBaseAuditor
    from app.kb.enums import ExclusionReason, ProcessingStatus
    from app.kb.integrity.contract import downstream_eligible
    from app.kb.models.canonical import CanonicalDocument
    from app.kb.policy.domain_policy import PRODUCTION_EXCLUDED_CANONICAL_URLS
    from app.kb.services.kb_service import KnowledgeBaseService
    from app.kb.storage.canonical_store import CanonicalStore

    kb = KnowledgeBaseService()
    canonical_store = CanonicalStore(settings.canonical_dir)

    before_statuses = _load_canonical_statuses(settings.canonical_dir)
    before_active_validated = {
        url: status
        for url, status in before_statuses.items()
        if status in (ProcessingStatus.ACTIVE.value, ProcessingStatus.VALIDATED.value)
    }

    processed = 0
    for path in sorted(kb.raw_store.base_dir.rglob("sha256_*.json")):
        artifact = kb.raw_store.read_path(path)
        if artifact.website not in TARGET_SITES:
            continue
        kb.process_raw_artifact(artifact)
        processed += 1

    after_docs: list[dict] = []
    status_counts: Counter[str] = Counter()
    excluded_list: list[dict] = []
    production_kb_count = 0
    raw_count = sum(1 for _ in kb.raw_store.base_dir.rglob("sha256_*.json"))

    for website in TARGET_SITES:
        site_dir = settings.canonical_dir / website
        if not site_dir.exists():
            continue
        for path in site_dir.glob("*/current.json"):
            data = json.loads(path.read_text(encoding="utf-8"))
            after_docs.append(data)
            status = data.get("processing_status", "")
            status_counts[status] += 1
            url = data.get("canonical_url") or data.get("source_url", "")
            if status in (ProcessingStatus.ACTIVE.value, ProcessingStatus.VALIDATED.value):
                production_kb_count += 1
            if status == ProcessingStatus.EXCLUDED.value:
                excluded_list.append(
                    {
                        "url": url,
                        "website": data.get("website"),
                        "exclusion_reason": data.get("metadata", {}).get("exclusion_reason"),
                        "title": data.get("title"),
                    }
                )

    # Verification
    errors: list[str] = []
    for url in PRODUCTION_EXCLUDED_URLS:
        doc = next((d for d in after_docs if (d.get("canonical_url") or d.get("source_url")) == url), None)
        if doc is None:
            errors.append(f"missing canonical for excluded URL: {url}")
            continue
        if doc.get("processing_status") != ProcessingStatus.EXCLUDED.value:
            errors.append(f"{url} not EXCLUDED (status={doc.get('processing_status')})")
        reason = doc.get("metadata", {}).get("exclusion_reason")
        if reason != ExclusionReason.NOT_REQUIRED_FOR_PRODUCTION_KB.value:
            errors.append(f"{url} wrong exclusion_reason: {reason}")
        canonical = CanonicalDocument.model_validate(doc)
        if downstream_eligible(canonical):
            errors.append(f"{url} is downstream eligible but should not be")

    for url in PRODUCTION_EXCLUDED_URLS:
        latest = kb.raw_store.get_latest_for_url("earkart.in", url)
        if latest is None:
            errors.append(f"RAW artifact missing for {url}")

    for url, prev_status in before_active_validated.items():
        if url in PRODUCTION_EXCLUDED_URLS:
            continue
        new_status = next(
            (d.get("processing_status") for d in after_docs if (d.get("canonical_url") or d.get("source_url")) == url),
            None,
        )
        if new_status == ProcessingStatus.FAILED.value:
            errors.append(f"regression: {url} was {prev_status}, now FAILED")

    unexpected_new_excluded = [
        item
        for item in excluded_list
        if item["exclusion_reason"] == ExclusionReason.NOT_REQUIRED_FOR_PRODUCTION_KB.value
        and item["url"] not in PRODUCTION_EXCLUDED_URLS
    ]
    if unexpected_new_excluded:
        errors.append(f"unexpected NOT_REQUIRED exclusions: {unexpected_new_excluded}")

    auditor = KnowledgeBaseAuditor(reports_dir=settings.reports_dir)
    audit_result = auditor.run_audit()

    report = {
        "processed_raw_artifacts": processed,
        "total_raw_artifacts": raw_count,
        "total_canonical_documents": len(after_docs),
        "status_counts": dict(status_counts),
        "production_kb_count": production_kb_count,
        "excluded_documents": sorted(excluded_list, key=lambda item: item["url"]),
        "production_excluded_urls": sorted(PRODUCTION_EXCLUDED_CANONICAL_URLS),
        "verification_errors": errors,
        "audit_summary": audit_result["summary"].to_dict(),
    }
    out_path = settings.reports_dir / "production_exclusion_report.json"
    out_path.write_text(json.dumps(report, ensure_ascii=False, indent=2), encoding="utf-8")

    print(f"Reprocessed {processed} RAW artifacts")
    print(f"Total RAW artifacts: {raw_count}")
    print(f"Total canonical documents: {len(after_docs)}")
    print(f"ACTIVE: {status_counts.get('ACTIVE', 0)}")
    print(f"VALIDATED: {status_counts.get('VALIDATED', 0)}")
    print(f"EXCLUDED: {status_counts.get('EXCLUDED', 0)}")
    print(f"FAILED: {status_counts.get('FAILED', 0)}")
    print(f"Production KB count (ACTIVE+VALIDATED): {production_kb_count}")
    print()
    print("Excluded documents:")
    for item in sorted(excluded_list, key=lambda x: x["url"]):
        print(f"  [{item['exclusion_reason']}] {item['url']}")
    print()
    if errors:
        print("VERIFICATION ERRORS:")
        for err in errors:
            print(f"  - {err}")
        print(f"\nReport: {out_path}")
        return 1

    print("All verification checks passed.")
    print(f"Report: {out_path}")
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
