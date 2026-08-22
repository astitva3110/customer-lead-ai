"""Re-run KB audit after Phase 4 fixes and compare with Phase 3 baseline."""

import json
import sys
from pathlib import Path

ROOT = Path(__file__).resolve().parents[1]
sys.path.insert(0, str(ROOT))

from dotenv import load_dotenv

load_dotenv(ROOT / ".env")


def main() -> int:
    from app.config import settings
    from app.kb.audit.runner import KnowledgeBaseAuditor
    from app.kb.services.kb_service import KnowledgeBaseService

    kb = KnowledgeBaseService()
    processed = 0
    for path in sorted(kb.raw_store.base_dir.rglob("sha256_*.json")):
        artifact = kb.raw_store.read_path(path)
        kb.process_raw_artifact(artifact)
        processed += 1

    auditor = KnowledgeBaseAuditor(reports_dir=settings.reports_dir)
    result = auditor.run_audit()

    phase3_path = settings.reports_dir / "phase3_baseline.json"
    if not phase3_path.exists():
        phase3_path = settings.reports_dir / "kb_audit_summary.json"
    phase4_path = settings.reports_dir / "phase4_audit_summary.json"
    review_path = settings.reports_dir / "phase4_review.json"

    phase4_summary = result["summary"].to_dict()
    phase4_path.write_text(json.dumps(phase4_summary, ensure_ascii=False, indent=2), encoding="utf-8")

    review_payload = {
        "generated_at": phase4_summary["generated_at"],
        "review_items": [
            {
                "document_id": item.document_id,
                "website": item.website,
                "url": item.url,
                "title": item.title,
                "source_type": item.source_type,
                "issue": item.issue,
                "severity": item.severity,
                "context": item.context,
            }
            for item in result["review_items"]
        ],
    }
    review_path.write_text(json.dumps(review_payload, ensure_ascii=False, indent=2), encoding="utf-8")

    comparison = {"phase4": phase4_summary, "delta": {}}
    if phase3_path.exists():
        phase3 = json.loads(phase3_path.read_text(encoding="utf-8"))
        comparison["phase3"] = phase3
        for key in (
            "failed_count",
            "active_count",
            "validated_count",
            "excluded_count",
            "boilerplate_document_count",
            "high_content_loss_count",
            "ocr_issue_count",
            "human_review_count",
            "review_critical",
            "review_warning",
        ):
            before = phase3.get(key, 0)
            after = phase4_summary.get(key, 0)
            comparison["delta"][key] = {"before": before, "after": after, "change": after - before}

    comparison_path = settings.reports_dir / "phase4_comparison.json"
    comparison_path.write_text(json.dumps(comparison, ensure_ascii=False, indent=2), encoding="utf-8")

    print(f"Reprocessed {processed} RAW artifacts")
    print(f"Phase 4 summary: {phase4_path}")
    print(f"Phase 4 review: {review_path}")
    print(f"Comparison: {comparison_path}")
    if comparison.get("delta"):
        print("Delta:", comparison["delta"])
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
