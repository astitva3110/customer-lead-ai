"""Run full KB processing and quality audit (Phase 3)."""

import argparse
import sys
from pathlib import Path

from dotenv import load_dotenv

ROOT = Path(__file__).resolve().parents[1]
sys.path.insert(0, str(ROOT))
load_dotenv(ROOT / ".env")


def main() -> int:
    parser = argparse.ArgumentParser(description="Process RAW KB documents and generate quality audit reports")
    parser.add_argument("--website", help="Limit import/process/audit to one website")
    parser.add_argument("--import-legacy", action="store_true", help="Import data/knowledge/ into RAW first")
    parser.add_argument("--legacy-dir", type=Path, help="Legacy knowledge directory (default: data/knowledge)")
    parser.add_argument("--skip-process", action="store_true", help="Skip RAW processing; audit existing data only")
    parser.add_argument("--reports-dir", type=Path, help="Output directory for audit reports (default: reports/)")
    args = parser.parse_args()

    from app.kb.audit.runner import run_kb_audit

    result = run_kb_audit(
        import_legacy=args.import_legacy,
        process=not args.skip_process,
        website=args.website,
        legacy_dir=args.legacy_dir,
        reports_dir=args.reports_dir,
    )

    summary = result["summary"]
    paths = result["report_paths"]

    print(f"Imported {result['imported']} legacy document(s) into RAW")
    print(f"Processed {result['processed']} RAW document(s)")
    print(f"Canonical documents audited: {summary.canonical_count}")
    print(f"Reports written to {paths['summary_json'].parent}")
    print(f"  {paths['summary_json'].name}")
    print(f"  {paths['documents_json'].name}")
    print(f"  {paths['duplicates_json'].name}")
    print(f"  {paths['review_json'].name}")
    print(f"  {paths['summary_txt'].name}")
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
