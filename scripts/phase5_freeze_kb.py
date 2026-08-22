"""Phase 5: freeze canonical KB and run integrity checks."""

import sys
from pathlib import Path

from dotenv import load_dotenv

ROOT = Path(__file__).resolve().parents[1]
sys.path.insert(0, str(ROOT))
load_dotenv(ROOT / ".env")


def main() -> int:
    from app.config import settings
    from app.kb.integrity.freeze import CanonicalKbFreezer
    from app.kb.storage.canonical_store import CanonicalStore
    from app.kb.storage.cleaned_store import CleanedStore
    from app.kb.storage.raw_store import RawStore

    freezer = CanonicalKbFreezer(
        raw_store=RawStore(settings.raw_dir),
        cleaned_store=CleanedStore(settings.cleaned_dir),
        canonical_store=CanonicalStore(settings.canonical_dir),
        reports_dir=settings.reports_dir,
    )
    result = freezer.run()

    print(f"KB dataset version: {result.integrity_report.get('kb_dataset_version')}")
    print(f"Excluded unrecoverable FAILED PDFs: {result.excluded_failed_count}")
    print(f"Failed PDF analysis: reports/phase5_failed_pdf_analysis.json")
    print(f"Manifest: reports/canonical_kb_manifest.json")
    print(f"Integrity: reports/phase5_integrity_report.json")
    print(f"Freeze acceptance: {'PASSED' if result.passed else 'FAILED'}")
    return 0 if result.passed else 1


if __name__ == "__main__":
    raise SystemExit(main())
