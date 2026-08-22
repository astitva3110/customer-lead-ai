"""Import legacy data/knowledge/ JSON into immutable RAW storage (manual, opt-in)."""

import argparse
import sys
from pathlib import Path

from dotenv import load_dotenv

ROOT = Path(__file__).resolve().parents[1]
sys.path.insert(0, str(ROOT))
load_dotenv(ROOT / ".env")


def main() -> int:
    parser = argparse.ArgumentParser(
        description="Import legacy data/knowledge/ JSON into data/raw/ (NOT run automatically)"
    )
    parser.add_argument("--legacy-dir", type=Path, default=ROOT / "data" / "knowledge")
    parser.add_argument("--website", help="Import only this website (e.g. earkart.in)")
    parser.add_argument("--dry-run", action="store_true", help="Plan import without writing files")
    args = parser.parse_args()

    from app.config import settings
    from app.kb.services.legacy_importer import import_legacy_directory, plan_legacy_import
    from app.kb.storage.raw_store import RawStore

    plans = plan_legacy_import(args.legacy_dir)
    if args.website:
        plans = [p for p in plans if p.website == args.website]

    print(f"Found {len(plans)} legacy documents to import")
    if args.dry_run:
        for plan in plans[:10]:
            print(f"  [{plan.source_type}] {plan.canonical_url} <- {plan.source_path.name}")
        if len(plans) > 10:
            print(f"  ... and {len(plans) - 10} more")
        return 0

    store = RawStore(settings.raw_dir)
    imported = import_legacy_directory(args.legacy_dir, store, website=args.website, dry_run=False)
    print(f"Imported {len(imported)} documents into {settings.raw_dir}")
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
