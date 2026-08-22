"""Process RAW artifacts through cleaning pipeline (manual/CLI)."""

import argparse
import sys
from pathlib import Path

from dotenv import load_dotenv

ROOT = Path(__file__).resolve().parents[1]
sys.path.insert(0, str(ROOT))
load_dotenv(ROOT / ".env")


def main() -> int:
    parser = argparse.ArgumentParser(description="Process RAW KB documents through cleaning pipeline")
    parser.add_argument("--website", help="Process only this website")
    parser.add_argument("--raw-path", type=Path, help="Process a single RAW JSON file")
    parser.add_argument(
        "--import-legacy",
        action="store_true",
        help="Import data/knowledge/ into RAW before processing (does not modify legacy data)",
    )
    parser.add_argument("--legacy-dir", type=Path, help="Legacy knowledge directory (default: data/knowledge)")
    args = parser.parse_args()

    from app.config import settings
    from app.kb.services.kb_service import KnowledgeBaseService
    from app.kb.services.legacy_importer import import_legacy_directory
    from app.kb.storage.raw_store import RawStore

    kb = KnowledgeBaseService()
    raw_store = RawStore(settings.raw_dir)

    if args.import_legacy:
        legacy_dir = args.legacy_dir or settings.knowledge_dir
        imported = import_legacy_directory(legacy_dir, raw_store, website=args.website, dry_run=False)
        print(f"Imported {len(imported)} legacy document(s) into RAW")

    paths: list[Path] = []
    if args.raw_path:
        paths = [args.raw_path]
    else:
        for json_path in sorted(settings.raw_dir.rglob("sha256_*.json")):
            paths.append(json_path)

    processed = 0
    for path in paths:
        artifact = raw_store.read_path(path)
        if args.website and artifact.website != args.website:
            continue
        kb.process_raw_artifact(artifact)
        processed += 1
        print(f"Processed {artifact.canonical_url}")

    print(f"Done. Processed {processed} document(s).")
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
