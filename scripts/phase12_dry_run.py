#!/usr/bin/env python3
"""Phase 12 dry-run over fixture documents — no production embedding."""

from __future__ import annotations

import json
import sys
from pathlib import Path

ROOT = Path(__file__).resolve().parents[1]
sys.path.insert(0, str(ROOT))

FIXTURES = Path(__file__).resolve().parents[1] / "tests" / "fixtures" / "phase12"


def main() -> int:
    from app.config import settings
    from app.kb.enums import DocumentType
    from app.kb.ingestion.reports import build_chunk_quality_report, build_dry_run_report, write_reports
    from app.kb.ingestion.service import DocumentIngestionService
    from tests.fixtures.phase12.generate_fixtures import ensure_fixtures

    ensure_fixtures(FIXTURES)
    service = DocumentIngestionService()
    manifest = json.loads((FIXTURES / "manifest.json").read_text(encoding="utf-8"))

    results = []
    for item in manifest["documents"]:
        path = FIXTURES / item["filename"]
        doc_type = DocumentType(item["document_type"])
        try:
            result = service.ingest_file(
                path,
                document_type=doc_type,
                dry_run=True,
                embed=False,
            )
            results.append(result)
            print(f"OK  {item['id']} {item['filename']} chunks={len(result.chunks)}")
        except Exception as exc:
            print(f"FAIL {item['id']} {item['filename']}: {exc}")
            raise

    dry_run = build_dry_run_report(results)
    chunk_quality = build_chunk_quality_report(results)
    paths = write_reports(settings.reports_dir, dry_run=dry_run, chunk_quality=chunk_quality)
    for path in paths:
        print(f"Wrote {path}")

    print("\nPHASE12_DRY_RUN: PASS")
    print("PRODUCTION_EMBEDDING: NOT_RUN")
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
