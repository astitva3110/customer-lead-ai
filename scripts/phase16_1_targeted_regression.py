#!/usr/bin/env python3
"""Phase 16.1 — targeted V3 chunk regression audit (read-only)."""

from __future__ import annotations

import argparse
import json
import os
import sys
from pathlib import Path

ROOT = Path(__file__).resolve().parents[1]
sys.path.insert(0, str(ROOT))


def main() -> int:
    parser = argparse.ArgumentParser(description="Phase 16.1 targeted regression audit")
    parser.add_argument("--device", default="cuda", choices=["cuda", "cpu"])
    args = parser.parse_args()

    os.environ["EMBEDDING_DEVICE"] = args.device

    from app.config import settings
    from app.kb.evaluation.phase16_1_regression_audit import (
        format_phase16_1_text,
        run_phase16_1_regression_audit,
    )
    from app.kb.ingestion.indexing import Phase12VectorStore
    from app.kb.vector.store import VectorStore

    v1_store = VectorStore(
        settings.database_url,
        settings.vector_table,
        settings.embedding_dimension,
    )
    v2_store = Phase12VectorStore(table_name=settings.phase12_vector_table)
    v1_before = v1_store.count()
    v2_before = v2_store.count(embedding_version=settings.phase12_embedding_version)

    report = run_phase16_1_regression_audit(device=args.device)

    v1_after = v1_store.count()
    v2_after = v2_store.count(embedding_version=settings.phase12_embedding_version)
    report["integrity"] = {
        "read_only": True,
        "v1_unchanged": v1_before == v1_after,
        "v2_unchanged": v2_before == v2_after,
        "pgvector_writes": 0,
    }

    reports_dir = settings.reports_dir
    reports_dir.mkdir(parents=True, exist_ok=True)
    json_path = reports_dir / "phase16_1_targeted_regression.json"
    txt_path = reports_dir / "phase16_1_targeted_regression.txt"
    json_path.write_text(json.dumps(report, indent=2), encoding="utf-8")
    txt_path.write_text(format_phase16_1_text(report), encoding="utf-8")

    print(txt_path.read_text(encoding="utf-8"))
    print(f"\nWrote {json_path}")
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
