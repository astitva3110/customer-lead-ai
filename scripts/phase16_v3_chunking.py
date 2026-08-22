#!/usr/bin/env python3
"""Phase 16 — KB V3 chunking experiment (no embed, no PGVector writes)."""

from __future__ import annotations

import argparse
import json
import logging
import os
import sys
from pathlib import Path

ROOT = Path(__file__).resolve().parents[1]
sys.path.insert(0, str(ROOT))


def main() -> int:
    parser = argparse.ArgumentParser(description="Phase 16 KB V3 chunking experiment")
    parser.add_argument("--device", default="cuda", choices=["cuda", "cpu"])
    parser.add_argument("--skip-inspection", action="store_true")
    args = parser.parse_args()

    os.environ["EMBEDDING_DEVICE"] = args.device
    logging.basicConfig(level=logging.INFO, format="%(asctime)s %(levelname)s %(name)s: %(message)s")

    from app.config import settings
    from app.kb.evaluation.phase16_v3_inspection import run_phase16_inspection
    from app.kb.ingestion.indexing import Phase12VectorStore
    from app.kb.ingestion.phase16_corpus import load_phase16_v3_corpus
    from app.kb.ingestion.phase16_quality import build_phase16_quality_report, format_phase16_quality_text
    from app.kb.vector.store import VectorStore

    v1_store = VectorStore(
        settings.database_url,
        settings.vector_table,
        settings.embedding_dimension,
    )
    v2_store = Phase12VectorStore(table_name=settings.phase12_vector_table)
    v1_before = v1_store.count()
    v2_before = v2_store.count(embedding_version=settings.phase12_embedding_version)

    corpus = load_phase16_v3_corpus()
    quality = build_phase16_quality_report(corpus)
    inspection = None if args.skip_inspection else run_phase16_inspection(device=args.device)

    v1_after = v1_store.count()
    v2_after = v2_store.count(embedding_version=settings.phase12_embedding_version)

    report = {
        **quality,
        "integrity": {
            "read_only": True,
            "v1_vector_count_before": v1_before,
            "v1_vector_count_after": v1_after,
            "v2_vector_count_before": v2_before,
            "v2_vector_count_after": v2_after,
            "old_5904_vectors_modified": v1_before != v1_after,
            "v2_vectors_modified": v2_before != v2_after,
            "pgvector_writes": 0,
        },
        "retrieval_inspection": inspection,
    }

    reports_dir = settings.reports_dir
    reports_dir.mkdir(parents=True, exist_ok=True)
    json_path = reports_dir / "phase16_v3_chunk_quality.json"
    txt_path = reports_dir / "phase16_v3_chunk_quality.txt"
    json_path.write_text(json.dumps(report, indent=2), encoding="utf-8")

    txt_lines = [format_phase16_quality_text(quality)]
    if inspection:
        txt_lines.extend(["", "In-memory retrieval inspection (V2 vs V3):", ""])
        for item in inspection["queries"]:
            txt_lines.extend(
                [
                    f"{item['id']}: {item['query']}",
                    f"  V2 rank={item['v2_expected_rank']} sim={item['v2_expected_similarity']}",
                    f"  V3 rank={item['v3_expected_rank']} sim={item['v3_expected_similarity']}",
                    f"  improved={item['improved']} delta={item['rank_delta_v2_to_v3']}",
                    "",
                ]
            )
        txt_lines.extend(
            [
                "Focus queries (BTE, Contact, OMNI, Why buy):",
                "",
            ]
        )
        for item in inspection["focus_comparison"]:
            txt_lines.append(
                f"  {item['id']}: V2 rank {item['v2_expected_rank']} -> V3 rank {item['v3_expected_rank']}"
            )

    txt_path.write_text("\n".join(txt_lines), encoding="utf-8")
    print(txt_path.read_text(encoding="utf-8"))
    print(f"\nWrote {json_path}")
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
