"""Phase 11.8 — targeted chunk inspection for four CHUNKING candidates."""

from __future__ import annotations

import argparse
import json
import os
import sys
from pathlib import Path

ROOT = Path(__file__).resolve().parents[1]
if str(ROOT) not in sys.path:
    sys.path.insert(0, str(ROOT))


def _parse_args() -> argparse.Namespace:
    parser = argparse.ArgumentParser(description="Read-only targeted chunk inspection (4 queries, top-20).")
    parser.add_argument("--device", choices=("cuda", "cpu"), default="cuda")
    return parser.parse_args()


def main() -> int:
    args = _parse_args()
    os.environ["EMBEDDING_DEVICE"] = args.device
    os.environ.setdefault("EMBEDDING_BATCH_SIZE", "8")

    from app.config import settings
    from app.kb.embedding.config import EmbeddingConfig
    from app.kb.evaluation.retrieval_diagnostic.targeted_inspection import (
        format_targeted_text_report,
        run_targeted_inspection,
    )
    from app.kb.vector.search import VectorSearchService
    from app.kb.vector.store import VectorStore

    config = EmbeddingConfig.from_settings()
    store = VectorStore(config.database_url, config.vector_table, config.dimension)
    search = VectorSearchService(config=config, store=store)

    print(f"Targeted inspection | device={config.device} | 4 queries | top-20", flush=True)
    report = run_targeted_inspection(
        config=config,
        search=search,
        reports_dir=settings.reports_dir,
    )

    reports_dir = settings.reports_dir
    reports_dir.mkdir(parents=True, exist_ok=True)
    json_path = reports_dir / "earkart_targeted_chunk_inspection.json"
    txt_path = reports_dir / "earkart_targeted_chunk_inspection.txt"

    json_path.write_text(json.dumps(report, ensure_ascii=False, indent=2), encoding="utf-8")
    text = format_targeted_text_report(report)
    txt_path.write_text(text, encoding="utf-8")

    summary = text.split("EAR_KART_TARGETED_CHUNK_INSPECTION", 1)[1]
    print("EAR_KART_TARGETED_CHUNK_INSPECTION" + summary.encode("ascii", "replace").decode("ascii"))
    print(f"Wrote {json_path}", flush=True)
    print(f"Wrote {txt_path}", flush=True)
    return 0


if __name__ == "__main__":
    sys.exit(main())
