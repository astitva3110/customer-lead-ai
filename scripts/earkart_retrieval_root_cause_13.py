"""Phase 11.7 — lightweight read-only Earkart retrieval root-cause audit."""

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
    parser = argparse.ArgumentParser(
        description="Fast vector-only root-cause audit for 13 Earkart benchmark questions.",
    )
    parser.add_argument("--top-k", type=int, default=100)
    parser.add_argument(
        "--device",
        choices=("cuda", "cpu"),
        default="cuda",
        help="Query embedding device only (default: cuda)",
    )
    return parser.parse_args()


def main() -> int:
    args = _parse_args()
    os.environ["EMBEDDING_DEVICE"] = args.device
    os.environ.setdefault("EMBEDDING_BATCH_SIZE", "8")

    from app.config import settings
    from app.kb.embedding.config import EmbeddingConfig
    from app.kb.embedding.loader import ChunkLoader
    from app.kb.evaluation.retrieval_diagnostic.lightweight_audit import (
        format_lightweight_text_report,
        run_lightweight_audit,
    )
    from app.kb.vector.search import VectorSearchService
    from app.kb.vector.store import VectorStore

    reports_dir = settings.reports_dir
    reports_dir.mkdir(parents=True, exist_ok=True)

    config = EmbeddingConfig.from_settings()
    print(f"Lightweight audit | device={config.device} | top_k={args.top_k}", flush=True)

    loader = ChunkLoader(settings.chunks_dir, config.embedding_input_manifest)
    chunks = loader.load_all().chunks
    store = VectorStore(config.database_url, config.vector_table, config.dimension)
    search = VectorSearchService(config=config, store=store)

    report = run_lightweight_audit(config=config, chunks=chunks, search=search, top_k=args.top_k)

    json_path = reports_dir / "earkart_retrieval_root_cause_13.json"
    txt_path = reports_dir / "earkart_retrieval_root_cause_13.txt"

    json_payload = {key: value for key, value in report.items() if not key.startswith("_")}
    json_path.write_text(json.dumps(json_payload, ensure_ascii=False, indent=2), encoding="utf-8")
    text = format_lightweight_text_report(
        json_payload,
        full_questions=report.get("_internal_questions"),
    )
    txt_path.write_text(text, encoding="utf-8")

    summary_block = text.split("EAR_KART_RETRIEVAL_ROOT_CAUSE_13", 1)[1]
    print("EAR_KART_RETRIEVAL_ROOT_CAUSE_13" + summary_block.encode("ascii", "replace").decode("ascii"))
    print(f"Wrote {json_path}", flush=True)
    print(f"Wrote {txt_path}", flush=True)
    return 0


if __name__ == "__main__":
    sys.exit(main())
