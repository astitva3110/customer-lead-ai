#!/usr/bin/env python3
"""Phase 15 — thin wrapper around unified retrieval evaluation (v2 / PGVector)."""

from __future__ import annotations

import argparse
import sys
from pathlib import Path

ROOT = Path(__file__).resolve().parents[1]
sys.path.insert(0, str(ROOT))


def main() -> int:
    parser = argparse.ArgumentParser(
        description="Phase 15 KB V2 retrieval evaluation (delegates to evaluate_retrieval.py)"
    )
    parser.add_argument("--device", default="cuda", choices=["cuda", "cpu"])
    args, extra = parser.parse_known_args()
    argv = [
        str(ROOT / "scripts" / "evaluate_retrieval.py"),
        "--corpus-version",
        "v2",
        "--mode",
        "pgvector",
        "--vector-table",
        "chunk_embeddings_phase12",
        "--device",
        args.device,
        *extra,
    ]
    import runpy

    sys.argv = argv
    runpy.run_path(str(ROOT / "scripts" / "evaluate_retrieval.py"), run_name="__main__")
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
