"""Phase 10 document-aware semantic chunking dry run."""

from __future__ import annotations

import json
from pathlib import Path

from dotenv import load_dotenv

from app.config import settings
from app.kb.chunking.config import ChunkingConfig
from app.kb.chunking.reports import run_and_write_reports
from app.kb.chunking.service import ChunkingDryRunService
from app.kb.integrity.version import KB_DATASET_VERSION
from app.kb.retrieval.storage import RetrievalStore


def main() -> None:
    load_dotenv()
    config = ChunkingConfig()
    service = ChunkingDryRunService(
        RetrievalStore(settings.retrieval_dir),
        kb_dataset_version=KB_DATASET_VERSION,
        config=config,
    )
    summary = run_and_write_reports(service, config, settings.reports_dir)
    print(json.dumps(summary, indent=2))


if __name__ == "__main__":
    main()
