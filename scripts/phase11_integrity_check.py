"""Phase 11 — verify frozen KB integrity after embedding work."""

from __future__ import annotations

import json
import sys
from pathlib import Path

from app.config import settings
from app.kb.integrity.phase11_integrity import verify_directory_unchanged


def main() -> int:
    baseline = Path("data/integrity/phase11_baseline.json")
    chunks_root = settings.chunks_dir / settings.embedding_input_manifest
    result = verify_directory_unchanged(chunks_root, baseline)
    print(json.dumps(result, indent=2))
    return 0 if result["unchanged"] else 1


if __name__ == "__main__":
    sys.exit(main())
