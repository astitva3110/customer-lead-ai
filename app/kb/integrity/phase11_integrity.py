"""Integrity verification for frozen KB assets."""

from __future__ import annotations

import hashlib
import json
from pathlib import Path


def directory_sha256(root: Path) -> tuple[int, str]:
    if not root.exists():
        raise FileNotFoundError(f"Directory not found: {root}")
    digest = hashlib.sha256()
    files = sorted(root.rglob("*"))
    file_count = 0
    for path in files:
        if path.is_file():
            file_count += 1
            digest.update(str(path.relative_to(root)).encode("utf-8"))
            digest.update(path.read_bytes())
    return file_count, digest.hexdigest()


def verify_directory_unchanged(root: Path, baseline_path: Path) -> dict:
    baseline = json.loads(baseline_path.read_text(encoding="utf-8"))
    file_count, sha = directory_sha256(root)
    expected_sha = baseline.get("chunks_sha256")
    expected_count = baseline.get("chunks_file_count")
    unchanged = sha == expected_sha and file_count == expected_count
    return {
        "path": str(root),
        "baseline_path": str(baseline_path),
        "file_count": file_count,
        "sha256": sha,
        "expected_file_count": expected_count,
        "expected_sha256": expected_sha,
        "unchanged": unchanged,
    }
