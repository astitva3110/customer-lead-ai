"""
Legacy knowledge base importer (placeholder).

Converts documents from data/knowledge/ into immutable RAW artifacts under data/raw/.

NOT executed automatically. Run manually when ready:

    python -m scripts.import_legacy_knowledge [--dry-run] [--website earkart.in]
"""

from __future__ import annotations

import json
from dataclasses import dataclass
from datetime import datetime, timezone
from pathlib import Path

from app.kb.enums import ExtractionMethod, SourceType
from app.kb.hashing import content_hash
from app.kb.models.raw import RawArtifact
from app.kb.storage.raw_store import RawStore
from app.kb.url_normalizer import extract_website, normalize_url


@dataclass
class LegacyImportPlan:
    source_path: Path
    website: str
    source_type: SourceType
    extraction_method: ExtractionMethod
    canonical_url: str


def _infer_legacy_fields(data: dict) -> tuple[SourceType, ExtractionMethod]:
    url = data.get("url", "")
    title = data.get("title", "")
    content_type = (data.get("content_type") or "").lower()

    if content_type == "html" or (url and not url.lower().endswith(".pdf")):
        return SourceType.HTML, ExtractionMethod.HTML_PARSER

    if "[OCR]" in title:
        return SourceType.SCANNED_PDF, ExtractionMethod.OCR

    return SourceType.PDF, ExtractionMethod.PDF_TEXT


def plan_legacy_import(legacy_dir: Path) -> list[LegacyImportPlan]:
    """Scan data/knowledge/ and build an import plan without writing files."""
    plans: list[LegacyImportPlan] = []

    for json_path in sorted(legacy_dir.rglob("*.json")):
        if json_path.name == "manifest.json":
            continue
        try:
            data = json.loads(json_path.read_text(encoding="utf-8"))
        except json.JSONDecodeError:
            continue

        url = data.get("url") or ""
        if not url:
            continue

        website = extract_website(url)
        source_type, extraction_method = _infer_legacy_fields(data)
        canonical_url = normalize_url(url)

        plans.append(
            LegacyImportPlan(
                source_path=json_path,
                website=website,
                source_type=source_type,
                extraction_method=extraction_method,
                canonical_url=canonical_url,
            )
        )

    return plans


def legacy_record_to_raw(data: dict, *, scraped_at: datetime | None = None) -> RawArtifact:
    """Convert a legacy knowledge JSON record to a RawArtifact."""
    url = data["url"]
    website = extract_website(url)
    canonical_url = normalize_url(url)
    source_type, extraction_method = _infer_legacy_fields(data)
    markdown = data.get("markdown") or data.get("content") or ""
    title = data.get("title") or url
    body_hash = content_hash(markdown)

    return RawArtifact(
        url=url,
        canonical_url=canonical_url,
        website=website,
        title=title,
        source_type=source_type,
        extraction_method=extraction_method,
        content=markdown,
        scraped_at=scraped_at or datetime.now(timezone.utc),
        content_hash=body_hash,
        crawl_root_url=data.get("source_url"),
        metadata={"legacy_import": True, "legacy_path": data.get("_legacy_path", "")},
    )


def import_legacy_file(raw_store: RawStore, legacy_path: Path, dry_run: bool = False) -> RawArtifact | None:
    """Import a single legacy JSON file into RAW storage."""
    data = json.loads(legacy_path.read_text(encoding="utf-8"))
    data["_legacy_path"] = str(legacy_path)
    artifact = legacy_record_to_raw(data)
    if dry_run:
        return artifact
    result = raw_store.write(artifact)
    return result.artifact


def import_legacy_directory(
    legacy_dir: Path,
    raw_store: RawStore,
    *,
    website: str | None = None,
    dry_run: bool = False,
) -> list[RawArtifact]:
    """
    Import all legacy JSON files from data/knowledge/ into RAW storage.

    This function is NOT called automatically.
    """
    artifacts: list[RawArtifact] = []
    for plan in plan_legacy_import(legacy_dir):
        if website and plan.website != website:
            continue
        artifact = import_legacy_file(raw_store, plan.source_path, dry_run=dry_run)
        if artifact is not None:
            artifacts.append(artifact)
    return artifacts
