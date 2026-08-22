"""Deterministic canonical KB manifest generation."""

from __future__ import annotations

from collections import Counter
from datetime import datetime, timezone
from typing import Any

from app.kb.enums import ProcessingStatus
from app.kb.integrity.version import KB_DATASET_VERSION
from app.kb.models.canonical import CanonicalDocument


def _utc_now_iso() -> str:
    return datetime.now(timezone.utc).isoformat()


def build_manifest(documents: list[CanonicalDocument]) -> dict[str, Any]:
    current = [d for d in documents if d.is_current]
    current.sort(key=lambda d: (d.website, d.document_id))

    by_status = Counter(d.processing_status.value for d in current)
    by_website = Counter(d.website for d in current)
    by_source_type = Counter(d.source_type.value for d in current)
    by_extraction = Counter(d.extraction_method.value for d in current)

    entries = [
        {
            "document_id": d.document_id,
            "website": d.website,
            "canonical_url": d.canonical_url,
            "source_url": d.source_url,
            "title": d.title,
            "source_type": d.source_type.value,
            "extraction_method": d.extraction_method.value,
            "processing_status": d.processing_status.value,
            "version": d.version,
            "content_hash": d.content_hash,
            "excluded": bool(d.metadata.get("excluded")),
            "exclusion_reason": d.metadata.get("exclusion_reason"),
        }
        for d in current
    ]

    production = [
        d
        for d in current
        if d.processing_status in (ProcessingStatus.ACTIVE, ProcessingStatus.VALIDATED)
    ]

    return {
        "generated_at": _utc_now_iso(),
        "kb_dataset_version": KB_DATASET_VERSION,
        "total_documents": len(current),
        "active_count": by_status.get("ACTIVE", 0),
        "validated_count": by_status.get("VALIDATED", 0),
        "excluded_count": by_status.get("EXCLUDED", 0),
        "failed_count": by_status.get("FAILED", 0),
        "production_count": len(production),
        "websites": dict(sorted(by_website.items())),
        "source_types": dict(sorted(by_source_type.items())),
        "extraction_methods": dict(sorted(by_extraction.items())),
        "processing_status": dict(sorted(by_status.items())),
        "documents": entries,
    }
