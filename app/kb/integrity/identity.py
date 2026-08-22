"""URL and document identity integrity checks across the canonical store."""

from __future__ import annotations

from dataclasses import dataclass, field
from typing import Any

from app.kb.models.canonical import CanonicalDocument
from app.kb.url_normalizer import make_document_id, normalize_url


@dataclass
class IdentityIntegrityReport:
    duplicate_document_ids: list[dict[str, Any]] = field(default_factory=list)
    duplicate_canonical_urls: list[dict[str, Any]] = field(default_factory=list)
    duplicate_current_versions: list[dict[str, Any]] = field(default_factory=list)
    document_id_mismatches: list[dict[str, Any]] = field(default_factory=list)
    url_normalization_collisions: list[dict[str, Any]] = field(default_factory=list)
    passed: bool = True

    def to_dict(self) -> dict[str, Any]:
        return {
            "passed": self.passed,
            "duplicate_document_ids": self.duplicate_document_ids,
            "duplicate_canonical_urls": self.duplicate_canonical_urls,
            "duplicate_current_versions": self.duplicate_current_versions,
            "document_id_mismatches": self.document_id_mismatches,
            "url_normalization_collisions": self.url_normalization_collisions,
        }


def check_identity_integrity(documents: list[CanonicalDocument]) -> IdentityIntegrityReport:
    report = IdentityIntegrityReport()

    by_id: dict[str, list[CanonicalDocument]] = {}
    by_canonical_url: dict[str, list[CanonicalDocument]] = {}
    by_source_normalized: dict[str, list[CanonicalDocument]] = {}

    for doc in documents:
        if not doc.is_current:
            continue

        by_id.setdefault(doc.document_id, []).append(doc)
        key = f"{doc.website}:{doc.canonical_url}"
        by_canonical_url.setdefault(key, []).append(doc)

        norm_key = f"{doc.website}:{normalize_url(doc.source_url)}"
        by_source_normalized.setdefault(norm_key, []).append(doc)

        expected_id = make_document_id(doc.website, doc.canonical_url)
        if doc.document_id != expected_id:
            report.document_id_mismatches.append(
                {
                    "document_id": doc.document_id,
                    "expected_document_id": expected_id,
                    "url": doc.canonical_url,
                }
            )
            report.passed = False

    for doc_id, group in by_id.items():
        if len(group) > 1:
            report.duplicate_document_ids.append(
                {
                    "document_id": doc_id,
                    "documents": [{"website": d.website, "url": d.canonical_url} for d in group],
                }
            )
            report.passed = False

    for key, group in by_canonical_url.items():
        if len(group) > 1:
            report.duplicate_canonical_urls.append(
                {
                    "key": key,
                    "documents": [{"document_id": d.document_id, "version": d.version} for d in group],
                }
            )
            report.passed = False

    for key, group in by_source_normalized.items():
        urls = {d.canonical_url for d in group}
        source_urls = {d.source_url for d in group}
        if len(group) > 1 and len(source_urls) > 1 and len(urls) == 1:
            report.url_normalization_collisions.append(
                {
                    "normalized_key": key,
                    "canonical_url": group[0].canonical_url,
                    "source_urls": sorted(source_urls),
                }
            )

    return report
