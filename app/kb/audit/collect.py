from dataclasses import dataclass
from pathlib import Path

from app.kb.enums import SourceType
from app.kb.models.canonical import CanonicalDocument
from app.kb.models.raw import CleanedArtifact, RawArtifact
from app.kb.storage.canonical_store import CanonicalStore
from app.kb.storage.cleaned_store import CleanedStore
from app.kb.storage.raw_store import RawStore


@dataclass
class DocumentBundle:
    canonical: CanonicalDocument
    raw: RawArtifact | None
    cleaned: CleanedArtifact | None


def iter_raw_artifacts(raw_store: RawStore) -> list[RawArtifact]:
    artifacts: list[RawArtifact] = []
    for path in sorted(raw_store.base_dir.rglob("sha256_*.json")):
        artifacts.append(raw_store.read_path(path))
    return artifacts


def iter_cleaned_artifacts(cleaned_store: CleanedStore) -> list[CleanedArtifact]:
    artifacts: list[CleanedArtifact] = []
    for path in sorted(cleaned_store.base_dir.rglob("sha256_*.json")):
        artifacts.append(cleaned_store.read_path(path))
    return artifacts


def iter_canonical_documents(canonical_store: CanonicalStore) -> list[CanonicalDocument]:
    documents: list[CanonicalDocument] = []
    for current_path in sorted(canonical_store.base_dir.rglob("current.json")):
        documents.append(canonical_store.read_path(current_path))
    return documents


def _find_raw(raw_store: RawStore, canonical: CanonicalDocument) -> RawArtifact | None:
    raw_hash = canonical.metadata.get("raw_content_hash")
    if not raw_hash:
        return None
    return raw_store.get_by_hash(canonical.website, canonical.source_type, raw_hash)


def _find_cleaned(cleaned_store: CleanedStore, canonical: CanonicalDocument) -> CleanedArtifact | None:
    cleaned_hash = canonical.metadata.get("cleaned_content_hash")
    if not cleaned_hash:
        return None
    return cleaned_store.get_by_hash(canonical.website, canonical.source_type, cleaned_hash)


def collect_document_bundles(
    raw_store: RawStore,
    cleaned_store: CleanedStore,
    canonical_store: CanonicalStore,
) -> list[DocumentBundle]:
    bundles: list[DocumentBundle] = []
    for canonical in iter_canonical_documents(canonical_store):
        bundles.append(
            DocumentBundle(
                canonical=canonical,
                raw=_find_raw(raw_store, canonical),
                cleaned=_find_cleaned(cleaned_store, canonical),
            )
        )
    return bundles


def count_raw_by_website_type(raw_store: RawStore) -> dict[str, dict[str, int]]:
    counts: dict[str, dict[str, int]] = {}
    for artifact in iter_raw_artifacts(raw_store):
        site = counts.setdefault(artifact.website, {"html": 0, "pdf": 0, "scanned_pdf": 0})
        site[artifact.source_type.value] += 1
    return counts
