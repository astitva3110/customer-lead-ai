from app.kb.infrastructure.repositories.canonical_repository import CanonicalRepository
from app.kb.infrastructure.repositories.cleaned_repository import CleanedRepository
from app.kb.infrastructure.repositories.paths import (
    canonical_website_dir,
    retrieval_dataset_dir,
    retrieval_document_path,
    website_source_dir,
)
from app.kb.infrastructure.repositories.raw_repository import RawRepository, RawWriteResult

# Backward-compatible aliases
CanonicalStore = CanonicalRepository
CleanedStore = CleanedRepository
RawStore = RawRepository

__all__ = [
    "CanonicalRepository",
    "CleanedRepository",
    "RawRepository",
    "RawWriteResult",
    "CanonicalStore",
    "CleanedStore",
    "RawStore",
    "canonical_website_dir",
    "retrieval_dataset_dir",
    "retrieval_document_path",
    "website_source_dir",
]
