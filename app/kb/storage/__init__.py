"""Backward-compatible re-exports — prefer app.kb.infrastructure.repositories."""

from app.kb.infrastructure.repositories import (
    CanonicalRepository,
    CanonicalStore,
    CleanedRepository,
    CleanedStore,
    RawRepository,
    RawStore,
    RawWriteResult,
    canonical_website_dir,
    retrieval_dataset_dir,
    retrieval_document_path,
    website_source_dir,
)

__all__ = [
    "CanonicalRepository",
    "CanonicalStore",
    "CleanedRepository",
    "CleanedStore",
    "RawRepository",
    "RawStore",
    "RawWriteResult",
    "canonical_website_dir",
    "retrieval_dataset_dir",
    "retrieval_document_path",
    "website_source_dir",
]
