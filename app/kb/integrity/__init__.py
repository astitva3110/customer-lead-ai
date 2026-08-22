"""Canonical KB integrity and freeze (Phase 5)."""

from app.kb.integrity.contract import DownstreamDocument, downstream_eligible
from app.kb.integrity.freeze import CanonicalKbFreezer, FreezeResult
from app.kb.integrity.version import KB_DATASET_VERSION

__all__ = [
    "CanonicalKbFreezer",
    "DownstreamDocument",
    "FreezeResult",
    "KB_DATASET_VERSION",
    "downstream_eligible",
]
