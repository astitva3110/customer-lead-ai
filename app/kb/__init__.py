"""Knowledge Base Management Layer."""

from app.kb.models.canonical import CanonicalDocument
from app.kb.infrastructure.repositories.raw_repository import RawRepository, RawWriteResult

RawStore = RawRepository

__all__ = ["CanonicalDocument", "RawRepository", "RawStore", "RawWriteResult"]
