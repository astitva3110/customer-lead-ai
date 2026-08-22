from abc import ABC, abstractmethod
from dataclasses import dataclass, field
from datetime import datetime, timezone

from app.kb.models.raw import RawArtifact


@dataclass
class CleaningResult:
    content: str
    pages: list | None = None
    ocr: object | None = None
    profile: str = "generic"
    profile_version: str = "1.0"
    removed_elements: list[str] = field(default_factory=list)
    removed_headers: list[str] = field(default_factory=list)
    removed_footers: list[str] = field(default_factory=list)
    warnings: list[str] = field(default_factory=list)
    cleaned_at: datetime = field(default_factory=lambda: datetime.now(timezone.utc))

    def cleaning_metadata(self) -> dict:
        return {
            "profile": self.profile,
            "version": self.profile_version,
            "removed_elements": self.removed_elements,
            "removed_headers": self.removed_headers,
            "removed_footers": self.removed_footers,
            "cleaned_at": self.cleaned_at.isoformat(),
            "warnings": self.warnings,
        }


class BaseCleaner(ABC):
    profile: str = "generic"
    profile_version: str = "1.0"

    @abstractmethod
    def clean(self, artifact: RawArtifact) -> CleaningResult:
        raise NotImplementedError
