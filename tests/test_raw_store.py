from datetime import datetime, timezone
from pathlib import Path

from app.kb.enums import ExtractionMethod, SourceType
from app.kb.hashing import content_hash
from app.kb.models.raw import RawArtifact
from app.kb.storage.raw_store import RawStore


def _artifact(content: str = "hello world") -> RawArtifact:
    now = datetime(2026, 1, 1, tzinfo=timezone.utc)
    body_hash = content_hash(content)
    return RawArtifact(
        url="https://earkart.in/about-us.html",
        canonical_url="https://earkart.in/about-us.html",
        website="earkart.in",
        title="About Us",
        source_type=SourceType.HTML,
        extraction_method=ExtractionMethod.HTML_PARSER,
        content=content,
        scraped_at=now,
        content_hash=body_hash,
    )


def test_raw_file_naming(tmp_path: Path) -> None:
    store = RawStore(tmp_path)
    artifact = _artifact()
    result = store.write(artifact)

    assert result.created is True
    assert result.path.name.startswith("sha256_")
    assert result.path.name.endswith(".json")
    assert result.path.parent.name == "html"


def test_duplicate_raw_not_overwritten(tmp_path: Path) -> None:
    store = RawStore(tmp_path)
    first = store.write(_artifact())
    second = store.write(_artifact())

    assert first.path == second.path
    assert second.created is False
    assert second.duplicate is True


def test_duplicate_content_detection(tmp_path: Path) -> None:
    store = RawStore(tmp_path)
    artifact = _artifact()
    store.write(artifact)

    assert store.is_duplicate_content(
        artifact.website,
        artifact.canonical_url,
        artifact.content_hash,
    )


def test_changed_content_new_hash(tmp_path: Path) -> None:
    store = RawStore(tmp_path)
    store.write(_artifact("version one"))
    result = store.write(_artifact("version two"))

    assert result.created is True
    assert result.path != store.artifact_path("earkart.in", SourceType.HTML, content_hash("version one"))
