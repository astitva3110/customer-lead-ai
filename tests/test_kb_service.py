import json
from datetime import datetime, timezone
from pathlib import Path

from app.kb.services.kb_service import KnowledgeBaseService
from app.kb.services.legacy_importer import legacy_record_to_raw
from app.kb.storage.canonical_store import CanonicalStore
from app.kb.storage.cleaned_store import CleanedStore
from app.kb.storage.raw_store import RawStore


FIXTURE_HTML = {
    "url": "https://earkart.in/about-us.html",
    "title": "About Us | earKART",
    "markdown": "## Our Story\n\nearKART is transforming hearing care.",
    "source_url": "https://earkart.in/",
    "content_type": "html",
}


def test_end_to_end_legacy_json_to_canonical(tmp_path: Path) -> None:
    raw_store = RawStore(tmp_path / "raw")
    cleaned_store = CleanedStore(tmp_path / "cleaned")
    canonical_store = CanonicalStore(tmp_path / "canonical")
    kb = KnowledgeBaseService(raw_store, cleaned_store, canonical_store)

    artifact = legacy_record_to_raw(FIXTURE_HTML, scraped_at=datetime(2026, 8, 17, tzinfo=timezone.utc))
    raw_result = raw_store.write(artifact)
    assert raw_result.created is True

    raw_dup = raw_store.write(artifact)
    assert raw_dup.duplicate is True

    kb_result, version_created = kb.ingest_html(
        FIXTURE_HTML["url"],
        FIXTURE_HTML["title"],
        FIXTURE_HTML["markdown"],
        crawl_root_url=FIXTURE_HTML["source_url"],
    )
    assert kb_result.duplicate is True
    assert version_created is False

    from app.kb.url_normalizer import make_document_id

    doc_id = make_document_id("earkart.in", "https://earkart.in/about-us.html")
    canonical = canonical_store.read_current("earkart.in", doc_id)
    assert canonical is not None
    assert canonical.website == "earkart.in"
    assert canonical.source_type.value == "html"
    assert canonical.extraction_method.value == "html_parser"
    assert canonical.structured_content.children

    # Example canonical JSON export for inspection
    example_path = tmp_path / "example_canonical.json"
    example_path.write_text(
        json.dumps(canonical.model_dump(mode="json"), ensure_ascii=False, indent=2),
        encoding="utf-8",
    )
    assert example_path.exists()
