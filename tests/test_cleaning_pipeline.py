import json
from datetime import datetime, timezone
from pathlib import Path

import pytest

from app.kb.models.structured_content import ListNode, SectionNode, TableNode
from app.kb.services.cleaning_pipeline import CleaningPipeline
from app.kb.services.legacy_importer import legacy_record_to_raw
from app.kb.storage.canonical_store import CanonicalStore
from app.kb.storage.cleaned_store import CleanedStore
from app.kb.validation.document_validator import validate_document

FIXTURES = Path(__file__).parent / "fixtures" / "cleaning"


def _artifact(name: str):
    data = json.loads((FIXTURES / name).read_text(encoding="utf-8"))
    return legacy_record_to_raw(data, scraped_at=datetime(2026, 1, 1, tzinfo=timezone.utc))


def test_pipeline_structures_investor_page(tmp_path) -> None:
    raw = _artifact("earkart_in_investor_html.json")
    pipeline = CleaningPipeline(CleanedStore(tmp_path / "cleaned"), CanonicalStore(tmp_path / "canonical"))
    cleaned, canonical = pipeline.process(raw)

    assert "Material Documents IPO" in cleaned.content
    assert canonical.structured_content.children
    assert any(
        isinstance(node, SectionNode) or node.type == "section"
        for node in canonical.structured_content.children
    )
    assert canonical.metadata.get("cleaning", {}).get("profile") == "earkart.in"


def test_pipeline_preserves_pdf_pages(tmp_path) -> None:
    raw = _artifact("earkart_in_native_pdf.json")
    pipeline = CleaningPipeline(CleanedStore(tmp_path / "cleaned"), CanonicalStore(tmp_path / "canonical"))
    _, canonical = pipeline.process(raw)
    assert canonical.structured_content.pages
    assert canonical.structured_content.pages[0].page_number == 1


def test_pipeline_validation_passes_terms_page(tmp_path) -> None:
    raw = _artifact("earkart_in_terms_html.json")
    pipeline = CleaningPipeline(CleanedStore(tmp_path / "cleaned"), CanonicalStore(tmp_path / "canonical"))
    cleaned, canonical = pipeline.process(raw)
    result = validate_document(raw, cleaned, canonical)
    assert result.passed


def _all_nodes(doc):
    from app.kb.models.structured_content import SectionNode

    nodes = list(doc.children)
    for child in doc.children:
        if isinstance(child, SectionNode):
            nodes.extend(child.children)
    return nodes


def test_markdown_table_parsing() -> None:
    from app.kb.structuring.markdown import markdown_to_structured

    md = "## Specs\n\n| Feature | Value |\n| --- | --- |\n| Channels | 16 |\n"
    doc = markdown_to_structured("Specs", md)
    tables = [n for n in _all_nodes(doc) if isinstance(n, TableNode)]
    assert tables
    assert tables[0].headers == ["Feature", "Value"]
    assert tables[0].rows[0] == ["Channels", "16"]


def test_markdown_list_parsing() -> None:
    from app.kb.structuring.markdown import markdown_to_structured

    md = "## Benefits\n\n* Free Insurance\n* Free UV Dehumidifier\n"
    doc = markdown_to_structured("Benefits", md)
    lists = [n for n in _all_nodes(doc) if isinstance(n, ListNode)]
    assert lists
    assert "Free Insurance" in lists[0].items
