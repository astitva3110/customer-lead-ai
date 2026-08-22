from datetime import datetime, timezone

from app.kb.enums import ExtractionMethod, SourceType
from app.kb.models.raw import RawArtifact
from app.kb.models.structured_content import (
    DocumentContent,
    OcrMetadata,
    PageContent,
    ParagraphNode,
    SectionNode,
)
from app.kb.services.document_builder import DocumentBuilder
from app.kb.storage.canonical_store import CanonicalStore


def test_pdf_page_structure_in_canonical(tmp_path) -> None:
    now = datetime(2026, 1, 1, tzinfo=timezone.utc)
    artifact = RawArtifact(
        url="https://earkart.in/spec.pdf",
        canonical_url="https://earkart.in/spec.pdf",
        website="earkart.in",
        title="Spec",
        source_type=SourceType.PDF,
        extraction_method=ExtractionMethod.PDF_TEXT,
        content="Page one\n\nPage two",
        scraped_at=now,
        content_hash="sha256:" + "b" * 64,
        pages=[
            PageContent(page_number=1, blocks=[ParagraphNode(text="Page one")]),
            PageContent(page_number=2, blocks=[ParagraphNode(text="Page two")]),
        ],
    )

    store = CanonicalStore(tmp_path)
    builder = DocumentBuilder(store)
    doc = builder.build_from_raw(artifact)

    assert doc.structured_content.pages is not None
    assert len(doc.structured_content.pages) == 2
    assert doc.structured_content.pages[0].page_number == 1
    assert doc.structured_content.pages[1].blocks[0].text == "Page two"


def test_ocr_metadata_preserved(tmp_path) -> None:
    now = datetime(2026, 1, 1, tzinfo=timezone.utc)
    ocr = OcrMetadata(engine="tesseract", dpi=200, language="eng", average_confidence=0.82)
    artifact = RawArtifact(
        url="https://earkart.in/policy.pdf",
        canonical_url="https://earkart.in/policy.pdf",
        website="earkart.in",
        title="Policy",
        source_type=SourceType.SCANNED_PDF,
        extraction_method=ExtractionMethod.OCR,
        content="Policy text",
        scraped_at=now,
        content_hash="sha256:" + "c" * 64,
        ocr=ocr,
        pages=[
            PageContent(
                page_number=1,
                ocr_confidence=0.79,
                blocks=[ParagraphNode(text="Policy text", confidence=0.79)],
            )
        ],
    )

    builder = DocumentBuilder(CanonicalStore(tmp_path))
    doc = builder.build_from_raw(artifact)

    assert doc.source_type == SourceType.SCANNED_PDF
    assert doc.extraction_method == ExtractionMethod.OCR
    assert doc.structured_content.ocr is not None
    assert doc.structured_content.ocr.average_confidence == 0.82
    assert doc.structured_content.pages[0].ocr_confidence == 0.79


def test_structured_content_not_flattened_only() -> None:
    content = DocumentContent(
        type="document",
        title="Doc",
        children=[
            SectionNode(
                heading="Intro",
                level=2,
                children=[ParagraphNode(text="Body")],
            )
        ],
    )
    assert content.children[0].heading == "Intro"  # type: ignore[attr-defined]
