"""Legacy .doc extraction — native first, pymupdf fallback."""

from __future__ import annotations

from pathlib import Path

import pymupdf

from app.kb.enums import ExtractionMethod
from app.kb.ingestion.extraction.docx import DocxDocumentExtractor
from app.kb.ingestion.models import ContentUnitType, ExtractedUnit, ExtractionResult


class DocDocumentExtractor:
    def extract(
        self,
        path: Path,
        *,
        document_id: str,
        document_version: int,
        title: str | None = None,
        allow_ocr: bool = True,
    ) -> ExtractionResult:
        resolved_title = title or path.stem

        # Attempt pymupdf text extraction (works for some .doc files).
        try:
            doc = pymupdf.open(path)
            try:
                text_parts = [page.get_text().strip() for page in doc if page.get_text().strip()]
                if text_parts:
                    units = [
                        ExtractedUnit(
                            document_id=document_id,
                            document_version=document_version,
                            page_number=index,
                            text=text,
                            content_type=ContentUnitType.PARAGRAPH,
                            extraction_method=ExtractionMethod.DOC_NATIVE,
                        )
                        for index, text in enumerate(text_parts, start=1)
                    ]
                    return ExtractionResult(
                        document_id=document_id,
                        document_version=document_version,
                        title=resolved_title,
                        units=units,
                        extraction_method=ExtractionMethod.DOC_NATIVE,
                        page_count=len(text_parts),
                        native_page_count=len(text_parts),
                    )
            finally:
                doc.close()
        except Exception:
            pass

        # Fallback: if file is actually OOXML renamed as .doc, try docx path.
        if path.read_bytes()[:2] == b"PK":
            return DocxDocumentExtractor().extract(
                path,
                document_id=document_id,
                document_version=document_version,
                title=resolved_title,
                allow_ocr=allow_ocr,
            )

        raise ValueError(
            f"Unable to extract legacy .doc file {path.name}. "
            "Convert to .docx or .pdf and re-upload."
        )
