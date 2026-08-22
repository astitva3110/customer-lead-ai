"""PDF extraction with selective per-page OCR."""

from __future__ import annotations

from pathlib import Path

import pymupdf

from app.kb.enums import ExtractionMethod
from app.kb.ingestion.extraction.ocr import extract_native_page, ocr_page, page_needs_ocr
from app.kb.ingestion.models import ContentUnitType, ExtractedUnit, ExtractionResult


def _blocks_to_units(
    *,
    document_id: str,
    document_version: int,
    page_number: int,
    blocks,
    extraction_method: ExtractionMethod,
    ocr_used: bool,
) -> list[ExtractedUnit]:
    units: list[ExtractedUnit] = []
    for block in blocks:
        block_type = getattr(block, "type", None) or block.get("type")
        if block_type == "heading":
            text = block.text if hasattr(block, "text") else block["text"]
            level = block.level if hasattr(block, "level") else block.get("level", 2)
            units.append(
                ExtractedUnit(
                    document_id=document_id,
                    document_version=document_version,
                    page_number=page_number,
                    text=text.strip(),
                    content_type=ContentUnitType.HEADING,
                    extraction_method=extraction_method,
                    ocr_used=ocr_used,
                    heading_level=level,
                )
            )
        elif block_type == "paragraph":
            text = block.text if hasattr(block, "text") else block["text"]
            if text.strip():
                units.append(
                    ExtractedUnit(
                        document_id=document_id,
                        document_version=document_version,
                        page_number=page_number,
                        text=text.strip(),
                        content_type=ContentUnitType.PARAGRAPH,
                        extraction_method=extraction_method,
                        ocr_used=ocr_used,
                    )
                )
        elif block_type == "list":
            items = block.items if hasattr(block, "items") else block.get("items", [])
            if items:
                units.append(
                    ExtractedUnit(
                        document_id=document_id,
                        document_version=document_version,
                        page_number=page_number,
                        text="\n".join(f"- {item}" for item in items),
                        content_type=ContentUnitType.LIST,
                        extraction_method=extraction_method,
                        ocr_used=ocr_used,
                        list_items=list(items),
                    )
                )
        elif block_type == "table":
            headers = block.headers if hasattr(block, "headers") else block.get("headers", [])
            rows = block.rows if hasattr(block, "rows") else block.get("rows", [])
            lines = []
            if headers:
                lines.append(" | ".join(headers))
            for row in rows:
                lines.append(" | ".join(row))
            units.append(
                ExtractedUnit(
                    document_id=document_id,
                    document_version=document_version,
                    page_number=page_number,
                    text="\n".join(lines),
                    content_type=ContentUnitType.TABLE,
                    extraction_method=extraction_method,
                    ocr_used=ocr_used,
                    table_headers=list(headers),
                    table_rows=[list(row) for row in rows],
                )
            )
    return units


class PdfDocumentExtractor:
    def extract(
        self,
        path: Path,
        *,
        document_id: str,
        document_version: int,
        title: str | None = None,
        allow_ocr: bool = True,
    ) -> ExtractionResult:
        doc = pymupdf.open(path)
        try:
            resolved_title = title or (doc.metadata or {}).get("title") or path.stem
            units: list[ExtractedUnit] = []
            native_pages = 0
            ocr_pages = 0

            for index, page in enumerate(doc, start=1):
                native_text, native_blocks = extract_native_page(page)
                use_ocr = allow_ocr and page_needs_ocr(page, native_text)
                if use_ocr:
                    ocr_text, ocr_blocks = ocr_page(page)
                    if len(ocr_text.strip()) > len(native_text.strip()):
                        text, blocks = ocr_text, ocr_blocks
                        method = ExtractionMethod.OCR
                        ocr_pages += 1
                        ocr_used = True
                    else:
                        text, blocks = native_text, native_blocks
                        method = ExtractionMethod.NATIVE
                        native_pages += 1
                        ocr_used = False
                else:
                    text, blocks = native_text, native_blocks
                    method = ExtractionMethod.NATIVE
                    native_pages += 1
                    ocr_used = False

                if not text.strip():
                    continue
                units.extend(
                    _blocks_to_units(
                        document_id=document_id,
                        document_version=document_version,
                        page_number=index,
                        blocks=blocks or [],
                        extraction_method=method,
                        ocr_used=ocr_used,
                    )
                )
                if not blocks and text.strip():
                    units.append(
                        ExtractedUnit(
                            document_id=document_id,
                            document_version=document_version,
                            page_number=index,
                            text=text.strip(),
                            content_type=ContentUnitType.PARAGRAPH,
                            extraction_method=method,
                            ocr_used=ocr_used,
                        )
                    )

            if ocr_pages and native_pages:
                overall_method = ExtractionMethod.MIXED
            elif ocr_pages:
                overall_method = ExtractionMethod.OCR
            else:
                overall_method = ExtractionMethod.NATIVE

            return ExtractionResult(
                document_id=document_id,
                document_version=document_version,
                title=resolved_title,
                units=units,
                extraction_method=overall_method,
                ocr_used=ocr_pages > 0,
                native_page_count=native_pages,
                ocr_page_count=ocr_pages,
                page_count=len(doc),
            )
        finally:
            doc.close()
