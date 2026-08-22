"""DOCX native extraction."""

from __future__ import annotations

from pathlib import Path

from app.kb.enums import ExtractionMethod
from app.kb.ingestion.models import ContentUnitType, ExtractedUnit, ExtractionResult


def _heading_level(style_name: str | None) -> int | None:
    if not style_name:
        return None
    lowered = style_name.lower()
    if lowered.startswith("heading"):
        digits = "".join(ch for ch in lowered if ch.isdigit())
        return int(digits) if digits else 2
    if lowered in {"title", "subtitle"}:
        return 1
    return None


class DocxDocumentExtractor:
    def extract(
        self,
        path: Path,
        *,
        document_id: str,
        document_version: int,
        title: str | None = None,
        allow_ocr: bool = True,
    ) -> ExtractionResult:
        from docx import Document
        from docx.table import Table
        from docx.text.paragraph import Paragraph

        document = Document(str(path))
        units: list[ExtractedUnit] = []
        resolved_title = title or path.stem

        def walk_table(table: Table) -> None:
            rows: list[list[str]] = []
            for row in table.rows:
                rows.append([cell.text.strip() for cell in row.cells])
            headers = rows[0] if rows else []
            body_rows = rows[1:] if len(rows) > 1 else []
            lines = []
            if headers:
                lines.append(" | ".join(headers))
            for row in body_rows:
                lines.append(" | ".join(row))
            units.append(
                ExtractedUnit(
                    document_id=document_id,
                    document_version=document_version,
                    page_number=None,
                    text="\n".join(lines),
                    content_type=ContentUnitType.TABLE,
                    extraction_method=ExtractionMethod.DOCX_NATIVE,
                    table_headers=headers,
                    table_rows=body_rows,
                )
            )

        for block in document.element.body:
            tag = block.tag.split("}")[-1]
            if tag == "p":
                paragraph = Paragraph(block, document)
                text = paragraph.text.strip()
                if not text:
                    continue
                level = _heading_level(paragraph.style.name if paragraph.style else None)
                if level:
                    units.append(
                        ExtractedUnit(
                            document_id=document_id,
                            document_version=document_version,
                            text=text,
                            content_type=ContentUnitType.HEADING,
                            extraction_method=ExtractionMethod.DOCX_NATIVE,
                            heading_level=level,
                        )
                    )
                else:
                    units.append(
                        ExtractedUnit(
                            document_id=document_id,
                            document_version=document_version,
                            text=text,
                            content_type=ContentUnitType.PARAGRAPH,
                            extraction_method=ExtractionMethod.DOCX_NATIVE,
                        )
                    )
            elif tag == "tbl":
                walk_table(Table(block, document))

        return ExtractionResult(
            document_id=document_id,
            document_version=document_version,
            title=resolved_title,
            units=units,
            extraction_method=ExtractionMethod.DOCX_NATIVE,
            ocr_used=False,
            native_page_count=1,
            ocr_page_count=0,
            page_count=1,
        )
