#!/usr/bin/env python3
"""Phase 13 golden PDF dry-run ingestion and chunk quality audit."""

from __future__ import annotations

import json
import sys
from pathlib import Path

ROOT = Path(__file__).resolve().parents[1]
sys.path.insert(0, str(ROOT))


def main() -> int:
    from app.config import settings
    from app.kb.enums import DocumentType
    from app.kb.ingestion.golden_audit import (
        Phase13DocumentReport,
        audit_chunks,
        audit_structure,
        build_phase13_report,
        discover_golden_pdfs,
        format_phase13_text,
        run_manual_golden_checks,
    )
    from app.kb.ingestion.service import DocumentIngestionService
    from app.kb.ingestion.storage import DocumentStorage
    from app.kb.ingestion.structure import build_structured_document

    v1_baseline_path = Path("data/integrity/phase11_baseline.json")
    v1_baseline = json.loads(v1_baseline_path.read_text(encoding="utf-8")) if v1_baseline_path.exists() else {}

    candidates = discover_golden_pdfs()
    if len(candidates) < 2:
        print(f"Expected 2 golden PDFs, found {len(candidates)}: {[c.path.name for c in candidates]}", file=sys.stderr)
        return 1

    import tempfile

    with tempfile.TemporaryDirectory(prefix="phase13_") as tmp:
        service = DocumentIngestionService(storage=DocumentStorage(base_dir=Path(tmp)))
        doc_reports: list[Phase13DocumentReport] = []
        chunks_by_label: dict[str, list] = {"merged": [], "terms": []}

        for candidate in candidates:
            print(f"Processing {candidate.label}: {candidate.path.name}", flush=True)
            result = service.ingest_file(
                candidate.path,
                document_type=candidate.document_type,
                dry_run=True,
                embed=False,
            )
            extraction = result.extraction
            if extraction is None:
                print(f"Extraction failed for {candidate.path.name}", file=sys.stderr)
                return 1

            structured, _ = build_structured_document(extraction)
            structure = audit_structure(structured)
            quality_issues = audit_chunks(
                result.chunks,
                document_label=candidate.label,
                document_title=extraction.title,
            )
            determinism_pass = not any(
                issue["category"].startswith("nondeterministic") for issue in quality_issues
            )

            doc_reports.append(
                Phase13DocumentReport(
                    label=candidate.label,
                    filename=candidate.path.name,
                    document_id=result.document.document_id,
                    document_type=result.document.document_type.value,
                    page_count=extraction.page_count,
                    native_page_count=extraction.native_page_count,
                    ocr_page_count=extraction.ocr_page_count,
                    extraction_method=extraction.extraction_method.value,
                    ocr_used=extraction.ocr_used,
                    structure=structure,
                    chunks=result.chunks,
                    suppressed=result.suppressed_chunks,
                    quality_issues=quality_issues,
                    determinism_pass=determinism_pass,
                )
            )
            chunks_by_label[candidate.label] = result.chunks
            print(
                f"  pages={extraction.page_count} method={extraction.extraction_method.value} "
                f"chunks={len(result.chunks)} issues={len(quality_issues)}",
                flush=True,
            )

        manual = run_manual_golden_checks(
            chunks_by_label.get("merged", []),
            chunks_by_label.get("terms", []),
        )
        report = build_phase13_report(doc_reports, manual)
        text = format_phase13_text(report)

    reports_dir = settings.reports_dir
    reports_dir.mkdir(parents=True, exist_ok=True)
    json_path = reports_dir / "phase13_golden_pdf_ingestion.json"
    txt_path = reports_dir / "phase13_golden_pdf_ingestion.txt"
    json_path.write_text(json.dumps(report, indent=2, default=str), encoding="utf-8")
    txt_path.write_text(text, encoding="utf-8")

    print(text)
    print(f"\nWrote {json_path}")
    print(f"Wrote {txt_path}")
    if v1_baseline:
        print(f"V1 integrity baseline unchanged: {v1_baseline_path}")
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
