#!/usr/bin/env python3
"""Phase 12 document ingestion CLI."""

from __future__ import annotations

import argparse
import sys
from pathlib import Path

ROOT = Path(__file__).resolve().parents[1]
sys.path.insert(0, str(ROOT))


def main() -> int:
    parser = argparse.ArgumentParser(description="Phase 12 document-first ingestion")
    parser.add_argument("--file", required=True, type=Path, help="Path to PDF/DOC/DOCX file")
    parser.add_argument(
        "--document-type",
        default="other",
        choices=[
            "product",
            "brochure",
            "company",
            "policy",
            "faq",
            "investor",
            "notice",
            "prospectus",
            "other",
        ],
    )
    parser.add_argument("--language", default="en")
    parser.add_argument("--no-ocr", action="store_true", help="Disable OCR fallback")
    parser.add_argument("--dry-run", action="store_true", help="Extract/chunk/validate only — no vectors")
    parser.add_argument("--document", dest="document_id", help="Process existing document_id instead of uploading")
    args = parser.parse_args()

    from app.kb.enums import DocumentType, DocumentUploadStatus
    from app.kb.ingestion.wiring import build_ingestion_service

    service = build_ingestion_service()
    doc_type = DocumentType(args.document_type)

    if args.document_id:
        result = service.process_document(
            args.document_id,
            allow_ocr=not args.no_ocr,
            dry_run=args.dry_run,
            embed=not args.dry_run,
        )
    else:
        if not args.file.exists():
            print(f"File not found: {args.file}", file=sys.stderr)
            return 1
        result = service.ingest_file(
            args.file,
            document_type=doc_type,
            language=args.language,
            allow_ocr=not args.no_ocr,
            dry_run=args.dry_run,
            embed=not args.dry_run,
        )

    print(f"Document ID: {result.document.document_id}")
    print(f"Status: {result.document.status.value}")
    print(f"Duplicate: {result.duplicate}")
    print(f"Chunks: {len(result.chunks)}")
    print(f"Dry run: {result.dry_run}")
    print(f"Embedded: {result.embedded}")
    print(f"Indexed: {result.indexed}")
    if result.document.error:
        print(f"Error: {result.document.error}", file=sys.stderr)
    if result.extraction:
        print(f"Extraction: {result.extraction.extraction_method.value}")
    if result.quality:
        print(f"Quality: {result.quality.get('statistics', {})}")
    if result.document.status == DocumentUploadStatus.FAILED:
        return 1
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
