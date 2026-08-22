from pathlib import Path

from app.kb.enums import SourceType


def website_source_dir(base: Path, website: str, source_type: SourceType) -> Path:
    folder = {
        SourceType.HTML: "html",
        SourceType.PDF: "pdf",
        SourceType.SCANNED_PDF: "scanned_pdf",
    }[source_type]
    return base / website / folder


def canonical_website_dir(base: Path, website: str) -> Path:
    return base / website


def retrieval_dataset_dir(base: Path, kb_dataset_version: str) -> Path:
    return base / kb_dataset_version


def retrieval_document_path(base: Path, kb_dataset_version: str, website: str, document_id: str) -> Path:
    return retrieval_dataset_dir(base, kb_dataset_version) / website / f"{document_id}.json"
