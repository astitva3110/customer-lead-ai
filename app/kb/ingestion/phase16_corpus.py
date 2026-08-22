"""Phase 16 V3 corpus loader."""

from __future__ import annotations

import tempfile
from dataclasses import dataclass
from pathlib import Path

from app.kb.chunking.v3.dedup import suppress_cross_document_duplicates
from app.kb.ingestion.chunking_v3 import chunk_document_v3
from app.kb.ingestion.extraction.base import get_extractor
from app.kb.ingestion.golden_audit import discover_golden_pdfs
from app.kb.ingestion.models import DocumentRecord, Phase12ChunkRecord
from app.kb.ingestion.storage import DocumentStorage
from app.kb.ingestion.structure import build_structured_document


@dataclass(frozen=True)
class Phase16CorpusDocument:
    label: str
    filename: str
    record: DocumentRecord
    title: str
    chunks: list[Phase12ChunkRecord]


@dataclass(frozen=True)
class Phase16Corpus:
    documents: list[Phase16CorpusDocument]
    suppressed_duplicates: list[dict]
    source_pdfs: list[str]

    @property
    def all_chunks(self) -> list[Phase12ChunkRecord]:
        return [chunk for doc in self.documents for chunk in doc.chunks]

    @property
    def chunk_count(self) -> int:
        return len(self.all_chunks)


def load_phase16_v3_corpus(*, storage_dir: Path | None = None) -> Phase16Corpus:
    candidates = discover_golden_pdfs()
    if len(candidates) < 2:
        raise RuntimeError(f"Expected 2 golden PDFs, found {len(candidates)}")

    if storage_dir is None:
        storage_dir = Path(tempfile.mkdtemp(prefix="phase16_v3_"))

    storage = DocumentStorage(base_dir=storage_dir)
    doc_outputs: list[tuple[str, str, DocumentRecord, str, list[Phase12ChunkRecord]]] = []

    for candidate in candidates:
        record, validation, _duplicate = storage.store_upload(
            candidate.path,
            document_type=candidate.document_type,
            source="phase16_v3",
        )
        if not validation.ok:
            raise RuntimeError(f"Validation failed for {candidate.path.name}: {validation.error}")

        path = Path(record.storage_path)
        extractor = get_extractor(path)
        extraction = extractor.extract(
            path,
            document_id=record.document_id,
            document_version=record.document_version,
            title=candidate.path.stem,
            allow_ocr=False,
        )
        structured, _ = build_structured_document(extraction)
        chunks = chunk_document_v3(record=record, extraction=extraction, structured=structured)
        doc_outputs.append((candidate.label, candidate.path.name, record, extraction.title, chunks))

    kept, suppressed = suppress_cross_document_duplicates(
        [(label, chunks) for label, _, _, _, chunks in doc_outputs]
    )
    kept_ids = {chunk.chunk_id for chunk in kept}
    documents = [
        Phase16CorpusDocument(
            label=label,
            filename=filename,
            record=record,
            title=title,
            chunks=[chunk for chunk in chunks if chunk.chunk_id in kept_ids],
        )
        for label, filename, record, title, chunks in doc_outputs
    ]

    return Phase16Corpus(
        documents=documents,
        suppressed_duplicates=suppressed,
        source_pdfs=[doc.filename for doc in documents],
    )
