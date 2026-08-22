"""Load Phase 13 validated golden-PDF chunk corpus for KB V2 embedding."""

from __future__ import annotations

import tempfile
from dataclasses import dataclass
from pathlib import Path

from app.kb.ingestion.golden_audit import discover_golden_pdfs
from app.kb.ingestion.models import DocumentRecord, Phase12ChunkRecord
from app.kb.ingestion.quality import ChunkQualityGate
from app.kb.ingestion.service import DocumentIngestionService
from app.kb.ingestion.storage import DocumentStorage


@dataclass(frozen=True)
class Phase13CorpusDocument:
    label: str
    filename: str
    record: DocumentRecord
    title: str
    chunks: list[Phase12ChunkRecord]


@dataclass(frozen=True)
class Phase13Corpus:
    documents: list[Phase13CorpusDocument]
    source_pdfs: list[str]

    @property
    def all_chunks(self) -> list[Phase12ChunkRecord]:
        return [chunk for doc in self.documents for chunk in doc.chunks]

    @property
    def chunk_count(self) -> int:
        return len(self.all_chunks)

    def record_for(self, document_id: str) -> DocumentRecord | None:
        for doc in self.documents:
            if doc.record.document_id == document_id:
                return doc.record
        return None

    def title_for(self, document_id: str) -> str:
        for doc in self.documents:
            if doc.record.document_id == document_id:
                return doc.title
        return ""


def load_phase13_validated_corpus(*, storage_dir: Path | None = None) -> Phase13Corpus:
    """Re-chunk golden PDFs and return quality-gate passed chunks only."""
    candidates = discover_golden_pdfs()
    if len(candidates) < 2:
        raise RuntimeError(
            f"Expected 2 golden PDFs, found {len(candidates)}: {[c.path.name for c in candidates]}"
        )

    gate = ChunkQualityGate()
    documents: list[Phase13CorpusDocument] = []

    if storage_dir is None:
        tmp = tempfile.TemporaryDirectory(prefix="phase13_corpus_")
        storage_dir = Path(tmp.name)

    service = DocumentIngestionService(storage=DocumentStorage(base_dir=storage_dir))

    for candidate in candidates:
        result = service.ingest_file(
            candidate.path,
            document_type=candidate.document_type,
            dry_run=True,
            embed=False,
        )
        if result.extraction is None:
            raise RuntimeError(f"Extraction failed for {candidate.path.name}")
        quality = gate.evaluate(result.chunks)
        passed = quality["passed"]
        if not passed:
            raise RuntimeError(
                f"Quality gate failed for {candidate.path.name}: {len(quality['failed'])} chunks rejected"
            )
        documents.append(
            Phase13CorpusDocument(
                label=candidate.label,
                filename=candidate.path.name,
                record=result.document,
                title=result.extraction.title,
                chunks=passed,
            )
        )

    return Phase13Corpus(
        documents=documents,
        source_pdfs=[doc.filename for doc in documents],
    )


def select_deterministic_chunks(
    chunks: list[Phase12ChunkRecord],
    limit: int,
) -> list[Phase12ChunkRecord]:
    ordered = sorted(chunks, key=lambda item: item.chunk_id)
    return ordered[:limit]
