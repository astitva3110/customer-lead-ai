"""Phase 16.1 V3.1 corpus — V3 atomic chunks plus one Why Choose earKART summary chunk."""

from __future__ import annotations

from datetime import datetime, timezone

from app.config import settings
from app.kb.chunking.v3.embedding_input import attach_v3_embedding_metadata
from app.kb.ingestion.chunking import deterministic_chunk_id
from app.kb.ingestion.models import Phase12ChunkRecord
from app.kb.ingestion.phase16_corpus import Phase16Corpus, Phase16CorpusDocument, load_phase16_v3_corpus

WHY_CHOOSE_SECTION_MARKERS = ("why choose earkart", "5. why choose earkart")
SUMMARY_CONTENT_TYPE = "section_summary"
V3_1_ALGORITHM_VERSION = "phase16.1.0"
V3_1_KB_DATASET_VERSION = "phase16.1-v1"


def _is_why_choose_atom(chunk: Phase12ChunkRecord) -> bool:
    path_text = " > ".join(chunk.section_path).lower()
    parent = (chunk.parent_section or "").lower()
    subsection = (chunk.subsection or "").lower()
    return any(
        marker in text
        for marker in WHY_CHOOSE_SECTION_MARKERS
        for text in (path_text, parent, subsection, chunk.content[:120].lower())
    )


def _build_why_choose_summary_content(atoms: list[Phase12ChunkRecord]) -> str:
    lines = ["Why Choose earKART"]
    seen: set[str] = set()
    for atom in sorted(atoms, key=lambda item: item.section_path):
        text = atom.content.strip()
        if not text or text in seen:
            continue
        seen.add(text)
        lines.append(text)
    return "\n".join(lines).strip()


def build_why_choose_summary_chunk(
    *,
    merged_doc: Phase16CorpusDocument,
    atoms: list[Phase12ChunkRecord],
    chunk_index: int,
) -> Phase12ChunkRecord | None:
    if not atoms:
        return None

    content = _build_why_choose_summary_content(atoms)
    section_path = list(atoms[0].section_path)
    if not any("why choose" in part.lower() for part in section_path):
        section_path = [merged_doc.title, "5. Why Choose earKART"]

    chunk_id = deterministic_chunk_id(
        kb_dataset_version=V3_1_KB_DATASET_VERSION,
        document_id=merged_doc.record.document_id,
        document_version=merged_doc.record.document_version,
        chunk_index=chunk_index,
        content=content,
    )
    base = Phase12ChunkRecord(
        chunk_id=chunk_id,
        document_id=merged_doc.record.document_id,
        document_version=merged_doc.record.document_version,
        document_type=merged_doc.record.document_type,
        chunking_algorithm_version=V3_1_ALGORITHM_VERSION,
        content=content,
        embedding_input="",
        embedding_input_hash="",
        token_count=max(1, len(content.split())),
        page_number=atoms[0].page_number,
        section_path=section_path,
        parent_section=atoms[0].parent_section or merged_doc.title,
        subsection="5. Why Choose earKART",
        content_type=SUMMARY_CONTENT_TYPE,
        source_file_hash=merged_doc.record.file_hash,
        extraction_method=atoms[0].extraction_method,
        ocr_used=atoms[0].ocr_used,
        created_at=datetime.now(timezone.utc),
        split_method="v3_summary",
    )
    return attach_v3_embedding_metadata(
        base,
        title=merged_doc.title,
        embedding_model=settings.embedding_model,
        embedding_model_revision=settings.embedding_model_revision,
        embedding_dimension=settings.embedding_dimension,
    )


def augment_corpus_with_v3_1_summary(corpus: Phase16Corpus) -> Phase16Corpus:
    documents: list[Phase16CorpusDocument] = []
    for doc in corpus.documents:
        if doc.label != "merged":
            documents.append(doc)
            continue

        atoms = [chunk for chunk in doc.chunks if _is_why_choose_atom(chunk)]
        summary = build_why_choose_summary_chunk(
            merged_doc=doc,
            atoms=atoms,
            chunk_index=len(doc.chunks),
        )
        extra = [summary] if summary else []
        documents.append(
            Phase16CorpusDocument(
                label=doc.label,
                filename=doc.filename,
                record=doc.record,
                title=doc.title,
                chunks=[*doc.chunks, *extra],
            )
        )
    return Phase16Corpus(
        documents=documents,
        suppressed_duplicates=corpus.suppressed_duplicates,
        source_pdfs=corpus.source_pdfs,
    )


def load_phase16_v3_1_corpus(*, storage_dir=None) -> Phase16Corpus:
    base = load_phase16_v3_corpus(storage_dir=storage_dir)
    return augment_corpus_with_v3_1_summary(base)
