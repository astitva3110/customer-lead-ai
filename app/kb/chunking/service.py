"""Dry-run chunking orchestration over retrieval manifest."""

from __future__ import annotations

import json
from dataclasses import dataclass, field
from pathlib import Path

from pydantic import TypeAdapter

from app.kb.chunking.config import ChunkingConfig
from app.kb.chunking.engine import ChunkingEngine
from app.kb.chunking.fidelity import (
    check_faq_atomicity,
    check_policy_fidelity,
    check_product_spec_fidelity,
    check_prospectus_fidelity,
)
from app.kb.chunking.models import ChunkRecord
from app.kb.chunking.semantic_units import effective_document_type
from app.kb.chunking.validators import ChunkValidationIssue, validate_chunks
from app.kb.integrity.version import KB_DATASET_VERSION
from app.kb.retrieval.models import RetrievalDocument
from app.kb.retrieval.storage import RetrievalStore


@dataclass
class DocumentChunkResult:
    document_id: str
    canonical_url: str
    document_type: str
    chunks: list[ChunkRecord] = field(default_factory=list)
    error: str | None = None


@dataclass
class DryRunResult:
    kb_dataset_version: str
    eligible_documents: int
    documents_processed: int
    documents_with_errors: int
    total_chunks: int
    chunks: list[ChunkRecord] = field(default_factory=list)
    document_results: list[DocumentChunkResult] = field(default_factory=list)
    validation_issues: list[ChunkValidationIssue] = field(default_factory=list)
    suppressed_chunks: list[dict] = field(default_factory=list)


class ChunkingDryRunService:
    """Generate in-memory chunks for eligible manifest documents only."""

    def __init__(
        self,
        retrieval_store: RetrievalStore,
        *,
        kb_dataset_version: str = KB_DATASET_VERSION,
        config: ChunkingConfig | None = None,
    ) -> None:
        self.retrieval_store = retrieval_store
        self.kb_dataset_version = kb_dataset_version
        self.config = config or ChunkingConfig()
        self.engine = ChunkingEngine(self.config)

    def load_manifest(self) -> list[dict]:
        manifest_path = self.retrieval_store.manifest_path(self.kb_dataset_version)
        payload = json.loads(manifest_path.read_text(encoding="utf-8"))
        return payload["documents"]

    def load_retrieval_document(self, website: str, document_id: str) -> RetrievalDocument | None:
        return self.retrieval_store.read(self.kb_dataset_version, website, document_id)

    def run(self) -> DryRunResult:
        manifest_entries = self.load_manifest()
        all_chunks: list[ChunkRecord] = []
        all_suppressed: list[dict] = []
        document_results: list[DocumentChunkResult] = []
        validation_issues: list[ChunkValidationIssue] = []
        errors = 0

        for entry in manifest_entries:
            document_id = entry["document_id"]
            website = entry["website"]
            doc_type = entry.get("document_type", "webpage")
            try:
                retrieval = self.load_retrieval_document(website, document_id)
                if retrieval is None:
                    raise FileNotFoundError(f"Missing retrieval document for {document_id}")
                chunks, suppressed = self.engine.chunk_document_with_suppressed(retrieval)
                validation = validate_chunks(chunks, self.config)
                validation_issues.extend(validation.issues)
                all_chunks.extend(chunks)
                all_suppressed.extend(suppressed)
                document_results.append(
                    DocumentChunkResult(
                        document_id=document_id,
                        canonical_url=entry["canonical_url"],
                        document_type=doc_type,
                        chunks=chunks,
                    )
                )
            except Exception as exc:  # noqa: BLE001 - dry-run records per-document failures
                errors += 1
                document_results.append(
                    DocumentChunkResult(
                        document_id=document_id,
                        canonical_url=entry.get("canonical_url", ""),
                        document_type=doc_type,
                        error=str(exc),
                    )
                )

        return DryRunResult(
            kb_dataset_version=self.kb_dataset_version,
            eligible_documents=len(manifest_entries),
            documents_processed=len(manifest_entries) - errors,
            documents_with_errors=errors,
            total_chunks=len(all_chunks),
            chunks=all_chunks,
            document_results=document_results,
            validation_issues=validation_issues,
            suppressed_chunks=all_suppressed,
        )


def chunks_to_json(chunks: list[ChunkRecord]) -> list[dict]:
    adapter = TypeAdapter(list[ChunkRecord])
    return adapter.dump_python(chunks, mode="json")
