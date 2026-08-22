"""Production embedding gate orchestration."""

from __future__ import annotations

from collections import Counter, defaultdict
from dataclasses import dataclass, field
from datetime import datetime, timezone

from app.kb.chunking.config import ChunkingConfig
from app.kb.chunking.eligibility import ChunkEligibilityStatus, ReviewCategory, classify_chunk_eligibility
from app.kb.chunking.fidelity import (
    RETURNS_DOC_ID,
    RADIUS_M16_DOC_ID,
    check_faq_atomicity,
    check_policy_fidelity,
    check_product_spec_fidelity,
    check_prospectus_fidelity,
)
from app.kb.chunking.models import ChunkRecord, ProductionChunkRecord
from app.kb.chunking.question_coverage import evaluate_question_coverage
from app.kb.chunking.semantic_units import effective_document_type
from app.kb.chunking.service import ChunkingDryRunService
from app.kb.chunking.storage import ChunkStore
from app.kb.chunking.validators import validate_chunks
from app.kb.integrity.version import KB_DATASET_VERSION
from app.kb.retrieval.models import RetrievalDocument
from app.kb.retrieval.storage import RetrievalStore

PROSPECTUS_IDS = (
    "01e76cb2-317f-509a-b8b5-f473f5bd5400",
    "8bd45b19-cb3d-5e56-ad5a-b67f1af7d2f9",
)
EQFY_DOC_ID = "a0fa1902-0871-5f43-9924-69da7d6036c2"


@dataclass
class ClassifiedChunk:
    chunk: ChunkRecord
    eligibility_status: ChunkEligibilityStatus
    reasons: list[str]
    flags: list[str]
    review_category: ReviewCategory | None = None


@dataclass
class EmbeddingGateResult:
    kb_dataset_version: str
    candidate_chunks: list[ChunkRecord]
    production_chunks: list[ProductionChunkRecord]
    classified: list[ClassifiedChunk]
    excluded_records: list[dict]
    review_records: list[dict]
    engine_suppressed: list[dict]
    gate_passed: bool
    ready_for_embeddings: bool
    statistics: dict
    validation: dict
    p0_findings: list[dict] = field(default_factory=list)
    p1_findings: list[dict] = field(default_factory=list)
    p2_findings: list[dict] = field(default_factory=list)
    question_coverage: list[dict] = field(default_factory=list)


def _structured_text(retrieval: RetrievalDocument) -> str:
    parts: list[str] = []

    def walk(nodes) -> None:
        for node in nodes:
            node_type = node.type
            if node_type == "section":
                if node.heading:
                    parts.append(str(node.heading))
                walk(node.children or [])
            elif node_type == "heading":
                parts.append(str(node.text))
            elif node_type == "paragraph":
                parts.append(str(node.text))
            elif node_type == "list":
                parts.extend(str(item) for item in node.items or [])
            elif node_type == "table":
                if node.headers:
                    parts.append(" | ".join(node.headers))
                for row in node.rows or []:
                    parts.append(" | ".join(row))

    content = retrieval.structured_content
    if content.pages:
        for page in content.pages:
            walk(page.blocks)
    else:
        walk(content.children)
    return "\n".join(part for part in parts if part.strip())


def investigate_p0_fidelity(*, chunk: ChunkRecord, retrieval: RetrievalDocument) -> dict:
    body = chunk.content.split("\n\n")[-1].strip() if "\n\n" in chunk.content else chunk.content.strip()
    retrieval_norm = " ".join(retrieval.retrieval_text.split()).lower()
    structured_norm = " ".join(_structured_text(retrieval).split()).lower()
    body_norm = " ".join(body.split()).lower()
    words = [w for w in body_norm.split() if len(w) > 2]

    in_retrieval = sum(1 for w in words if w in retrieval_norm) / max(1, len(words))
    in_structured = sum(1 for w in words if w in structured_norm) / max(1, len(words))

    if in_structured >= 0.65 and in_retrieval < 0.65:
        classification = "FALSE_POSITIVE_FORMATTING"
        detail = "Chunk content is present in structured_content but missing/truncated in retrieval_text"
    elif in_structured >= 0.65:
        classification = "FALSE_POSITIVE_CONTEXT_INHERITANCE"
        detail = "Chunk body matches structured source; audit heuristic used incomplete retrieval_text"
    elif in_structured < 0.65 and in_retrieval < 0.65:
        classification = "REAL_CONTENT_ADDITION"
        detail = "Chunk content not found in structured_content or retrieval_text"
    else:
        classification = "FALSE_POSITIVE_OCR_VARIATION"
        detail = "Word overlap variance from OCR/formatting differences"

    return {
        "document_id": retrieval.document_id,
        "chunk_id": chunk.chunk_id,
        "classification": classification,
        "detail": detail,
        "retrieval_overlap": round(in_retrieval, 3),
        "structured_overlap": round(in_structured, 3),
        "chunk_preview": body[:200],
    }


def classify_candidates(chunks: list[ChunkRecord]) -> list[ClassifiedChunk]:
    seen_strict: dict[str, ChunkRecord] = {}
    classified: list[ClassifiedChunk] = []
    for chunk in chunks:
        status, reasons, flags, review_category = classify_chunk_eligibility(chunk, seen_strict=seen_strict)
        classified.append(
            ClassifiedChunk(
                chunk=chunk,
                eligibility_status=status,
                reasons=reasons,
                flags=flags,
                review_category=review_category,
            )
        )
    return classified


class EmbeddingGateService:
    def __init__(
        self,
        retrieval_store: RetrievalStore,
        chunk_store: ChunkStore,
        *,
        kb_dataset_version: str = KB_DATASET_VERSION,
        config: ChunkingConfig | None = None,
    ) -> None:
        self.retrieval_store = retrieval_store
        self.chunk_store = chunk_store
        self.kb_dataset_version = kb_dataset_version
        self.config = config or ChunkingConfig()
        self.dry_run = ChunkingDryRunService(retrieval_store, kb_dataset_version=kb_dataset_version, config=config)

    def run(self, *, write_production: bool = True) -> EmbeddingGateResult:
        dry = self.dry_run.run()
        candidate_chunks = dry.chunks
        engine_suppressed = dry.suppressed_chunks
        classified = classify_candidates(candidate_chunks)

        production: list[ProductionChunkRecord] = []
        excluded_records: list[dict] = []
        review_records: list[dict] = []

        for item in classified:
            chunk = item.chunk
            record_base = {
                "chunk_id": chunk.chunk_id,
                "document_id": chunk.document_id,
                "document_version": chunk.document_version,
                "kb_dataset_version": chunk.kb_dataset_version,
                "website": chunk.website,
                "canonical_url": chunk.canonical_url,
                "source_url": chunk.source_url,
                "section_path": chunk.section_path,
                "token_count": chunk.token_count,
                "split_method": chunk.split_method.value,
                "content_preview": chunk.content[:200],
                "provenance": chunk.provenance.model_dump(),
                "page_number": chunk.page_number,
            }
            if item.eligibility_status == ChunkEligibilityStatus.EMBED_READY:
                production.append(
                    ProductionChunkRecord(
                        **chunk.model_dump(),
                        eligibility_status=ChunkEligibilityStatus.EMBED_READY.value,
                    )
                )
            elif item.eligibility_status == ChunkEligibilityStatus.REVIEW:
                review_records.append(
                    {
                        **record_base,
                        "eligibility_status": item.eligibility_status.value,
                        "review_category": (item.review_category or ReviewCategory.OTHER).value,
                        "reasons": item.reasons,
                        "flags": item.flags,
                    }
                )
            else:
                excluded_records.append(
                    {
                        **record_base,
                        "eligibility_status": item.eligibility_status.value,
                        "classification": item.reasons[0] if item.reasons else "DO_NOT_EMBED",
                        "reasons": item.reasons,
                        "flags": item.flags,
                    }
                )

        for suppressed in engine_suppressed:
            excluded_records.append(
                {
                    "chunk_id": None,
                    "document_id": suppressed.get("document_id"),
                    "eligibility_status": ChunkEligibilityStatus.DO_NOT_EMBED.value,
                    "classification": suppressed.get("suppression_reason"),
                    "reasons": [suppressed.get("detail", "")],
                    "content_preview": suppressed.get("content_preview"),
                    "provenance": suppressed.get("provenance"),
                    "source": "engine_pre_filter",
                }
            )

        production_validation = validate_chunks(
            [ChunkRecord(**p.model_dump()) for p in production],
            self.config,
        )

        p0_findings: list[dict] = []
        eqfy_retrieval = self.retrieval_store.read(self.kb_dataset_version, "earkart.in", EQFY_DOC_ID)
        eqfy_items = [c for c in classified if c.chunk.document_id == EQFY_DOC_ID]
        if eqfy_retrieval and eqfy_items:
            p0_findings.append(
                investigate_p0_fidelity(chunk=eqfy_items[0].chunk, retrieval=eqfy_retrieval)
            )

        review_excluded_chunks = [
            item.chunk
            for item in classified
            if item.eligibility_status != ChunkEligibilityStatus.EMBED_READY
        ]
        coverage = evaluate_question_coverage(
            [ChunkRecord(**p.model_dump()) for p in production],
            review_excluded=review_excluded_chunks,
        )
        policy_ok = check_policy_fidelity([ChunkRecord(**p.model_dump()) for p in production])
        product_ok = check_product_spec_fidelity([ChunkRecord(**p.model_dump()) for p in production])
        prospectus_ok = check_prospectus_fidelity(
            [ChunkRecord(**p.model_dump()) for p in production],
            max_chunk_tokens=self.config.max_chunk_tokens,
        )
        faq_ok = check_faq_atomicity([ChunkRecord(**p.model_dump()) for p in production])

        qc08 = next(r for r in coverage["results"] if r["id"] == "QC08")
        qc25 = next(r for r in coverage["results"] if r["id"] == "QC25")

        p1_findings: list[dict] = []
        non_ready_in_production = sum(
            1 for item in classified
            if item.eligibility_status != ChunkEligibilityStatus.EMBED_READY
            and item.chunk.chunk_id in {p.chunk_id for p in production}
        )
        if non_ready_in_production:
            p1_findings.append({"category": "non_embed_ready_in_production", "count": non_ready_in_production})
        if coverage.get("coverage_failures"):
            p1_findings.append(
                {
                    "category": "question_coverage_failure",
                    "count": coverage["coverage_failures"],
                    "detail": "Knowledge exists only in REVIEW/DO_NOT_EMBED chunks",
                }
            )
        failed_q = [r for r in coverage["results"] if not r["passed"]]
        if failed_q:
            p1_findings.append(
                {"category": "question_coverage_gap", "count": len(failed_q), "ids": [q["id"] for q in failed_q[:10]]}
            )

        p2_findings: list[dict] = []
        if not qc08["single_production_chunk"]:
            p2_findings.append({"category": "QC08_address_split", "detail": qc08})
        if not qc25["single_production_chunk"]:
            p2_findings.append(
                {
                    "category": "QC25_drhp_public_issue_split",
                    "detail": qc25,
                    "verdict": "VALID_SPLIT" if qc25["knowledge_present_in_production"] else "INVALID_SPLIT",
                    "explanation": (
                        "DRHP and Public Issue appear in distinct prospectus sections; "
                        "separate EMBED_READY chunks preserve section context."
                        if qc25["knowledge_present_in_production"]
                        else "Required prospectus terms not co-located in production corpus."
                    ),
                }
            )

        real_p0 = [f for f in p0_findings if f["classification"].startswith("REAL_")]
        token_violations = sum(1 for p in production if p.token_count > self.config.max_chunk_tokens)
        empty_chunks = sum(1 for p in production if not p.content.strip())
        duplicate_ids = len(production) - len({p.chunk_id for p in production})
        review_in_manifest = sum(
            1 for p in production if p.eligibility_status != ChunkEligibilityStatus.EMBED_READY.value
        )

        gate_passed = (
            not real_p0
            and not p1_findings
            and token_violations == 0
            and empty_chunks == 0
            and duplicate_ids == 0
            and review_in_manifest == 0
            and production_validation.valid
            and all(p.eligibility_status == ChunkEligibilityStatus.EMBED_READY.value for p in production)
            and policy_ok.passed
            and product_ok.passed
            and prospectus_ok.passed
            and faq_ok.passed
            and coverage["all_passed"]
            and not qc08.get("inappropriate_split")
        )

        statistics = self._statistics(candidate_chunks, production, classified, review_records, excluded_records)

        self._write_review_manifest(review_records)

        if gate_passed and write_production:
            self._write_production(production, statistics)

        return EmbeddingGateResult(
            kb_dataset_version=self.kb_dataset_version,
            candidate_chunks=candidate_chunks,
            production_chunks=production,
            classified=classified,
            excluded_records=excluded_records,
            review_records=review_records,
            engine_suppressed=engine_suppressed,
            gate_passed=gate_passed,
            ready_for_embeddings=gate_passed,
            statistics=statistics,
            validation={
                "production_valid": production_validation.valid,
                "token_violations": token_violations,
                "empty_chunks": empty_chunks,
                "duplicate_ids": duplicate_ids,
            },
            p0_findings=p0_findings,
            p1_findings=p1_findings,
            p2_findings=p2_findings,
            question_coverage=coverage["results"],
        )

    def _statistics(
        self,
        candidate: list[ChunkRecord],
        production: list[ProductionChunkRecord],
        classified: list[ClassifiedChunk],
        review_records: list[dict],
        excluded_records: list[dict],
    ) -> dict:
        by_status = Counter(c.eligibility_status.value for c in classified)
        by_type: dict[str, Counter] = defaultdict(Counter)
        by_source: dict[str, Counter] = defaultdict(Counter)
        by_website: dict[str, Counter] = defaultdict(Counter)
        for item in classified:
            doc_type = effective_document_type(
                item.chunk.document_type, item.chunk.title, item.chunk.canonical_url
            )
            by_type[doc_type][item.eligibility_status.value] += 1
            by_source[item.chunk.source_type.value][item.eligibility_status.value] += 1
            by_website[item.chunk.website][item.eligibility_status.value] += 1

        return {
            "candidate_total": len(candidate),
            "production_total": len(production),
            "review_total": len(review_records),
            "excluded_total": len(excluded_records),
            "engine_suppressed_total": len([e for e in excluded_records if e.get("source") == "engine_pre_filter"]),
            "candidate_by_status": dict(by_status),
            "production_percentage": round(100 * len(production) / max(1, len(candidate)), 2),
            "by_document_type": {k: dict(v) for k, v in by_type.items()},
            "by_source_type": {k: dict(v) for k, v in by_source.items()},
            "by_website": {k: dict(v) for k, v in by_website.items()},
        }

    def _write_production(self, production: list[ProductionChunkRecord], statistics: dict) -> None:
        for chunk in production:
            self.chunk_store.write_production_chunk(chunk)
        manifest = {
            "generated_at": datetime.now(timezone.utc).isoformat(),
            "kb_dataset_version": self.kb_dataset_version,
            "eligibility_status": "EMBED_READY",
            "total_chunks": len(production),
            "statistics": statistics,
            "chunks": [
                {
                    "chunk_id": c.chunk_id,
                    "document_id": c.document_id,
                    "document_version": c.document_version,
                    "website": c.website,
                    "chunk_index": c.chunk_index,
                    "token_count": c.token_count,
                    "canonical_url": c.canonical_url,
                    "eligibility_status": c.eligibility_status,
                }
                for c in production
            ],
        }
        self.chunk_store.write_manifest(self.kb_dataset_version, manifest)

    def _write_review_manifest(self, review_records: list[dict]) -> None:
        by_category: dict[str, int] = defaultdict(int)
        for record in review_records:
            by_category[record.get("review_category", "OTHER")] += 1
        self.chunk_store.write_review_manifest(
            self.kb_dataset_version,
            {
                "generated_at": datetime.now(timezone.utc).isoformat(),
                "kb_dataset_version": self.kb_dataset_version,
                "total_review_chunks": len(review_records),
                "by_category": dict(by_category),
                "chunks": review_records,
            },
        )
