"""Fast read-only retrieval root-cause audit (13 queries, vector-only)."""

from __future__ import annotations

from datetime import datetime, timezone
from typing import Any

from app.kb.chunking.models import ProductionChunkRecord
from app.kb.embedding.config import EmbeddingConfig
from app.kb.embedding.provider import EmbeddingProvider
from app.kb.evaluation.earkart_13_v1 import EARKART_13_CASES
from app.kb.evaluation.retrieval_diagnostic.corpus_check import (
    resolve_expected_documents,
    resolve_expected_status,
)
from app.kb.evaluation.retrieval_diagnostic.lightweight_root_cause import classify_lightweight_root_cause
from app.kb.evaluation.retrieval_diagnostic.ranking import (
    best_rank_for_documents,
    best_rank_for_ids,
    top_competing_chunks,
)
from app.kb.vector.search import VectorSearchService


def format_hit_metadata(
    *,
    rank: int,
    item: dict[str, Any],
    chunk: ProductionChunkRecord | None,
) -> dict[str, Any]:
    return {
        "rank": rank,
        "chunk_id": item["chunk_id"],
        "document_id": item.get("document_id") or (chunk.document_id if chunk else None),
        "title": item.get("title") or (chunk.title if chunk else ""),
        "canonical_url": item.get("canonical_url") or (chunk.canonical_url if chunk else ""),
        "document_type": item.get("document_type") or (chunk.document_type.value if chunk else ""),
        "section_path": item.get("section_path") or (chunk.section_path if chunk else []),
        "similarity": item.get("similarity"),
        "token_count": chunk.token_count if chunk else item.get("token_count", 0),
    }


def format_hit_with_text(
    *,
    rank: int,
    item: dict[str, Any],
    chunk: ProductionChunkRecord | None,
) -> dict[str, Any]:
    meta = format_hit_metadata(rank=rank, item=item, chunk=chunk)
    meta["text"] = item.get("content") or (chunk.content if chunk else "")
    return meta


def top_competing_document(hits: list[dict[str, Any]], expected_document_ids: set[str]) -> tuple[str | None, float | None]:
    for hit in hits:
        if hit.get("document_id") in expected_document_ids:
            continue
        return hit.get("title") or hit.get("document_id"), hit.get("similarity")
    return None, None


def run_lightweight_audit(
    *,
    config: EmbeddingConfig,
    chunks: list[ProductionChunkRecord],
    search: VectorSearchService,
    provider: EmbeddingProvider | None = None,
    top_k: int = 100,
) -> dict[str, Any]:
    chunk_by_id = {chunk.chunk_id: chunk for chunk in chunks}
    known_chunk_ids = set(chunk_by_id)
    embedding_provider = provider or search.provider
    store = search.store
    vector_count = store.count(embedding_version=config.embedding_version)
    config_issues = _retrieval_config_issues(vector_count, len(chunks))

    questions: list[dict[str, Any]] = []
    for case in EARKART_13_CASES:
        expected = resolve_expected_status(
            case=case,
            chunks=chunks,
            chunk_by_id=chunk_by_id,
            known_chunk_ids=known_chunk_ids,
        )
        expected_chunk_ids = [
            chunk_id for chunk_id in expected["expected_chunk_ids"] if chunk_id in known_chunk_ids
        ]
        expected_document_ids = resolve_expected_documents(case, chunk_by_id)

        query_vector = embedding_provider.embed_queries([case["query"]])[0]
        raw_hits = store.search(
            query_vector,
            top_k=top_k,
            embedding_version=config.embedding_version,
        )

        hits_meta = [
            format_hit_metadata(
                rank=index,
                item=item,
                chunk=chunk_by_id.get(item["chunk_id"]),
            )
            for index, item in enumerate(raw_hits[:top_k], start=1)
        ]
        top_10 = [
            format_hit_with_text(
                rank=index,
                item=item,
                chunk=chunk_by_id.get(item["chunk_id"]),
            )
            for index, item in enumerate(raw_hits[:10], start=1)
        ]

        ranked_ids = [item["chunk_id"] for item in hits_meta]
        chunk_rank = best_rank_for_ids(ranked_ids, set(expected_chunk_ids))
        chunk_similarity = None
        if chunk_rank is not None:
            chunk_similarity = hits_meta[chunk_rank - 1].get("similarity")

        document_rank, document_similarity = best_rank_for_documents(
            hits_meta,
            set(expected_document_ids),
        )

        diagnosis = classify_lightweight_root_cause(
            case=case,
            expected_status=expected["status"],
            mapping_issues=expected.get("mapping_issues", []),
            chunk_rank=chunk_rank,
            chunk_similarity=chunk_similarity,
            document_rank=document_rank,
            document_similarity=document_similarity,
            top_hits=hits_meta,
            expected_chunk_ids=set(expected_chunk_ids),
            expected_document_ids=set(expected_document_ids),
            retrieval_config_issues=config_issues,
        )

        competing_title, competing_sim = top_competing_document(hits_meta, set(expected_document_ids))

        questions.append(
            {
                "id": case["id"],
                "question": case["query"],
                "expected_chunk_ids": expected_chunk_ids,
                "expected_document_ids": expected_document_ids,
                "expected_chunk_analysis": {
                    "best_rank": chunk_rank,
                    "best_similarity": chunk_similarity,
                },
                "expected_document_analysis": {
                    "best_rank": document_rank,
                    "best_similarity": document_similarity,
                },
                "top_100_metadata": hits_meta,
                "top_10": top_10,
                "top_competing_document": competing_title,
                "top_competing_similarity": competing_sim,
                "diagnosis": diagnosis,
            }
        )

    summary = _build_summary(questions)
    return {
        "report_version": "1.0",
        "question_count": len(EARKART_13_CASES),
        "generated_at": datetime.now(timezone.utc).isoformat(),
        "index": {
            "chunks": vector_count,
            "model": config.model,
            "revision": config.model_revision,
            "dimension": config.dimension,
            "similarity": "cosine",
            "vector_table": config.vector_table,
        },
        "summary": summary,
        "questions": [_public_question(item) for item in questions],
        "final_verdict": _final_verdict(questions),
        "_internal_questions": questions,
    }


def _public_question(item: dict[str, Any]) -> dict[str, Any]:
    """JSON output excludes full top-100 metadata per spec."""
    return {
        "id": item["id"],
        "question": item["question"],
        "expected_chunk_ids": item["expected_chunk_ids"],
        "expected_document_ids": item["expected_document_ids"],
        "expected_chunk_analysis": item["expected_chunk_analysis"],
        "expected_document_analysis": item["expected_document_analysis"],
        "top_10": item["top_10"],
        "diagnosis": item["diagnosis"],
    }


def _build_summary(questions: list[dict[str, Any]]) -> dict[str, Any]:
    def in_top(threshold: int) -> int:
        return sum(
            1
            for item in questions
            if item["expected_chunk_analysis"]["best_rank"] is not None
            and item["expected_chunk_analysis"]["best_rank"] <= threshold
        )

    def doc_in_top(threshold: int) -> int:
        return sum(
            1
            for item in questions
            if item["expected_document_analysis"]["best_rank"] is not None
            and item["expected_document_analysis"]["best_rank"] <= threshold
        )

    root_causes = {
        "CORPUS_GAP": 0,
        "EVALUATION_MAPPING": 0,
        "CHUNKING": 0,
        "RETRIEVAL_COMPETITION": 0,
        "QUERY_MISMATCH": 0,
        "EMBEDDING_REPRESENTATION": 0,
        "RETRIEVAL_CONFIGURATION": 0,
        "UNRESOLVED": 0,
    }
    for item in questions:
        primary = item["diagnosis"]["primary_root_cause"]
        root_causes[primary] = root_causes.get(primary, 0) + 1

    return {
        "top10_expected_chunk": in_top(10),
        "top20_expected_chunk": in_top(20),
        "top50_expected_chunk": in_top(50),
        "top100_expected_chunk": in_top(100),
        "top10_expected_document": doc_in_top(10),
        "root_causes": root_causes,
    }


def _retrieval_config_issues(vector_count: int, manifest_count: int) -> list[str]:
    if vector_count != manifest_count:
        return [
            f"Production vector count ({vector_count}) differs from manifest count ({manifest_count})."
        ]
    return []


def _final_verdict(questions: list[dict[str, Any]]) -> str:
    failed = [
        item
        for item in questions
        if item["expected_chunk_analysis"]["best_rank"] is None
        or item["expected_chunk_analysis"]["best_rank"] > 10
    ]
    if not failed:
        return "ROOT_CAUSE_IDENTIFIED"
    unresolved = sum(1 for item in failed if item["diagnosis"]["primary_root_cause"] == "UNRESOLVED")
    if unresolved >= max(2, len(failed) // 2):
        return "NEEDS_MORE_EVIDENCE"
    return "ROOT_CAUSE_IDENTIFIED"


def format_lightweight_text_report(report: dict[str, Any], *, full_questions: list[dict[str, Any]] | None = None) -> str:
    summary = report["summary"]
    rows = full_questions or report["questions"]
    lines = [
        "Earkart Lightweight Retrieval Root-Cause Audit (Phase 11.7)",
        f"Generated: {report.get('generated_at', '')}",
        f"Index: {report['index']['chunks']} vectors | {report['index']['model']}",
        "",
        "Summary table",
        "Question | Exp chunk rank | Exp doc rank | Top competing doc | Top sim | Root cause | Conf",
        "---------|----------------|--------------|-------------------|---------|------------|-----",
    ]
    for item in rows:
        chunk_rank = item["expected_chunk_analysis"]["best_rank"]
        doc_rank = item["expected_document_analysis"]["best_rank"]
        diagnosis = item["diagnosis"]
        competing = item.get("top_competing_document") or "-"
        competing_sim = item.get("top_competing_similarity")
        lines.append(
            f"{item['id']} | {chunk_rank or '>100'} | {doc_rank or '>100'} | {competing!r} | "
            f"{competing_sim if competing_sim is not None else '-'} | "
            f"{diagnosis['primary_root_cause']} | {diagnosis['confidence']}"
        )

    lines.extend(["", "Failed query explanations"])
    for item in rows:
        chunk_rank = item["expected_chunk_analysis"]["best_rank"]
        if chunk_rank is not None and chunk_rank <= 10:
            continue
        lines.extend(
            [
                "",
                f"{item['id']}: {item['question']}",
                f"  Root cause: {item['diagnosis']['primary_root_cause']} (confidence={item['diagnosis']['confidence']})",
                f"  Expected chunk rank: {chunk_rank or 'not in top-100'}",
                f"  Expected document rank: {item['expected_document_analysis']['best_rank'] or 'not in top-100'}",
            ]
        )
        for note in item["diagnosis"]["evidence"]:
            lines.append(f"  - {note}")
        lines.append(f"  Recommendation: {item['diagnosis']['recommendation']}")

    lines.extend(
        [
            "",
            "EAR_KART_RETRIEVAL_ROOT_CAUSE_13",
            "",
            f"Top-10 expected chunks: {summary['top10_expected_chunk']}/13",
            f"Top-20 expected chunks: {summary['top20_expected_chunk']}/13",
            f"Top-50 expected chunks: {summary['top50_expected_chunk']}/13",
            f"Top-100 expected chunks: {summary['top100_expected_chunk']}/13",
            "",
            "Root causes:",
        ]
    )
    for label, count in summary["root_causes"].items():
        lines.append(f"{label}: {count}")
    lines.extend(["", f"Final verdict: {report['final_verdict']}"])
    return "\n".join(lines)
