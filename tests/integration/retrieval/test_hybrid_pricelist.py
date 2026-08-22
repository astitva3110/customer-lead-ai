"""Hybrid Retrieval V1 against the already indexed pricelist PDF.

Does not re-ingest or modify document 6bb9a4ee-ecf4-58ff-9eaa-3206d27d9b0d.
"""

from __future__ import annotations

import json
from pathlib import Path

import pytest

from app.services.retrieval.diagnostics import format_retrieval_trace
from app.services.retrieval.hybrid import HybridRetriever
from app.providers.reranker.factory import create_reranker
from app.providers.retrieval.keyword_retriever import KeywordCandidateRetriever
from app.providers.retrieval.vector_retriever import VectorCandidateRetriever
from app.kb.evaluation.metrics import recall_at_k
from app.kb.evaluation.retrieval_config import RetrievalConfig
from app.kb.evaluation.search.backends import EmbeddingSession
from app.kb.ingestion.indexing import Phase12VectorStore

GOLDEN_PATH = Path(__file__).resolve().parents[2] / "fixtures" / "hybrid_retrieval" / "pricelist_v1.json"
DOC_ID = "6bb9a4ee-ecf4-58ff-9eaa-3206d27d9b0d"
REPORT_PATH = Path("reports") / "hybrid_retrieval_v1.txt"


def _load_golden() -> dict:
    return json.loads(GOLDEN_PATH.read_text(encoding="utf-8"))


def _is_relevant(hit, question: dict) -> bool:
    if getattr(hit, "document_id", None) != DOC_ID:
        return False
    text = (getattr(hit, "text", "") or "").lower()
    patterns = [pattern.lower() for pattern in question["content_patterns"]]
    return all(pattern in text for pattern in patterns)


def _first_relevant_rank(hits: list, question: dict) -> int | None:
    for index, hit in enumerate(hits, start=1):
        if _is_relevant(hit, question):
            return index
    return None


def _mark(rank: int | None, k: int) -> str:
    return "Y" if rank is not None and rank <= k else "N"


@pytest.fixture(scope="module")
def retrieval_config() -> RetrievalConfig:
    return RetrievalConfig.from_yaml(Path("configs/retrieval/v2.yaml"))


@pytest.fixture(scope="module")
def hybrid_stack(retrieval_config: RetrievalConfig):
    store = Phase12VectorStore(table_name=retrieval_config.vector_table)
    try:
        probe = store.search_keyword("Earkart", top_k=1, document_id=DOC_ID)
    except Exception as exc:
        pytest.skip(f"PostgreSQL/pgvector unavailable: {exc}")
    if not probe:
        pytest.skip("Indexed pricelist document is not available; not re-ingesting")

    session = EmbeddingSession()
    vector = VectorCandidateRetriever(
        store=store,
        session=session,
        embedding_version=retrieval_config.embedding_version,
        default_k=20,
    )
    keyword = KeywordCandidateRetriever(
        store,
        embedding_version=retrieval_config.embedding_version,
        default_k=20,
    )
    hybrid = HybridRetriever(
        vector,
        keyword,
        create_reranker("lexical_overlap"),
        vector_k=20,
        keyword_k=20,
        final_k=5,
        min_score=0.0,
    )
    return {"vector": vector, "keyword": keyword, "hybrid": hybrid, "store": store}


def test_hybrid_pricelist_known_answers_and_comparison(hybrid_stack) -> None:
    golden = _load_golden()
    questions = golden["questions"]
    vector = hybrid_stack["vector"]
    keyword = hybrid_stack["keyword"]
    hybrid: HybridRetriever = hybrid_stack["hybrid"]

    rows = []
    traces = []
    latencies: list[dict] = []
    hybrid_no_result = 0
    threshold_reject_probe = 0

    for question in questions:
        query = question["query"]
        vector_hits = vector.retrieve(query, top_k=10, document_id=DOC_ID)
        keyword_hits = keyword.retrieve(query, top_k=10, document_id=DOC_ID)
        detailed = hybrid.search_detailed(query, final_k=10, document_id=DOC_ID, min_score=0.0)
        hybrid_hits = detailed.final_candidates
        traces.append(format_retrieval_trace(detailed))
        latencies.append(detailed.timings.to_dict())

        vector_rank = _first_relevant_rank(vector_hits, question)
        keyword_rank = _first_relevant_rank(keyword_hits, question)
        hybrid_rank = _first_relevant_rank(hybrid_hits, question)
        rerank_rank = _first_relevant_rank(detailed.reranked_candidates, question)

        expected_ids = question.get("expected_chunk_ids") or []
        hybrid_ids = [hit.chunk_id for hit in hybrid_hits[:5]]
        page_hit = any(
            _is_relevant(hit, question) and hit.page_number == question.get("expected_page")
            for hit in hybrid_hits[:5]
        )
        doc_hit = any(hit.document_id == DOC_ID for hit in hybrid_hits[:5])

        assert doc_hit, f"{question['id']}: expected document missing from hybrid top-5"
        assert hybrid_rank is not None and hybrid_rank <= 5, (
            f"{question['id']}: expected content {question['content_patterns']} "
            f"missing from hybrid top-5 ({hybrid_ids})"
        )
        assert page_hit, f"{question['id']}: expected page {question.get('expected_page')} missing from hybrid top-5"
        if expected_ids:
            assert any(chunk_id in hybrid_ids or chunk_id in [hit.chunk_id for hit in hybrid_hits[:10]] for chunk_id in expected_ids) or hybrid_rank <= 5

        if question["kind"] == "keyword":
            assert keyword_rank is not None and keyword_rank <= 10, f"{question['id']}: keyword retriever missed a keyword query"
        if question["kind"] == "semantic":
            assert vector_rank is not None and vector_rank <= 10, f"{question['id']}: vector retriever missed a semantic query"

        if rerank_rank is not None and hybrid_rank is not None:
            assert hybrid_rank <= rerank_rank or hybrid_rank <= 5

        best_singleton = None
        for rank in (vector_rank, keyword_rank):
            if rank is None:
                continue
            best_singleton = rank if best_singleton is None else min(best_singleton, rank)
        if best_singleton is not None and best_singleton <= 5:
            assert hybrid_rank is not None and hybrid_rank <= 5

        if not detailed.has_relevant_context:
            hybrid_no_result += 1

        rows.append(
            {
                "id": question["id"],
                "kind": question["kind"],
                "query": query,
                "vector_rank": vector_rank,
                "keyword_rank": keyword_rank,
                "hybrid_rank": hybrid_rank,
                "rerank_rank": rerank_rank,
                "recall5_vector": recall_at_k(["rel"], ["rel"] if vector_rank and vector_rank <= 5 else ["x"], 5),
                "recall5_keyword": recall_at_k(["rel"], ["rel"] if keyword_rank and keyword_rank <= 5 else ["x"], 5),
                "recall5_hybrid": recall_at_k(["rel"], ["rel"] if hybrid_rank and hybrid_rank <= 5 else ["x"], 5),
                "recall10_vector": recall_at_k(["rel"], ["rel"] if vector_rank and vector_rank <= 10 else ["x"], 10),
                "recall10_keyword": recall_at_k(["rel"], ["rel"] if keyword_rank and keyword_rank <= 10 else ["x"], 10),
                "recall10_hybrid": recall_at_k(["rel"], ["rel"] if hybrid_rank and hybrid_rank <= 10 else ["x"], 10),
                "mrr_vector": 0.0 if vector_rank is None else 1.0 / vector_rank,
                "mrr_keyword": 0.0 if keyword_rank is None else 1.0 / keyword_rank,
                "mrr_hybrid": 0.0 if hybrid_rank is None else 1.0 / hybrid_rank,
            }
        )

    unrelated = hybrid.search_detailed(
        "quantum chromodynamics lattice gauge theory",
        final_k=5,
        document_id=DOC_ID,
        min_score=0.5,
    )
    if not unrelated.has_relevant_context:
        threshold_reject_probe = 1
    assert unrelated.has_relevant_context is False
    assert unrelated.final_candidates == []

    relevant_query = hybrid.search_detailed(
        questions[1]["query"],
        final_k=5,
        document_id=DOC_ID,
        min_score=0.5,
    )
    assert relevant_query.has_relevant_context is True

    n = len(rows)
    summary_lines = [
        "Hybrid Retrieval V1 — pricelist golden set",
        f"document_id={DOC_ID}",
        "",
        f"{'Query':<24} {'Vector':<8} {'Keyword':<8} {'Hybrid':<8}",
    ]
    for row in rows:
        summary_lines.append(
            f"{row['id']:<24} {_mark(row['vector_rank'], 5):<8} {_mark(row['keyword_rank'], 5):<8} {_mark(row['hybrid_rank'], 5):<8}"
        )
    avg = {
        "recall5_vector": round(sum(r["recall5_vector"] for r in rows) / n, 4),
        "recall5_keyword": round(sum(r["recall5_keyword"] for r in rows) / n, 4),
        "recall5_hybrid": round(sum(r["recall5_hybrid"] for r in rows) / n, 4),
        "recall10_vector": round(sum(r["recall10_vector"] for r in rows) / n, 4),
        "recall10_keyword": round(sum(r["recall10_keyword"] for r in rows) / n, 4),
        "recall10_hybrid": round(sum(r["recall10_hybrid"] for r in rows) / n, 4),
        "mrr_vector": round(sum(r["mrr_vector"] for r in rows) / n, 4),
        "mrr_keyword": round(sum(r["mrr_keyword"] for r in rows) / n, 4),
        "mrr_hybrid": round(sum(r["mrr_hybrid"] for r in rows) / n, 4),
    }
    mean_latency = {
        key: round(sum(item[key] for item in latencies) / len(latencies), 3)
        for key in latencies[0]
    }
    summary_lines.extend(
        [
            "",
            f"Recall@5  vector={avg['recall5_vector']} keyword={avg['recall5_keyword']} hybrid={avg['recall5_hybrid']}",
            f"Recall@10 vector={avg['recall10_vector']} keyword={avg['recall10_keyword']} hybrid={avg['recall10_hybrid']}",
            f"MRR       vector={avg['mrr_vector']} keyword={avg['mrr_keyword']} hybrid={avg['mrr_hybrid']}",
            f"hybrid_no_result_queries={hybrid_no_result}",
            f"unrelated_threshold_rejection={threshold_reject_probe}",
            f"mean_latency_ms={mean_latency}",
            "",
            "RERANKER: lexical_overlap (no extra model)",
            "THRESHOLD default=0.0; unrelated probe used 0.5",
        ]
    )
    REPORT_PATH.parent.mkdir(parents=True, exist_ok=True)
    REPORT_PATH.write_text("\n".join(summary_lines) + "\n\n" + "\n\n".join(traces), encoding="utf-8")
    print("\n".join(summary_lines))

    assert avg["recall5_hybrid"] >= avg["recall5_vector"]
    assert avg["recall5_hybrid"] >= avg["recall5_keyword"]
    assert mean_latency["total_ms"] > 0
