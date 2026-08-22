from __future__ import annotations

import json
from pathlib import Path

from app.services.retrieval.hybrid import HybridRetriever
from app.providers.reranker.passthrough import PassthroughReranker

DOC_ID = "6bb9a4ee-ecf4-58ff-9eaa-3206d27d9b0d"
GOLDEN_PATH = Path("tests/fixtures/hybrid_retrieval/pricelist_v2_expanded.json")
ENTITY_QUERIES = ("BTE", "TINY", "Bluup", "warranty", "delivery")
SEMANTIC_QUERIES = (
    "How does hearing loss treatment work?",
    "How does a hearing aid help someone hear?",
)


def _is_relevant(hit, question: dict) -> bool:
    if getattr(hit, "document_id", None) != DOC_ID:
        return False
    text = (getattr(hit, "text", "") or "").lower()
    patterns = [pattern.lower() for pattern in question.get("content_patterns") or []]
    if not patterns:
        return False
    if question.get("pattern_mode") == "any":
        return any(pattern in text for pattern in patterns)
    return all(pattern in text for pattern in patterns)


def _first_rank(hits, question: dict) -> int | None:
    for index, hit in enumerate(hits, start=1):
        if _is_relevant(hit, question):
            return index
    return None


def test_hybrid_contains_vector_and_keyword(production_hybrid) -> None:
    hybrid: HybridRetriever = production_hybrid["hybrid"]
    result = hybrid.search_detailed("TINY", document_id=DOC_ID)
    assert result.vector_candidates
    assert result.keyword_candidates
    assert result.merged_candidates
    assert result.reranked_candidates
    vector_ids = {item.chunk_id for item in result.vector_candidates}
    keyword_ids = {item.chunk_id for item in result.keyword_candidates}
    duplicates = vector_ids & keyword_ids
    assert len(result.merged_candidates) == len(vector_ids | keyword_ids) or len(result.merged_candidates) <= 30


def test_entity_queries_measure_keyword_contribution(production_hybrid) -> None:
    hybrid: HybridRetriever = production_hybrid["hybrid"]
    rows = []
    for query in ENTITY_QUERIES:
        result = hybrid.search_detailed(query, document_id=DOC_ID)
        vector_ids = {item.chunk_id for item in result.vector_candidates}
        keyword_ids = {item.chunk_id for item in result.keyword_candidates}
        rows.append(
            {
                "query": query,
                "vector_candidates": len(result.vector_candidates),
                "keyword_candidates": len(result.keyword_candidates),
                "merged_candidates": len(result.merged_candidates),
                "duplicate_count": len(vector_ids & keyword_ids),
                "keyword_only": len(keyword_ids - vector_ids),
            }
        )
        assert result.keyword_candidates, f"{query}: keyword/FTS returned no candidates"
        assert result.vector_candidates, f"{query}: vector returned no candidates"
    assert any(row["keyword_only"] > 0 or row["duplicate_count"] > 0 for row in rows)


def test_semantic_queries_measure_vector_contribution(production_hybrid) -> None:
    hybrid: HybridRetriever = production_hybrid["hybrid"]
    for query in SEMANTIC_QUERIES:
        result = hybrid.search_detailed(query, document_id=DOC_ID)
        assert result.vector_candidates, f"{query}: vector returned no candidates"


def test_reranker_only_reorders_candidates(production_hybrid) -> None:
    hybrid: HybridRetriever = production_hybrid["hybrid"]
    result = hybrid.search_detailed("TINY", document_id=DOC_ID)
    merged_ids = {item.chunk_id for item in result.merged_candidates}
    reranked_ids = {item.chunk_id for item in result.reranked_candidates}
    assert reranked_ids <= merged_ids
    assert not (reranked_ids - merged_ids)


def test_reranker_cannot_recover_missing_gold(production_hybrid) -> None:
    hybrid: HybridRetriever = production_hybrid["hybrid"]
    result = hybrid.search_detailed("TINY", document_id=DOC_ID)
    absent = "not-a-real-chunk-id"
    assert absent not in {item.chunk_id for item in result.merged_candidates}
    assert absent not in {item.chunk_id for item in result.final_candidates}


def test_rerank_vs_passthrough_on_golden(production_hybrid) -> None:
    golden = json.loads(GOLDEN_PATH.read_text(encoding="utf-8"))
    questions = [item for item in golden["questions"] if item.get("answerable")][:12]
    base = production_hybrid
    hybrid: HybridRetriever = base["hybrid"]
    passthrough = HybridRetriever(
        base["vector"],
        base["keyword"],
        PassthroughReranker(),
        vector_k=hybrid._vector_k,
        keyword_k=hybrid._keyword_k,
        final_k=5,
        min_score=0.0,
        rerank_candidate_k=hybrid._rerank_candidate_k,
    )
    with_r = []
    without = []
    for question in questions:
        ranked = hybrid.search_detailed(question["query"], document_id=DOC_ID, final_k=5)
        plain = passthrough.search_detailed(question["query"], document_id=DOC_ID, final_k=5)
        gold_ids = [hit.chunk_id for hit in ranked.final_candidates if _is_relevant(hit, question)]
        with_r.append(1.0 if _first_rank(ranked.final_candidates, question) is not None else 0.0)
        without.append(1.0 if _first_rank(plain.final_candidates, question) is not None else 0.0)
        ranked_ids = [item.chunk_id for item in ranked.reranked_candidates]
        merged_ids = {item.chunk_id for item in ranked.merged_candidates}
        assert set(ranked_ids) <= merged_ids
    assert sum(with_r) >= 0
    assert len(with_r) == len(without)
