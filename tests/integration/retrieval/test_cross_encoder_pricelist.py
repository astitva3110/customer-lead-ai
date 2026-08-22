"""Compare vector vs hybrid-lexical vs hybrid-cross-encoder on the indexed pricelist.

Does not re-ingest document 6bb9a4ee-ecf4-58ff-9eaa-3206d27d9b0d.
Does not change VECTOR/KEYWORD/FINAL K or RERANKER_MIN_SCORE.
"""

from __future__ import annotations

import json
import time
from pathlib import Path

import pytest

from app.config import settings
from app.services.retrieval.diagnostics import format_retrieval_trace
from app.services.retrieval.hybrid import HybridRetriever
from app.services.retrieval.models import RetrievalCandidate
from app.providers.reranker.cross_encoder import CrossEncoderReranker
from app.providers.reranker.factory import create_reranker
from app.providers.retrieval.keyword_retriever import KeywordCandidateRetriever
from app.providers.retrieval.vector_retriever import VectorCandidateRetriever
from app.kb.evaluation.retrieval_config import RetrievalConfig
from app.kb.evaluation.search.backends import EmbeddingSession
from app.kb.ingestion.indexing import Phase12VectorStore
from tests.integration.retrieval.test_hybrid_pricelist import (
    DOC_ID,
    _first_relevant_rank,
    _load_golden,
)

REPORT_PATH = Path("reports") / "hybrid_retrieval_v2_cross_encoder.txt"
ABOUT_CHUNK = "ee142645808820f93ddb7671a4881aaefd6d335f14eeebdb9a27b4c41210787e"
OMNI_CHUNK = "e2a2d91b59f7f6eea4cb811f2b0e78e309c80d8187e3b50f130cf1fdd256fc96"
WARM_REPEATS = 5
LATENCY_QUERY = "What is Earkart and how does it deliver hearing care?"


def _metrics(ranks: list[int | None]) -> dict[str, float]:
    n = len(ranks)

    def recall(k: int) -> float:
        return round(sum(1 for rank in ranks if rank is not None and rank <= k) / n, 4)

    mrr = round(sum(0.0 if rank is None else 1.0 / rank for rank in ranks) / n, 4)
    return {
        "recall_at_1": recall(1),
        "recall_at_5": recall(5),
        "recall_at_10": recall(10),
        "mrr": mrr,
    }


def _rank_delta(lexical_rank: int | None, ce_rank: int | None) -> str:
    if lexical_rank is None and ce_rank is None:
        return "unchanged"
    if lexical_rank is None:
        return "improved"
    if ce_rank is None:
        return "worsened"
    if ce_rank < lexical_rank:
        return "improved"
    if ce_rank > lexical_rank:
        return "worsened"
    return "unchanged"


@pytest.fixture(scope="module")
def retrieval_config() -> RetrievalConfig:
    return RetrievalConfig.from_yaml(Path("configs/retrieval/v2.yaml"))


@pytest.fixture(scope="module")
def retrievers(retrieval_config: RetrievalConfig):
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
    lexical = HybridRetriever(
        vector,
        keyword,
        create_reranker("lexical_overlap"),
        vector_k=20,
        keyword_k=20,
        final_k=5,
        min_score=0.0,
    )
    device = (settings.reranker_device or settings.embedding_device).strip() or "cpu"
    gpu_peak = None
    try:
        from app.kb.embedding.device import peak_gpu_memory_mb, reset_peak_gpu_memory

        try:
            reset_peak_gpu_memory()
        except Exception:
            pass
        ce = CrossEncoderReranker(
            model_name=settings.reranker_model,
            device=device,
            batch_size=settings.reranker_batch_size,
            max_length=settings.reranker_max_length,
        )
        load_started = time.perf_counter()
        ce.rerank("warmup", [RetrievalCandidate(chunk_id="w", document_id="w", text="warmup chunk")])
        load_ms = (time.perf_counter() - load_started) * 1000
        try:
            gpu_peak = peak_gpu_memory_mb()
        except Exception:
            gpu_peak = None
    except Exception as exc:
        pytest.skip(f"cross-encoder model unavailable: {exc}")

    hybrid_ce = HybridRetriever(
        vector,
        keyword,
        ce,
        vector_k=20,
        keyword_k=20,
        final_k=5,
        min_score=0.0,
    )
    return {
        "vector": vector,
        "lexical": lexical,
        "cross": hybrid_ce,
        "load_ms": load_ms,
        "gpu_peak_mb": gpu_peak,
        "device": device,
    }


def test_cross_encoder_pricelist_comparison(retrievers) -> None:
    questions = _load_golden()["questions"]
    vector = retrievers["vector"]
    lexical: HybridRetriever = retrievers["lexical"]
    hybrid_ce: HybridRetriever = retrievers["cross"]

    vector_ranks: list[int | None] = []
    lexical_ranks: list[int | None] = []
    ce_ranks: list[int | None] = []
    deltas: list[tuple[str, str, int | None, int | None]] = []
    traces: list[str] = []
    about_case: dict = {}

    for question in questions:
        query = question["query"]
        vector_hits = vector.retrieve(query, top_k=10, document_id=DOC_ID)
        lexical_result = lexical.search_detailed(query, final_k=10, document_id=DOC_ID, min_score=0.0)
        ce_result = hybrid_ce.search_detailed(query, final_k=10, document_id=DOC_ID, min_score=0.0)
        traces.append("=== LEXICAL ===\n" + format_retrieval_trace(lexical_result))
        traces.append("=== CROSS-ENCODER ===\n" + format_retrieval_trace(ce_result))

        v_rank = _first_relevant_rank(vector_hits, question)
        l_rank = _first_relevant_rank(lexical_result.final_candidates, question)
        c_rank = _first_relevant_rank(ce_result.final_candidates, question)
        vector_ranks.append(v_rank)
        lexical_ranks.append(l_rank)
        ce_ranks.append(c_rank)
        deltas.append((question["id"], _rank_delta(l_rank, c_rank), l_rank, c_rank))

        if question["id"] == "semantic_about_earkart":
            lexical_ids = [hit.chunk_id for hit in lexical_result.reranked_candidates[:5]]
            ce_ids = [hit.chunk_id for hit in ce_result.reranked_candidates[:5]]
            about_case = {
                "query": query,
                "vector_top": vector_hits[0].chunk_id if vector_hits else None,
                "lexical_top": lexical_result.reranked_candidates[0].chunk_id
                if lexical_result.reranked_candidates
                else None,
                "ce_top": ce_result.reranked_candidates[0].chunk_id if ce_result.reranked_candidates else None,
                "lexical_about_rank": next((i for i, cid in enumerate(lexical_ids, 1) if cid == ABOUT_CHUNK), None),
                "ce_about_rank": next((i for i, cid in enumerate(ce_ids, 1) if cid == ABOUT_CHUNK), None),
                "lexical_omni_rank": next((i for i, cid in enumerate(lexical_ids, 1) if cid == OMNI_CHUNK), None),
                "ce_omni_rank": next((i for i, cid in enumerate(ce_ids, 1) if cid == OMNI_CHUNK), None),
            }

        assert c_rank is not None and c_rank <= 5, f"{question['id']}: cross-encoder missed Recall@5"

    hybrid_ce.search_detailed(LATENCY_QUERY, final_k=5, document_id=DOC_ID, min_score=0.0)
    warm_samples: list[dict[str, float]] = []
    for _ in range(WARM_REPEATS):
        result = hybrid_ce.search_detailed(LATENCY_QUERY, final_k=5, document_id=DOC_ID, min_score=0.0)
        warm_samples.append(result.timings.to_dict())

    def _agg(key: str) -> dict[str, float]:
        values = [sample[key] for sample in warm_samples]
        return {
            "avg": round(sum(values) / len(values), 3),
            "min": round(min(values), 3),
            "max": round(max(values), 3),
        }

    vector_m = _metrics(vector_ranks)
    lexical_m = _metrics(lexical_ranks)
    ce_m = _metrics(ce_ranks)
    improved = [row[0] for row in deltas if row[1] == "improved"]
    worsened = [row[0] for row in deltas if row[1] == "worsened"]
    unchanged = [row[0] for row in deltas if row[1] == "unchanged"]
    promote = (
        ce_m["recall_at_5"] >= lexical_m["recall_at_5"]
        and ce_m["mrr"] > lexical_m["mrr"]
        and about_case.get("ce_about_rank") is not None
        and (
            about_case.get("lexical_about_rank") is None
            or about_case["ce_about_rank"] <= about_case["lexical_about_rank"]
        )
        and _agg("rerank_ms")["avg"] < 500
    )

    lines = [
        "Hybrid Retrieval V2 — cross-encoder benchmark",
        f"document_id={DOC_ID}",
        f"model={settings.reranker_model}",
        f"device={retrievers['device']}",
        f"batch_size={settings.reranker_batch_size}",
        "score=sigmoid(logit); raw logit stored in metadata rerank_logit",
        f"cold_load_or_first_infer_ms={round(retrievers['load_ms'], 3)}",
        f"gpu_peak_mb_after_load={retrievers['gpu_peak_mb']}",
        "",
        f"{'Metric':<12} {'Vector':<10} {'Lexical':<10} {'CrossEnc':<10}",
        f"{'Recall@1':<12} {vector_m['recall_at_1']:<10} {lexical_m['recall_at_1']:<10} {ce_m['recall_at_1']:<10}",
        f"{'Recall@5':<12} {vector_m['recall_at_5']:<10} {lexical_m['recall_at_5']:<10} {ce_m['recall_at_5']:<10}",
        f"{'Recall@10':<12} {vector_m['recall_at_10']:<10} {lexical_m['recall_at_10']:<10} {ce_m['recall_at_10']:<10}",
        f"{'MRR':<12} {vector_m['mrr']:<10} {lexical_m['mrr']:<10} {ce_m['mrr']:<10}",
        "",
        f"improved={improved}",
        f"worsened={worsened}",
        f"unchanged={unchanged}",
        "",
        "Known case: What is Earkart?",
        json.dumps(about_case, indent=2),
        "",
        "Warm latency ms (5 repeats after warmup):",
        json.dumps({key: _agg(key) for key in warm_samples[0]}, indent=2),
        "",
        f"promote_cross_encoder={promote}",
        "Production default remains RERANKER_PROVIDER=lexical_overlap (not auto-switched).",
        "RERANKER_MIN_SCORE remains 0.0; threshold calibration is out of scope.",
    ]
    REPORT_PATH.parent.mkdir(parents=True, exist_ok=True)
    REPORT_PATH.write_text("\n".join(lines) + "\n\n" + "\n\n".join(traces), encoding="utf-8")
    print("\n".join(lines))

    assert ce_m["recall_at_5"] >= lexical_m["recall_at_5"]
    assert about_case.get("ce_about_rank") is not None
