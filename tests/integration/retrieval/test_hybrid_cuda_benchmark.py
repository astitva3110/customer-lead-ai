"""CUDA benchmark of the full hybrid pipeline. Does not change production defaults."""

from __future__ import annotations

import json
import math
import statistics
import time
from pathlib import Path

import pytest
import torch

from app.config import settings
from app.services.retrieval.hybrid import HybridRetriever
from app.services.retrieval.models import RetrievalCandidate
from app.helpers.retrieval_hits import candidate_from_store_item
from app.providers.reranker.cross_encoder import CrossEncoderReranker
from app.providers.reranker.factory import create_reranker
from app.providers.retrieval.keyword_retriever import KeywordCandidateRetriever
from app.providers.retrieval.vector_retriever import VectorCandidateRetriever
from app.kb.evaluation.retrieval_config import RetrievalConfig
from app.kb.evaluation.search.backends import EmbeddingSession
from app.kb.ingestion.indexing import Phase12VectorStore
from tests.integration.retrieval.test_hybrid_pricelist import DOC_ID, _first_relevant_rank, _load_golden

REPORT_PATH = Path("reports") / "hybrid_retrieval_cuda_benchmark.txt"
ABOUT_CHUNK = "ee142645808820f93ddb7671a4881aaefd6d335f14eeebdb9a27b4c41210787e"
OMNI_CHUNK = "e2a2d91b59f7f6eea4cb811f2b0e78e309c80d8187e3b50f130cf1fdd256fc96"
WARM_ITERS = 5
CPU_CE_BASELINE = {
    "vector_ms": 709.0,
    "keyword_ms": 22.0,
    "merge_ms": 0.14,
    "rerank_ms": 1301.0,
    "total_ms": 2032.0,
}


def _summarize(values: list[float]) -> dict[str, float]:
    if not values:
        return {"n": 0, "mean": 0.0, "median": 0.0, "p95": 0.0, "min": 0.0, "max": 0.0}
    ordered = sorted(values)
    n = len(ordered)
    p95_index = min(n - 1, max(0, math.ceil(0.95 * n) - 1))
    return {
        "n": n,
        "mean": round(statistics.fmean(ordered), 3),
        "median": round(statistics.median(ordered), 3),
        "p95": round(ordered[p95_index], 3),
        "min": round(ordered[0], 3),
        "max": round(ordered[-1], 3),
    }


def _vram() -> dict[str, float]:
    if not torch.cuda.is_available():
        return {"allocated_mb": 0.0, "reserved_mb": 0.0}
    return {
        "allocated_mb": round(torch.cuda.memory_allocated() / (1024 * 1024), 2),
        "reserved_mb": round(torch.cuda.memory_reserved() / (1024 * 1024), 2),
    }


def _param_device(model) -> str:
    return str(next(model.parameters()).device)


def _metrics(ranks: list[int | None]) -> dict[str, float]:
    n = len(ranks)

    def recall(k: int) -> float:
        return round(sum(1 for rank in ranks if rank is not None and rank <= k) / n, 4)

    mrr = round(sum(0.0 if rank is None else 1.0 / rank for rank in ranks) / n, 4)
    return {"recall_at_1": recall(1), "recall_at_5": recall(5), "recall_at_10": recall(10), "mrr": mrr}


def _top_ids(hits, k: int = 5) -> list[str]:
    return [hit.chunk_id for hit in hits[:k]]


def _split_vector_search(session: EmbeddingSession, store: Phase12VectorStore, query: str, embedding_version: str):
    started = time.perf_counter()
    query_vector = session.embed_queries([query])[0]
    embed_ms = (time.perf_counter() - started) * 1000
    started = time.perf_counter()
    raw = store.search(query_vector, top_k=20, embedding_version=embedding_version, document_id=DOC_ID)
    search_ms = (time.perf_counter() - started) * 1000
    candidates = [
        candidate_from_store_item(item, vector_score=float(item.get("similarity") or 0.0)) for item in raw
    ]
    return candidates, embed_ms, search_ms


@pytest.mark.skipif(not torch.cuda.is_available(), reason="CUDA is required for this benchmark")
def test_hybrid_cuda_pipeline_benchmark() -> None:
    assert settings.reranker_provider == "lexical_overlap"
    questions = _load_golden()["questions"]
    config = RetrievalConfig.from_yaml(Path("configs/retrieval/v2.yaml"))
    store = Phase12VectorStore(table_name=config.vector_table)
    try:
        probe = store.search_keyword("Earkart", top_k=1, document_id=DOC_ID)
    except Exception as exc:
        pytest.skip(f"PostgreSQL/pgvector unavailable: {exc}")
    if not probe:
        pytest.skip("Indexed pricelist document is not available; not re-ingesting")

    gpu_name = torch.cuda.get_device_name(0)
    vram_before = _vram()

    qwen_load_started = time.perf_counter()
    session = EmbeddingSession(device="cuda")
    qwen_load_ms = (time.perf_counter() - qwen_load_started) * 1000
    qwen_device = _param_device(session.provider._model)
    if not qwen_device.startswith("cuda"):
        pytest.fail(f"Qwen remained on {qwen_device}; CUDA wiring is broken")
    vram_after_qwen = _vram()

    ce = CrossEncoderReranker(
        model_name=settings.reranker_model,
        device="cuda",
        batch_size=16,
        max_length=settings.reranker_max_length,
    )
    ce_load_started = time.perf_counter()
    ce.rerank("warmup", [RetrievalCandidate(chunk_id="w", document_id="w", text="warmup chunk")])
    ce_load_ms = (time.perf_counter() - ce_load_started) * 1000
    if ce.parameter_device is None or not ce.parameter_device.startswith("cuda"):
        pytest.fail(f"Cross-encoder remained on {ce.parameter_device}; CUDA wiring is broken")
    vram_after_ce = _vram()

    vector = VectorCandidateRetriever(
        store=store,
        session=session,
        embedding_version=config.embedding_version,
        default_k=20,
    )
    keyword = KeywordCandidateRetriever(store, embedding_version=config.embedding_version, default_k=20)
    hybrid_lexical = HybridRetriever(
        vector, keyword, create_reranker("lexical_overlap"),
        vector_k=20, keyword_k=20, final_k=5, min_score=0.0,
    )
    hybrid_ce = HybridRetriever(vector, keyword, ce, vector_k=20, keyword_k=20, final_k=5, min_score=0.0)

    qwen_id_before = id(session.provider._model)
    ce_id_before = id(ce._model)
    assert ce.load_count == 1

    for query in [questions[0]["query"], questions[1]["query"]]:
        hybrid_ce.search_detailed(query, final_k=5, document_id=DOC_ID, min_score=0.0)
    assert id(session.provider._model) == qwen_id_before
    assert id(ce._model) == ce_id_before
    assert ce.load_count == 1

    samples = {
        "vector": {"vector_ms": [], "total_ms": []},
        "lexical": {"vector_ms": [], "keyword_ms": [], "merge_ms": [], "rerank_ms": [], "threshold_ms": [], "total_ms": []},
        "ce_cuda": {
            "vector_ms": [],
            "embed_ms": [],
            "pgvector_ms": [],
            "keyword_ms": [],
            "merge_ms": [],
            "rerank_ms": [],
            "threshold_ms": [],
            "total_ms": [],
        },
    }
    peak_embed = 0.0
    peak_rerank = 0.0

    for question in questions:
        query = question["query"]
        for _ in range(WARM_ITERS):
            started = time.perf_counter()
            vector.retrieve(query, top_k=20, document_id=DOC_ID)
            elapsed = (time.perf_counter() - started) * 1000
            samples["vector"]["vector_ms"].append(elapsed)
            samples["vector"]["total_ms"].append(elapsed)

            lexical_result = hybrid_lexical.search_detailed(query, final_k=5, document_id=DOC_ID, min_score=0.0)
            for key in ("vector_ms", "keyword_ms", "merge_ms", "rerank_ms", "threshold_ms", "total_ms"):
                samples["lexical"][key].append(lexical_result.timings.to_dict()[key])

            _, embed_ms, pgvector_ms = _split_vector_search(session, store, query, config.embedding_version)
            peak_embed = max(peak_embed, _vram()["allocated_mb"])
            ce_result = hybrid_ce.search_detailed(query, final_k=5, document_id=DOC_ID, min_score=0.0)
            peak_rerank = max(peak_rerank, _vram()["allocated_mb"])
            timings = ce_result.timings.to_dict()
            samples["ce_cuda"]["embed_ms"].append(embed_ms)
            samples["ce_cuda"]["pgvector_ms"].append(pgvector_ms)
            for key in ("vector_ms", "keyword_ms", "merge_ms", "rerank_ms", "threshold_ms", "total_ms"):
                samples["ce_cuda"][key].append(timings[key])

    assert id(session.provider._model) == qwen_id_before
    assert id(ce._model) == ce_id_before
    assert ce.load_count == 1

    quality = {"vector": [], "lexical": [], "ce_cuda": []}
    earkart_tops: dict[str, list[str]] = {}
    cuda_vs_cpu_rank: list[dict] = []
    cpu_ce = CrossEncoderReranker(
        model_name=settings.reranker_model,
        device="cpu",
        batch_size=16,
        max_length=settings.reranker_max_length,
    )

    for question in questions:
        query = question["query"]
        vector_hits = vector.retrieve(query, top_k=10, document_id=DOC_ID)
        lexical_result = hybrid_lexical.search_detailed(query, final_k=10, document_id=DOC_ID, min_score=0.0)
        cuda_result = hybrid_ce.search_detailed(query, final_k=10, document_id=DOC_ID, min_score=0.0)
        cpu_ranked = cpu_ce.rerank(query, list(cuda_result.merged_candidates))
        quality["vector"].append(_first_relevant_rank(vector_hits, question))
        quality["lexical"].append(_first_relevant_rank(lexical_result.final_candidates, question))
        quality["ce_cuda"].append(_first_relevant_rank(cuda_result.final_candidates, question))
        cuda_ids = _top_ids(cuda_result.final_candidates)
        cpu_ids = _top_ids(cpu_ranked)
        cuda_vs_cpu_rank.append(
            {
                "id": question["id"],
                "cuda_top5": cuda_ids,
                "cpu_top5": cpu_ids,
                "same_top5": cuda_ids == cpu_ids,
                "same_top1": cuda_ids[:1] == cpu_ids[:1],
            }
        )
        if question["id"] == "semantic_about_earkart":
            earkart_tops = {
                "vector": _top_ids(vector_hits, 3),
                "lexical": _top_ids(lexical_result.reranked_candidates, 3),
                "ce_cuda": _top_ids(cuda_result.reranked_candidates, 3),
                "ce_cpu_same_pool": _top_ids(cpu_ranked, 3),
            }

    vector_m = _metrics(quality["vector"])
    lexical_m = _metrics(quality["lexical"])
    ce_m = _metrics(quality["ce_cuda"])
    ranking_matches = sum(1 for row in cuda_vs_cpu_rank if row["same_top5"])
    vram_end = _vram()
    vram_growth = round(vram_end["allocated_mb"] - vram_after_ce["allocated_mb"], 2)
    ce_mean = statistics.fmean(samples["ce_cuda"]["rerank_ms"])
    total_mean = statistics.fmean(samples["ce_cuda"]["total_ms"])
    ce_speedup = round(CPU_CE_BASELINE["rerank_ms"] / ce_mean, 2) if ce_mean else 0.0
    total_speedup = round(CPU_CE_BASELINE["total_ms"] / total_mean, 2) if total_mean else 0.0

    about_ok = bool(earkart_tops.get("ce_cuda")) and earkart_tops["ce_cuda"][0] == ABOUT_CHUNK
    quality_ok = ce_m["recall_at_5"] >= lexical_m["recall_at_5"] and ce_m["mrr"] >= lexical_m["mrr"]
    close_to_hybrid_v1 = total_mean <= 479 * 1.5
    memory_ok = vram_end["allocated_mb"] < 5500 and vram_growth < 400
    if not memory_ok:
        decision = "C. CUDA memory is unsafe; keep lexical_overlap"
    elif not about_ok or not quality_ok:
        decision = "D. CUDA setup is not working correctly"
    elif close_to_hybrid_v1:
        decision = "A. Cross-encoder CUDA is ready for further production evaluation"
    else:
        decision = "B. Cross-encoder CUDA is too slow; keep lexical_overlap"

    summaries = {name: {key: _summarize(vals) for key, vals in stages.items()} for name, stages in samples.items()}
    lines = [
        "Hybrid Retrieval CUDA benchmark",
        f"GPU: {gpu_name}",
        "CUDA: True",
        f"Qwen parameter device: {qwen_device}",
        f"Cross-encoder parameter device: {ce.parameter_device}",
        f"Current production RERANKER_PROVIDER={settings.reranker_provider}",
        "Benchmark overrides: EmbeddingSession(device=cuda) + CrossEncoderReranker(device=cuda). .env not changed.",
        "",
        "VRAM MB:",
        f"  before={vram_before}",
        f"  after_qwen={vram_after_qwen}",
        f"  after_cross_encoder={vram_after_ce}",
        f"  peak_after_query_embed={peak_embed}",
        f"  peak_after_rerank={peak_rerank}",
        f"  end={vram_end}",
        f"  allocated_growth_after_load={vram_growth}",
        "",
        f"Cold startup: qwen_load_ms={round(qwen_load_ms, 3)} cross_encoder_load_ms={round(ce_load_ms, 3)}",
        "p95 note: 8 queries x 5 warm iterations (n=40). Statistically weak for tail latency.",
        "",
        "Latency mean ms:",
        f"{'config':<16} {'vector':<10} {'keyword':<10} {'merge':<10} {'rerank':<10} {'total':<10}",
        f"{'Vector-only':<16} {summaries['vector']['vector_ms']['mean']:<10} {'-':<10} {'-':<10} {'-':<10} {summaries['vector']['total_ms']['mean']:<10}",
        f"{'Hybrid lexical':<16} {summaries['lexical']['vector_ms']['mean']:<10} {summaries['lexical']['keyword_ms']['mean']:<10} {summaries['lexical']['merge_ms']['mean']:<10} {summaries['lexical']['rerank_ms']['mean']:<10} {summaries['lexical']['total_ms']['mean']:<10}",
        f"{'Hybrid CE CPU*':<16} {CPU_CE_BASELINE['vector_ms']:<10} {CPU_CE_BASELINE['keyword_ms']:<10} {CPU_CE_BASELINE['merge_ms']:<10} {CPU_CE_BASELINE['rerank_ms']:<10} {CPU_CE_BASELINE['total_ms']:<10}",
        f"{'Hybrid CE CUDA':<16} {summaries['ce_cuda']['vector_ms']['mean']:<10} {summaries['ce_cuda']['keyword_ms']['mean']:<10} {summaries['ce_cuda']['merge_ms']['mean']:<10} {summaries['ce_cuda']['rerank_ms']['mean']:<10} {summaries['ce_cuda']['total_ms']['mean']:<10}",
        "*CPU CE row is the previous CPU benchmark (Qwen CPU + MiniLM CPU).",
        "",
        "CUDA CE split (mean/median/p95/min/max ms):",
        f"  qwen_embed={summaries['ce_cuda']['embed_ms']}",
        f"  pgvector={summaries['ce_cuda']['pgvector_ms']}",
        f"  keyword={summaries['ce_cuda']['keyword_ms']}",
        f"  merge={summaries['ce_cuda']['merge_ms']}",
        f"  cross_encoder={summaries['ce_cuda']['rerank_ms']}",
        f"  total={summaries['ce_cuda']['total_ms']}",
        f"  speedup vs CPU CE: rerank={ce_speedup}x total={total_speedup}x",
        "",
        "Quality:",
        f"{'config':<16} {'R@1':<8} {'R@5':<8} {'R@10':<8} {'MRR':<8}",
        f"{'Vector':<16} {vector_m['recall_at_1']:<8} {vector_m['recall_at_5']:<8} {vector_m['recall_at_10']:<8} {vector_m['mrr']:<8}",
        f"{'Hybrid lexical':<16} {lexical_m['recall_at_1']:<8} {lexical_m['recall_at_5']:<8} {lexical_m['recall_at_10']:<8} {lexical_m['mrr']:<8}",
        f"{'Hybrid CE CUDA':<16} {ce_m['recall_at_1']:<8} {ce_m['recall_at_5']:<8} {ce_m['recall_at_10']:<8} {ce_m['mrr']:<8}",
        f"CUDA vs CPU CE same top-5 on identical merged pool: {ranking_matches}/8",
        json.dumps(cuda_vs_cpu_rank, indent=2),
        "",
        "Earkart top-3 chunk ids:",
        json.dumps(earkart_tops, indent=2),
        f"About chunk expected={ABOUT_CHUNK}",
        f"OMNI chunk={OMNI_CHUNK}",
        "",
        f"model_reuse ce_load_count={ce.load_count}",
        f"decision={decision}",
        "Production default unchanged: RERANKER_PROVIDER=lexical_overlap RERANKER_MIN_SCORE=0.0",
        "Manual enable (not applied): EMBEDDING_DEVICE=cuda RERANKER_DEVICE=cuda RERANKER_PROVIDER=cross_encoder",
    ]
    REPORT_PATH.parent.mkdir(parents=True, exist_ok=True)
    REPORT_PATH.write_text("\n".join(lines) + "\n", encoding="utf-8")
    print("\n".join(lines))

    assert qwen_device.startswith("cuda")
    assert ce.parameter_device and ce.parameter_device.startswith("cuda")
    assert ce.load_count == 1
    assert ce_m["recall_at_5"] >= lexical_m["recall_at_5"]
    assert earkart_tops["ce_cuda"][0] == ABOUT_CHUNK
    assert ranking_matches >= 6
    assert settings.reranker_provider == "lexical_overlap"
