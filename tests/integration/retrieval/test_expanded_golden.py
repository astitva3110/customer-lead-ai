"""Expanded 45-query golden eval. Threshold stays 0.0. Does not promote the reranker."""

from __future__ import annotations

import json
import math
import statistics
import time
from collections import defaultdict
from pathlib import Path

import pytest
import torch

from app.config import settings
from app.services.retrieval.hybrid import HybridRetriever
from app.services.retrieval.models import RetrievalCandidate
from app.providers.reranker.cross_encoder import CrossEncoderReranker
from app.providers.reranker.factory import create_reranker
from app.providers.retrieval.keyword_retriever import KeywordCandidateRetriever
from app.providers.retrieval.vector_retriever import VectorCandidateRetriever
from app.kb.evaluation.retrieval_config import RetrievalConfig
from app.kb.evaluation.search.backends import EmbeddingSession
from app.kb.ingestion.indexing import Phase12VectorStore

GOLDEN_PATH = (
    Path(__file__).resolve().parents[2] / "fixtures" / "hybrid_retrieval" / "pricelist_v2_expanded.json"
)
DOC_ID = "6bb9a4ee-ecf4-58ff-9eaa-3206d27d9b0d"
REPORT_PATH = Path("reports") / "hybrid_retrieval_expanded_golden.txt"
JSON_PATH = Path("reports") / "hybrid_retrieval_expanded_golden.json"
CONFIGS = ("vector", "lexical", "ce_cuda")
ANSWERABLE_CATEGORIES = (
    "exact_factual",
    "semantic",
    "numerical",
    "multi_condition",
    "ambiguous",
    "terminology",
)


def _load_golden() -> dict:
    return json.loads(GOLDEN_PATH.read_text(encoding="utf-8"))


def _is_relevant(hit, question: dict) -> bool:
    if not question.get("answerable", True):
        return False
    if getattr(hit, "document_id", None) != DOC_ID:
        return False
    patterns = [pattern.lower() for pattern in question.get("content_patterns") or []]
    if not patterns:
        return False
    text = (getattr(hit, "text", "") or "").lower()
    if question.get("pattern_mode") == "any":
        return any(pattern in text for pattern in patterns)
    return all(pattern in text for pattern in patterns)


def _first_relevant_rank(hits: list, question: dict) -> int | None:
    for index, hit in enumerate(hits, start=1):
        if _is_relevant(hit, question):
            return index
    return None


def _hit_score(hit) -> float | None:
    if getattr(hit, "rerank_score", None) is not None:
        return float(hit.rerank_score)
    if getattr(hit, "vector_score", None) is not None:
        return float(hit.vector_score)
    if getattr(hit, "keyword_score", None) is not None:
        return float(hit.keyword_score)
    return None


def _percentile(values: list[float], pct: float) -> float | None:
    if not values:
        return None
    ordered = sorted(values)
    index = min(len(ordered) - 1, max(0, math.ceil(pct / 100 * len(ordered)) - 1))
    return round(ordered[index], 4)


def _summarize_scores(values: list[float]) -> dict:
    if not values:
        return {"n": 0}
    return {
        "n": len(values),
        "mean": round(statistics.fmean(values), 4),
        "min": round(min(values), 4),
        "p10": _percentile(values, 10),
        "p50": _percentile(values, 50),
        "p90": _percentile(values, 90),
        "p95": _percentile(values, 95),
        "max": round(max(values), 4),
    }


def _summarize_latency(values: list[float]) -> dict:
    if not values:
        return {"n": 0, "mean": 0.0, "p95": 0.0}
    ordered = sorted(values)
    p95_index = min(len(ordered) - 1, max(0, math.ceil(0.95 * len(ordered)) - 1))
    return {
        "n": len(ordered),
        "mean": round(statistics.fmean(ordered), 3),
        "p95": round(ordered[p95_index], 3),
        "min": round(ordered[0], 3),
        "max": round(ordered[-1], 3),
    }


def _histogram(values: list[float]) -> dict[str, int]:
    buckets = {f"{i / 10:.1f}-{(i + 1) / 10:.1f}": 0 for i in range(10)}
    for value in values:
        clamped = min(0.9999, max(0.0, float(value)))
        key = f"{math.floor(clamped * 10) / 10:.1f}-{(math.floor(clamped * 10) + 1) / 10:.1f}"
        buckets[key] += 1
    return buckets


def _metrics_from_ranks(ranks: list[int | None]) -> dict[str, float]:
    n = len(ranks)
    if n == 0:
        return {"n": 0, "recall_at_1": 0.0, "recall_at_5": 0.0, "recall_at_10": 0.0, "mrr": 0.0}

    def recall(k: int) -> float:
        return round(sum(1 for rank in ranks if rank is not None and rank <= k) / n, 4)

    mrr = round(sum(0.0 if rank is None else 1.0 / rank for rank in ranks) / n, 4)
    return {
        "n": n,
        "recall_at_1": recall(1),
        "recall_at_5": recall(5),
        "recall_at_10": recall(10),
        "mrr": mrr,
        "fp_at_1": round(1.0 - recall(1), 4),
    }


def _fmt(value) -> str:
    if value is None:
        return "-"
    return str(value)


def _build_stack(store: Phase12VectorStore, embedding_version: str):
    session = EmbeddingSession(device="cuda")
    qwen_device = str(next(session.provider._model.parameters()).device)
    if not qwen_device.startswith("cuda"):
        pytest.fail(f"Qwen remained on {qwen_device}; CUDA wiring is broken")

    vector = VectorCandidateRetriever(
        store=store,
        session=session,
        embedding_version=embedding_version,
        default_k=20,
    )
    keyword = KeywordCandidateRetriever(store, embedding_version=embedding_version, default_k=20)
    ce = CrossEncoderReranker(
        model_name=settings.reranker_model,
        device="cuda",
        batch_size=16,
        max_length=settings.reranker_max_length,
    )
    ce.rerank("warmup", [RetrievalCandidate(chunk_id="w", document_id="w", text="warmup chunk")])
    if ce.parameter_device is None or not ce.parameter_device.startswith("cuda"):
        pytest.fail(f"Cross-encoder remained on {ce.parameter_device}; CUDA wiring is broken")

    hybrid_lexical = HybridRetriever(
        vector,
        keyword,
        create_reranker("lexical_overlap"),
        vector_k=20,
        keyword_k=20,
        final_k=5,
        min_score=0.0,
    )
    hybrid_ce = HybridRetriever(vector, keyword, ce, vector_k=20, keyword_k=20, final_k=5, min_score=0.0)
    return {
        "session": session,
        "vector": vector,
        "hybrid_lexical": hybrid_lexical,
        "hybrid_ce": hybrid_ce,
        "ce": ce,
        "qwen_device": qwen_device,
    }


def _retrieve(stack: dict, name: str, query: str):
    started = time.perf_counter()
    if name == "vector":
        hits = stack["vector"].retrieve(query, top_k=10, document_id=DOC_ID)
        elapsed = (time.perf_counter() - started) * 1000
        return hits, elapsed, None
    retriever = stack["hybrid_lexical"] if name == "lexical" else stack["hybrid_ce"]
    detailed = retriever.search_detailed(query, final_k=10, document_id=DOC_ID, min_score=0.0)
    return detailed.final_candidates, detailed.timings.total_ms, detailed


def _rank_key(rank: int | None) -> int:
    return 999 if rank is None else rank


def _head_to_head(rows: list[dict], left: str, right: str) -> dict[str, int]:
    better = tie = worse = 0
    for row in rows:
        if not row["answerable"]:
            continue
        left_rank = _rank_key(row[left]["rank"])
        right_rank = _rank_key(row[right]["rank"])
        if left_rank < right_rank:
            better += 1
        elif left_rank > right_rank:
            worse += 1
        else:
            tie += 1
    return {"better": better, "tie": tie, "worse": worse}


def _promotion_decision(payload: dict) -> dict:
    """45-query set is the real CE-vs-lexical production decision."""
    ce = payload["overall_answerable"]["ce_cuda"]
    lex = payload["overall_answerable"]["lexical"]
    cuda_ok = str(payload["qwen_device"]).startswith("cuda") and str(payload["ce_device"]).startswith("cuda")
    quality_ok = (
        ce["recall_at_5"] >= lex["recall_at_5"]
        and ce["recall_at_1"] >= lex["recall_at_1"]
        and ce["mrr"] >= lex["mrr"]
    )
    latency_ok = ce["latency"]["mean"] <= 500.0
    category_ok = all(
        payload["per_category"]["ce_cuda"][category]["recall_at_5"]
        >= payload["per_category"]["lexical"][category]["recall_at_5"]
        for category in ANSWERABLE_CATEGORIES
    )
    promote = cuda_ok and quality_ok and latency_ok and category_ok
    if not cuda_ok:
        verdict = "KEEP lexical_overlap — CUDA wiring failed"
    elif not quality_ok:
        verdict = "KEEP lexical_overlap — cross-encoder did not beat lexical on R@1/R@5/MRR"
    elif not category_ok:
        verdict = "KEEP lexical_overlap — cross-encoder lost Recall@5 in at least one category"
    elif not latency_ok:
        verdict = "KEEP lexical_overlap — cross-encoder mean latency exceeded 500 ms"
    else:
        verdict = (
            "PROMOTE cross-encoder CUDA over lexical_overlap. "
            "Keep RERANKER_MIN_SCORE=0.0 until a threshold sweep."
        )
    return {
        "promote": promote,
        "verdict": verdict,
        "checks": {
            "cuda_ok": cuda_ok,
            "quality_ok": quality_ok,
            "category_r5_ok": category_ok,
            "latency_mean_le_500ms": latency_ok,
        },
        "head_to_head_ce_vs_lexical": payload["head_to_head"],
    }


@pytest.mark.skipif(not torch.cuda.is_available(), reason="CUDA is required for this expanded eval")
def test_expanded_golden_vector_lexical_ce_cuda() -> None:
    assert settings.reranker_provider == "lexical_overlap"
    assert settings.reranker_min_score == 0.0

    golden = _load_golden()
    questions = golden["questions"]
    assert len(questions) == 45
    by_category = defaultdict(int)
    for question in questions:
        by_category[question["category"]] += 1
    assert by_category["exact_factual"] == 10
    assert by_category["semantic"] == 10
    assert by_category["numerical"] == 5
    assert by_category["multi_condition"] == 5
    assert by_category["negative"] == 5
    assert by_category["ambiguous"] == 5
    assert by_category["terminology"] == 5

    config = RetrievalConfig.from_yaml(Path("configs/retrieval/v2.yaml"))
    store = Phase12VectorStore(table_name=config.vector_table)
    try:
        probe = store.search_keyword("Earkart", top_k=1, document_id=DOC_ID)
    except Exception as exc:
        pytest.skip(f"PostgreSQL/pgvector unavailable: {exc}")
    if not probe:
        pytest.skip("Indexed pricelist document is not available; not re-ingesting")

    stack = _build_stack(store, config.embedding_version)
    gpu_name = torch.cuda.get_device_name(0)

    warmup_queries = [questions[0]["query"], questions[10]["query"]]
    for query in warmup_queries:
        for name in CONFIGS:
            _retrieve(stack, name, query)

    rows: list[dict] = []
    latencies: dict[str, list[float]] = {name: [] for name in CONFIGS}

    for question in questions:
        query = question["query"]
        row: dict = {
            "id": question["id"],
            "category": question["category"],
            "answerable": bool(question.get("answerable", True)),
            "query": query,
        }
        for name in CONFIGS:
            hits, elapsed, detailed = _retrieve(stack, name, query)
            latencies[name].append(elapsed)
            rank = _first_relevant_rank(hits, question)
            top1 = hits[0] if hits else None
            rel_hit = next((hit for hit in hits if _is_relevant(hit, question)), None)
            irrelevant_at_5 = sum(1 for hit in hits[:5] if not _is_relevant(hit, question))
            row[name] = {
                "rank": rank,
                "returned": len(hits),
                "empty": len(hits) == 0,
                "latency_ms": round(elapsed, 3),
                "top1_score": None if top1 is None else _hit_score(top1),
                "relevant_score": None if rel_hit is None else _hit_score(rel_hit),
                "irrelevant_at_5": irrelevant_at_5,
                "top1_chunk_id": None if top1 is None else top1.chunk_id,
                "has_relevant_context": None if detailed is None else detailed.has_relevant_context,
            }
        rows.append(row)

    answerable = [row for row in rows if row["answerable"]]
    negatives = [row for row in rows if not row["answerable"]]
    overall = {}
    per_category = {}
    negative_metrics = {}
    score_dist = {}

    for name in CONFIGS:
        overall[name] = _metrics_from_ranks([row[name]["rank"] for row in answerable])
        overall[name]["mean_irrelevant_at_5"] = round(
            statistics.fmean([row[name]["irrelevant_at_5"] for row in answerable]),
            4,
        )
        overall[name]["latency"] = _summarize_latency(latencies[name])
        per_category[name] = {}
        for category in ANSWERABLE_CATEGORIES:
            ranks = [row[name]["rank"] for row in answerable if row["category"] == category]
            per_category[name][category] = _metrics_from_ranks(ranks)

        neg_empty = sum(1 for row in negatives if row[name]["empty"])
        neg_returned = len(negatives) - neg_empty
        negative_metrics[name] = {
            "n": len(negatives),
            "returned_any": neg_returned,
            "empty_final": neg_empty,
            "no_result_accuracy": round(neg_empty / len(negatives), 4) if negatives else 0.0,
            "false_positive_rate": round(neg_returned / len(negatives), 4) if negatives else 0.0,
            "mean_irrelevant_at_5": round(
                statistics.fmean([row[name]["irrelevant_at_5"] for row in negatives]),
                4,
            )
            if negatives
            else 0.0,
            "top1_scores": _summarize_scores(
                [row[name]["top1_score"] for row in negatives if row[name]["top1_score"] is not None]
            ),
        }
        relevant_scores = [
            row[name]["relevant_score"]
            for row in answerable
            if row[name]["relevant_score"] is not None
        ]
        miss_top1 = [
            row[name]["top1_score"]
            for row in answerable
            if row[name]["rank"] != 1 and row[name]["top1_score"] is not None
        ]
        hit_top1 = [
            row[name]["top1_score"]
            for row in answerable
            if row[name]["rank"] == 1 and row[name]["top1_score"] is not None
        ]
        neg_top1 = [row[name]["top1_score"] for row in negatives if row[name]["top1_score"] is not None]
        score_dist[name] = {
            "answerable_first_relevant": _summarize_scores(relevant_scores),
            "answerable_first_relevant_hist": _histogram(relevant_scores),
            "answerable_top1_correct": _summarize_scores(hit_top1),
            "answerable_top1_incorrect": _summarize_scores(miss_top1),
            "negative_top1": _summarize_scores(neg_top1),
            "negative_top1_hist": _histogram(neg_top1),
        }

    head_to_head = {
        "ce_vs_lexical": _head_to_head(rows, "ce_cuda", "lexical"),
        "ce_vs_vector": _head_to_head(rows, "ce_cuda", "vector"),
        "lexical_vs_vector": _head_to_head(rows, "lexical", "vector"),
    }
    payload = {
        "document_id": DOC_ID,
        "threshold": 0.0,
        "vector_k": 20,
        "keyword_k": 20,
        "final_k_measured": 10,
        "production_reranker": settings.reranker_provider,
        "gpu": gpu_name,
        "qwen_device": stack["qwen_device"],
        "ce_device": stack["ce"].parameter_device,
        "query_count": len(questions),
        "category_counts": dict(by_category),
        "overall_answerable": overall,
        "per_category": per_category,
        "negative": negative_metrics,
        "score_distributions": score_dist,
        "head_to_head": head_to_head,
        "rows": rows,
    }
    payload["promotion"] = _promotion_decision(payload)

    lines = _format_report(payload)
    REPORT_PATH.parent.mkdir(parents=True, exist_ok=True)
    REPORT_PATH.write_text("\n".join(lines) + "\n", encoding="utf-8")
    JSON_PATH.write_text(json.dumps(payload, indent=2), encoding="utf-8")
    print("\n".join(lines))

    assert settings.reranker_provider == "lexical_overlap"
    assert settings.reranker_min_score == 0.0
    assert stack["ce"].load_count == 1
    assert all(overall[name]["n"] == 40 for name in CONFIGS)


def _format_report(payload: dict) -> list[str]:
    overall = payload["overall_answerable"]
    promo = payload["promotion"]
    h2h = payload["head_to_head"]
    lines = [
        "Expanded golden set — production reranker decision",
        "This 45-query set is the real benchmark for CE CUDA vs lexical. The 8-query set only proved the pipeline.",
        f"document_id={payload['document_id']}",
        f"queries={payload['query_count']}  threshold={payload['threshold']}  "
        f"vector_k={payload['vector_k']} keyword_k={payload['keyword_k']}",
        "Recall@10 uses final_k=10; production FINAL_RETRIEVAL_K remains 5. Rerank cost is unchanged.",
        f"GPU: {payload['gpu']}",
        f"Qwen device: {payload['qwen_device']}  Cross-encoder device: {payload['ce_device']}",
        f"Current production RERANKER_PROVIDER={payload['production_reranker']}",
        "Benchmark overrides: EmbeddingSession(device=cuda) + CrossEncoderReranker(device=cuda).",
        "",
        f"DECISION: {promo['verdict']}",
        f"promote={promo['promote']} checks={promo['checks']}",
        "Head-to-head on 40 answerable queries (better = gold chunk ranked higher):",
        f"  CE vs lexical: better={h2h['ce_vs_lexical']['better']} tie={h2h['ce_vs_lexical']['tie']} worse={h2h['ce_vs_lexical']['worse']}",
        f"  CE vs vector:  better={h2h['ce_vs_vector']['better']} tie={h2h['ce_vs_vector']['tie']} worse={h2h['ce_vs_vector']['worse']}",
        f"  lexical vs vector: better={h2h['lexical_vs_vector']['better']} tie={h2h['lexical_vs_vector']['tie']} worse={h2h['lexical_vs_vector']['worse']}",
        "",
        "Category counts: " + ", ".join(f"{k}={v}" for k, v in payload["category_counts"].items()),
        "",
        "=== Overall answerable (n=40) ===",
        f"{'config':<16} {'R@1':<8} {'R@5':<8} {'R@10':<8} {'MRR':<8} {'FP@1':<8} {'mean irr@5':<12} {'mean ms':<10} {'p95 ms':<10}",
    ]
    labels = {"vector": "Vector", "lexical": "Hybrid lexical", "ce_cuda": "Hybrid CE CUDA"}
    for name in CONFIGS:
        m = overall[name]
        lat = m["latency"]
        lines.append(
            f"{labels[name]:<16} {m['recall_at_1']:<8} {m['recall_at_5']:<8} {m['recall_at_10']:<8} "
            f"{m['mrr']:<8} {m['fp_at_1']:<8} {m['mean_irrelevant_at_5']:<12} {lat['mean']:<10} {lat['p95']:<10}"
        )
    lines.extend(["", "=== Per-category Recall@1 / Recall@5 / MRR ==="])
    header = f"{'category':<18}" + "".join(f"{labels[name]:<28}" for name in CONFIGS)
    lines.append(header)
    for category in ANSWERABLE_CATEGORIES:
        cells = []
        for name in CONFIGS:
            m = payload["per_category"][name][category]
            cells.append(f"R1={m['recall_at_1']} R5={m['recall_at_5']} MRR={m['mrr']}")
        lines.append(f"{category:<18}" + "".join(f"{cell:<28}" for cell in cells))

    lines.extend(
        [
            "",
            "=== Negative / no-result (n=5, corpus gaps) ===",
            "Threshold is 0.0, so surviving candidates are not dropped. No-result accuracy at 0.0 is expected to be 0.",
            f"{'config':<16} {'returned_any':<14} {'empty':<8} {'no-result acc':<16} {'neg FP rate':<12} {'neg top1':<40}",
        ]
    )
    for name in CONFIGS:
        n = payload["negative"][name]
        lines.append(
            f"{labels[name]:<16} {n['returned_any']:<14} {n['empty_final']:<8} "
            f"{n['no_result_accuracy']:<16} {n['false_positive_rate']:<12} {n['top1_scores']}"
        )

    lines.extend(["", "=== Score distributions (do not guess 0.5 from this; use these for a later sweep) ==="])
    for name in CONFIGS:
        dist = payload["score_distributions"][name]
        lines.append(f"{labels[name]}:")
        lines.append(f"  first-relevant {dist['answerable_first_relevant']}")
        lines.append(f"  first-relevant hist {dist['answerable_first_relevant_hist']}")
        lines.append(f"  top1 correct {dist['answerable_top1_correct']}")
        lines.append(f"  top1 incorrect {dist['answerable_top1_incorrect']}")
        lines.append(f"  negative top1 {dist['negative_top1']}")
        lines.append(f"  negative top1 hist {dist['negative_top1_hist']}")

    lines.extend(
        [
            "",
            "=== Per-query ranks ===",
            f"{'id':<28} {'cat':<16} {'V':<4} {'L':<4} {'CE':<4} {'CE top1':<10} {'CE rel':<10}",
        ]
    )
    for row in payload["rows"]:
        lines.append(
            f"{row['id']:<28} {row['category']:<16} "
            f"{_fmt(row['vector']['rank']):<4} {_fmt(row['lexical']['rank']):<4} {_fmt(row['ce_cuda']['rank']):<4} "
            f"{_fmt(row['ce_cuda']['top1_score']):<10} {_fmt(row['ce_cuda']['relevant_score']):<10}"
        )
    lines.extend(
        [
            "",
            "False positives: FP@1 = answerable queries whose top-1 is not a gold hit (1 - Recall@1).",
            "Negative FP rate = gap queries that still returned chunks at threshold 0.0.",
            "CE rerank_score is sigmoid(logit); lexical score is term overlap. Do not compare them as the same scale.",
            "This run does not change production. Apply only if DECISION says PROMOTE.",
        ]
    )
    return lines
