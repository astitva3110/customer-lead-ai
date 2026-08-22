from __future__ import annotations

import json
from pathlib import Path

from app.services.retrieval.hybrid import HybridRetriever
from app.helpers.validation_report import write_json_txt
from app.kb.evaluation.metrics import reciprocal_rank

DOC_ID = "6bb9a4ee-ecf4-58ff-9eaa-3206d27d9b0d"
GOLDEN_PATH = Path("tests/fixtures/hybrid_retrieval/pricelist_v2_expanded.json")
KS = (5, 10, 20, 30)


def _is_relevant(hit, question: dict) -> bool:
    if not question.get("answerable", True):
        return False
    if getattr(hit, "document_id", None) != DOC_ID:
        return False
    text = (getattr(hit, "text", "") or "").lower()
    patterns = [pattern.lower() for pattern in question.get("content_patterns") or []]
    if not patterns:
        return False
    if question.get("pattern_mode") == "any":
        return any(pattern in text for pattern in patterns)
    return all(pattern in text for pattern in patterns)


def _gold_ids(hits, question: dict) -> list[str]:
    return [hit.chunk_id for hit in hits if _is_relevant(hit, question)]


def test_retrieval_depth_offline_benchmark(production_hybrid) -> None:
    golden = json.loads(GOLDEN_PATH.read_text(encoding="utf-8"))
    questions = [item for item in golden["questions"] if item.get("answerable")]
    base = production_hybrid
    rows = []
    for candidate_k in KS:
        hybrid = HybridRetriever(
            base["vector"],
            base["keyword"],
            base["hybrid"]._reranker,
            vector_k=candidate_k,
            keyword_k=candidate_k,
            final_k=candidate_k,
            min_score=0.0,
            rerank_candidate_k=min(candidate_k * 2, 40) if candidate_k >= 20 else None,
        )
        recalls = {1: [], 3: [], 5: [], 10: [], 20: [], 30: []}
        mrrs = []
        retrieval_ms = []
        rerank_ms = []
        total_ms = []
        for question in questions:
            detailed = hybrid.search_detailed(question["query"], document_id=DOC_ID, final_k=candidate_k)
            ids = [item.chunk_id for item in detailed.final_candidates]
            gold = _gold_ids(detailed.final_candidates, question)
            expected = gold[:1] if gold else ["__missing__"]
            for k in recalls:
                hits = 1.0 if any(_is_relevant(item, question) for item in detailed.final_candidates[:k]) else 0.0
                recalls[k].append(hits)
            mrrs.append(reciprocal_rank(expected if gold else [], ids))
            retrieval_ms.append(detailed.timings.vector_ms + detailed.timings.keyword_ms + detailed.timings.merge_ms)
            rerank_ms.append(detailed.timings.rerank_ms)
            total_ms.append(detailed.timings.total_ms)
        row = {
            "candidate_k": candidate_k,
            "recall_at_1": round(sum(recalls[1]) / len(recalls[1]), 4),
            "recall_at_3": round(sum(recalls[3]) / len(recalls[3]), 4),
            "recall_at_5": round(sum(recalls[5]) / len(recalls[5]), 4),
            "recall_at_10": round(sum(recalls[10]) / len(recalls[10]), 4),
            "recall_at_20": round(sum(recalls[20]) / len(recalls[20]), 4),
            "recall_at_30": round(sum(recalls[30]) / len(recalls[30]), 4),
            "mrr": round(sum(mrrs) / len(mrrs), 4),
            "mean_retrieval_ms": round(sum(retrieval_ms) / len(retrieval_ms), 3),
            "mean_rerank_ms": round(sum(rerank_ms) / len(rerank_ms), 3),
            "mean_total_ms": round(sum(total_ms) / len(total_ms), 3),
        }
        rows.append(row)
    plateau = max(item["recall_at_5"] for item in rows)
    optimal = min(
        (item for item in rows if item["recall_at_5"] >= plateau - 0.01),
        key=lambda item: (item["candidate_k"], item["mean_total_ms"]),
    )
    payload = {
        "note": "Offline read-only benchmark. Production config was not changed. Starting knowledge k values remain 20/20/30/6.",
        "document_id": DOC_ID,
        "rows": rows,
        "optimal_retrieval_k": optimal["candidate_k"],
        "optimal_reason": (
            f"Smallest candidate K whose Recall@5 ({optimal['recall_at_5']}) "
            f"is within 0.01 of the measured plateau {plateau}."
        ),
    }
    text_lines = [
        "RETRIEVAL DEPTH BENCHMARK",
        f"OPTIMAL_RETRIEVAL_K: {payload['optimal_retrieval_k']}",
        payload["optimal_reason"],
        "",
    ]
    for row in rows:
        text_lines.append(
            f"K={row['candidate_k']:>2}  R@1={row['recall_at_1']} R@3={row['recall_at_3']} "
            f"R@5={row['recall_at_5']} R@10={row['recall_at_10']} R@20={row['recall_at_20']} "
            f"R@30={row['recall_at_30']} MRR={row['mrr']} "
            f"retr={row['mean_retrieval_ms']}ms rerank={row['mean_rerank_ms']}ms total={row['mean_total_ms']}ms"
        )
    write_json_txt(Path("reports/phase19/retrieval_depth"), payload, "\n".join(text_lines))
    assert payload["optimal_retrieval_k"] in KS
    assert rows[0]["recall_at_5"] >= 0
