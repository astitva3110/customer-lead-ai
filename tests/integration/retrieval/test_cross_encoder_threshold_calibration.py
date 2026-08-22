"""Calibrate RERANKER_MIN_SCORE from CE CUDA scores. Does not change production."""

from __future__ import annotations

import json
import math
import statistics
from pathlib import Path

import pytest
import torch

from app.config import settings
from app.services.retrieval.hybrid import HybridRetriever
from app.services.retrieval.models import RetrievalCandidate
from app.services.retrieval.threshold import apply_rerank_threshold, take_final_k
from app.providers.reranker.cross_encoder import CrossEncoderReranker
from app.providers.retrieval.keyword_retriever import KeywordCandidateRetriever
from app.providers.retrieval.vector_retriever import VectorCandidateRetriever
from app.kb.evaluation.retrieval_config import RetrievalConfig
from app.kb.evaluation.search.backends import EmbeddingSession
from app.kb.ingestion.indexing import Phase12VectorStore
from tests.integration.retrieval.test_expanded_golden import (
    DOC_ID,
    GOLDEN_PATH,
    _first_relevant_rank,
    _is_relevant,
    _load_golden,
)

REPORT_PATH = Path("reports") / "hybrid_retrieval_threshold_calibration.txt"
JSON_PATH = Path("reports") / "hybrid_retrieval_threshold_calibration.json"
SWEEP = [
    0.0,
    0.05,
    0.10,
    0.15,
    0.20,
    0.25,
    0.30,
    0.35,
    0.40,
    0.45,
    0.50,
    0.60,
    0.70,
    0.80,
    0.85,
    0.90,
    0.95,
]


def _percentile(values: list[float], pct: float) -> float | None:
    if not values:
        return None
    ordered = sorted(values)
    index = min(len(ordered) - 1, max(0, math.ceil(pct / 100 * len(ordered)) - 1))
    return round(ordered[index], 6)


def _summarize(values: list[float]) -> dict:
    if not values:
        return {"n": 0}
    return {
        "n": len(values),
        "min": round(min(values), 6),
        "p10": _percentile(values, 10),
        "p25": _percentile(values, 25),
        "median": _percentile(values, 50),
        "p75": _percentile(values, 75),
        "p90": _percentile(values, 90),
        "p95": _percentile(values, 95),
        "max": round(max(values), 6),
        "mean": round(statistics.fmean(values), 6),
    }


def _candidate_row(hit: RetrievalCandidate, rank: int, question: dict) -> dict:
    logit = hit.metadata.get("rerank_logit") if hit.metadata else None
    return {
        "rank": rank,
        "chunk_id": hit.chunk_id,
        "page_number": hit.page_number,
        "score": None if hit.rerank_score is None else float(hit.rerank_score),
        "logit": None if logit is None else float(logit),
        "relevant": _is_relevant(hit, question),
        "text_preview": (hit.text or "").replace("\n", " ")[:180],
    }


def _metrics_from_ranks(ranks: list[int | None]) -> dict:
    n = len(ranks)
    if n == 0:
        return {"n": 0, "recall_at_1": 0.0, "recall_at_5": 0.0, "mrr": 0.0, "fp_at_1": 0.0}

    def recall(k: int) -> float:
        return round(sum(1 for rank in ranks if rank is not None and rank <= k) / n, 4)

    mrr = round(sum(0.0 if rank is None else 1.0 / rank for rank in ranks) / n, 4)
    return {"n": n, "recall_at_1": recall(1), "recall_at_5": recall(5), "mrr": mrr, "fp_at_1": round(1.0 - recall(1), 4)}


def _evaluate_threshold(records: list[dict], threshold: float, final_k: int) -> dict:
    answerable_ranks: list[int | None] = []
    answerable_empty = 0
    gap_empty = 0
    gap_total = 0
    for record in records:
        reconstructed = [
            RetrievalCandidate(
                chunk_id=row["chunk_id"],
                document_id=DOC_ID,
                text=row["text_preview"],
                rerank_score=row["score"],
                metadata={"relevant": row["relevant"]},
            )
            for row in record["reranked"]
        ]
        surviving = apply_rerank_threshold(reconstructed, threshold)
        final = take_final_k(surviving, final_k)
        gold_rank = None
        for index, candidate in enumerate(final, start=1):
            if candidate.metadata.get("relevant"):
                gold_rank = index
                break
        if record["answerable"]:
            answerable_ranks.append(gold_rank)
            if not final:
                answerable_empty += 1
        else:
            gap_total += 1
            if not final:
                gap_empty += 1
    answerable_n = len(answerable_ranks)
    metrics = _metrics_from_ranks(answerable_ranks)
    metrics["answerable_no_result_rate"] = round(answerable_empty / answerable_n, 4) if answerable_n else 0.0
    metrics["gap_no_result_accuracy"] = round(gap_empty / gap_total, 4) if gap_total else 0.0
    metrics["threshold"] = threshold
    return metrics


def _recommend(sweep_rows: list[dict], lowest_first_relevant: float | None, highest_gap: float | None) -> dict:
    overlap = (
        lowest_first_relevant is not None
        and highest_gap is not None
        and lowest_first_relevant <= highest_gap
    )
    keep_recall = [row for row in sweep_rows if row["recall_at_5"] >= 0.975 and row["answerable_no_result_rate"] == 0.0]
    chosen = max(keep_recall, key=lambda row: (row["gap_no_result_accuracy"], row["threshold"])) if keep_recall else sweep_rows[0]
    reject_gaps = [row for row in sweep_rows if row["gap_no_result_accuracy"] == 1.0]
    gap_choice = (
        min(reject_gaps, key=lambda row: (-row["recall_at_5"], row["answerable_no_result_rate"], row["threshold"]))
        if reject_gaps
        else None
    )
    if overlap:
        rationale = (
            "First-relevant gold scores overlap gap-query scores, so no single cutoff both keeps all answers "
            "and rejects all gaps. Recommend the highest threshold that does not empty an answerable query. "
            "A gap-rejecting alternative is reported separately."
        )
    else:
        rationale = "Highest threshold that keeps Recall@5, does not empty answerable queries, and sits above the gap mass."
    return {
        "recommended": chosen["threshold"],
        "overlap": overlap,
        "rationale": rationale,
        "at_recommended": chosen,
        "gap_rejecting_alternative": gap_choice,
    }


@pytest.mark.skipif(not torch.cuda.is_available(), reason="CUDA is required for CE threshold calibration")
def test_cross_encoder_threshold_calibration() -> None:
    production_before = {
        "reranker_provider": settings.reranker_provider,
        "reranker_min_score": settings.reranker_min_score,
        "embedding_device": settings.embedding_device,
        "reranker_device": settings.reranker_device,
        "final_retrieval_k": settings.final_retrieval_k,
        "vector_candidate_k": settings.vector_candidate_k,
        "keyword_candidate_k": settings.keyword_candidate_k,
    }
    assert settings.reranker_min_score == 0.0
    assert settings.final_retrieval_k == 5
    assert settings.vector_candidate_k == 20
    assert settings.keyword_candidate_k == 20

    questions = _load_golden()["questions"]
    assert len(questions) == 45
    answerable_qs = [q for q in questions if q.get("answerable", True)]
    gap_qs = [q for q in questions if not q.get("answerable", True)]
    assert len(answerable_qs) == 40
    assert len(gap_qs) == 5

    config = RetrievalConfig.from_yaml(Path("configs/retrieval/v2.yaml"))
    store = Phase12VectorStore(table_name=config.vector_table)
    try:
        probe = store.search_keyword("Earkart", top_k=1, document_id=DOC_ID)
    except Exception as exc:
        pytest.skip(f"PostgreSQL/pgvector unavailable: {exc}")
    if not probe:
        pytest.skip("Indexed pricelist document is not available; not re-ingesting")

    session = EmbeddingSession(device="cuda")
    qwen_device = str(next(session.provider._model.parameters()).device)
    if not qwen_device.startswith("cuda"):
        pytest.fail(f"Qwen remained on {qwen_device}")
    ce = CrossEncoderReranker(
        model_name=settings.reranker_model,
        device="cuda",
        batch_size=16,
        max_length=settings.reranker_max_length,
    )
    ce.rerank("warmup", [RetrievalCandidate(chunk_id="w", document_id="w", text="warmup chunk")])
    if ce.parameter_device is None or not ce.parameter_device.startswith("cuda"):
        pytest.fail(f"Cross-encoder remained on {ce.parameter_device}")

    hybrid = HybridRetriever(
        VectorCandidateRetriever(
            store=store,
            session=session,
            embedding_version=config.embedding_version,
            default_k=20,
        ),
        KeywordCandidateRetriever(store, embedding_version=config.embedding_version, default_k=20),
        ce,
        vector_k=20,
        keyword_k=20,
        final_k=5,
        min_score=0.0,
    )
    hybrid.search_detailed(questions[0]["query"], final_k=5, document_id=DOC_ID, min_score=0.0)

    records: list[dict] = []
    class_a: list[float] = []
    class_a_first: list[float] = []
    class_b: list[float] = []
    class_c: list[float] = []
    class_c_top1: list[float] = []

    for question in questions:
        detailed = hybrid.search_detailed(question["query"], final_k=5, document_id=DOC_ID, min_score=0.0)
        reranked_rows = [_candidate_row(hit, index, question) for index, hit in enumerate(detailed.reranked_candidates, start=1)]
        first_rank = _first_relevant_rank(detailed.reranked_candidates, question)
        first_score = None
        if first_rank is not None:
            first_score = reranked_rows[first_rank - 1]["score"]
            class_a_first.append(first_score)
        for row in reranked_rows:
            score = row["score"]
            if score is None:
                continue
            if not question.get("answerable", True):
                class_c.append(score)
            elif row["relevant"]:
                class_a.append(score)
            else:
                class_b.append(score)
        if not question.get("answerable", True) and reranked_rows and reranked_rows[0]["score"] is not None:
            class_c_top1.append(reranked_rows[0]["score"])
        records.append(
            {
                "id": question["id"],
                "category": question["category"],
                "answerable": bool(question.get("answerable", True)),
                "query": question["query"],
                "content_patterns": question.get("content_patterns") or [],
                "pool_size": len(reranked_rows),
                "final_size_at_0": len(detailed.final_candidates),
                "first_relevant_rank_before_threshold": first_rank,
                "first_relevant_score": first_score,
                "top1_score": None if not reranked_rows else reranked_rows[0]["score"],
                "reranked": reranked_rows,
            }
        )

    dist = {
        "A_relevant_all_gold_in_pool": _summarize(class_a),
        "A_first_relevant_per_query": _summarize(class_a_first),
        "B_nonrelevant_answerable": _summarize(class_b),
        "C_gap_all_candidates": _summarize(class_c),
        "C_gap_top1": _summarize(class_c_top1),
    }
    lowest_relevant_all = dist["A_relevant_all_gold_in_pool"].get("min")
    lowest_first_relevant = dist["A_first_relevant_per_query"].get("min")
    highest_irrelevant = dist["B_nonrelevant_answerable"].get("max")
    highest_gap = dist["C_gap_all_candidates"].get("max")
    highest_gap_top1 = dist["C_gap_top1"].get("max")

    lowest_first_row = min(
        (row for row in records if row["answerable"] and row["first_relevant_score"] is not None),
        key=lambda row: row["first_relevant_score"],
    )
    highest_gap_row = max(
        (row for row in records if not row["answerable"] and row["top1_score"] is not None),
        key=lambda row: row["top1_score"],
    )

    sweep_rows = [_evaluate_threshold(records, threshold, settings.final_retrieval_k) for threshold in SWEEP]
    recommendation = _recommend(sweep_rows, lowest_first_relevant, highest_gap)

    payload = {
        "document_id": DOC_ID,
        "golden_path": str(GOLDEN_PATH),
        "collection_min_score": 0.0,
        "vector_k": 20,
        "keyword_k": 20,
        "final_k": 5,
        "gpu": torch.cuda.get_device_name(0),
        "qwen_device": qwen_device,
        "ce_device": ce.parameter_device,
        "production_before": production_before,
        "query_count": len(questions),
        "answerable": len(answerable_qs),
        "gap": len(gap_qs),
        "distributions": dist,
        "key_numbers": {
            "lowest_relevant_score_all_gold_in_pool": lowest_relevant_all,
            "lowest_first_relevant_score": lowest_first_relevant,
            "lowest_first_relevant_query": lowest_first_row["id"],
            "highest_irrelevant_score": highest_irrelevant,
            "highest_gap_query_score": highest_gap,
            "highest_gap_top1_score": highest_gap_top1,
            "highest_gap_query": highest_gap_row["id"],
            "clean_gap_first_relevant_vs_gap_pool": (
                lowest_first_relevant is not None
                and highest_gap is not None
                and lowest_first_relevant > highest_gap
            ),
            "clean_gap_all_gold_vs_irrelevant": (
                lowest_relevant_all is not None
                and highest_irrelevant is not None
                and lowest_relevant_all > highest_irrelevant
            ),
        },
        "sweep": sweep_rows,
        "recommendation": recommendation,
        "records": records,
    }
    lines = _format_report(payload)
    REPORT_PATH.parent.mkdir(parents=True, exist_ok=True)
    REPORT_PATH.write_text("\n".join(lines) + "\n", encoding="utf-8")
    JSON_PATH.write_text(json.dumps(payload, indent=2), encoding="utf-8")
    print("\n".join(lines))

    production_after = {
        "reranker_provider": settings.reranker_provider,
        "reranker_min_score": settings.reranker_min_score,
        "embedding_device": settings.embedding_device,
        "reranker_device": settings.reranker_device,
        "final_retrieval_k": settings.final_retrieval_k,
    }
    assert production_after["reranker_min_score"] == production_before["reranker_min_score"] == 0.0
    assert production_after["reranker_provider"] == production_before["reranker_provider"]
    assert production_after["final_retrieval_k"] == 5
    assert len(records) == 45


def _format_report(payload: dict) -> list[str]:
    key = payload["key_numbers"]
    rec = payload["recommendation"]
    lines = [
        "Cross-encoder threshold calibration — 45-query golden set",
        "Evaluation only. Production configuration was not changed.",
        f"document_id={payload['document_id']}",
        f"queries={payload['query_count']} answerable={payload['answerable']} gap={payload['gap']}",
        "Collection: VECTOR_K=20 KEYWORD_K=20 FINAL_K=5 CE CUDA min_score=0.0",
        f"GPU: {payload['gpu']} Qwen={payload['qwen_device']} CE={payload['ce_device']}",
        f"Recorded production (unchanged): {payload['production_before']}",
        "",
        "=== Key numbers ===",
        f"lowest relevant score (all gold in pool): {key['lowest_relevant_score_all_gold_in_pool']}",
        f"lowest first-relevant score (recall-critical): {key['lowest_first_relevant_score']}  query={key['lowest_first_relevant_query']}",
        f"highest irrelevant score (answerable non-gold): {key['highest_irrelevant_score']}",
        f"highest gap-query score (all gap candidates): {key['highest_gap_query_score']}",
        f"highest gap-query top-1 score: {key['highest_gap_top1_score']}  query={key['highest_gap_query']}",
        f"clean gap first-relevant vs gap pool: {key['clean_gap_first_relevant_vs_gap_pool']}",
        f"clean gap all-gold vs irrelevant: {key['clean_gap_all_gold_vs_irrelevant']}",
        "",
        "=== Score distributions ===",
    ]
    labels = {
        "A_relevant_all_gold_in_pool": "A relevant / gold (every matching chunk in the reranked pool)",
        "A_first_relevant_per_query": "A first relevant per answerable query",
        "B_nonrelevant_answerable": "B non-relevant candidates from answerable queries",
        "C_gap_all_candidates": "C all candidates from gap/unanswerable queries",
        "C_gap_top1": "C gap-query top-1 only",
    }
    for name, title in labels.items():
        lines.append(f"{title}: {payload['distributions'][name]}")

    lines.extend(
        [
            "",
            "=== Threshold sweep (offline on saved CE ranks, FINAL_K=5) ===",
            f"{'thr':<8} {'R@1':<8} {'R@5':<8} {'MRR':<8} {'FP@1':<8} {'ans empty':<12} {'gap no-result':<14}",
        ]
    )
    for row in payload["sweep"]:
        lines.append(
            f"{row['threshold']:<8} {row['recall_at_1']:<8} {row['recall_at_5']:<8} {row['mrr']:<8} "
            f"{row['fp_at_1']:<8} {row['answerable_no_result_rate']:<12} {row['gap_no_result_accuracy']:<14}"
        )
    lines.extend(
        [
            "",
            f"RECOMMENDED RERANKER_MIN_SCORE={rec['recommended']}  (not applied)",
            f"overlap_between_first_relevant_and_gap={rec['overlap']}",
            rec["rationale"],
            f"metrics at recommended: {rec['at_recommended']}",
            f"gap-rejecting alternative: {rec['gap_rejecting_alternative']}",
            "",
            "=== Per-query first relevant (before threshold) ===",
            f"{'id':<28} {'cat':<16} {'rank':<6} {'rel score':<12} {'top1':<12}",
        ]
    )
    for row in payload["records"]:
        lines.append(
            f"{row['id']:<28} {row['category']:<16} {str(row['first_relevant_rank_before_threshold']):<6} "
            f"{str(row['first_relevant_score']):<12} {str(row['top1_score']):<12}"
        )
    lines.append("Do not modify .env from this report. Threshold is a recommendation only.")
    return lines
