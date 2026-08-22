"""Compare retrieval evaluation results across corpus versions."""

from __future__ import annotations

from typing import Any

from app.kb.evaluation.retrieval_config import RetrievalConfig
from app.kb.evaluation.retrieval_runner import run_retrieval_evaluation
from app.kb.evaluation.resolver.expected import load_evaluation_questions


from app.kb.evaluation.report.comprehensive_diagnostic import _category_metrics


def _expected_rank_similarity(query_row: dict[str, Any]) -> float | None:
    rank = query_row.get("expected_chunk_rank")
    if rank is None:
        return None
    for hit in query_row.get("top_results", []):
        if hit.get("rank") == rank:
            return float(hit.get("similarity") or 0.0)
    return None


def _category_pass_at_5(report: dict[str, Any], category: str) -> float:
    return _category_metrics(report.get("queries", []), category).get("pass_at_5_pct", 0.0)


def compare_retrieval(
    *,
    baseline_config: RetrievalConfig,
    candidate_config: RetrievalConfig,
    questions_path: str,
    device: str | None = None,
) -> dict[str, Any]:
    questions = load_evaluation_questions(questions_path)
    baseline = run_retrieval_evaluation(config=baseline_config, questions=questions, device=device)
    candidate = run_retrieval_evaluation(config=candidate_config, questions=questions, device=device)

    baseline_by_id = {row["id"]: row for row in baseline["queries"]}
    candidate_by_id = {row["id"]: row for row in candidate["queries"]}

    per_query: list[dict[str, Any]] = []
    newly_passed: list[str] = []
    newly_failed: list[str] = []

    for question in questions:
        base = baseline_by_id[question.id]
        cand = candidate_by_id[question.id]
        base_rank = base.get("expected_chunk_rank")
        cand_rank = cand.get("expected_chunk_rank")
        rank_delta = None
        if base_rank is not None and cand_rank is not None:
            rank_delta = base_rank - cand_rank

        base_pass = base.get("passed_at_10", False)
        cand_pass = cand.get("passed_at_10", False)
        base_similarity = _expected_rank_similarity(base)
        cand_similarity = _expected_rank_similarity(cand)
        if not base_pass and cand_pass:
            newly_passed.append(question.id)
        if base_pass and not cand_pass:
            newly_failed.append(question.id)

        status = "unchanged"
        if rank_delta is not None and rank_delta > 0:
            status = "improved"
        elif rank_delta is not None and rank_delta < 0:
            status = "regressed"

        per_query.append(
            {
                "id": question.id,
                "question": question.question,
                "category": question.category,
                "baseline_rank": base_rank,
                "candidate_rank": cand_rank,
                "rank_delta": rank_delta,
                "baseline_similarity": base_similarity,
                "candidate_similarity": cand_similarity,
                "baseline_passed_at_10": base_pass,
                "candidate_passed_at_10": cand_pass,
                "baseline_root_cause": base.get("root_cause"),
                "candidate_root_cause": cand.get("root_cause"),
                "improved": rank_delta is not None and rank_delta > 0,
                "regressed": rank_delta is not None and rank_delta < 0,
                "status": status,
            }
        )

    base_metrics = baseline["metrics"]
    cand_metrics = candidate["metrics"]

    return {
        "baseline_version": baseline_config.corpus_version,
        "candidate_version": candidate_config.corpus_version,
        "dataset_questions": len(questions),
        "metric_deltas": {
            "recall_at_1": round(cand_metrics["recall_at_1"] - base_metrics["recall_at_1"], 4),
            "recall_at_3": round(cand_metrics["recall_at_3"] - base_metrics["recall_at_3"], 4),
            "recall_at_5": round(cand_metrics["recall_at_5"] - base_metrics["recall_at_5"], 4),
            "recall_at_10": round(cand_metrics["recall_at_10"] - base_metrics["recall_at_10"], 4),
            "mrr": round(cand_metrics["mrr"] - base_metrics["mrr"], 4),
        },
        "category_pass_at_5": {
            "PRODUCT": _category_pass_at_5(baseline, "PRODUCT"),
            "SPECIFICATION": _category_pass_at_5(baseline, "SPECIFICATION"),
            "HEARING_AID": _category_pass_at_5(baseline, "HEARING_AID"),
            "COMPANY": _category_pass_at_5(baseline, "COMPANY"),
            "POLICY": _category_pass_at_5(baseline, "POLICY"),
            "SUMMARY": _category_pass_at_5(baseline, "SUMMARY"),
            "NOISY_QUERY": _category_pass_at_5(baseline, "NOISY_QUERY"),
        },
        "category_pass_at_5_candidate": {
            "PRODUCT": _category_pass_at_5(candidate, "PRODUCT"),
            "SPECIFICATION": _category_pass_at_5(candidate, "SPECIFICATION"),
            "HEARING_AID": _category_pass_at_5(candidate, "HEARING_AID"),
            "COMPANY": _category_pass_at_5(candidate, "COMPANY"),
            "POLICY": _category_pass_at_5(candidate, "POLICY"),
            "SUMMARY": _category_pass_at_5(candidate, "SUMMARY"),
            "NOISY_QUERY": _category_pass_at_5(candidate, "NOISY_QUERY"),
        },
        "category_pass_at_5_delta": {
            key: round(
                _category_pass_at_5(candidate, key) - _category_pass_at_5(baseline, key),
                1,
            )
            for key in ("PRODUCT", "SPECIFICATION", "HEARING_AID", "COMPANY", "POLICY", "SUMMARY", "NOISY_QUERY")
        },
        "baseline_metrics": base_metrics,
        "candidate_metrics": cand_metrics,
        "newly_passed": newly_passed,
        "newly_failed": newly_failed,
        "per_query": per_query,
        "baseline_report": baseline,
        "candidate_report": candidate,
    }
