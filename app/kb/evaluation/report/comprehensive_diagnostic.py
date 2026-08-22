"""Category and chunking diagnostic analysis for comprehensive retrieval audits."""

from __future__ import annotations

import json
from pathlib import Path
from typing import Any


CATEGORY_ORDER = [
    "PRODUCT",
    "SPECIFICATION",
    "HEARING_AID",
    "COMPANY",
    "POLICY",
    "SUMMARY",
    "NOISY_QUERY",
    "CORPUS_GAP",
]


def _pct(numerator: int, denominator: int) -> float:
    if denominator == 0:
        return 0.0
    return round(100.0 * numerator / denominator, 1)


def _category_metrics(queries: list[dict[str, Any]], category: str) -> dict[str, Any]:
    pool = [row for row in queries if row.get("category") == category]
    answerable = [row for row in pool if row.get("answerability") != "CORPUS_GAP"]
    total = len(answerable) or len(pool)
    if not pool:
        return {"count": 0, "answerable": 0, "pass_at_5_pct": 0.0, "pass_at_10_pct": 0.0}

    pass5 = sum(1 for row in answerable if row.get("passed_at_5"))
    pass10 = sum(1 for row in answerable if row.get("passed_at_10"))
    return {
        "count": len(pool),
        "answerable": len(answerable),
        "pass_at_5_pct": _pct(pass5, len(answerable)) if answerable else 0.0,
        "pass_at_10_pct": _pct(pass10, len(answerable)) if answerable else 0.0,
        "passed_at_5": pass5,
        "passed_at_10": pass10,
    }


def _rate(queries: list[dict[str, Any]], predicate) -> float:
    answerable = [row for row in queries if row.get("answerability") != "CORPUS_GAP"]
    if not answerable:
        return 0.0
    return round(sum(1 for row in answerable if predicate(row)) / len(answerable), 4)


def build_comprehensive_diagnostic(report: dict[str, Any]) -> dict[str, Any]:
    queries = report.get("queries", [])
    answerable = [row for row in queries if row.get("answerability") != "CORPUS_GAP"]
    corpus_gaps = [row for row in queries if row.get("answerability") == "CORPUS_GAP"]

    category_table = {category: _category_metrics(queries, category) for category in CATEGORY_ORDER}

    root_causes = report.get("root_cause_summary", {})
    embedding_share = root_causes.get("EMBEDDING_REPRESENTATION", 0)
    total_failures = sum(
        1 for row in answerable if not row.get("passed_at_10") and row.get("root_cause")
    )

    summary_missing = sum(
        1 for row in answerable if row.get("root_cause") == "SUMMARY_MISSING"
    )
    fragmentation = sum(
        1 for row in answerable if row.get("root_cause") in {"CHUNK_FRAGMENTATION", "CHUNK_GRANULARITY"}
    )

    chunking_quality = "GOOD"
    if fragmentation >= 3 or summary_missing >= 2:
        chunking_quality = "NEEDS_IMPROVEMENT"

    pass5_rate = _rate(answerable, lambda row: row.get("passed_at_5"))
    retrieval_quality = "GOOD" if pass5_rate >= 0.55 else "NEEDS_IMPROVEMENT"

    summary_pass = category_table.get("SUMMARY", {}).get("pass_at_5_pct", 0.0)
    summary_rep = "GOOD" if summary_pass >= 60.0 else "NEEDS_IMPROVEMENT"

    corpus_quality = "GOOD" if len(corpus_gaps) <= 4 else "NEEDS_IMPROVEMENT"

    embedding_model = "DO_NOT_CHANGE"
    if total_failures and embedding_share / max(total_failures, 1) >= 0.35:
        embedding_model = "INVESTIGATE"

    return {
        "dataset": report.get("dataset"),
        "corpus_version": report.get("corpus_version"),
        "questions_total": len(queries),
        "answerable_count": len(answerable),
        "corpus_gap_count": len(corpus_gaps),
        "overall_metrics": report.get("metrics", {}),
        "analysis": {
            "exact_lookup_pass_at_5": category_table.get("PRODUCT", {}).get("pass_at_5_pct", 0.0),
            "broad_query_pass_at_5": category_table.get("SUMMARY", {}).get("pass_at_5_pct", 0.0),
            "policy_pass_at_5": category_table.get("POLICY", {}).get("pass_at_5_pct", 0.0),
            "product_pass_at_5": category_table.get("PRODUCT", {}).get("pass_at_5_pct", 0.0),
            "company_pass_at_5": category_table.get("COMPANY", {}).get("pass_at_5_pct", 0.0),
            "noisy_query_pass_at_5": category_table.get("NOISY_QUERY", {}).get("pass_at_5_pct", 0.0),
            "expected_document_in_top_5_rate": _rate(
                answerable, lambda row: (row.get("expected_document_best_rank") or 999) <= 5
            ),
            "expected_chunk_in_top_5_rate": _rate(answerable, lambda row: row.get("passed_at_5")),
            "expected_document_in_top_10_rate": _rate(
                answerable, lambda row: (row.get("expected_document_best_rank") or 999) <= 10
            ),
            "expected_chunk_in_top_10_rate": _rate(answerable, lambda row: row.get("passed_at_10")),
        },
        "category_pass_at_5": {
            category: category_table[category]["pass_at_5_pct"] for category in CATEGORY_ORDER
        },
        "category_table": category_table,
        "failed_answerable_queries": [
            {
                "id": row["id"],
                "question": row["question"],
                "category": row.get("category"),
                "root_cause": row.get("root_cause"),
                "expected_chunk_rank": row.get("expected_chunk_rank"),
                "expected_document_best_rank": row.get("expected_document_best_rank"),
                "failure_analysis": row.get("failure_analysis"),
            }
            for row in answerable
            if not row.get("passed_at_10")
        ],
        "chunking_observations": _chunking_observations(queries, category_table),
        "verdicts": {
            "CHUNKING_QUALITY": chunking_quality,
            "RETRIEVAL_QUALITY": retrieval_quality,
            "SUMMARY_REPRESENTATION": summary_rep,
            "CORPUS_QUALITY": corpus_quality,
            "EMBEDDING_MODEL": embedding_model,
        },
    }


def _chunking_observations(queries: list[dict[str, Any]], category_table: dict[str, Any]) -> list[str]:
    observations: list[str] = []
    if category_table.get("PRODUCT", {}).get("pass_at_5_pct", 0) >= 70:
        observations.append("Atomic product/hearing-aid type chunks retrieve well for exact lookup queries.")
    if category_table.get("SUMMARY", {}).get("pass_at_5_pct", 0) < 60:
        observations.append("Broad summary queries still struggle — verify section_summary chunk coverage.")
    if category_table.get("SPECIFICATION", {}).get("pass_at_5_pct", 0) < 50:
        observations.append("Product specification chunks may be too fragmented or lack dedicated summary context.")
    omni_gap = any(row["id"] == "ERK-C03" and row.get("answerability") == "CORPUS_GAP" for row in queries)
    if omni_gap:
        observations.append("OMNI product knowledge is absent beyond merged metadata — true corpus gap.")
    battery_gap = any(
        row["id"] in {"ERK-C07", "ERK-C43"} and row.get("answerability") == "CORPUS_GAP" for row in queries
    )
    if battery_gap:
        observations.append("Battery-life duration specifications are not present in the two-PDF corpus.")
    if not observations:
        observations.append("No major chunking structural issues detected at Pass@5 threshold.")
    return observations


def write_chunking_diagnostic(
    *,
    reports_dir: Path,
    diagnostic: dict[str, Any],
) -> tuple[Path, Path]:
    out_dir = reports_dir / "retrieval"
    out_dir.mkdir(parents=True, exist_ok=True)
    json_path = out_dir / "earkart_kb_v3_1_chunking_diagnostic.json"
    txt_path = out_dir / "earkart_kb_v3_1_chunking_diagnostic.txt"
    json_path.write_text(json.dumps(diagnostic, indent=2), encoding="utf-8")
    txt_path.write_text(format_chunking_diagnostic(diagnostic), encoding="utf-8")
    return json_path, txt_path


def format_chunking_diagnostic(diagnostic: dict[str, Any]) -> str:
    metrics = diagnostic.get("overall_metrics", {})
    analysis = diagnostic.get("analysis", {})
    category = diagnostic.get("category_pass_at_5", {})
    verdicts = diagnostic.get("verdicts", {})
    lines = [
        "PHASE17_RETRIEVAL_AUDIT",
        "",
        f"Questions: {diagnostic.get('questions_total')}",
        f"Answerable: {diagnostic.get('answerable_count')}",
        f"Corpus gaps: {diagnostic.get('corpus_gap_count')}",
        "",
        f"Recall@1: {metrics.get('recall_at_1')}",
        f"Recall@3: {metrics.get('recall_at_3')}",
        f"Recall@5: {metrics.get('recall_at_5')}",
        f"Recall@10: {metrics.get('recall_at_10')}",
        f"MRR: {metrics.get('mrr')}",
        "",
        f"Expected chunk in top-5: {analysis.get('expected_chunk_in_top_5_rate')}",
        f"Expected chunk in top-10: {analysis.get('expected_chunk_in_top_10_rate')}",
        "",
        "CATEGORY                  PASS@5",
        "--------------------------------",
    ]
    labels = {
        "PRODUCT": "Product",
        "SPECIFICATION": "Specification",
        "HEARING_AID": "Hearing Aid",
        "COMPANY": "Company",
        "POLICY": "Policy",
        "SUMMARY": "Summary",
        "NOISY_QUERY": "Noisy Query",
    }
    for key, label in labels.items():
        lines.append(f"{label:<26}{category.get(key, 0.0)}%")
    lines.extend(
        [
            "",
            f"CHUNKING_QUALITY: {verdicts.get('CHUNKING_QUALITY')}",
            f"RETRIEVAL_QUALITY: {verdicts.get('RETRIEVAL_QUALITY')}",
            f"SUMMARY_REPRESENTATION: {verdicts.get('SUMMARY_REPRESENTATION')}",
            f"CORPUS_QUALITY: {verdicts.get('CORPUS_QUALITY')}",
            f"EMBEDDING_MODEL: {verdicts.get('EMBEDDING_MODEL')}",
            "",
            "Chunking observations:",
        ]
    )
    for item in diagnostic.get("chunking_observations", []):
        lines.append(f"  - {item}")
    lines.append("")
    lines.append("Failed answerable queries:")
    for row in diagnostic.get("failed_answerable_queries", []):
        lines.append(
            f"  {row['id']} [{row.get('category')}] rank={row.get('expected_chunk_rank')} "
            f"root_cause={row.get('root_cause')}"
        )
    return "\n".join(lines)


def print_phase17_summary(diagnostic: dict[str, Any]) -> None:
    print(format_chunking_diagnostic(diagnostic))
