"""Unified retrieval evaluation report writer."""

from __future__ import annotations

import json
from pathlib import Path
from typing import Any


def report_paths(reports_dir: Path, dataset: str, corpus_version: str) -> tuple[Path, Path]:
    out_dir = reports_dir / "retrieval"
    out_dir.mkdir(parents=True, exist_ok=True)
    stem = f"{dataset}_{corpus_version.replace('.', '_')}"
    return out_dir / f"{stem}.json", out_dir / f"{stem}.txt"


def write_retrieval_report(
    *,
    reports_dir: Path,
    dataset: str,
    corpus_version: str,
    payload: dict[str, Any],
) -> tuple[Path, Path]:
    json_path, txt_path = report_paths(reports_dir, dataset, corpus_version)
    json_path.write_text(json.dumps(payload, indent=2), encoding="utf-8")
    txt_path.write_text(format_retrieval_report(payload), encoding="utf-8")
    return json_path, txt_path


def format_retrieval_report(report: dict[str, Any]) -> str:
    metrics = report["metrics"]
    lines = [
        "RETRIEVAL_EVALUATION",
        f"Dataset: {report.get('dataset')}",
        f"Corpus version: {report.get('corpus_version')}",
        f"Mode: {report.get('mode')}",
        f"Vector table: {report.get('vector_table')}",
        "",
        f"Questions: {report.get('total_questions')}",
        f"Recall@1: {metrics['recall_at_1']}",
        f"Recall@3: {metrics['recall_at_3']}",
        f"Recall@5: {metrics['recall_at_5']}",
        f"Recall@10: {metrics['recall_at_10']}",
        f"MRR: {metrics['mrr']}",
        "",
        f"Passed@1: {metrics['passed_at_1']}",
        f"Passed@3: {metrics['passed_at_3']}",
        f"Passed@5: {metrics['passed_at_5']}",
        f"Passed@10: {metrics['passed_at_10']}",
        "",
        "Root-cause summary:",
    ]
    for label, count in report.get("root_cause_summary", {}).items():
        lines.append(f"  {label}: {count}")
    lines.extend(["", "Per-query:"])
    for query in report.get("queries", []):
        lines.extend(
            [
                f"  {query['id']}: {query['question']}",
                f"    rank={query.get('expected_chunk_rank')} "
                f"doc_rank={query.get('expected_document_best_rank')} "
                f"recall@10={query.get('recall_at_10')} root_cause={query.get('root_cause')}",
            ]
        )
    return "\n".join(lines)


def format_comparison_report(comparison: dict[str, Any]) -> str:
    deltas = comparison["metric_deltas"]
    lines = [
        "RETRIEVAL_COMPARISON",
        f"Baseline: {comparison['baseline_version']}",
        f"Candidate: {comparison['candidate_version']}",
        "",
        f"Recall@1 delta: {deltas['recall_at_1']}",
        f"Recall@3 delta: {deltas['recall_at_3']}",
        f"Recall@5 delta: {deltas['recall_at_5']}",
        f"Recall@10 delta: {deltas['recall_at_10']}",
        f"MRR delta: {deltas['mrr']}",
        "",
        f"Newly passed: {comparison['newly_passed']}",
        f"Newly failed: {comparison['newly_failed']}",
        "",
        "Per-query rank changes:",
    ]
    for row in comparison["per_query"]:
        lines.append(
            f"  {row['id']}: {row['baseline_rank']} -> {row['candidate_rank']} "
            f"(delta={row['rank_delta']})"
        )
    return "\n".join(lines)
