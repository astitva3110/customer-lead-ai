"""Baseline report helpers for embedding model comparison."""

from __future__ import annotations

from typing import Any


def build_baseline_row(eval_report: dict[str, Any]) -> dict[str, Any]:
    metrics = eval_report["metrics"]
    return {
        "embedding_model": eval_report["embedding_model"],
        "embedding_version": eval_report["embedding_version"],
        "embedding_model_revision": eval_report["embedding_model_revision"],
        "evaluation_dataset_version": eval_report["evaluation_dataset_version"],
        "recall_at_1": metrics["recall_at_1"],
        "recall_at_3": metrics["recall_at_3"],
        "recall_at_5": metrics["recall_at_5"],
        "recall_at_10": metrics["recall_at_10"],
        "mrr": metrics["mrr"],
    }


def format_baseline_table(rows: list[dict[str, Any]]) -> str:
    header = "Embedding Model | Recall@1 | Recall@3 | Recall@5 | Recall@10 | MRR"
    lines = [header, "-" * len(header)]
    for row in rows:
        lines.append(
            f"{row['embedding_model']} | {row['recall_at_1']} | {row['recall_at_3']} | "
            f"{row['recall_at_5']} | {row['recall_at_10']} | {row['mrr']}"
        )
    return "\n".join(lines)
