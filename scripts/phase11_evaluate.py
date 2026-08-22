"""Phase 11 — retrieval evaluation and baseline report."""

from __future__ import annotations

import json
import sys

from app.config import settings
from app.kb.embedding.config import EmbeddingConfig
from app.kb.embedding.loader import ChunkLoader
from app.kb.evaluation.baseline import build_baseline_row, format_baseline_table
from app.kb.evaluation.regression import run_regression_checks
from app.kb.evaluation.runner import RetrievalEvaluationRunner
from app.kb.vector.search import VectorSearchService
from app.kb.vector.store import VectorStore


def main() -> int:
    config = EmbeddingConfig.from_settings()
    reports_dir = settings.reports_dir
    reports_dir.mkdir(parents=True, exist_ok=True)

    loader = ChunkLoader(settings.chunks_dir, config.embedding_input_manifest)
    store = VectorStore(config.database_url, config.vector_table, config.dimension)
    search = VectorSearchService(config=config, store=store)

    eval_report = RetrievalEvaluationRunner(config=config, loader=loader, search=search).run(top_k=10)
    regression = run_regression_checks(search, top_k=10)
    baseline_row = build_baseline_row(eval_report)

    eval_report["regression"] = regression
    eval_report["baseline_row"] = baseline_row

    json_path = reports_dir / "phase11_retrieval_evaluation.json"
    txt_path = reports_dir / "phase11_retrieval_evaluation.txt"
    json_path.write_text(json.dumps(eval_report, ensure_ascii=False, indent=2), encoding="utf-8")

    metrics = eval_report["metrics"]
    lines = [
        "Phase 11 Retrieval Evaluation",
        f"Evaluation dataset: {eval_report['evaluation_dataset_version']}",
        f"Embedding model: {config.model}",
        f"Embedding version: {config.embedding_version}",
        "",
        f"Recall@1: {metrics['recall_at_1']}",
        f"Recall@3: {metrics['recall_at_3']}",
        f"Recall@5: {metrics['recall_at_5']}",
        f"Recall@10: {metrics['recall_at_10']}",
        f"MRR: {metrics['mrr']}",
        "",
        f"Passed queries: {eval_report['passed_queries']}/{eval_report['total_evaluation_queries']}",
        f"Critical regression: {'PASS' if regression['passed'] else 'FAIL'}",
        "",
        "Baseline table:",
        format_baseline_table([baseline_row]),
    ]
    txt_path.write_text("\n".join(lines), encoding="utf-8")
    print("\n".join(lines))

    passed = eval_report["passed"] and regression["passed"]
    return 0 if passed else 1


if __name__ == "__main__":
    sys.exit(main())
