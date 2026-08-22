"""Combined retrieval recall: qc35 question coverage + golden Excel Q&A."""

from __future__ import annotations

import argparse
import json
import sys
from pathlib import Path

from app.config import settings
from app.kb.embedding.config import EmbeddingConfig
from app.kb.embedding.loader import ChunkLoader
from app.kb.evaluation.baseline import build_baseline_row, format_baseline_table
from app.kb.evaluation.combined_runner import CombinedRetrievalEvaluationRunner
from app.kb.evaluation.golden_excel import DEFAULT_GOLDEN_EXCEL_PATH, DEFAULT_GOLDEN_SHEET
from app.kb.vector.search import VectorSearchService
from app.kb.vector.store import VectorStore


def _build_parser() -> argparse.ArgumentParser:
    parser = argparse.ArgumentParser(
        description="Evaluate combined chunk-ID recall on qc35 + golden Excel questions.",
    )
    parser.add_argument(
        "--excel",
        type=Path,
        default=DEFAULT_GOLDEN_EXCEL_PATH,
        help=f"Golden Excel path (default: {DEFAULT_GOLDEN_EXCEL_PATH})",
    )
    parser.add_argument(
        "--sheet",
        default=DEFAULT_GOLDEN_SHEET,
        help=f"Golden Excel sheet (default: {DEFAULT_GOLDEN_SHEET})",
    )
    parser.add_argument(
        "--top-k",
        type=int,
        default=10,
        help="Top-k retrieval depth (default: 10)",
    )
    return parser


def main() -> int:
    args = _build_parser().parse_args()
    config = EmbeddingConfig.from_settings()
    reports_dir = settings.reports_dir
    reports_dir.mkdir(parents=True, exist_ok=True)

    loader = ChunkLoader(settings.chunks_dir, config.embedding_input_manifest)
    store = VectorStore(config.database_url, config.vector_table, config.dimension)
    search = VectorSearchService(config=config, store=store)

    report = CombinedRetrievalEvaluationRunner(
        config=config,
        loader=loader,
        search=search,
        excel_path=args.excel,
        sheet_name=args.sheet,
    ).run(top_k=args.top_k)

    baseline_row = build_baseline_row(report)
    report["baseline_row"] = baseline_row

    json_path = reports_dir / "combined_retrieval_evaluation.json"
    txt_path = reports_dir / "combined_retrieval_evaluation.txt"
    json_path.write_text(json.dumps(report, ensure_ascii=False, indent=2), encoding="utf-8")

    metrics = report["metrics"]
    qc35 = report["datasets"]["qc35_v1"]["metrics"]
    golden = report["datasets"]["golden_excel_v1"]["metrics"]

    lines = [
        "Combined Retrieval Evaluation",
        f"Evaluation dataset: {report['evaluation_dataset_version']}",
        f"Embedding model: {config.model}",
        f"Embedding version: {config.embedding_version}",
        "",
        f"Total questions: {report['total_evaluation_queries']}",
        f"  qc35_v1: {report['datasets']['qc35_v1']['total_queries']}",
        f"  golden_excel_v1: {report['datasets']['golden_excel_v1']['total_queries']} "
        f"(skipped {report['datasets']['golden_excel_v1']['skipped_queries']} Excel rows)",
        "",
        "Combined recall",
        f"Recall@1: {metrics['recall_at_1']}",
        f"Recall@3: {metrics['recall_at_3']}",
        f"Recall@5: {metrics['recall_at_5']}",
        f"Recall@10: {metrics['recall_at_10']}",
        f"MRR: {metrics['mrr']}",
        f"Passed queries: {report['passed_queries']}/{report['total_evaluation_queries']}",
        "",
        "qc35_v1 recall",
        f"Recall@1: {qc35['recall_at_1']}",
        f"Recall@3: {qc35['recall_at_3']}",
        f"Recall@5: {qc35['recall_at_5']}",
        f"Recall@10: {qc35['recall_at_10']}",
        f"MRR: {qc35['mrr']}",
        f"Passed: {report['datasets']['qc35_v1']['passed_queries']}/{qc35['total']}",
        "",
        "golden_excel_v1 recall",
        f"Recall@1: {golden['recall_at_1']}",
        f"Recall@3: {golden['recall_at_3']}",
        f"Recall@5: {golden['recall_at_5']}",
        f"Recall@10: {golden['recall_at_10']}",
        f"MRR: {golden['mrr']}",
        f"Passed: {report['datasets']['golden_excel_v1']['passed_queries']}/{golden['total']}",
        "",
        "Baseline table:",
        format_baseline_table([baseline_row]),
        "",
        f"Report JSON: {json_path}",
    ]
    txt_path.write_text("\n".join(lines), encoding="utf-8")
    print("\n".join(lines))
    return 0 if report["passed"] else 1


if __name__ == "__main__":
    sys.exit(main())
