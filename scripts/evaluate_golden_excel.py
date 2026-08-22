"""Evaluate retrieval recall against a golden Q&A Excel dataset."""

from __future__ import annotations

import argparse
import json
import sys
from pathlib import Path

from app.config import settings
from app.kb.embedding.config import EmbeddingConfig
from app.kb.embedding.loader import ChunkLoader
from app.kb.evaluation.baseline import build_baseline_row, format_baseline_table
from app.kb.evaluation.golden_excel import DEFAULT_GOLDEN_EXCEL_PATH, DEFAULT_GOLDEN_SHEET
from app.kb.evaluation.golden_runner import GoldenExcelEvaluationRunner
from app.kb.vector.search import VectorSearchService
from app.kb.vector.store import VectorStore


def _build_parser() -> argparse.ArgumentParser:
    parser = argparse.ArgumentParser(
        description="Evaluate chunk-ID retrieval recall using a golden Q&A Excel dataset.",
    )
    parser.add_argument(
        "--excel",
        type=Path,
        default=DEFAULT_GOLDEN_EXCEL_PATH,
        help=f"Path to golden dataset Excel (default: {DEFAULT_GOLDEN_EXCEL_PATH})",
    )
    parser.add_argument(
        "--sheet",
        default=DEFAULT_GOLDEN_SHEET,
        help=f"Worksheet name (default: {DEFAULT_GOLDEN_SHEET})",
    )
    parser.add_argument(
        "--top-k",
        type=int,
        default=10,
        help="Top-k retrieval depth for recall (default: 10)",
    )
    parser.add_argument(
        "--resolve-only",
        action="store_true",
        help="Only resolve golden rows to expected chunk IDs; skip vector search",
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

    runner = GoldenExcelEvaluationRunner(
        config=config,
        loader=loader,
        search=search,
        excel_path=args.excel,
        sheet_name=args.sheet,
    )
    report = runner.run(top_k=args.top_k, resolve_only=args.resolve_only)

    if args.resolve_only:
        json_path = reports_dir / "golden_excel_chunk_resolution.json"
        txt_path = reports_dir / "golden_excel_chunk_resolution.txt"
    else:
        json_path = reports_dir / "golden_excel_retrieval_evaluation.json"
        txt_path = reports_dir / "golden_excel_retrieval_evaluation.txt"

    json_path.write_text(json.dumps(report, ensure_ascii=False, indent=2), encoding="utf-8")

    if args.resolve_only:
        lines = [
            "Golden Excel Chunk Resolution",
            f"Excel: {report['excel_path']}",
            f"Sheet: {report['sheet_name']}",
            f"Total rows: {report['total_rows']}",
            f"Resolved cases: {report['resolved_cases']}",
            f"Skipped cases: {report['skipped_cases']}",
        ]
        if report["skipped_cases"]:
            lines.extend(["", "Skipped IDs:"])
            for case in report["skipped"][:15]:
                lines.append(f"  - {case['id']}: {case.get('skipped_reason')}")
        txt_path.write_text("\n".join(lines), encoding="utf-8")
        print("\n".join(lines))
        return 0

    metrics = report["metrics"]
    baseline_row = build_baseline_row(report)
    report["baseline_row"] = baseline_row
    json_path.write_text(json.dumps(report, ensure_ascii=False, indent=2), encoding="utf-8")

    lines = [
        "Golden Excel Retrieval Evaluation",
        f"Evaluation dataset: {report['evaluation_dataset_version']}",
        f"Excel: {report['excel_path']}",
        f"Sheet: {report['sheet_name']}",
        f"Embedding model: {config.model}",
        f"Embedding version: {config.embedding_version}",
        "",
        f"Resolved rows: {report['total_evaluation_queries']}/{report['total_rows']}",
        f"Skipped rows: {report['skipped_rows']}",
        "",
        f"Recall@1: {metrics['recall_at_1']}",
        f"Recall@3: {metrics['recall_at_3']}",
        f"Recall@5: {metrics['recall_at_5']}",
        f"Recall@10: {metrics['recall_at_10']}",
        f"MRR: {metrics['mrr']}",
        "",
        f"Passed queries: {report['passed_queries']}/{report['total_evaluation_queries']}",
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
