#!/usr/bin/env python3
"""Unified retrieval evaluation CLI — corpus version is configuration."""

from __future__ import annotations

import argparse
import sys
from pathlib import Path

ROOT = Path(__file__).resolve().parents[1]
if str(ROOT) not in sys.path:
    sys.path.insert(0, str(ROOT))

from app.config import settings
from app.kb.evaluation.report.retrieval_report import write_retrieval_report
from app.kb.evaluation.resolver.expected import load_evaluation_questions
from app.kb.evaluation.retrieval_config import RetrievalConfig
from app.kb.evaluation.retrieval_runner import run_retrieval_evaluation


def _default_questions_path() -> Path:
    return ROOT / "data" / "evaluations" / "earkart_kb_v2_9.json"


def _default_config_path(corpus_version: str) -> Path:
    safe = corpus_version.replace(".", "_")
    return ROOT / "configs" / "retrieval" / f"{safe}.yaml"


def _dataset_name(questions_path: Path) -> str:
    import json

    payload = json.loads(questions_path.read_text(encoding="utf-8"))
    if isinstance(payload, dict) and payload.get("dataset"):
        return str(payload["dataset"])
    return questions_path.stem


def build_parser() -> argparse.ArgumentParser:
    parser = argparse.ArgumentParser(description="Evaluate retrieval for any corpus version.")
    parser.add_argument("--corpus-version", required=True, help="Corpus version (v2, v3, v3.1, …)")
    parser.add_argument("--config", type=Path, help="Retrieval YAML config (defaults to configs/retrieval/<version>.yaml)")
    parser.add_argument("--vector-table", help="Override vector table for pgvector mode")
    parser.add_argument(
        "--questions",
        type=Path,
        default=_default_questions_path(),
        help="Question dataset JSON with stable expected anchors",
    )
    parser.add_argument("--top-k", type=int, help="Override retrieval top-k")
    parser.add_argument(
        "--mode",
        choices=["pgvector", "in-memory"],
        help="Retrieval backend mode (defaults from config)",
    )
    parser.add_argument("--chunks", type=Path, help="Optional chunks directory hint (in-memory mode)")
    parser.add_argument("--device", default=None, help="Embedding device override")
    parser.add_argument("--reports-dir", type=Path, default=settings.reports_dir)
    parser.add_argument("--report-dataset", help="Override dataset name used in report filenames")
    parser.add_argument("--dry-run", action="store_true", help="Resolve anchors only; skip embedding/search")
    parser.add_argument(
        "--diagnostic",
        action="store_true",
        help="Emit comprehensive chunking diagnostic report (uses top-100 hit detail)",
    )
    return parser


def main() -> int:
    args = build_parser().parse_args()
    config_path = args.config or _default_config_path(args.corpus_version)
    config = RetrievalConfig.from_yaml(config_path)

    overrides: dict = {"corpus_version": args.corpus_version}
    if args.vector_table:
        overrides["vector_table"] = args.vector_table
    if args.top_k:
        overrides["top_k"] = args.top_k
    if args.mode:
        overrides["mode"] = args.mode
    if args.chunks:
        overrides["chunks_path"] = str(args.chunks)

    from dataclasses import replace

    config = replace(config, **overrides)

    questions = load_evaluation_questions(args.questions)
    dataset = args.report_dataset or _dataset_name(args.questions)

    if args.dry_run:
        from app.kb.evaluation.corpus_registry import load_corpus
        from app.kb.evaluation.resolver.expected import build_document_label_map, resolve_question_expected

        corpus = load_corpus(config.corpus_version)
        labels = build_document_label_map(corpus.chunks, corpus.source_pdfs)
        for question in questions:
            resolved = resolve_question_expected(question, corpus.chunks, document_labels=labels)
            print(
                f"{question.id}: {resolved.answerability} "
                f"category={question.category} chunks={len(resolved.expected_chunk_ids)}"
            )
        return 0

    top_n_report = 100 if args.diagnostic else 10
    report = run_retrieval_evaluation(
        config=config,
        questions=questions,
        device=args.device,
        top_n_report=top_n_report,
    )
    report["dataset"] = dataset
    json_path, txt_path = write_retrieval_report(
        reports_dir=args.reports_dir,
        dataset=dataset,
        corpus_version=config.corpus_version,
        payload=report,
    )
    metrics = report["metrics"]
    print(f"Recall@10={metrics['recall_at_10']} MRR={metrics['mrr']} Passed@10={metrics['passed_at_10']}")
    print(f"Report: {json_path}")
    print(f"Report: {txt_path}")

    if args.diagnostic:
        from app.kb.evaluation.report.comprehensive_diagnostic import (
            build_comprehensive_diagnostic,
            print_phase17_summary,
            write_chunking_diagnostic,
        )

        diagnostic = build_comprehensive_diagnostic(report)
        diag_json, diag_txt = write_chunking_diagnostic(reports_dir=args.reports_dir, diagnostic=diagnostic)
        print(f"Diagnostic: {diag_json}")
        print(f"Diagnostic: {diag_txt}")
        print()
        print_phase17_summary(diagnostic)

    return 0


if __name__ == "__main__":
    raise SystemExit(main())
