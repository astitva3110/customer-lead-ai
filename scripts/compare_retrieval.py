#!/usr/bin/env python3
"""Compare retrieval evaluation results across corpus versions."""

from __future__ import annotations

import argparse
import json
import sys
from pathlib import Path

ROOT = Path(__file__).resolve().parents[1]
if str(ROOT) not in sys.path:
    sys.path.insert(0, str(ROOT))

from app.config import settings
from app.kb.evaluation.comparison_runner import compare_retrieval
from app.kb.evaluation.report.retrieval_report import format_comparison_report, report_paths
from app.kb.evaluation.retrieval_config import RetrievalConfig


def _default_questions_path(dataset: str) -> Path:
    return ROOT / "data" / "evaluations" / f"{dataset}.json"


def _config_for_version(version: str) -> RetrievalConfig:
    safe = version.replace(".", "_")
    return RetrievalConfig.from_yaml(ROOT / "configs" / "retrieval" / f"{safe}.yaml")


def build_parser() -> argparse.ArgumentParser:
    parser = argparse.ArgumentParser(description="Compare retrieval metrics between corpus versions.")
    parser.add_argument("--baseline", required=True, help="Baseline corpus version (e.g. v3)")
    parser.add_argument("--candidate", required=True, help="Candidate corpus version (e.g. v3.1)")
    parser.add_argument("--dataset", default="earkart_kb_v2_9", help="Evaluation dataset name")
    parser.add_argument("--questions", type=Path, help="Override questions JSON path")
    parser.add_argument("--device", default=None)
    parser.add_argument("--reports-dir", type=Path, default=settings.reports_dir)
    parser.add_argument("--report-stem", help="Override comparison report filename stem")
    return parser


def main() -> int:
    args = build_parser().parse_args()
    questions_path = args.questions or _default_questions_path(args.dataset)
    baseline_config = _config_for_version(args.baseline)
    candidate_config = _config_for_version(args.candidate)

    comparison = compare_retrieval(
        baseline_config=baseline_config,
        candidate_config=candidate_config,
        questions_path=questions_path,
        device=args.device,
    )

    out_dir = args.reports_dir / "retrieval"
    out_dir.mkdir(parents=True, exist_ok=True)
    stem = args.report_stem or (
        f"{args.dataset}_{args.baseline.replace('.', '_')}_vs_{args.candidate.replace('.', '_')}"
    )
    json_path = out_dir / f"{stem}.json"
    txt_path = out_dir / f"{stem}.txt"
    json_path.write_text(json.dumps(comparison, indent=2), encoding="utf-8")
    txt_path.write_text(format_comparison_report(comparison), encoding="utf-8")

    deltas = comparison["metric_deltas"]
    print(f"Recall@10 delta: {deltas['recall_at_10']}  MRR delta: {deltas['mrr']}")
    print(f"Newly passed: {comparison['newly_passed']}")
    print(f"Newly failed: {comparison['newly_failed']}")
    print(f"Report: {json_path}")
    print(f"Report: {txt_path}")
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
