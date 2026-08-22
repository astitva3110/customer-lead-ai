#!/usr/bin/env python3
"""Observe saved exploratory scenarios through the existing /chat orchestrator."""

from __future__ import annotations

import argparse
import sys
from pathlib import Path

ROOT = Path(__file__).resolve().parents[1]
if str(ROOT) not in sys.path:
    sys.path.insert(0, str(ROOT))

from app.evaluation.analyze import print_analysis
from app.evaluation.config import load_mass_eval_config
from app.evaluation.runner import run_mass_evaluation
from app.evaluation.schema import load_generated_dataset


def main() -> None:
    parser = argparse.ArgumentParser()
    parser.add_argument("--config", default="configs/evaluation/mass_conversation.yaml")
    parser.add_argument("--dataset", default=None)
    parser.add_argument("--run-id", default=None)
    parser.add_argument("--resume", action="store_true")
    parser.add_argument("--force", action="store_true")
    parser.add_argument("--concurrency", type=int, default=None)
    args = parser.parse_args()
    config = load_mass_eval_config(args.config)
    if args.concurrency:
        payload = config.to_dict()
        payload["concurrency"] = args.concurrency
        from app.evaluation.config import MassEvalConfig

        config = MassEvalConfig.from_dict(payload)
    dataset_path = Path(args.dataset) if args.dataset else config.generated_path()
    dataset = load_generated_dataset(dataset_path)
    summary = run_mass_evaluation(
        dataset,
        config,
        run_id=args.run_id,
        resume=True if args.resume else None,
        force=args.force,
    )
    print(print_analysis(summary))
    print(f"summary: {summary.get('reports', {}).get('summary_json')}")


if __name__ == "__main__":
    main()
