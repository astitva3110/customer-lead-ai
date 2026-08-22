#!/usr/bin/env python3
"""Print clustered results for an existing mass-eval run. Does not re-run /chat."""

from __future__ import annotations

import argparse
import sys
from pathlib import Path

ROOT = Path(__file__).resolve().parents[1]
if str(ROOT) not in sys.path:
    sys.path.insert(0, str(ROOT))

from app.evaluation.analyze import analyze_run, print_analysis, reaggregate_run
from app.evaluation.config import load_mass_eval_config
from app.evaluation.schema import load_generated_dataset


def main() -> None:
    parser = argparse.ArgumentParser()
    parser.add_argument("--run-id", required=True)
    parser.add_argument("--config", default="configs/evaluation/mass_conversation.yaml")
    parser.add_argument("--dataset", default=None)
    parser.add_argument("--reaggregate", action="store_true")
    args = parser.parse_args()
    config = load_mass_eval_config(args.config)
    if args.reaggregate:
        dataset_path = Path(args.dataset) if args.dataset else config.generated_path()
        dataset = load_generated_dataset(dataset_path)
        summary = reaggregate_run(config.output_path(), args.run_id, dataset, config)
    else:
        summary = analyze_run(config.output_path(), args.run_id)
    print(print_analysis(summary))


if __name__ == "__main__":
    main()
