#!/usr/bin/env python3
"""Generate an exploratory scenario dataset. Offline. Not part of /chat."""

from __future__ import annotations

import argparse
import sys
from pathlib import Path

ROOT = Path(__file__).resolve().parents[1]
if str(ROOT) not in sys.path:
    sys.path.insert(0, str(ROOT))

from app.evaluation.config import load_mass_eval_config
from app.evaluation.generator import generate_dataset
from app.evaluation.schema import dump_generated_dataset, validate_dataset_payload


def main() -> None:
    parser = argparse.ArgumentParser()
    parser.add_argument("--config", default="configs/evaluation/mass_conversation.yaml")
    parser.add_argument("--count", type=int, default=None)
    parser.add_argument("--seed", type=int, default=None)
    parser.add_argument("--output", default=None)
    parser.add_argument("--use-llm", action="store_true")
    parser.add_argument("--no-llm", action="store_true")
    args = parser.parse_args()
    config = load_mass_eval_config(args.config)
    updates = {}
    if args.seed is not None:
        updates["seed"] = args.seed
    if args.count is not None:
        updates["count"] = args.count
    if updates:
        payload = config.to_dict()
        payload.update(updates)
        from app.evaluation.config import MassEvalConfig

        config = MassEvalConfig.from_dict(payload)
    use_llm = False if args.no_llm else (True if args.use_llm else config.use_llm)
    dataset = generate_dataset(config, count=args.count, use_llm=use_llm)
    validate_dataset_payload(dataset)
    output = Path(args.output) if args.output else config.generated_path()
    dump_generated_dataset(dataset, output)
    print(f"wrote {len(dataset.conversations)} conversations to {output}")


if __name__ == "__main__":
    main()
