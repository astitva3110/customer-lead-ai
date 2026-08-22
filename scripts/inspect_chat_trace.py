from __future__ import annotations

import argparse
import json
import sys
from pathlib import Path

ROOT = Path(__file__).resolve().parents[1]
if str(ROOT) not in sys.path:
    sys.path.insert(0, str(ROOT))

from app.config import settings
from app.services.diagnostics.report import format_chat_trace_text, load_chat_trace


def main(argv: list[str] | None = None) -> int:
    parser = argparse.ArgumentParser(description="Print a saved chat diagnostic trace.")
    parser.add_argument("--trace-id", required=True)
    parser.add_argument("--json", action="store_true")
    parser.add_argument("--output-dir", default=None)
    args = parser.parse_args(argv)
    directory = Path(args.output_dir) if args.output_dir else settings.chat_trace_output_dir
    payload = load_chat_trace(args.trace_id, directory)
    if args.json:
        sys.stdout.write(json.dumps(payload, indent=2, default=str) + "\n")
    else:
        sys.stdout.write(format_chat_trace_text(payload) + "\n")
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
