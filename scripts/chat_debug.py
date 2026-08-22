from __future__ import annotations

import argparse
import json
import sys
from pathlib import Path

ROOT = Path(__file__).resolve().parents[1]
if str(ROOT) not in sys.path:
    sys.path.insert(0, str(ROOT))

from app.config import settings


def main(argv: list[str] | None = None) -> int:
    parser = argparse.ArgumentParser(description="Run one /chat turn with diagnostics enabled.")
    parser.add_argument("--conversation-id", required=True)
    parser.add_argument("--message", required=True)
    parser.add_argument("--output-dir", default=None)
    parser.add_argument("--full-context", action="store_true")
    args = parser.parse_args(argv)
    settings.chat_trace_enabled = True
    if args.output_dir:
        settings.chat_trace_output_dir = Path(args.output_dir)
    if args.full_context:
        settings.chat_trace_include_full_context = True
    from app.dependencies import get_orchestrator

    result = get_orchestrator().handle(args.conversation_id, args.message)
    print(
        json.dumps(
            {
                "conversation_id": result.conversation_id,
                "debug_trace_id": (result.trace or {}).get("trace_id"),
                "mode": result.mode,
                "response": result.response,
                "sources": result.sources,
            },
            indent=2,
            default=str,
        )
    )
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
