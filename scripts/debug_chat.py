from __future__ import annotations

import argparse
import sys
from pathlib import Path

ROOT = Path(__file__).resolve().parents[1]
if str(ROOT) not in sys.path:
    sys.path.insert(0, str(ROOT))

from app.config import settings


def main(argv: list[str] | None = None) -> int:
    parser = argparse.ArgumentParser(
        description="Run one existing /chat turn and print the ChatTrace debug pipeline."
    )
    parser.add_argument("--conversation-id", required=True)
    parser.add_argument("--message", required=True)
    parser.add_argument("--output-dir", default=None)
    args = parser.parse_args(argv)
    settings.chat_trace_enabled = True
    settings.chat_debug_console = True
    settings.chat_trace_include_full_context = True
    settings.chat_trace_include_prompt = True
    if args.output_dir:
        settings.chat_trace_output_dir = Path(args.output_dir)
    from app.dependencies import get_orchestrator

    get_orchestrator().handle(args.conversation_id, args.message)
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
