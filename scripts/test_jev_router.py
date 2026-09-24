"""Live worst-case Jev router evaluation (OpenRouter semantic + guardrails).

Usage:
    set PYTHONPATH=d:\\code\\office\\chatbot
    python scripts/test_jev_router.py

    python scripts/test_jev_router.py --category hinglish_acquisition
    python scripts/test_jev_router.py --delay 2.0
    python scripts/test_jev_router.py --failures-only
"""

from __future__ import annotations

import argparse
import json
import sys
import time
from collections import defaultdict
from datetime import UTC, datetime
from pathlib import Path

from app.config import settings
from app.providers.llm.factory import get_semantic_router_provider, semantic_router_model_label
from app.services.conversation.models import ConversationState
from app.services.conversation.query_rewriter import QueryRewriter
from app.services.conversation.router import ChatRouter
from tests.unit.conversation.worst_case_routing_cases import WORST_CASE_ROUTING_CASES, WorstCaseRoutingCase


def _matches_expected(routed, expected: str) -> bool:
    if expected == "GENERAL":
        return routed.current_turn_intent == "GENERAL"
    return routed.mode.value == expected


def _route_one(router: ChatRouter, case: WorstCaseRoutingCase) -> dict:
    state = ConversationState(user_message=case.message)
    QueryRewriter().normalize(state)
    routed = router.route(state)
    sem = (routed.trace or {}).get("semantic_router") or {}
    ok = _matches_expected(routed, case.expected)
    return {
        "id": case.id,
        "message": case.message,
        "category": case.category,
        "expected": case.expected,
        "actual_mode": routed.mode.value,
        "actual_turn": routed.current_turn_intent,
        "ok": ok,
        "semantic_used": routed.trace.get("semantic_router_used"),
        "semantic_route": sem.get("route"),
        "fallback": sem.get("fallback_reason"),
        "llm_raw": (sem.get("raw_output") or "")[:200],
        "hearing_symptom": (routed.trace or {}).get("hearing_symptom"),
    }


def main() -> int:
    parser = argparse.ArgumentParser(description="Worst-case Jev router live test")
    parser.add_argument("--category", default="", help="Filter by category substring")
    parser.add_argument("--delay", type=float, default=1.5, help="Seconds between OpenRouter calls")
    parser.add_argument("--failures-only", action="store_true")
    parser.add_argument("--output", default="reports/jev_router_worst_case.json")
    args = parser.parse_args()

    cases = [
        case
        for case in WORST_CASE_ROUTING_CASES
        if not args.category or args.category in case.category
    ]
    if not cases:
        print("No cases matched filter.", file=sys.stderr)
        return 1

    router = ChatRouter(router_llm=get_semantic_router_provider(settings))
    print("Jev worst-case router test")
    print("Router model:", semantic_router_model_label(settings))
    print("Generation model:", settings.generation_model)
    print("Cases:", len(cases))
    print("-" * 72)

    results: list[dict] = []
    by_category: dict[str, list[bool]] = defaultdict(list)

    for index, case in enumerate(cases, start=1):
        try:
            row = _route_one(router, case)
        except Exception as exc:
            row = {
                "id": case.id,
                "message": case.message,
                "category": case.category,
                "expected": case.expected,
                "ok": False,
                "error": f"{type(exc).__name__}: {exc}",
            }
        results.append(row)
        by_category[case.category].append(bool(row.get("ok")))

        if not args.failures_only or not row.get("ok"):
            label = "OK" if row.get("ok") else "MISS"
            if row.get("error"):
                label = "ERR"
            print(f"[{index:02d}/{len(cases)}] [{label}] {case.id} ({case.category})")
            print(f"     msg: {case.message}")
            if row.get("error"):
                print(f"     err: {row['error']}")
            else:
                print(
                    f"     got mode={row['actual_mode']} turn={row['actual_turn']} "
                    f"expected={case.expected}"
                )
                print(
                    f"     semantic_used={row.get('semantic_used')} "
                    f"route={row.get('semantic_route')} fallback={row.get('fallback')}"
                )
                if row.get("llm_raw"):
                    print(f"     llm: {row['llm_raw']}")
            print()

        if index < len(cases):
            time.sleep(max(0.0, args.delay))

    passed = sum(1 for row in results if row.get("ok"))
    total = len(results)
    accuracy = round(100.0 * passed / total, 1) if total else 0.0

    print("=" * 72)
    print(f"RESULT: {passed}/{total} correct ({accuracy}%)")
    print("\nBy category:")
    for category in sorted(by_category):
        cat_pass = sum(by_category[category])
        cat_total = len(by_category[category])
        cat_pct = round(100.0 * cat_pass / cat_total, 1) if cat_total else 0.0
        print(f"  {category:30s} {cat_pass}/{cat_total} ({cat_pct}%)")

    failures = [row for row in results if not row.get("ok")]
    if failures:
        print(f"\nFailures ({len(failures)}):")
        for row in failures:
            print(f"  - {row['id']}: {row['message'][:60]}")
            if row.get("error"):
                print(f"    error: {row['error']}")
            else:
                print(
                    f"    expected={row['expected']} got={row['actual_mode']}/"
                    f"{row['actual_turn']} fallback={row.get('fallback')}"
                )

    report = {
        "timestamp": datetime.now(UTC).isoformat(),
        "router_model": semantic_router_model_label(settings),
        "generation_model": settings.generation_model,
        "passed": passed,
        "total": total,
        "accuracy_pct": accuracy,
        "results": results,
    }
    out = Path(args.output)
    out.parent.mkdir(parents=True, exist_ok=True)
    out.write_text(json.dumps(report, ensure_ascii=False, indent=2), encoding="utf-8")
    print(f"\nReport saved: {out}")
    return 0 if passed == total else 1


if __name__ == "__main__":
    raise SystemExit(main())
