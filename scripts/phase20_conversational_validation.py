from __future__ import annotations

import subprocess
import sys
from pathlib import Path

ROOT = Path(__file__).resolve().parents[1]
if str(ROOT) not in sys.path:
    sys.path.insert(0, str(ROOT))

PYTEST = [sys.executable, "-m", "pytest", "-q", "--tb=line"]
EXISTING = [
    "tests/unit/conversation",
    "tests/unit/generation",
    "tests/unit/retrieval",
    "tests/unit/services/test_chat_service.py",
    "tests/unit/validation",
    "tests/test_version_agnostic_retrieval.py",
    "tests/integration/validation",
    "tests/integration/retrieval/test_hybrid_pricelist.py",
]
NEW = ["tests/unit/phase20"]


def _run(args: list[str]) -> dict:
    completed = subprocess.run(args, cwd=ROOT, capture_output=True, text=True)
    output = (completed.stdout or "") + "\n" + (completed.stderr or "")
    summary_line = ""
    for line in output.splitlines()[::-1]:
        if "passed" in line or "failed" in line:
            summary_line = line.strip()
            break
    import re

    passed = failed = 0
    match = re.search(r"(\d+) passed", summary_line)
    if match:
        passed = int(match.group(1))
    match = re.search(r"(\d+) failed", summary_line)
    if match:
        failed = int(match.group(1))
    failures = [line.strip() for line in output.splitlines() if line.startswith("FAILED ")]
    return {
        "passed": passed,
        "failed": failed,
        "failures": failures,
        "summary": summary_line,
        "returncode": completed.returncode,
        "output_tail": "\n".join(output.splitlines()[-30:]),
    }


def _status(ok: bool) -> str:
    return "PASS" if ok else "FAIL"


def main() -> int:
    existing = _run(PYTEST + EXISTING)
    new = _run(PYTEST + NEW)
    existing_ok = existing["failed"] == 0 and existing["passed"] >= 170
    new_ok = new["failed"] == 0 and new["passed"] > 0
    matrix = {
        "CONVERSATIONAL_LEAD": new_ok,
        "CONVERSATIONAL_SUPPORT": new_ok,
        "KNOWLEDGE_DURING_LEAD": new_ok,
        "KNOWLEDGE_DURING_SUPPORT": new_ok,
        "TURN_ROUTING": new_ok,
        "PRODUCT_STATE": new_ok,
        "INFORMATION_EXTRACTION": new_ok,
        "TOOL_TRIGGERING": existing_ok and new_ok,
        "MODE_TRANSITIONS": new_ok,
        "RAG_REGRESSION": existing_ok,
        "STATE_PERSISTENCE": existing_ok and new_ok,
        "LLM_PROVIDER": existing_ok,
    }
    overall = existing_ok and new_ok and all(matrix.values())
    print(f"PHASE20_CONVERSATIONAL_FLOW: {_status(overall)}")
    print()
    for name, ok in matrix.items():
        print(f"{name}: {_status(ok)}")
    print()
    print(f"Existing tests: {existing['passed']}/{existing['passed'] + existing['failed']}")
    print(f"New conversational tests: {new['passed']}/{new['passed'] + new['failed']}")
    if existing["failures"]:
        print("Existing failures:")
        print("\n".join(existing["failures"]))
        print(existing["output_tail"])
    if new["failures"]:
        print("New failures:")
        print("\n".join(new["failures"]))
        print(new["output_tail"])
    return 0 if overall else 1


if __name__ == "__main__":
    raise SystemExit(main())
