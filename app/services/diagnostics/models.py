from __future__ import annotations

from dataclasses import asdict, dataclass, field
from typing import Any


@dataclass
class ChatTrace:
    request: dict[str, Any] = field(default_factory=dict)
    state_before: dict[str, Any] = field(default_factory=dict)
    guardrail: dict[str, Any] = field(default_factory=dict)
    turn_understanding: dict[str, Any] = field(default_factory=dict)
    query: dict[str, Any] = field(default_factory=dict)
    retrieval: dict[str, Any] = field(default_factory=dict)
    reranking: dict[str, Any] = field(default_factory=dict)
    final_context: dict[str, Any] = field(default_factory=dict)
    generation: dict[str, Any] = field(default_factory=dict)
    grounding: dict[str, Any] = field(default_factory=dict)
    tool_execution: dict[str, Any] = field(default_factory=lambda: {"tool_executed": False})
    response: dict[str, Any] = field(default_factory=dict)
    state_after: dict[str, Any] = field(default_factory=dict)
    latency: dict[str, Any] = field(default_factory=dict)
    errors: list[dict[str, Any]] = field(default_factory=list)
    llamaindex: dict[str, Any] = field(default_factory=dict)
    observability: dict[str, Any] = field(default_factory=dict)
    semantic_router: dict[str, Any] = field(default_factory=dict)

    @property
    def trace_id(self) -> str:
        return str(self.request.get("trace_id") or "")

    def to_dict(self) -> dict[str, Any]:
        payload = asdict(self)
        payload["trace_id"] = self.trace_id
        return payload
