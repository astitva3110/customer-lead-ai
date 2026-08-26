from __future__ import annotations

import logging
import os
from typing import Any

logger = logging.getLogger(__name__)


def configure_langsmith() -> None:
    """Copy settings into the env vars LangGraph / LangSmith / LiteLLM already read."""
    from app.config import settings

    tracing = _truthy(getattr(settings, "langsmith_tracing", False))
    api_key = str(getattr(settings, "langsmith_api_key", "") or "").strip()
    project = str(getattr(settings, "langsmith_project", "") or "").strip() or "earkart-chatbot"
    endpoint = str(getattr(settings, "langsmith_endpoint", "") or "").strip()
    if not tracing:
        logger.info("LangSmith tracing off (LANGSMITH_TRACING is false)")
        return
    if not api_key:
        logger.warning("LangSmith tracing requested but LANGSMITH_API_KEY is empty")
        return
    os.environ["LANGSMITH_TRACING"] = "true"
    os.environ["LANGSMITH_API_KEY"] = api_key
    os.environ["LANGSMITH_PROJECT"] = project
    os.environ["LANGCHAIN_TRACING_V2"] = "true"
    os.environ["LANGCHAIN_API_KEY"] = api_key
    os.environ["LANGCHAIN_PROJECT"] = project
    if endpoint:
        os.environ["LANGSMITH_ENDPOINT"] = endpoint
        os.environ["LANGCHAIN_ENDPOINT"] = endpoint
    _configure_litellm_callback()
    logger.info("LangSmith tracing on project=%s endpoint=%s", project, endpoint or "default")


def langsmith_enabled() -> bool:
    from app.config import settings

    return _truthy(getattr(settings, "langsmith_tracing", False)) and bool(
        str(getattr(settings, "langsmith_api_key", "") or "").strip()
    )


def current_langsmith_run_id() -> str | None:
    try:
        from langsmith.run_helpers import get_current_run_tree
    except Exception:
        return None
    try:
        tree = get_current_run_tree()
    except Exception:
        return None
    if tree is None:
        return None
    run_id = getattr(tree, "id", None) or getattr(tree, "trace_id", None)
    return str(run_id) if run_id else None


def usage_payload(usage: Any) -> dict[str, int | None]:
    if not usage:
        return {"input_tokens": None, "output_tokens": None, "total_tokens": None}
    if isinstance(usage, dict):
        prompt = usage.get("prompt_tokens", usage.get("input_tokens"))
        completion = usage.get("completion_tokens", usage.get("output_tokens"))
        total = usage.get("total_tokens")
    else:
        prompt = getattr(usage, "prompt_tokens", None) or getattr(usage, "input_tokens", None)
        completion = getattr(usage, "completion_tokens", None) or getattr(usage, "output_tokens", None)
        total = getattr(usage, "total_tokens", None)
    prompt_n = _int(prompt)
    completion_n = _int(completion)
    total_n = _int(total)
    if total_n is None and prompt_n is not None and completion_n is not None:
        total_n = prompt_n + completion_n
    return {"input_tokens": prompt_n, "output_tokens": completion_n, "total_tokens": total_n}


def add_token_usage(trace: Any, usage: dict[str, int | None] | None) -> None:
    if trace is None or not usage:
        return
    current = dict(getattr(trace, "observability", None) or {})
    tokens = dict(current.get("token_usage") or {})
    for key in ("input_tokens", "output_tokens", "total_tokens"):
        added = usage.get(key)
        if added is None:
            continue
        tokens[key] = int(tokens.get(key) or 0) + int(added)
    current["token_usage"] = tokens
    run_id = current_langsmith_run_id()
    if run_id:
        current["langsmith_run_id"] = run_id
    trace.observability = current


def _configure_litellm_callback() -> None:
    try:
        import litellm
    except Exception:
        return
    callbacks = list(getattr(litellm, "success_callback", None) or [])
    if "langsmith" not in callbacks:
        callbacks.append("langsmith")
        litellm.success_callback = callbacks
    failures = list(getattr(litellm, "failure_callback", None) or [])
    if "langsmith" not in failures:
        failures.append("langsmith")
        litellm.failure_callback = failures


def _truthy(value: Any) -> bool:
    if isinstance(value, bool):
        return value
    return str(value or "").strip().lower() in {"1", "true", "yes", "on"}


def _int(value: Any) -> int | None:
    if value is None or value == "":
        return None
    try:
        return int(value)
    except (TypeError, ValueError):
        return None
