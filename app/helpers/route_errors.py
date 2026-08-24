from __future__ import annotations

from typing import Any

from fastapi import Request
from fastapi.exceptions import RequestValidationError


def request_label(request: Request) -> str:
    path = request.url.path
    if request.url.query:
        path = f"{path}?{request.url.query}"
    client = request.client.host if request.client else "unknown"
    return f"{request.method} {path} client={client}"


def detail_to_message(detail: Any) -> str:
    if detail is None:
        return ""
    if isinstance(detail, str):
        return detail
    if isinstance(detail, list):
        parts: list[str] = []
        for item in detail:
            if isinstance(item, dict):
                loc = ".".join(str(part) for part in item.get("loc", ()))
                msg = item.get("msg", "")
                parts.append(f"{loc}: {msg}" if loc else str(msg))
            else:
                parts.append(str(item))
        return "; ".join(parts)
    if isinstance(detail, dict):
        message = detail.get("message") or detail.get("detail")
        if message is not None:
            return str(message)
        return str(detail)
    return str(detail)


def validation_errors_to_message(exc: RequestValidationError) -> str:
    return detail_to_message(exc.errors())


def http_error_log_message(request: Request, status_code: int, detail: Any) -> str:
    message = detail_to_message(detail)
    suffix = f" detail={message}" if message else ""
    return f"HTTP {status_code} | {request_label(request)}{suffix}"


def unhandled_error_log_message(request: Request, exc: Exception) -> str:
    return f"Unhandled {type(exc).__name__}: {exc} | {request_label(request)}"
