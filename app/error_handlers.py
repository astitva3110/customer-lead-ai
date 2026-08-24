from __future__ import annotations

import logging

from fastapi import FastAPI, HTTPException, Request
from fastapi.exception_handlers import (
    http_exception_handler,
    request_validation_exception_handler,
)
from fastapi.exceptions import RequestValidationError
from fastapi.responses import JSONResponse

from app.helpers.route_errors import (
    http_error_log_message,
    request_label,
    unhandled_error_log_message,
    validation_errors_to_message,
)

logger = logging.getLogger("app.http")


def _log_level_for_status(status_code: int) -> int:
    if status_code >= 500:
        return logging.ERROR
    if status_code >= 400:
        return logging.WARNING
    return logging.INFO


def register_error_handlers(app: FastAPI) -> None:
    @app.exception_handler(HTTPException)
    async def log_http_exception(request: Request, exc: HTTPException) -> JSONResponse:
        logger.log(
            _log_level_for_status(exc.status_code),
            http_error_log_message(request, exc.status_code, exc.detail),
        )
        return await http_exception_handler(request, exc)

    @app.exception_handler(RequestValidationError)
    async def log_validation_exception(request: Request, exc: RequestValidationError) -> JSONResponse:
        logger.warning(
            "HTTP 422 | %s detail=%s",
            request_label(request),
            validation_errors_to_message(exc),
        )
        return await request_validation_exception_handler(request, exc)

    @app.exception_handler(Exception)
    async def log_unhandled_exception(request: Request, exc: Exception) -> JSONResponse:
        logger.exception(unhandled_error_log_message(request, exc))
        return JSONResponse(status_code=500, content={"detail": "Internal server error"})
