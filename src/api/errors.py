"""Typed error envelope for the apex read surface.

Every failure returns ``{"error": {"code": ..., "message": ..., ...}}`` with a stable
machine-readable code, so a consumer can branch on the reason instead of guessing from
a bare 500. Replaces the previous behaviour where ``AdjustedDataUnavailable`` escaped
the route layer uncaught -- a condition 243 equity symbols hit in production, including
HON, MMM, CMCSA, AIG, ECL, MSI, WY and LEN.
"""

from __future__ import annotations

import logging
from enum import Enum

from fastapi import FastAPI, Request
from fastapi.exceptions import RequestValidationError
from fastapi.responses import JSONResponse

logger = logging.getLogger(__name__)


class ApiErrorCode(str, Enum):
    """Stable, machine-readable failure reasons. Values are part of the contract.

    A code names the thing that was wrong. ``INVALID_PARAMETER`` exists because a
    malformed query value is not an unknown symbol and not an upstream outage --
    reusing either code would send the caller debugging the wrong subject.
    """

    INVALID_PARAMETER = "invalid_parameter"
    INTERNAL_ERROR = "internal_error"
    UNSUPPORTED_TIMEFRAME = "unsupported_timeframe"
    UNSUPPORTED_ASSET_CLASS = "unsupported_asset_class"
    ADJUSTED_NOT_SUPPORTED = "adjusted_not_supported"
    UNKNOWN_SYMBOL = "unknown_symbol"
    AMBIGUOUS_SYMBOL = "ambiguous_symbol"
    NOT_YET_AVAILABLE = "not_yet_available"
    PROVIDER_NOT_CONFIGURED = "provider_not_configured"
    ADJUSTED_UNAVAILABLE = "adjusted_unavailable"


# 503 for ADJUSTED_UNAVAILABLE is deliberate: a missing or quarantined Silver artifact
# is an upstream condition livewire may repair, so "retry later" is the honest
# semantic. A 4xx would tell the caller their request was wrong; it was not.
STATUS_BY_CODE: dict[ApiErrorCode, int] = {
    ApiErrorCode.INVALID_PARAMETER: 400,
    ApiErrorCode.INTERNAL_ERROR: 500,
    ApiErrorCode.UNSUPPORTED_TIMEFRAME: 400,
    ApiErrorCode.UNSUPPORTED_ASSET_CLASS: 400,
    ApiErrorCode.ADJUSTED_NOT_SUPPORTED: 400,
    ApiErrorCode.UNKNOWN_SYMBOL: 404,
    ApiErrorCode.AMBIGUOUS_SYMBOL: 409,
    ApiErrorCode.NOT_YET_AVAILABLE: 501,
    ApiErrorCode.PROVIDER_NOT_CONFIGURED: 503,
    ApiErrorCode.ADJUSTED_UNAVAILABLE: 503,
}


class ApiError(Exception):
    """A read-surface failure with a typed reason code."""

    def __init__(
        self,
        code: ApiErrorCode,
        message: str,
        *,
        symbol: str | None = None,
        asset_class: str | None = None,
    ) -> None:
        super().__init__(message)
        self.code = code
        self.message = message
        self.symbol = symbol
        self.asset_class = asset_class

    @property
    def status_code(self) -> int:
        return STATUS_BY_CODE[self.code]


def api_error_response(exc: ApiError) -> JSONResponse:
    """Render ``exc`` as the error envelope, omitting absent context fields."""
    error: dict[str, str] = {"code": exc.code.value, "message": exc.message}
    if exc.symbol is not None:
        error["symbol"] = exc.symbol
    if exc.asset_class is not None:
        error["asset_class"] = exc.asset_class
    return JSONResponse(status_code=exc.status_code, content={"error": error})


def install_error_handlers(app: FastAPI) -> None:
    """Register the error handlers so routes can raise instead of building responses."""

    @app.exception_handler(ApiError)
    async def _handle(request: Request, exc: ApiError) -> JSONResponse:  # pragma: no cover
        logger.warning("api error %s on %s: %s", exc.code.value, request.url.path, exc.message)
        return api_error_response(exc)

    @app.exception_handler(RequestValidationError)
    async def _handle_validation(request: Request, exc: RequestValidationError) -> JSONResponse:
        """FastAPI's own 422 body is ``{"detail": [...]}`` -- a second, undocumented
        error shape on the same surface. Re-render it as the envelope so a consumer
        parses one shape for every failure.

        The 422 status is preserved: it is the documented FastAPI contract for request
        validation and some callers already branch on it. Only the body changes.
        """
        detail = "; ".join(
            f"{'.'.join(str(p) for p in e.get('loc', ())[1:])}: {e.get('msg', '')}".strip(": ")
            for e in exc.errors()
        )
        logger.warning("request validation failed on %s: %s", request.url.path, detail)
        return JSONResponse(
            status_code=422,
            content={"error": {"code": ApiErrorCode.INVALID_PARAMETER.value, "message": detail}},
        )

    @app.exception_handler(Exception)
    async def _handle_unexpected(request: Request, exc: Exception) -> JSONResponse:
        """Anything unanticipated still leaves as the envelope, never a bare 500.

        The lake is a Parquet tree on an EXTERNAL volume read through DuckDB. If that
        volume unmounts, a file is truncated mid-write, or a permission changes, DuckDB
        raises and the exception would otherwise escape as an untyped 500 whose body
        shape no consumer can parse -- the exact failure this module exists to remove.

        The message is deliberately generic: ``exc`` carries absolute lake paths and SQL
        text, and the client is not entitled to either. The detail goes to the log.
        """
        logger.exception("unhandled error on %s", request.url.path)
        return api_error_response(
            ApiError(ApiErrorCode.INTERNAL_ERROR, "internal error; see server logs")
        )
