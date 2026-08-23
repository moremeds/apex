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
from fastapi.responses import JSONResponse

logger = logging.getLogger(__name__)


class ApiErrorCode(str, Enum):
    """Stable, machine-readable failure reasons. Values are part of the contract."""

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
    """Register the ApiError handler so routes can raise instead of building responses."""

    @app.exception_handler(ApiError)
    async def _handle(request: Request, exc: ApiError) -> JSONResponse:  # pragma: no cover
        logger.warning("api error %s on %s: %s", exc.code.value, request.url.path, exc.message)
        return api_error_response(exc)
