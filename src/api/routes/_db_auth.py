"""Bearer authentication shared by the PostgreSQL read routes."""

from __future__ import annotations

import os
import secrets

from fastapi import Request

from src.api.errors import ApiError, ApiErrorCode


def require_db_token(request: Request) -> None:
    expected = os.environ.get("APEX_PG_READ_TOKEN")
    if not expected:
        raise ApiError(
            ApiErrorCode.PROVIDER_NOT_CONFIGURED,
            "PostgreSQL read API authentication not configured",
        )

    authorization = request.headers.get("Authorization", "")
    parts = authorization.split(" ")
    if len(parts) != 2 or parts[0].casefold() != "bearer" or not parts[1]:
        raise ApiError(ApiErrorCode.UNAUTHORIZED, "valid bearer token required")
    if not secrets.compare_digest(parts[1].encode("utf-8"), expected.encode("utf-8")):
        raise ApiError(ApiErrorCode.UNAUTHORIZED, "valid bearer token required")
