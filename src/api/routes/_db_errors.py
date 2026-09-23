"""Shared asyncpg failure mapping for the PostgreSQL read routes.

The response never carries the driver message or SQL: only a typed code and a fixed
sentence. The caller logs the exception class name, which is enough to debug.
"""

from __future__ import annotations

import asyncpg

from src.api.errors import ApiError, ApiErrorCode

_UNAVAILABLE = (
    asyncpg.PostgresConnectionError,
    asyncpg.CannotConnectNowError,
    asyncpg.TooManyConnectionsError,
    # Stale password or a role without LOGIN: the pool starts lazily (min_size=0), so
    # this surfaces on first acquire, not at startup.
    asyncpg.InvalidAuthorizationSpecificationError,
    ConnectionError,
    OSError,
)


def map_driver_error(exc: Exception) -> ApiError | None:
    """Return the typed error for a known driver failure, or ``None`` to re-raise."""
    # A stale catalog names a table or column the database no longer has.
    if isinstance(exc, (asyncpg.UndefinedTableError, asyncpg.UndefinedColumnError)):
        return ApiError(ApiErrorCode.INVALID_PARAMETER, "requested table is no longer readable")
    if isinstance(exc, asyncpg.InsufficientPrivilegeError):
        return ApiError(ApiErrorCode.FORBIDDEN, "read access denied")
    if isinstance(exc, _UNAVAILABLE):
        return ApiError(ApiErrorCode.PROVIDER_NOT_CONFIGURED, "database unavailable")
    return None
