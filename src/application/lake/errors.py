"""Transport-neutral lake failures.

``code`` values are the REST error-envelope codes (``src/api/errors.py``), so REST
maps a ``LakeError`` to its HTTP status without a translation table and MCP reports
the same code in its tool error. The application never imports REST exceptions.
"""

from __future__ import annotations

from typing import Any, Literal, Mapping, Optional

LakeErrorCode = Literal[
    "invalid_parameter",
    "query_timeout",
    "unsupported_timeframe",
    "unsupported_asset_class",
    "adjusted_not_supported",
    "unknown_symbol",
    "ambiguous_symbol",
    "not_yet_available",
    "provider_not_configured",
    "adjusted_unavailable",
    "unknown_index",
    "ambiguous_security",
    "membership_unavailable",
    "unknown_revision",
    "pit_unavailable",
    "revision_not_supported",
]


class LakeError(Exception):
    """A lake query failure with a stable, machine-readable code."""

    def __init__(
        self,
        code: LakeErrorCode,
        message: str,
        *,
        symbol: Optional[str] = None,
        asset_class: Optional[str] = None,
        details: Optional[Mapping[str, Any]] = None,
    ) -> None:
        super().__init__(message)
        self.code = code
        self.message = message
        self.symbol = symbol
        self.asset_class = asset_class
        self.details = dict(details) if details else None
