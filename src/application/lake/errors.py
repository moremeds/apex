"""Transport-neutral lake failures.

``code`` values are the REST error-envelope codes (``src/api/errors.py``), so REST
maps a ``LakeError`` to its HTTP status without a translation table and MCP reports
the same code in its tool error. The application never imports REST exceptions.
"""

from __future__ import annotations

import re
from typing import Any, Literal, Mapping, Optional

# Absolute POSIX paths. Quoted ones (how OSError and DuckDB render filenames) are
# replaced whole, spaces included; unquoted ones up to whitespace or punctuation.
# Relative Silver paths ("generations/.../1d.parquet") are public manifest content and
# kept; "/v1/..." is an API route a message may point to.
_QUOTED_PATH = re.compile(r"(['\"])/(?!v1/)[^'\"]*\1")
_BARE_PATH = re.compile(r"(?<![^\s=:(\[])/(?!v1/)[^\s'\"(),:;]+")


def redact_paths(text: str) -> str:
    """Remove absolute host paths from client-visible text (design §6)."""
    return _BARE_PATH.sub("<path>", _QUOTED_PATH.sub(r"\1<path>\1", text))


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
        message = redact_paths(message)
        super().__init__(message)
        self.code = code
        self.message = message
        self.symbol = symbol
        self.asset_class = asset_class
        self.details = dict(details) if details else None
