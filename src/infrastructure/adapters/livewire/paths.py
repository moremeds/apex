"""Resolve (symbol, timeframe) to a livewire bronze parquet path.

The per-ticker Hive layout IS the read contract, confirmed against livewire's
``clients/bronze_client.py`` + ``clients/intraday_bronze_client.py`` (2026-06-14):

  <root>/asset_class=equity/symbol=<encode_symbol(SYM)>/<tf>.parquet

where ``<root>`` is livewire's ``data-lake/bronze`` directory (``APEX_LIVEWIRE_ROOT``).
"""

from __future__ import annotations

from pathlib import Path

# Timeframes livewire warehouses for equities: daily via BronzeClient (1d), intraday
# via IntradayBronzeClient (1m/5m/30m/1h). Matches livewire's INTRADAY_TIMEFRAMES + 1d.
SUPPORTED_TIMEFRAMES = ("1m", "5m", "30m", "1h", "1d")

# Mirrors livewire's clients/symbol_paths.py exactly: keep these characters literal,
# percent-encode everything else as UTF-8 bytes. Reversible + safe on case-insensitive
# filesystems. Uppercase tickers with `.`/`-` (AAPL, BRK.B) are identity; `/` etc. encode.
_CASE_SAFE = frozenset("ABCDEFGHIJKLMNOPQRSTUVWXYZ0123456789._-")


def encode_symbol(symbol: str) -> str:
    """Encode a symbol to its livewire bronze partition name (matches livewire 1:1)."""
    parts: list[str] = []
    for character in symbol:
        if character in _CASE_SAFE:
            parts.append(character)
        else:
            parts.extend(f"%{byte:02X}" for byte in character.encode("utf-8"))
    return "".join(parts)


def parquet_path(bronze_root: Path, symbol: str, timeframe: str) -> Path:
    if timeframe not in SUPPORTED_TIMEFRAMES:
        raise ValueError(f"unsupported timeframe: {timeframe!r} (have {SUPPORTED_TIMEFRAMES})")
    return (
        bronze_root
        / "asset_class=equity"
        / f"symbol={encode_symbol(symbol)}"
        / f"{timeframe}.parquet"
    )


def daily_silver_path(silver_root: Path, symbol: str) -> Path:
    """Return the materialized adjusted-daily artifact for ``symbol``."""
    return silver_root / "asset_class=equity" / f"symbol={encode_symbol(symbol)}" / "1d.parquet"


def factor_path(silver_root: Path, symbol: str) -> Path:
    """Return the compact adjustment-factor artifact for ``symbol``."""
    return (
        silver_root
        / "adjustments"
        / "asset_class=equity"
        / f"symbol={encode_symbol(symbol)}"
        / "factors.parquet"
    )
