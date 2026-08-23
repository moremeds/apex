"""Resolve (asset_class, symbol, timeframe) to a livewire parquet path.

The per-ticker Hive layout IS the read contract, confirmed against livewire's
``clients/bronze_client.py`` + ``clients/intraday_bronze_client.py`` (2026-06-14) and
re-verified across all six asset classes on 2026-08-23:

  <root>/asset_class=<class>/symbol=<encode_symbol(SYM)>/<tf>.parquet

where ``<root>`` is livewire's ``data-lake/bronze`` directory (``APEX_LIVEWIRE_ROOT``).
Silver is equity-only, so the two silver builders take no asset_class.
"""

from __future__ import annotations

from pathlib import Path

from .asset_classes import DEFAULT_ASSET_CLASS, get_asset_class

# Retained for backward compatibility: the equity timeframe ladder. New code should
# read ``get_asset_class(name).timeframes`` -- ladders differ per class (cmdty,
# futures and rates are daily-only; volatility has no 1m).
SUPPORTED_TIMEFRAMES = ("1m", "5m", "30m", "1h", "1d")

# Mirrors livewire's clients/symbol_paths.py exactly: keep these characters literal,
# percent-encode everything else as UTF-8 bytes. Reversible + safe on case-insensitive
# filesystems. Uppercase tickers with `.`/`-` (AAPL, BRK.B) are identity; `/` etc. encode.
_CASE_SAFE = frozenset("ABCDEFGHIJKLMNOPQRSTUVWXYZ0123456789._-")


class UnsupportedTimeframe(ValueError):
    """Raised when a timeframe is not warehoused for the requested asset class."""


def encode_symbol(symbol: str) -> str:
    """Encode a symbol to its livewire partition name (matches livewire 1:1)."""
    parts: list[str] = []
    for character in symbol:
        if character in _CASE_SAFE:
            parts.append(character)
        else:
            parts.extend(f"%{byte:02X}" for byte in character.encode("utf-8"))
    return "".join(parts)


def parquet_path(
    bronze_root: Path,
    symbol: str,
    timeframe: str,
    asset_class: str = DEFAULT_ASSET_CLASS,
) -> Path:
    """Return the bronze artifact path, raising on an unknown class or timeframe."""
    spec = get_asset_class(asset_class)
    if timeframe not in spec.timeframes:
        raise UnsupportedTimeframe(
            f"unsupported timeframe {timeframe!r} for {spec.name} "
            f"(have {list(spec.timeframes)})"
        )
    return bronze_root / spec.partition / f"symbol={encode_symbol(symbol)}" / f"{timeframe}.parquet"


def delisted_bronze_path(
    delisted_root: Path,
    symbol: str,
    timeframe: str = "1d",
    asset_class: str = DEFAULT_ASSET_CLASS,
) -> Path:
    """Return the archived-delisted artifact path.

    livewire's ``archive_otc_symbols.py`` moves symbols under bronze-delisted/. The tree
    is overwhelmingly equity (8620 symbols on 2026-08-23) but is NOT equity-only --
    asset_class=fx holds USDEUR -- so the partition is parameterized rather than fixed.
    """
    spec = get_asset_class(asset_class)
    return (
        delisted_root / spec.partition / f"symbol={encode_symbol(symbol)}" / f"{timeframe}.parquet"
    )


def daily_silver_path(silver_root: Path, symbol: str) -> Path:
    """Return the materialized adjusted-daily artifact for ``symbol`` (equity only)."""
    return silver_root / "asset_class=equity" / f"symbol={encode_symbol(symbol)}" / "1d.parquet"


def factor_path(silver_root: Path, symbol: str) -> Path:
    """Return the compact adjustment-factor artifact for ``symbol`` (equity only)."""
    return (
        silver_root
        / "adjustments"
        / "asset_class=equity"
        / f"symbol={encode_symbol(symbol)}"
        / "factors.parquet"
    )
