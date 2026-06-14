"""Resolve (symbol, timeframe) to a livewire bronze parquet path.

The per-ticker Hive layout IS the read contract (livewire-adaptation.md 3, 5):
  <root>/asset_class=equity/symbol=<SYM>/<tf>.parquet
"""

from __future__ import annotations

from pathlib import Path

# Timeframes livewire warehouses for equities (livewire-adaptation.md 2).
SUPPORTED_TIMEFRAMES = ("1m", "5m", "30m", "1h", "1d")


def parquet_path(bronze_root: Path, symbol: str, timeframe: str) -> Path:
    if timeframe not in SUPPORTED_TIMEFRAMES:
        raise ValueError(f"unsupported timeframe: {timeframe!r} (have {SUPPORTED_TIMEFRAMES})")
    return bronze_root / "asset_class=equity" / f"symbol={symbol}" / f"{timeframe}.parquet"
