from __future__ import annotations

from pathlib import Path

import pytest

from src.infrastructure.adapters.livewire.paths import (
    SUPPORTED_TIMEFRAMES,
    parquet_path,
)


def test_parquet_path_layout() -> None:
    root = Path("/data/bronze")
    p = parquet_path(root, "AAPL", "1d")
    assert p == root / "asset_class=equity" / "symbol=AAPL" / "1d.parquet"


def test_unsupported_timeframe_raises() -> None:
    with pytest.raises(ValueError, match="unsupported timeframe"):
        parquet_path(Path("/data/bronze"), "AAPL", "3m")


def test_supported_timeframes_match_livewire() -> None:
    assert SUPPORTED_TIMEFRAMES == ("1m", "5m", "30m", "1h", "1d")
