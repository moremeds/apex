from __future__ import annotations

from pathlib import Path

import pytest

from src.infrastructure.adapters.livewire.paths import (
    SUPPORTED_TIMEFRAMES,
    encode_symbol,
    parquet_path,
)


def test_parquet_path_layout() -> None:
    root = Path("/data/bronze")
    p = parquet_path(root, "AAPL", "1d")
    assert p == root / "asset_class=equity" / "symbol=AAPL" / "1d.parquet"


def test_encode_symbol_matches_livewire() -> None:
    # case-safe chars pass through (uppercase tickers, dots, hyphens)
    assert encode_symbol("AAPL") == "AAPL"
    assert encode_symbol("BRK.B") == "BRK.B"
    assert encode_symbol("BF-A") == "BF-A"
    # everything else is percent-encoded as UTF-8 bytes (matches livewire 1:1)
    assert encode_symbol("BF/B") == "BF%2FB"


def test_parquet_path_encodes_special_symbol() -> None:
    p = parquet_path(Path("/data/bronze"), "BF/B", "1d")
    assert p == Path("/data/bronze") / "asset_class=equity" / "symbol=BF%2FB" / "1d.parquet"


def test_unsupported_timeframe_raises() -> None:
    with pytest.raises(ValueError, match="unsupported timeframe"):
        parquet_path(Path("/data/bronze"), "AAPL", "3m")


def test_supported_timeframes_match_livewire() -> None:
    assert SUPPORTED_TIMEFRAMES == ("1m", "5m", "30m", "1h", "1d")
