from __future__ import annotations

from pathlib import Path

import pytest

from src.infrastructure.adapters.livewire.asset_classes import UnknownAssetClass
from src.infrastructure.adapters.livewire.paths import (
    SUPPORTED_TIMEFRAMES,
    UnsupportedTimeframe,
    daily_silver_path,
    delisted_bronze_path,
    encode_symbol,
    factor_path,
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


def test_daily_silver_path_layout() -> None:
    root = Path("/data/silver")
    assert daily_silver_path(root, "AAPL") == (
        root / "asset_class=equity" / "symbol=AAPL" / "1d.parquet"
    )


def test_factor_path_layout_and_symbol_encoding() -> None:
    root = Path("/data/silver")
    assert factor_path(root, "BF/B") == (
        root / "adjustments" / "asset_class=equity" / "symbol=BF%2FB" / "factors.parquet"
    )


def test_unsupported_timeframe_raises() -> None:
    with pytest.raises(ValueError, match="unsupported timeframe"):
        parquet_path(Path("/data/bronze"), "AAPL", "3m")


def test_supported_timeframes_match_livewire() -> None:
    assert SUPPORTED_TIMEFRAMES == ("1m", "5m", "30m", "1h", "1d")


def test_parquet_path_defaults_to_equity() -> None:
    root = Path("/data/bronze")
    assert parquet_path(root, "AAPL", "1d") == (
        root / "asset_class=equity" / "symbol=AAPL" / "1d.parquet"
    )


def test_parquet_path_resolves_each_asset_class() -> None:
    root = Path("/data/bronze")
    assert parquet_path(root, "VIX", "1d", "volatility") == (
        root / "asset_class=volatility" / "symbol=VIX" / "1d.parquet"
    )
    assert parquet_path(root, "DGS10", "1d", "rates") == (
        root / "asset_class=rates" / "symbol=DGS10" / "1d.parquet"
    )
    assert parquet_path(root, "BZ_202609", "1d", "futures") == (
        root / "asset_class=futures" / "symbol=BZ_202609" / "1d.parquet"
    )
    assert parquet_path(root, "XAUUSD", "1d", "cmdty") == (
        root / "asset_class=cmdty" / "symbol=XAUUSD" / "1d.parquet"
    )


def test_fx_intraday_resolves() -> None:
    """All 21 production FX pairs publish the full ladder; rejecting it would
    make 84 real parquet files unreachable."""
    root = Path("/data/bronze")
    for timeframe in ("1m", "5m", "30m", "1h"):
        assert parquet_path(root, "DXY", timeframe, "fx") == (
            root / "asset_class=fx" / "symbol=DXY" / f"{timeframe}.parquet"
        )


def test_volatility_serves_intraday_but_not_one_minute() -> None:
    """Measured union across all 44 volatility symbols: 5m/30m/1h/1d, never 1m."""
    root = Path("/data/bronze")
    assert parquet_path(root, "VIX", "5m", "volatility") == (
        root / "asset_class=volatility" / "symbol=VIX" / "5m.parquet"
    )
    with pytest.raises(UnsupportedTimeframe, match="volatility"):
        parquet_path(root, "VIX", "1m", "volatility")


def test_intraday_is_rejected_for_daily_only_classes() -> None:
    for asset_class, symbol in (("cmdty", "XAUUSD"), ("futures", "BZ_202609"), ("rates", "DGS10")):
        with pytest.raises(UnsupportedTimeframe, match=asset_class):
            parquet_path(Path("/data/bronze"), symbol, "1h", asset_class)


def test_unknown_asset_class_raises() -> None:
    with pytest.raises(UnknownAssetClass):
        parquet_path(Path("/data/bronze"), "AAPL", "1d", "crypto")


def test_delisted_path_layout() -> None:
    root = Path("/data/bronze-delisted")
    assert delisted_bronze_path(root, "BBBY", "1d") == (
        root / "asset_class=equity" / "symbol=BBBY" / "1d.parquet"
    )


def test_delisted_path_is_not_equity_only() -> None:
    """bronze-delisted holds asset_class=equity (8620) and asset_class=fx (USDEUR)."""
    root = Path("/data/bronze-delisted")
    assert delisted_bronze_path(root, "USDEUR", "1d", "fx") == (
        root / "asset_class=fx" / "symbol=USDEUR" / "1d.parquet"
    )
