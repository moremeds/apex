"""The asset-class registry is the single seam every other layer reads from.

Timeframe ladders are NOT guesses. They were measured exhaustively against the
production lake (macmini:/Volumes/DATA_LAKE/livewire/data-lake) on 2026-08-23 by
enumerating every symbol directory in each non-equity partition:

    volatility  44 symbols  union {1d, 1h, 30m, 5m}   (13 carry full intraday)
    fx          21 symbols  union {1d, 1h, 1m, 30m, 5m}  (all 21 carry it)
    cmdty        1 symbol   union {1d}
    futures     14 symbols  union {1d}
    rates        4 symbols  union {1d}

A registry that declared every non-equity class daily-only would make 126 real
parquet files unreachable through the API.
"""

from __future__ import annotations

import pytest

from src.infrastructure.adapters.livewire.asset_classes import (
    ASSET_CLASSES,
    DEFAULT_ASSET_CLASS,
    UnknownAssetClass,
    get_asset_class,
)


def test_all_six_lake_classes_are_registered() -> None:
    assert set(ASSET_CLASSES) == {"equity", "volatility", "fx", "cmdty", "futures", "rates"}


def test_equity_is_the_default_and_the_only_adjusted_class() -> None:
    assert DEFAULT_ASSET_CLASS == "equity"
    adjusted = {name for name, spec in ASSET_CLASSES.items() if spec.supports_adjusted}
    assert adjusted == {"equity"}, "Silver exists only for equity; nothing else may claim adjusted"


def test_equity_carries_the_full_intraday_ladder() -> None:
    assert get_asset_class("equity").timeframes == ("1m", "5m", "30m", "1h", "1d")


def test_fx_carries_the_full_intraday_ladder() -> None:
    """All 21 production FX pairs publish 1m/5m/30m/1h/1d."""
    assert get_asset_class("fx").timeframes == ("1m", "5m", "30m", "1h", "1d")


def test_volatility_carries_intraday_but_no_one_minute() -> None:
    """Measured union across all 44 volatility symbols: no 1m file exists anywhere."""
    assert get_asset_class("volatility").timeframes == ("5m", "30m", "1h", "1d")


def test_genuinely_daily_only_classes() -> None:
    for name in ("cmdty", "futures", "rates"):
        assert get_asset_class(name).timeframes == ("1d",), name


def test_partition_matches_the_hive_layout() -> None:
    assert get_asset_class("futures").partition == "asset_class=futures"


def test_rates_uses_the_series_payload_everything_else_uses_bars() -> None:
    assert get_asset_class("rates").payload == "rates_series"
    for name in ("equity", "volatility", "fx", "cmdty", "futures"):
        assert get_asset_class(name).payload == "bars", name


def test_futures_declares_its_extra_columns() -> None:
    assert get_asset_class("futures").extra_bar_fields == ("settlement", "open_interest")
    assert get_asset_class("equity").extra_bar_fields == ()


def test_timeframes_are_ordered_finest_to_coarsest() -> None:
    """Ordering is load-bearing: error messages list them and consumers render them."""
    order = ["1m", "5m", "30m", "1h", "1d"]
    for name, spec in ASSET_CLASSES.items():
        idx = [order.index(t) for t in spec.timeframes]
        assert idx == sorted(idx), f"{name} timeframes out of order: {spec.timeframes}"


def test_unknown_class_raises() -> None:
    with pytest.raises(UnknownAssetClass, match="crypto"):
        get_asset_class("crypto")
