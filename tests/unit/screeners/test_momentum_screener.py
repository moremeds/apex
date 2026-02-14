"""Tests for the quantitative momentum screener.

Uses synthetic OHLCV data that simulates real market patterns
(uptrend, downtrend, choppy) for deterministic testing.
"""

from __future__ import annotations

from datetime import datetime
from pathlib import Path

import numpy as np
import pytest

from src.domain.screeners.momentum.compute import (
    compute_adaptive_momentum,
    compute_fip,
    compute_momentum_12_1,
)
from src.domain.screeners.momentum.config import MomentumConfig
from src.domain.screeners.momentum.models import MomentumScreenResult
from src.domain.screeners.momentum.scorer import (
    classify_quality,
    compute_composite_rank,
    compute_percentile_ranks,
)
from src.domain.screeners.momentum.screener import MomentumScreener

# ── Test Data Factories ───────────────────────────────────────────────


def _make_uptrend(
    n: int = 300, start_price: float = 100.0, daily_drift: float = 0.001
) -> np.ndarray:
    """Generate uptrending price series with realistic noise."""
    rng = np.random.RandomState(42)
    returns = daily_drift + rng.normal(0, 0.015, n)
    prices = start_price * np.cumprod(1 + returns)
    return prices


def _make_downtrend(
    n: int = 300, start_price: float = 100.0, daily_drift: float = -0.001
) -> np.ndarray:
    """Generate downtrending price series."""
    rng = np.random.RandomState(43)
    returns = daily_drift + rng.normal(0, 0.015, n)
    prices = start_price * np.cumprod(1 + returns)
    return prices


def _make_choppy(n: int = 300, start_price: float = 100.0) -> np.ndarray:
    """Generate choppy (sideways) price series."""
    rng = np.random.RandomState(44)
    returns = rng.normal(0, 0.02, n)
    prices = start_price * np.cumprod(1 + returns)
    return prices


def _make_volumes(n: int = 300, avg_vol: float = 1_000_000) -> np.ndarray:
    """Generate realistic volume data."""
    rng = np.random.RandomState(45)
    volumes = rng.lognormal(np.log(avg_vol), 0.5, n)
    return volumes.astype(np.int64)


DEFAULT_CONFIG: dict = {
    "data_source": {
        "lookback_trading_days": 252,
        "skip_recent_trading_days": 21,
    },
    "filters": {
        "min_market_cap": 500_000_000,
        "min_avg_daily_dollar_volume": 1_000_000,
        "min_price": 10.0,
        "min_daily_turnover_rate": 0.002,
    },
    "scoring": {
        "momentum_weight": 0.5,
        "fip_weight": 0.5,
        "top_n": 5,
    },
    "regime_rules": {
        "r0_min_composite_percentile": 0.0,
        "r0_position_size_factor": 1.0,
        "r1_min_composite_percentile": 0.70,
        "r1_position_size_factor": 0.5,
        "r2_block_entirely": True,
    },
    "quality_thresholds": {
        "strong": 0.80,
        "moderate": 0.60,
    },
}


# ── compute.py Tests ──────────────────────────────────────────────────


class TestMomentum12_1:
    def test_uptrend_positive(self) -> None:
        closes = _make_uptrend(300)
        result = compute_momentum_12_1(closes, skip=21, lookback=252)
        assert result is not None
        assert result > 0, "Uptrend should have positive momentum"

    def test_downtrend_negative(self) -> None:
        closes = _make_downtrend(300)
        result = compute_momentum_12_1(closes, skip=21, lookback=252)
        assert result is not None
        assert result < 0, "Downtrend should have negative momentum"

    def test_insufficient_data_returns_none(self) -> None:
        closes = np.linspace(100, 110, 200)  # Only 200 bars, need 273
        result = compute_momentum_12_1(closes, skip=21, lookback=252)
        assert result is None

    def test_zero_start_price_returns_none(self) -> None:
        closes = np.zeros(300)
        result = compute_momentum_12_1(closes, skip=21, lookback=252)
        assert result is None


class TestFIP:
    def test_uptrend_positive_fip(self) -> None:
        closes = _make_uptrend(300)
        result = compute_fip(closes, skip=21, lookback=252)
        assert result is not None
        assert result > 0, "Uptrend should have positive FIP"

    def test_fip_bounded(self) -> None:
        closes = _make_uptrend(300)
        result = compute_fip(closes, skip=21, lookback=252)
        assert result is not None
        assert -1.0 <= result <= 1.0, "FIP must be in [-1, 1]"

    def test_downtrend_negative_fip(self) -> None:
        closes = _make_downtrend(300)
        result = compute_fip(closes, skip=21, lookback=252)
        assert result is not None
        assert result < 0, "Downtrend should have negative FIP"

    def test_insufficient_data_returns_none(self) -> None:
        result = compute_fip(np.array([100.0, 101.0]), skip=21, lookback=252)
        assert result is None


class TestAdaptiveMomentum:
    def test_returns_tuple_with_actual_lookback(self) -> None:
        closes = _make_uptrend(200)  # Less than 252+21 but >= 126+21
        result = compute_adaptive_momentum(closes, skip=21, target=252, floor=126)
        assert result is not None
        mom, actual = result
        assert isinstance(mom, float)
        assert 126 <= actual <= 179  # 200 - 21 = 179

    def test_insufficient_for_floor_returns_none(self) -> None:
        closes = _make_uptrend(100)  # < 126 + 21
        result = compute_adaptive_momentum(closes, skip=21, target=252, floor=126)
        assert result is None

    def test_full_history_uses_target(self) -> None:
        closes = _make_uptrend(300)
        result = compute_adaptive_momentum(closes, skip=21, target=252, floor=126)
        assert result is not None
        _, actual = result
        assert actual == 252


# ── scorer.py Tests ───────────────────────────────────────────────────


class TestPercentileRanks:
    def test_empty_list(self) -> None:
        assert compute_percentile_ranks([]) == []

    def test_single_element(self) -> None:
        assert compute_percentile_ranks([42.0]) == [0.5]

    def test_ascending_order(self) -> None:
        ranks = compute_percentile_ranks([10.0, 20.0, 30.0])
        assert ranks == [0.0, 0.5, 1.0]

    def test_descending_order(self) -> None:
        ranks = compute_percentile_ranks([30.0, 20.0, 10.0])
        assert ranks == [1.0, 0.5, 0.0]

    def test_ties_get_average_rank(self) -> None:
        ranks = compute_percentile_ranks([10.0, 20.0, 20.0, 30.0])
        # Tied values at 20.0 should get average of rank 1 and 2 = 1.5
        assert ranks[1] == ranks[2]

    def test_percentile_range(self) -> None:
        values = list(range(100))
        ranks = compute_percentile_ranks([float(v) for v in values])
        assert all(0.0 <= r <= 1.0 for r in ranks)


class TestCompositeRank:
    def test_equal_weights(self) -> None:
        result = compute_composite_rank(0.8, 0.6, 0.5, 0.5)
        assert result == pytest.approx(0.7)

    def test_bounded(self) -> None:
        for _ in range(100):
            m = np.random.random()
            f = np.random.random()
            result = compute_composite_rank(m, f, 0.5, 0.5)
            assert 0.0 <= result <= 1.0

    def test_zero_weights(self) -> None:
        result = compute_composite_rank(0.8, 0.6, 0.0, 0.0)
        assert result == 0.0


class TestClassifyQuality:
    def test_strong(self) -> None:
        assert classify_quality(0.90, 0.80, 0.60) == "STRONG"

    def test_moderate(self) -> None:
        assert classify_quality(0.70, 0.80, 0.60) == "MODERATE"

    def test_marginal(self) -> None:
        assert classify_quality(0.50, 0.80, 0.60) == "MARGINAL"

    def test_boundary_strong(self) -> None:
        assert classify_quality(0.80, 0.80, 0.60) == "STRONG"

    def test_boundary_moderate(self) -> None:
        assert classify_quality(0.60, 0.80, 0.60) == "MODERATE"


# ── screener.py Tests ─────────────────────────────────────────────────


def _build_screener_data(n_symbols: int = 10) -> tuple[
    dict[str, np.ndarray],
    dict[str, np.ndarray],
    dict[str, float],
]:
    """Build multi-symbol test data for the screener."""
    price_data: dict[str, np.ndarray] = {}
    volume_data: dict[str, np.ndarray] = {}
    market_caps: dict[str, float] = {}

    symbols = [f"SYM{i:02d}" for i in range(n_symbols)]
    for i, sym in enumerate(symbols):
        drift = 0.0005 + i * 0.0002  # Increasing drift per symbol
        rng = np.random.RandomState(100 + i)
        returns = drift + rng.normal(0, 0.015, 300)
        price_data[sym] = 100 * np.cumprod(1 + returns)
        volume_data[sym] = rng.lognormal(np.log(1_000_000), 0.5, 300).astype(np.int64)
        market_caps[sym] = 5_000_000_000 + i * 1_000_000_000  # $5B-$14B

    return price_data, volume_data, market_caps


class TestMomentumScreener:
    def test_produces_ranked_output(self) -> None:
        price_data, volume_data, market_caps = _build_screener_data()
        screener = MomentumScreener(DEFAULT_CONFIG)
        result = screener.screen(price_data, volume_data, "R0", market_caps)

        assert isinstance(result, MomentumScreenResult)
        assert len(result.candidates) > 0
        # Verify descending composite rank
        composites = [c.signal.composite_rank for c in result.candidates]
        assert composites == sorted(composites, reverse=True)

    def test_ranks_sequential(self) -> None:
        price_data, volume_data, market_caps = _build_screener_data()
        screener = MomentumScreener(DEFAULT_CONFIG)
        result = screener.screen(price_data, volume_data, "R0", market_caps)

        ranks = [c.rank for c in result.candidates]
        assert ranks == list(range(1, len(ranks) + 1))

    def test_r2_blocks_all_candidates(self) -> None:
        price_data, volume_data, market_caps = _build_screener_data()
        screener = MomentumScreener(DEFAULT_CONFIG)
        result = screener.screen(price_data, volume_data, "R2", market_caps)

        assert len(result.candidates) == 0
        assert result.regime == "R2"

    def test_r1_raises_threshold(self) -> None:
        price_data, volume_data, market_caps = _build_screener_data()
        screener = MomentumScreener(DEFAULT_CONFIG)

        r0_result = screener.screen(price_data, volume_data, "R0", market_caps)
        r1_result = screener.screen(price_data, volume_data, "R1", market_caps)

        assert len(r1_result.candidates) <= len(r0_result.candidates)

    def test_r1_reduces_position_size(self) -> None:
        price_data, volume_data, market_caps = _build_screener_data()
        screener = MomentumScreener(DEFAULT_CONFIG)
        result = screener.screen(price_data, volume_data, "R1", market_caps)

        for c in result.candidates:
            assert c.position_size_factor == 0.5

    def test_price_filter_excludes(self) -> None:
        price_data, volume_data, market_caps = _build_screener_data()
        # Set one symbol to have very low prices
        price_data["SYM00"] = np.full(300, 5.0)  # Below $10 min
        screener = MomentumScreener(DEFAULT_CONFIG)
        result = screener.screen(price_data, volume_data, "R0", market_caps)

        symbols = [c.signal.symbol for c in result.candidates]
        assert "SYM00" not in symbols

    def test_market_cap_filter_excludes(self) -> None:
        price_data, volume_data, market_caps = _build_screener_data()
        market_caps["SYM00"] = 100_000_000  # Below $500M min
        screener = MomentumScreener(DEFAULT_CONFIG)
        result = screener.screen(price_data, volume_data, "R0", market_caps)

        symbols = [c.signal.symbol for c in result.candidates]
        assert "SYM00" not in symbols

    def test_top_n_limits_output(self) -> None:
        price_data, volume_data, market_caps = _build_screener_data(20)
        config = {**DEFAULT_CONFIG, "scoring": {**DEFAULT_CONFIG["scoring"], "top_n": 3}}
        screener = MomentumScreener(config)
        result = screener.screen(price_data, volume_data, "R0", market_caps)

        assert len(result.candidates) <= 3

    def test_is_backtest_disables_turnover(self) -> None:
        """Verify turnover filter is skipped in backtest mode."""
        price_data, volume_data, market_caps = _build_screener_data()
        # Set very low turnover that would fail live filter
        for sym in volume_data:
            volume_data[sym] = np.full(300, 100, dtype=np.int64)  # Very low volume
            market_caps[sym] = 100_000_000_000  # High cap = low turnover rate

        screener = MomentumScreener(DEFAULT_CONFIG)

        # Live mode: should filter out low-turnover stocks
        live_result = screener.screen(price_data, volume_data, "R0", market_caps, is_backtest=False)

        # Backtest mode: should skip turnover filter
        bt_result = screener.screen(price_data, volume_data, "R0", market_caps, is_backtest=True)

        assert len(bt_result.candidates) >= len(live_result.candidates)

    def test_empty_universe(self) -> None:
        screener = MomentumScreener(DEFAULT_CONFIG)
        result = screener.screen({}, {}, "R0", {})
        assert len(result.candidates) == 0
        assert result.universe_size == 0

    def test_insufficient_history(self) -> None:
        """Stocks with too little history should be filtered out."""
        price_data = {"SHORT": np.linspace(100, 110, 100)}  # Only 100 bars
        volume_data = {"SHORT": np.full(100, 1_000_000, dtype=np.int64)}
        market_caps = {"SHORT": 10_000_000_000.0}
        screener = MomentumScreener(DEFAULT_CONFIG)
        result = screener.screen(price_data, volume_data, "R0", market_caps)
        assert len(result.candidates) == 0


# ── config.py Tests ───────────────────────────────────────────────────


class TestConfig:
    def test_from_yaml(self) -> None:
        config_path = (
            Path(__file__).resolve().parent.parent.parent.parent / "config/momentum_screener.yaml"
        )
        if config_path.exists():
            config = MomentumConfig.from_yaml(config_path)
            assert config.data_source.lookback_trading_days == 252
            assert config.scoring.top_n == 30
            assert config.regime_rules.r2_block_entirely is True

    def test_from_dict_defaults(self) -> None:
        config = MomentumConfig.from_dict({})
        assert config.data_source.lookback_trading_days == 252
        assert config.filters.min_price == 10.0

    def test_from_dict_overrides(self) -> None:
        config = MomentumConfig.from_dict({"scoring": {"top_n": 50}})
        assert config.scoring.top_n == 50
        # Other defaults preserved
        assert config.scoring.momentum_weight == 0.5

    def test_unknown_keys_ignored(self) -> None:
        config = MomentumConfig.from_dict({"unknown_section": {"foo": "bar"}})
        assert config.scoring.top_n == 30  # Default preserved
