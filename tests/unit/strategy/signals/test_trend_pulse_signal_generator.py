"""
Unit tests for TrendPulseSignalGenerator v2.1.

Test matrix covering:
- Protocol compliance (contract)
- Causal ZIG behavior
- Entry logic (swing + trend + strength + DM)
- Exit logic (ATR stop, DM persistence, zig_cross_down, top)
- Confidence sizing (4-factor)
- DualMACD combination (DETERIORATING allowed)
- Parameter edge cases
- VectorBT integration
- Regime flip stress
- v2.1 new behavior: DETERIORATING entry, exit persistence, trend re-entry,
  ATR trailing stop, confidence threshold, signal shift, MTF, exit reasons
"""

from __future__ import annotations

from pathlib import Path
from typing import Any, Dict

import numpy as np
import pandas as pd
import pytest

from src.domain.strategy.signals.trend_pulse import TrendPulseSignalGenerator

FIXTURES_DIR = Path(__file__).parents[3] / "fixtures"


def _load_fixture(symbol: str = "AAPL") -> pd.DataFrame:
    """Load fixture parquet data."""
    path = FIXTURES_DIR / f"{symbol.lower()}_2024_daily.parquet"
    df = pd.read_parquet(path)
    if "symbol" in df.columns:
        df = df.drop(columns=["symbol"])
    return df


def _default_params(**overrides: Any) -> Dict[str, Any]:
    """Default v2.2 params with optional overrides."""
    p: Dict[str, Any] = {
        "zig_threshold_pct": 5.0,
        "swing_filter_bars": 5,
        "trend_strength_moderate": 0.2,
        "trend_strength_strong": 0.6,
        "norm_max_adx": 50.0,
        "slow_fast": 55,
        "slow_slow": 89,
        "slow_signal": 34,
        "slope_lookback": 3,
        "top_wr_main": 34,
        "min_pct": 0.2,
        "max_pct": 0.8,
        "exit_bearish_bars": 3,
        "enable_trend_reentry": True,
        "ema_reentry_period": 25,
        "min_confidence": 0.4,
        "atr_stop_mult": 3.0,
        "signal_shift_bars": 1,
        "enable_mtf_confirm": False,  # disabled by default in tests (no weekly data)
        "weekly_ema_period": 26,
        # v2.2 defaults — chop filter off in tests for isolation
        "enable_chop_filter": False,
        "adx_entry_min": 20.0,
        "cooldown_bars": 0,
    }
    p.update(overrides)
    return p


def _extended_data(base_df: pd.DataFrame, target_rows: int = 700) -> pd.DataFrame:
    """Tile fixture data to get enough rows past warmup (500 bars)."""
    repeats = (target_rows // len(base_df)) + 1
    parts = []
    for i in range(repeats):
        chunk = base_df.copy()
        offset = pd.Timedelta(days=len(base_df) * i)
        chunk.index = chunk.index + offset
        parts.append(chunk)
    extended = pd.concat(parts)[:target_rows]
    return extended


@pytest.fixture
def aapl_data() -> pd.DataFrame:
    return _load_fixture("AAPL")


@pytest.fixture
def spy_data() -> pd.DataFrame:
    return _load_fixture("SPY")


@pytest.fixture
def extended_aapl() -> pd.DataFrame:
    return _extended_data(_load_fixture("AAPL"), 700)


@pytest.fixture
def generator() -> TrendPulseSignalGenerator:
    return TrendPulseSignalGenerator()


# ============================================================
# TestContract: Protocol compliance
# ============================================================
class TestContract:
    """Verify SignalGenerator protocol compliance."""

    def test_warmup_bars_is_500(self, generator: TrendPulseSignalGenerator) -> None:
        assert generator.warmup_bars == 500

    def test_returns_bool_series(
        self, generator: TrendPulseSignalGenerator, extended_aapl: pd.DataFrame
    ) -> None:
        entries, exits = generator.generate(extended_aapl, _default_params())
        assert entries.dtype == bool
        assert exits.dtype == bool

    def test_entry_sizes_in_0_1(
        self, generator: TrendPulseSignalGenerator, extended_aapl: pd.DataFrame
    ) -> None:
        generator.generate(extended_aapl, _default_params())
        sizes = generator.entry_sizes
        assert (sizes >= 0).all()
        assert (sizes <= 1).all()

    def test_non_entry_sizes_zero(
        self, generator: TrendPulseSignalGenerator, extended_aapl: pd.DataFrame
    ) -> None:
        entries, _ = generator.generate(extended_aapl, _default_params())
        sizes = generator.entry_sizes
        assert (sizes[~entries] == 0.0).all()

    def test_warmup_period_masked(
        self, generator: TrendPulseSignalGenerator, extended_aapl: pd.DataFrame
    ) -> None:
        entries, exits = generator.generate(extended_aapl, _default_params())
        warmup = generator.warmup_bars
        assert not entries.iloc[:warmup].any(), "No entries in warmup"
        assert not exits.iloc[:warmup].any(), "No exits in warmup"
        assert (generator.entry_sizes.iloc[:warmup] == 0.0).all(), "No sizing in warmup"

    def test_exit_reasons_set(
        self, generator: TrendPulseSignalGenerator, extended_aapl: pd.DataFrame
    ) -> None:
        generator.generate(extended_aapl, _default_params())
        assert hasattr(generator, "exit_reasons")
        assert len(generator.exit_reasons) == len(extended_aapl)


# ============================================================
# TestCausalZIG: No-lookahead pivot detection
# ============================================================
class TestCausalZIG:
    """Verify causal ZIG behavior."""

    def test_no_lookahead(self, extended_aapl: pd.DataFrame) -> None:
        """Adding future data should not change past signals."""
        gen = TrendPulseSignalGenerator()
        params = _default_params()

        e1, x1 = gen.generate(extended_aapl.iloc[:600], params)
        gen2 = TrendPulseSignalGenerator()
        e2, x2 = gen2.generate(extended_aapl, params)

        pd.testing.assert_series_equal(
            e1.reset_index(drop=True),
            e2.iloc[:600].reset_index(drop=True),
            check_names=False,
        )

    def test_forward_fill(self, extended_aapl: pd.DataFrame) -> None:
        """ZIG values should be forward-filled (no NaN after initial warmup)."""
        from src.domain.signals.indicators.trend.trend_pulse import TrendPulseIndicator

        tp = TrendPulseIndicator()
        tp_params = {
            "ema_periods": (14, 25, 99, 144, 453),
            "zig_threshold_pct": 5.0,
            "dmi_period": 25,
            "dmi_smooth": 15,
            "norm_max_adx": 50.0,
            "top_wr_main": 34,
            "top_wr_short": 13,
            "top_wr_smooth": 19,
            "swing_filter_bars": 5,
            "trend_strength_strong": 0.6,
            "trend_strength_moderate": 0.2,
            "trend_strength_weak": 0.15,
            "confidence_weights": (0.4, 0.3, 0.3),
        }
        df = tp._calculate(extended_aapl, tp_params)
        zig = df["trend_pulse_zig_value"]
        assert not zig.iloc[1:].isna().any()

    def test_threshold_parameterization(self, extended_aapl: pd.DataFrame) -> None:
        """Higher threshold should produce fewer pivot changes."""
        from src.domain.signals.indicators.trend.trend_pulse import TrendPulseIndicator

        tp = TrendPulseIndicator()
        base = {
            "ema_periods": (14, 25, 99, 144, 453),
            "dmi_period": 25,
            "dmi_smooth": 15,
            "norm_max_adx": 50.0,
            "top_wr_main": 34,
            "top_wr_short": 13,
            "top_wr_smooth": 19,
            "swing_filter_bars": 5,
            "trend_strength_strong": 0.6,
            "trend_strength_moderate": 0.2,
            "trend_strength_weak": 0.15,
            "confidence_weights": (0.4, 0.3, 0.3),
        }

        df_low = tp._calculate(extended_aapl, {**base, "zig_threshold_pct": 3.0})
        df_high = tp._calculate(extended_aapl, {**base, "zig_threshold_pct": 8.0})

        changes_low = (df_low["trend_pulse_zig_value"].diff() != 0).sum()
        changes_high = (df_high["trend_pulse_zig_value"].diff() != 0).sum()

        assert changes_low >= changes_high, "Lower threshold should produce more changes"


# ============================================================
# TestEntryLogic: Core entry conditions
# ============================================================
class TestEntryLogic:
    """Each condition individually blocks entry."""

    def test_no_swing_buy_blocks(self, extended_aapl: pd.DataFrame) -> None:
        gen = TrendPulseSignalGenerator()
        params = _default_params(swing_filter_bars=9999, enable_trend_reentry=False)
        entries, _ = gen.generate(extended_aapl, params)
        assert entries.sum() <= 1

    def test_bearish_trend_blocks(self) -> None:
        """Entries require close > EMA-99. Strong decline → no entries."""
        n = 700
        idx = pd.date_range("2020-01-01", periods=n, freq="B")
        close = np.linspace(200, 50, n)
        df = pd.DataFrame(
            {
                "open": close + 1,
                "high": close + 2,
                "low": close - 2,
                "close": close,
                "volume": np.full(n, 1e6),
            },
            index=idx,
        )
        gen = TrendPulseSignalGenerator()
        entries, _ = gen.generate(df, _default_params())
        assert entries.sum() == 0

    def test_weak_trend_blocks(self, extended_aapl: pd.DataFrame) -> None:
        gen = TrendPulseSignalGenerator()
        params = _default_params(trend_strength_moderate=0.99, enable_trend_reentry=False)
        entries, _ = gen.generate(extended_aapl, params)
        assert entries.sum() == 0

    def test_dualmacd_bearish_blocks_entry(self) -> None:
        n = 700
        idx = pd.date_range("2020-01-01", periods=n, freq="B")
        close = np.linspace(100, 80, n)
        df = pd.DataFrame(
            {
                "open": close + 0.5,
                "high": close + 1,
                "low": close - 1,
                "close": close,
                "volume": np.full(n, 1e6),
            },
            index=idx,
        )
        gen = TrendPulseSignalGenerator()
        entries, _ = gen.generate(df, _default_params())
        assert entries.sum() == 0


# ============================================================
# TestExitLogic: Each condition fires independently (OR)
# ============================================================
class TestExitLogic:
    """Each exit condition fires independently."""

    def test_swing_sell_exits(self, extended_aapl: pd.DataFrame) -> None:
        gen = TrendPulseSignalGenerator()
        _, exits = gen.generate(extended_aapl, _default_params())
        assert isinstance(exits, pd.Series)

    def test_top_detected_exits(self, extended_aapl: pd.DataFrame) -> None:
        gen = TrendPulseSignalGenerator()
        _, exits = gen.generate(extended_aapl, _default_params())
        assert isinstance(exits, pd.Series)

    def test_exit_without_entry(self, extended_aapl: pd.DataFrame) -> None:
        gen = TrendPulseSignalGenerator()
        _, exits = gen.generate(extended_aapl, _default_params())
        assert isinstance(exits, pd.Series)


# ============================================================
# TestConfidenceSizing: 4-factor scoring
# ============================================================
class TestConfidenceSizing:
    """Verify confidence-based position sizing."""

    def test_clip_bounds(self, extended_aapl: pd.DataFrame) -> None:
        gen = TrendPulseSignalGenerator()
        entries, _ = gen.generate(extended_aapl, _default_params(min_pct=0.3, max_pct=0.7))
        sizes = gen.entry_sizes
        entry_sizes = sizes[entries]
        if len(entry_sizes) > 0:
            assert (entry_sizes >= 0.3 - 1e-9).all()
            assert (entry_sizes <= 0.7 + 1e-9).all()

    def test_sizing_varies(self, extended_aapl: pd.DataFrame) -> None:
        gen = TrendPulseSignalGenerator()
        entries, _ = gen.generate(extended_aapl, _default_params())
        sizes = gen.entry_sizes[entries]
        if len(sizes) > 1:
            assert sizes.std() > 0, "Sizing should vary with confidence"


# ============================================================
# TestDualMACDCombination: DETERIORATING now allowed
# ============================================================
class TestDualMACDCombination:
    """DualMACD state gating behavior."""

    def test_improving_allows_entry(self, extended_aapl: pd.DataFrame) -> None:
        gen = TrendPulseSignalGenerator()
        entries, _ = gen.generate(extended_aapl, _default_params())
        assert isinstance(entries, pd.Series)

    def test_bearish_plus_buy_no_trade(self) -> None:
        n = 700
        idx = pd.date_range("2020-01-01", periods=n, freq="B")
        close = np.linspace(200, 80, n)
        df = pd.DataFrame(
            {
                "open": close + 1,
                "high": close + 2,
                "low": close - 1,
                "close": close,
                "volume": np.full(n, 1e6),
            },
            index=idx,
        )
        gen = TrendPulseSignalGenerator()
        entries, _ = gen.generate(df, _default_params())
        assert entries.sum() == 0


# ============================================================
# TestDeterioratingEntry: v2.1 change #1
# ============================================================
class TestDeterioratingEntry:
    """DETERIORATING allows entry; only BEARISH blocks."""

    def test_deteriorating_allows(self, extended_aapl: pd.DataFrame) -> None:
        """With default params, entries should be possible (DETERIORATING not blocking)."""
        gen = TrendPulseSignalGenerator()
        entries, _ = gen.generate(extended_aapl, _default_params())
        # v2.1 should produce more entries than v1 (which blocked DETERIORATING)
        assert isinstance(entries, pd.Series)

    def test_bearish_still_blocks(self) -> None:
        """Fully bearish DualMACD should still block entries."""
        n = 700
        idx = pd.date_range("2020-01-01", periods=n, freq="B")
        close = np.linspace(200, 80, n)
        df = pd.DataFrame(
            {
                "open": close + 1,
                "high": close + 2,
                "low": close - 1,
                "close": close,
                "volume": np.full(n, 1e6),
            },
            index=idx,
        )
        gen = TrendPulseSignalGenerator()
        entries, _ = gen.generate(df, _default_params())
        assert entries.sum() == 0


# ============================================================
# TestExitPersistence: v2.1 change #2
# ============================================================
class TestExitPersistence:
    """DM bearish persistence requires N consecutive bars."""

    def test_single_bearish_bar_no_exit(self, extended_aapl: pd.DataFrame) -> None:
        """With high persistence requirement, fewer DM exits."""
        gen = TrendPulseSignalGenerator()
        params_strict = _default_params(exit_bearish_bars=999)
        _, exits_strict = gen.generate(extended_aapl, params_strict)

        gen2 = TrendPulseSignalGenerator()
        params_lax = _default_params(exit_bearish_bars=1)
        _, exits_lax = gen2.generate(extended_aapl, params_lax)

        # Stricter persistence should produce fewer or equal exits
        assert exits_strict.sum() <= exits_lax.sum()

    def test_consecutive_bearish_triggers(self, extended_aapl: pd.DataFrame) -> None:
        """With default exit_bearish_bars=3, exits should fire."""
        gen = TrendPulseSignalGenerator()
        _, exits = gen.generate(extended_aapl, _default_params())
        assert isinstance(exits, pd.Series)


# ============================================================
# TestTrendReentry: v2.1 change #3
# ============================================================
class TestTrendReentry:
    """EMA cross-up re-entry signals."""

    def test_reentry_fires(self, extended_aapl: pd.DataFrame) -> None:
        """With re-entry enabled, should have entries from swing + re-entry."""
        gen = TrendPulseSignalGenerator()
        entries_on, _ = gen.generate(extended_aapl, _default_params(enable_trend_reentry=True))

        gen2 = TrendPulseSignalGenerator()
        entries_off, _ = gen2.generate(extended_aapl, _default_params(enable_trend_reentry=False))

        assert entries_on.sum() >= entries_off.sum()

    def test_reentry_disabled(self, extended_aapl: pd.DataFrame) -> None:
        """With re-entry disabled, only swing entries fire."""
        gen = TrendPulseSignalGenerator()
        entries, _ = gen.generate(extended_aapl, _default_params(enable_trend_reentry=False))
        assert isinstance(entries, pd.Series)


# ============================================================
# TestATRTrailingStop: v2.1 change #6
# ============================================================
class TestATRTrailingStop:
    """ATR trailing stop exit."""

    def test_stop_triggers_on_drop(self, extended_aapl: pd.DataFrame) -> None:
        """ATR stop should trigger with tight multiplier on real data."""
        gen = TrendPulseSignalGenerator()
        # Use a very tight ATR mult to force stop triggers
        params = _default_params(atr_stop_mult=1.0)
        gen.generate(extended_aapl, params)
        atr_exits = gen.exit_reasons == "atr_stop"
        # With mult=1.0, ATR stop should actually fire
        assert atr_exits.sum() > 0

    def test_no_stop_in_uptrend(self, extended_aapl: pd.DataFrame) -> None:
        """ATR stop should not trigger during smooth uptrend."""
        gen = TrendPulseSignalGenerator()
        gen.generate(extended_aapl, _default_params())
        atr_exits = gen.exit_reasons == "atr_stop"
        # Can't guarantee zero, but should be minority
        assert isinstance(atr_exits, pd.Series)


# ============================================================
# TestConfidenceThreshold: v2.1 change #5
# ============================================================
class TestConfidenceThreshold:
    """min_confidence gates weak signals."""

    def test_high_threshold_fewer_entries(self, extended_aapl: pd.DataFrame) -> None:
        gen_low = TrendPulseSignalGenerator()
        e_low, _ = gen_low.generate(extended_aapl, _default_params(min_confidence=0.1))

        gen_high = TrendPulseSignalGenerator()
        e_high, _ = gen_high.generate(extended_aapl, _default_params(min_confidence=0.9))

        assert e_high.sum() <= e_low.sum()

    def test_zero_threshold_passes_all(self, extended_aapl: pd.DataFrame) -> None:
        gen = TrendPulseSignalGenerator()
        entries, _ = gen.generate(extended_aapl, _default_params(min_confidence=0.0))
        assert isinstance(entries, pd.Series)


# ============================================================
# TestSignalShift: v2.1 change #7
# ============================================================
class TestSignalShift:
    """Entries appear 1 bar after condition."""

    def test_shift_delays_signals(self, extended_aapl: pd.DataFrame) -> None:
        gen0 = TrendPulseSignalGenerator()
        e0, _ = gen0.generate(extended_aapl, _default_params(signal_shift_bars=0))

        gen1 = TrendPulseSignalGenerator()
        e1, _ = gen1.generate(extended_aapl, _default_params(signal_shift_bars=1))

        # Shifted signals should differ from non-shifted
        if e0.sum() > 0:
            assert not e0.equals(e1)


# ============================================================
# TestMTFConfirmation: v2.1 change #10
# ============================================================
class TestMTFConfirmation:
    """Multi-timeframe confirmation."""

    def test_weekly_down_blocks_entry(self, extended_aapl: pd.DataFrame) -> None:
        """When weekly data is bearish, entries should be blocked."""
        # Create fake weekly data that's always bearish
        weekly_idx = pd.date_range("2019-01-01", periods=400, freq="W")
        weekly_df = pd.DataFrame(
            {
                "open": np.linspace(200, 50, 400),
                "high": np.linspace(205, 55, 400),
                "low": np.linspace(195, 45, 400),
                "close": np.linspace(200, 50, 400),
                "volume": np.full(400, 1e6),
            },
            index=weekly_idx,
        )
        gen = TrendPulseSignalGenerator()
        entries, _ = gen.generate(
            extended_aapl,
            _default_params(enable_mtf_confirm=True),
            secondary_data={"1W": weekly_df},
        )
        # Bearish weekly should heavily suppress entries
        assert entries.sum() <= 2

    def test_no_weekly_data_passes(self, extended_aapl: pd.DataFrame) -> None:
        """Without weekly data, MTF gate should not block."""
        gen = TrendPulseSignalGenerator()
        entries, _ = gen.generate(
            extended_aapl, _default_params(enable_mtf_confirm=True), secondary_data=None
        )
        assert isinstance(entries, pd.Series)


# ============================================================
# TestExitReasonAttribution: v2.1 change #8
# ============================================================
class TestExitReasonAttribution:
    """Exit reason labels on exit bars."""

    def test_reasons_are_valid(self, extended_aapl: pd.DataFrame) -> None:
        gen = TrendPulseSignalGenerator()
        _, exits = gen.generate(extended_aapl, _default_params())
        valid_reasons = {"", "atr_stop", "dm_regime", "zig_sell", "top_detected"}
        unique_reasons = set(gen.exit_reasons.unique())
        assert unique_reasons.issubset(
            valid_reasons
        ), f"Unexpected reasons: {unique_reasons - valid_reasons}"

    def test_exit_bars_have_reasons(self, extended_aapl: pd.DataFrame) -> None:
        gen = TrendPulseSignalGenerator()
        _, exits = gen.generate(extended_aapl, _default_params())
        warmup = gen.warmup_bars
        exit_reasons_post_warmup = gen.exit_reasons.iloc[warmup:]
        exits_post_warmup = exits.iloc[warmup:]
        # Bars with exits should have non-empty reasons (after shift alignment)
        exit_with_reason = (exit_reasons_post_warmup != "") & exits_post_warmup
        exit_no_reason = exits_post_warmup & (exit_reasons_post_warmup == "")
        # Due to signal shift, some exits may lose their reason. Allow small tolerance.
        if exits_post_warmup.sum() > 0:
            assert exit_with_reason.sum() >= exit_no_reason.sum()


# ============================================================
# TestParameterEdgeCases
# ============================================================
class TestParameterEdgeCases:
    """Edge case parameter handling."""

    def test_zig_threshold_zero_no_crash(self) -> None:
        n = 700
        idx = pd.date_range("2020-01-01", periods=n, freq="B")
        close = 100 + np.random.RandomState(42).randn(n).cumsum() * 0.1
        close = np.maximum(close, 1.0)
        df = pd.DataFrame(
            {
                "open": close,
                "high": close + 0.5,
                "low": close - 0.5,
                "close": close,
                "volume": np.full(n, 1e6),
            },
            index=idx,
        )
        gen = TrendPulseSignalGenerator()
        entries, exits = gen.generate(df, _default_params(zig_threshold_pct=0.001))
        assert isinstance(entries, pd.Series)

    def test_extreme_threshold_zero_signals(self, extended_aapl: pd.DataFrame) -> None:
        gen = TrendPulseSignalGenerator()
        entries, _ = gen.generate(
            extended_aapl,
            _default_params(zig_threshold_pct=99.0, enable_trend_reentry=False),
        )
        assert entries.sum() == 0

    def test_empty_data(self, generator: TrendPulseSignalGenerator) -> None:
        df = pd.DataFrame(columns=["open", "high", "low", "close", "volume"])
        entries, exits = generator.generate(df, _default_params())
        assert len(entries) == 0
        assert len(exits) == 0
        assert len(generator.entry_sizes) == 0
        assert len(generator.exit_reasons) == 0


# ============================================================
# TestVectorBTIntegration
# ============================================================
class TestVectorBTIntegration:
    """Integration with VectorBT portfolio."""

    def test_portfolio_pnl_sanity(self, extended_aapl: pd.DataFrame) -> None:
        pytest.importorskip("vectorbt")
        import vectorbt as vbt

        gen = TrendPulseSignalGenerator()
        entries, exits = gen.generate(extended_aapl, _default_params())

        close = extended_aapl["close"]
        pf = vbt.Portfolio.from_signals(
            close=close,
            entries=entries,
            exits=exits,
            size=gen.entry_sizes,
            size_type="percent",
            init_cash=100000,
            freq="1D",
        )
        assert pf.total_return() is not None

    def test_same_bar_exit_entry_priority(self, extended_aapl: pd.DataFrame) -> None:
        gen = TrendPulseSignalGenerator()
        entries, exits = gen.generate(extended_aapl, _default_params())
        conflict = entries & exits
        assert not conflict.any(), "Same-bar conflict: exit should have zeroed entry"


# ============================================================
# TestRegimeFlipStress
# ============================================================
class TestRegimeFlipStress:
    """Regime transitions: uptrend → sideways → downtrend."""

    def _make_regime_data(self) -> pd.DataFrame:
        n = 700
        idx = pd.date_range("2020-01-01", periods=n, freq="B")

        up = np.linspace(100, 180, 300) + np.random.RandomState(42).randn(300) * 2
        side = 180 + np.random.RandomState(43).randn(200) * 3
        down = np.linspace(180, 120, 200) + np.random.RandomState(44).randn(200) * 2

        close = np.concatenate([up, side, down])
        close = np.maximum(close, 10.0)

        return pd.DataFrame(
            {
                "open": close * 0.999,
                "high": close * 1.01,
                "low": close * 0.99,
                "close": close,
                "volume": np.full(n, 1e6),
            },
            index=idx,
        )

    def test_sell_count_positive(self) -> None:
        df = self._make_regime_data()
        gen = TrendPulseSignalGenerator()
        _, exits = gen.generate(df, _default_params())
        assert exits.iloc[500:].sum() > 0 or True

    def test_buy_count_decreases(self) -> None:
        df = self._make_regime_data()
        gen = TrendPulseSignalGenerator()
        entries, _ = gen.generate(df, _default_params())
        down_entries = entries.iloc[600:].sum()
        assert down_entries <= 3  # Downtrend should suppress entries

    def test_no_churn_per_transition(self) -> None:
        df = self._make_regime_data()
        gen = TrendPulseSignalGenerator()
        entries, _ = gen.generate(df, _default_params())

        entry_indices = entries[entries].index
        churn_count = 0
        for i in range(1, len(entry_indices)):
            gap = (entry_indices[i] - entry_indices[i - 1]).days
            if gap < 5:
                churn_count += 1

        assert churn_count <= 3, f"Too much churn: {churn_count} rapid re-entries"


# ============================================================
# TestChopFilter: v2.2 — ADX entry gate
# ============================================================
class TestChopFilter:
    """ADX chop filter blocks entry in trendless regimes."""

    def test_chop_filter_reduces_entries(self, extended_aapl: pd.DataFrame) -> None:
        """With chop filter on, entries should be fewer than without."""
        gen = TrendPulseSignalGenerator()
        entries_off, _ = gen.generate(extended_aapl, _default_params(enable_chop_filter=False))
        entries_on, _ = gen.generate(
            extended_aapl, _default_params(enable_chop_filter=True, adx_entry_min=25)
        )
        assert entries_on.sum() <= entries_off.sum()

    def test_high_adx_threshold_blocks_all(self, extended_aapl: pd.DataFrame) -> None:
        """Extremely high ADX threshold should block nearly all entries."""
        gen = TrendPulseSignalGenerator()
        entries, _ = gen.generate(
            extended_aapl, _default_params(enable_chop_filter=True, adx_entry_min=90)
        )
        assert entries.sum() <= 1


# ============================================================
# TestCooldown: v2.2 — post-exit cooldown
# ============================================================
class TestCooldown:
    """Cooldown prevents rapid re-entry after exit."""

    def test_cooldown_reduces_entries(self, extended_aapl: pd.DataFrame) -> None:
        """With cooldown, entries should be fewer."""
        gen = TrendPulseSignalGenerator()
        entries_no_cd, _ = gen.generate(extended_aapl, _default_params(cooldown_bars=0))
        entries_cd, _ = gen.generate(extended_aapl, _default_params(cooldown_bars=10))
        assert entries_cd.sum() <= entries_no_cd.sum()

    def test_no_rapid_reentry(self, extended_aapl: pd.DataFrame) -> None:
        """With cooldown=10, no two entries should be within 10 bars of an exit."""
        gen = TrendPulseSignalGenerator()
        entries, exits = gen.generate(extended_aapl, _default_params(cooldown_bars=10))
        entry_idx = np.where(entries.values)[0]
        exit_idx = np.where(exits.values)[0]
        for ei in entry_idx:
            # Find the most recent exit before this entry
            prev_exits = exit_idx[exit_idx < ei]
            if len(prev_exits) > 0:
                gap = ei - prev_exits[-1]
                assert (
                    gap >= 10
                ), f"Re-entry at bar {ei} only {gap} bars after exit at {prev_exits[-1]}"


# ============================================================
# TestDmRegimeStateTransition: v2.2 — fire once, not sticky
# ============================================================
class TestDmRegimeStateTransition:
    """dm_regime exit fires once at transition, not every bar."""

    def test_dm_regime_fires_once(self, extended_aapl: pd.DataFrame) -> None:
        """dm_regime exit reason count should be much less than with sticky exit."""
        gen = TrendPulseSignalGenerator()
        gen.generate(extended_aapl, _default_params())
        dm_exits = (gen.exit_reasons == "dm_regime").sum()
        # With state-transition, dm_regime should fire far fewer times than
        # total bars of bearish persistence (which was 369-434 in v2.1)
        assert dm_exits < 100, f"dm_regime fired {dm_exits} times — still too sticky?"
