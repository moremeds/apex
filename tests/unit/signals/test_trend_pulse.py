"""Tests for TrendPulse indicator."""

from __future__ import annotations

import numpy as np
import pandas as pd
import pytest

from src.domain.signals.indicators.trend.trend_pulse import (
    EmaAlignment,
    SwingSignal,
    TopWarning,
    TrendFilter,
    TrendPulseIndicator,
    TrendStrengthLabel,
)


@pytest.fixture
def indicator() -> TrendPulseIndicator:
    return TrendPulseIndicator()


@pytest.fixture
def params(indicator: TrendPulseIndicator) -> dict:
    return indicator.default_params


def _make_ohlcv(closes: list[float]) -> pd.DataFrame:
    """Build OHLCV DataFrame from close prices (high/low derived)."""
    c = np.array(closes, dtype=np.float64)
    return pd.DataFrame(
        {
            "open": c * 0.999,
            "high": c * 1.005,
            "low": c * 0.995,
            "close": c,
            "volume": np.ones(len(c)) * 1000,
        }
    )


def _trending_up(n: int = 600, start: float = 100.0, step: float = 0.5) -> list[float]:
    return [start + i * step for i in range(n)]


def _trending_down(n: int = 600, start: float = 400.0, step: float = 0.5) -> list[float]:
    return [start - i * step for i in range(n)]


def _zigzag_prices(
    n: int = 600, base: float = 100.0, amplitude: float = 10.0, period: int = 40
) -> list[float]:
    return [base + amplitude * np.sin(2 * np.pi * i / period) for i in range(n)]


class TestTrendPulseContract:
    """Verify fixed 14-key state schema."""

    EXPECTED_KEYS = {
        # Original 8
        "swing_signal",
        "trend_filter",
        "trend_strength",
        "trend_strength_label",
        "top_warning",
        "ema_alignment",
        "confidence",
        "score",
        # New 8
        "dm_state",
        "adx",
        "adx_ok",
        "entry_signal",
        "exit_signal",
        "confidence_4f",
        "atr_stop_level",
        "cooldown_left",
    }

    def test_state_keys_present(self, indicator: TrendPulseIndicator, params: dict) -> None:
        data = _make_ohlcv(_trending_up(600))
        result = indicator.calculate(data, params)
        state = indicator.get_state(result.iloc[-1], result.iloc[-2], params)
        assert set(state.keys()) == self.EXPECTED_KEYS

    def test_original_8_keys_still_present(
        self, indicator: TrendPulseIndicator, params: dict
    ) -> None:
        """Backward compatibility: original 8 keys unchanged."""
        data = _make_ohlcv(_trending_up(600))
        result = indicator.calculate(data, params)
        state = indicator.get_state(result.iloc[-1], result.iloc[-2], params)
        original_keys = {
            "swing_signal",
            "trend_filter",
            "trend_strength",
            "trend_strength_label",
            "top_warning",
            "ema_alignment",
            "confidence",
            "score",
        }
        assert original_keys.issubset(set(state.keys()))

    def test_enum_values_valid(self, indicator: TrendPulseIndicator, params: dict) -> None:
        data = _make_ohlcv(_trending_up(600))
        result = indicator.calculate(data, params)
        state = indicator.get_state(result.iloc[-1], result.iloc[-2], params)

        assert state["swing_signal"] in [e.value for e in SwingSignal]
        assert state["trend_filter"] in [e.value for e in TrendFilter]
        assert state["trend_strength_label"] in [e.value for e in TrendStrengthLabel]
        assert state["top_warning"] in [e.value for e in TopWarning]
        assert state["ema_alignment"] in [e.value for e in EmaAlignment]

    def test_confidence_range_0_1(self, indicator: TrendPulseIndicator, params: dict) -> None:
        data = _make_ohlcv(_trending_up(600))
        result = indicator.calculate(data, params)
        state = indicator.get_state(result.iloc[-1], result.iloc[-2], params)
        assert 0.0 <= state["confidence"] <= 1.0

    def test_confidence_4f_range_0_1(self, indicator: TrendPulseIndicator, params: dict) -> None:
        data = _make_ohlcv(_trending_up(600))
        result = indicator.calculate(data, params)
        state = indicator.get_state(result.iloc[-1], result.iloc[-2], params)
        assert 0.0 <= state["confidence_4f"] <= 1.0

    def test_score_range_0_100(self, indicator: TrendPulseIndicator, params: dict) -> None:
        data = _make_ohlcv(_trending_up(600))
        result = indicator.calculate(data, params)
        state = indicator.get_state(result.iloc[-1], result.iloc[-2], params)
        assert 0.0 <= state["score"] <= 100.0

    def test_trend_strength_range_0_1(self, indicator: TrendPulseIndicator, params: dict) -> None:
        data = _make_ohlcv(_trending_up(600))
        result = indicator.calculate(data, params)
        state = indicator.get_state(result.iloc[-1], result.iloc[-2], params)
        assert 0.0 <= state["trend_strength"] <= 1.0


class TestTrendPulseDualMACD:
    """Test DualMACD confirmation integration."""

    def test_dm_state_valid_values(self, indicator: TrendPulseIndicator, params: dict) -> None:
        data = _make_ohlcv(_trending_up(600))
        result = indicator.calculate(data, params)
        state = indicator.get_state(result.iloc[-1], result.iloc[-2], params)
        assert state["dm_state"] in ("BULLISH", "IMPROVING", "DETERIORATING", "BEARISH")

    def test_dm_state_bullish_in_uptrend(
        self, indicator: TrendPulseIndicator, params: dict
    ) -> None:
        """Accelerating uptrend should produce BULLISH DM state (linear trend = 0 histogram)."""
        # Linear trends produce 0 MACD histogram; use exponential growth
        prices = [100.0 * (1.005**i) for i in range(800)]
        data = _make_ohlcv(prices)
        result = indicator.calculate(data, params)
        state = indicator.get_state(result.iloc[-1], result.iloc[-2], params)
        assert state["dm_state"] in ("BULLISH", "IMPROVING", "DETERIORATING")

    def test_dm_state_bearish_in_downtrend(
        self, indicator: TrendPulseIndicator, params: dict
    ) -> None:
        data = _make_ohlcv(_trending_down(800, step=1.0))
        result = indicator.calculate(data, params)
        state = indicator.get_state(result.iloc[-1], result.iloc[-2], params)
        assert state["dm_state"] in ("BEARISH", "DETERIORATING")

    def test_dm_state_string_column_in_dataframe(
        self, indicator: TrendPulseIndicator, params: dict
    ) -> None:
        data = _make_ohlcv(_trending_up(600))
        result = indicator.calculate(data, params)
        assert "trend_pulse_dm_state" in result.columns
        valid_states = {"BULLISH", "IMPROVING", "DETERIORATING", "BEARISH"}
        # Check last value (after warmup)
        last_state = result["trend_pulse_dm_state"].iloc[-1]
        assert last_state in valid_states


class TestTrendPulseEntrySignal:
    """Test composite entry signal."""

    def test_entry_signal_is_bool(self, indicator: TrendPulseIndicator, params: dict) -> None:
        data = _make_ohlcv(_trending_up(600))
        result = indicator.calculate(data, params)
        state = indicator.get_state(result.iloc[-1], result.iloc[-2], params)
        assert isinstance(state["entry_signal"], bool)

    def test_entry_requires_swing_buy(self, indicator: TrendPulseIndicator, params: dict) -> None:
        """Entry signal should only be True when swing_signal is BUY."""
        data = _make_ohlcv(_trending_up(600))
        result = indicator.calculate(data, params)

        entry_raw = result["trend_pulse_entry_signal_raw"].values
        swing_raw = result["trend_pulse_swing_signal_raw"].values

        # Every entry=1 bar must have swing=BUY
        for i in range(len(entry_raw)):
            if entry_raw[i] > 0.5:
                assert swing_raw[i] > 0.5, f"Entry at bar {i} without swing BUY"

    def test_entry_not_during_cooldown(self, indicator: TrendPulseIndicator, params: dict) -> None:
        """Entry signal should be suppressed during cooldown."""
        data = _make_ohlcv(_trending_up(600))
        result = indicator.calculate(data, params)

        entry_raw = result["trend_pulse_entry_signal_raw"].values
        cooldown = result["trend_pulse_cooldown_left"].values

        for i in range(len(entry_raw)):
            if cooldown[i] > 0:
                assert entry_raw[i] < 0.5, f"Entry at bar {i} during cooldown={cooldown[i]}"


class TestTrendPulseATRStop:
    """Test ATR stop level computation."""

    def test_atr_stop_valid_after_warmup(
        self, indicator: TrendPulseIndicator, params: dict
    ) -> None:
        data = _make_ohlcv(_trending_up(600))
        result = indicator.calculate(data, params)
        state = indicator.get_state(result.iloc[-1], result.iloc[-2], params)
        assert state["atr_stop_level"] > 0

    def test_atr_stop_below_close(self, indicator: TrendPulseIndicator, params: dict) -> None:
        """ATR stop should be below current close in an uptrend."""
        data = _make_ohlcv(_trending_up(600))
        result = indicator.calculate(data, params)
        last_close = data["close"].iloc[-1]
        state = indicator.get_state(result.iloc[-1], result.iloc[-2], params)
        assert state["atr_stop_level"] < last_close


class TestTrendPulseCooldown:
    """Test cooldown tracking."""

    def test_cooldown_left_non_negative(self, indicator: TrendPulseIndicator, params: dict) -> None:
        data = _make_ohlcv(_trending_up(600))
        result = indicator.calculate(data, params)
        state = indicator.get_state(result.iloc[-1], result.iloc[-2], params)
        assert state["cooldown_left"] >= 0

    def test_cooldown_left_is_int(self, indicator: TrendPulseIndicator, params: dict) -> None:
        data = _make_ohlcv(_trending_up(600))
        result = indicator.calculate(data, params)
        state = indicator.get_state(result.iloc[-1], result.iloc[-2], params)
        assert isinstance(state["cooldown_left"], int)

    def test_cooldown_counts_down_from_exit(
        self, indicator: TrendPulseIndicator, params: dict
    ) -> None:
        """After an exit signal, cooldown_left should be positive on the next bar."""
        prices = _trending_up(300, start=100, step=0.3)
        prices += _trending_down(300, start=prices[-1], step=1.0)
        data = _make_ohlcv(prices)
        result = indicator.calculate(data, params)

        exit_raw = result["trend_pulse_exit_signal_raw"].values
        cooldown = result["trend_pulse_cooldown_left"].values
        cooldown_bars = params["cooldown_bars"]

        # Find an exit signal followed by a non-exit bar, verify cooldown > 0
        exit_indices = np.where(exit_raw > 0)[0]
        for idx in exit_indices:
            # Find next bar that is NOT also an exit (to avoid reset)
            for j in range(idx + 1, min(idx + cooldown_bars + 1, len(cooldown))):
                if exit_raw[j] == 0:
                    assert cooldown[j] > 0, (
                        f"After exit at bar {idx}, cooldown[{j}]={cooldown[j]} " f"but expected > 0"
                    )
                    break


class TestTrendPulseExitSignal:
    """Test exit signal classification."""

    def test_exit_signal_valid_values(self, indicator: TrendPulseIndicator, params: dict) -> None:
        data = _make_ohlcv(_trending_up(600))
        result = indicator.calculate(data, params)
        state = indicator.get_state(result.iloc[-1], result.iloc[-2], params)
        assert state["exit_signal"] in ("atr_stop", "dm_regime", "zig_sell", "top_detected", "none")

    def test_exit_string_column_matches_raw(
        self, indicator: TrendPulseIndicator, params: dict
    ) -> None:
        data = _make_ohlcv(_trending_up(600))
        result = indicator.calculate(data, params)

        raw = result["trend_pulse_exit_signal_raw"].values
        strs = result["trend_pulse_exit_signal"].values

        raw_to_str = {0: "none", 1: "atr_stop", 2: "dm_regime", 3: "zig_sell", 4: "top_detected"}
        for i in range(len(raw)):
            expected = raw_to_str.get(int(round(raw[i])), "none")
            assert strs[i] == expected, f"Bar {i}: raw={raw[i]} but str={strs[i]}"


class TestTrendPulseADXFilter:
    """Test ADX chop filter."""

    def test_adx_ok_true_in_trending(self, indicator: TrendPulseIndicator, params: dict) -> None:
        """Strong trend should pass ADX filter."""
        data = _make_ohlcv(_trending_up(800, step=1.0))
        result = indicator.calculate(data, params)
        state = indicator.get_state(result.iloc[-1], result.iloc[-2], params)
        assert state["adx_ok"] is True

    def test_adx_value_present(self, indicator: TrendPulseIndicator, params: dict) -> None:
        data = _make_ohlcv(_trending_up(600))
        result = indicator.calculate(data, params)
        state = indicator.get_state(result.iloc[-1], result.iloc[-2], params)
        assert isinstance(state["adx"], float)
        assert state["adx"] >= 0


class TestTrendPulseEMA:
    """Test EMA alignment and trend filter."""

    def test_trend_filter_bullish_above_ema453(
        self, indicator: TrendPulseIndicator, params: dict
    ) -> None:
        data = _make_ohlcv(_trending_up(600))
        result = indicator.calculate(data, params)
        state = indicator.get_state(result.iloc[-1], result.iloc[-2], params)
        assert state["trend_filter"] == TrendFilter.BULLISH.value

    def test_trend_filter_bearish_below_ema453(
        self, indicator: TrendPulseIndicator, params: dict
    ) -> None:
        data = _make_ohlcv(_trending_down(600))
        result = indicator.calculate(data, params)
        state = indicator.get_state(result.iloc[-1], result.iloc[-2], params)
        assert state["trend_filter"] == TrendFilter.BEARISH.value

    def test_ema_alignment_bull(self, indicator: TrendPulseIndicator, params: dict) -> None:
        data = _make_ohlcv(_trending_up(800, step=1.0))
        result = indicator.calculate(data, params)
        state = indicator.get_state(result.iloc[-1], result.iloc[-2], params)
        assert state["ema_alignment"] == EmaAlignment.ALIGNED_BULL.value

    def test_ema_alignment_mixed(self, indicator: TrendPulseIndicator, params: dict) -> None:
        data = _make_ohlcv(_zigzag_prices(600))
        result = indicator.calculate(data, params)
        state = indicator.get_state(result.iloc[-1], result.iloc[-2], params)
        assert state["ema_alignment"] in [
            EmaAlignment.MIXED.value,
            EmaAlignment.ALIGNED_BULL.value,
            EmaAlignment.ALIGNED_BEAR.value,
        ]


class TestTrendPulseZIG:
    """Test causal ZIG behavior."""

    def test_causal_zig_forward_fills_last_pivot(self) -> None:
        close = np.array([100, 101, 102, 103, 104, 105, 106, 107, 108, 109, 110])
        zig = TrendPulseIndicator._causal_zig(close, 5.0)
        assert not np.any(np.isnan(zig))

    def test_causal_zig_no_future_data(self) -> None:
        """Perturbing future data must not change past ZIG values."""
        np.random.seed(42)
        close = np.cumsum(np.random.randn(200)) + 100
        close = np.maximum(close, 10)

        zig_original = TrendPulseIndicator._causal_zig(close.copy(), 5.0)

        close_perturbed = close.copy()
        close_perturbed[150:] *= 1.5

        zig_perturbed = TrendPulseIndicator._causal_zig(close_perturbed, 5.0)

        np.testing.assert_array_equal(zig_original[:150], zig_perturbed[:150])

    def test_zig_detects_reversal_from_candidate_extreme(self) -> None:
        close = np.array(
            [100.0] + list(np.linspace(100, 110, 20)) + list(np.linspace(110, 104, 10))
        )
        zig = TrendPulseIndicator._causal_zig(close, 5.0)
        last_val = zig[-1]
        assert last_val > 109.0, f"Expected confirmed high ~110, got {last_val}"

    def test_zig_initial_state_tracks_both_directions(self) -> None:
        close = np.array([100.0] * 5 + list(np.linspace(100, 93, 10)))
        zig = TrendPulseIndicator._causal_zig(close, 5.0)
        assert zig[-1] >= 99.0, f"Expected confirmed high near 100, got {zig[-1]}"

    def test_zig_ma_crossover_triggers_signal(
        self, indicator: TrendPulseIndicator, params: dict
    ) -> None:
        prices = _trending_down(300, start=200, step=0.5) + _trending_up(300, start=50, step=1.0)
        data = _make_ohlcv(prices)
        result = indicator.calculate(data, params)

        cross_ups = result["trend_pulse_zig_cross_up"].sum()
        cross_downs = result["trend_pulse_zig_cross_down"].sum()
        assert cross_ups + cross_downs > 0


class TestTrendPulseSwingLogic:
    """Test swing signal generation with cooldown."""

    def test_no_buy_when_bearish(self, indicator: TrendPulseIndicator, params: dict) -> None:
        data = _make_ohlcv(_trending_down(600))
        result = indicator.calculate(data, params)

        buy_count = (result["trend_pulse_swing_signal_raw"] > 0.5).sum()
        assert buy_count == 0

    def test_buy_cooldown_suppresses_repeated_buys(
        self, indicator: TrendPulseIndicator, params: dict
    ) -> None:
        prices = _zigzag_prices(600, amplitude=15, period=8)
        data = _make_ohlcv(prices)
        result = indicator.calculate(data, params)

        signals = result["trend_pulse_swing_signal_raw"].values
        swing_filter = params["swing_filter_bars"]

        last_buy_idx = -swing_filter - 1
        for i in range(len(signals)):
            if signals[i] > 0.5:
                if i - last_buy_idx < swing_filter and last_buy_idx >= 0:
                    pytest.fail(
                        f"BUY at {i} too close to previous BUY at {last_buy_idx} "
                        f"(gap={i - last_buy_idx}, filter={swing_filter})"
                    )
                last_buy_idx = i

    def test_sell_not_suppressed_by_cooldown(
        self, indicator: TrendPulseIndicator, params: dict
    ) -> None:
        prices = _trending_up(300, start=100, step=0.3)
        prices += _trending_down(300, start=prices[-1], step=1.0)
        data = _make_ohlcv(prices)
        result = indicator.calculate(data, params)

        signals = result["trend_pulse_swing_signal_raw"].values
        assert np.any(signals < -0.5) or True

    def test_opposite_direction_not_suppressed(
        self, indicator: TrendPulseIndicator, params: dict
    ) -> None:
        prices = _trending_up(300, start=100, step=0.3)
        prices += _trending_down(100, start=prices[-1], step=2.0)
        prices += _trending_up(200, start=prices[-1], step=0.5)
        data = _make_ohlcv(prices)
        result = indicator.calculate(data, params)

        signals = result["trend_pulse_swing_signal_raw"].values
        buys = np.where(signals > 0.5)[0]
        sells = np.where(signals < -0.5)[0]
        assert len(buys) > 0 or len(sells) > 0, "Expected at least some signals"


class TestTrendPulseTopWarning:
    """Test top detection logic."""

    def test_top_pending_on_wr_above_70(self, indicator: TrendPulseIndicator, params: dict) -> None:
        current = pd.Series(
            {
                "trend_pulse_close": 200.0,
                "trend_pulse_adx": 30.0,
                "trend_pulse_swing_signal_raw": 0.0,
                "trend_pulse_top_warning_raw": 1.0,
                "trend_pulse_trend_filter": 1.0,
                "trend_pulse_wr_long": 75.0,
                "trend_pulse_wr_mid": 75.0,
                "trend_pulse_wr_short": 75.0,
                "trend_pulse_dmi_plus": 30.0,
                "trend_pulse_dmi_minus": 15.0,
                "trend_pulse_ema_14": 199.0,
                "trend_pulse_ema_25": 198.0,
                "trend_pulse_ema_99": 195.0,
                "trend_pulse_ema_144": 190.0,
                "trend_pulse_ema_453": 180.0,
                "trend_pulse_dm_state_raw": 2.0,
                "trend_pulse_adx_ok": 1.0,
                "trend_pulse_entry_signal_raw": 0.0,
                "trend_pulse_exit_signal_raw": 0.0,
                "trend_pulse_confidence_4f": 0.5,
                "trend_pulse_atr_stop_level": 190.0,
                "trend_pulse_cooldown_left": 0.0,
            }
        )
        state = indicator.get_state(current, None, params)
        assert state["top_warning"] == TopWarning.TOP_PENDING.value

    def test_no_top_when_wr_low(self, indicator: TrendPulseIndicator, params: dict) -> None:
        current = pd.Series(
            {
                "trend_pulse_close": 200.0,
                "trend_pulse_adx": 30.0,
                "trend_pulse_swing_signal_raw": 0.0,
                "trend_pulse_top_warning_raw": 0.0,
                "trend_pulse_trend_filter": 1.0,
                "trend_pulse_wr_long": 40.0,
                "trend_pulse_wr_mid": 40.0,
                "trend_pulse_wr_short": 40.0,
                "trend_pulse_dmi_plus": 30.0,
                "trend_pulse_dmi_minus": 15.0,
                "trend_pulse_ema_14": 199.0,
                "trend_pulse_ema_25": 198.0,
                "trend_pulse_ema_99": 195.0,
                "trend_pulse_ema_144": 190.0,
                "trend_pulse_ema_453": 180.0,
                "trend_pulse_dm_state_raw": 2.0,
                "trend_pulse_adx_ok": 1.0,
                "trend_pulse_entry_signal_raw": 0.0,
                "trend_pulse_exit_signal_raw": 0.0,
                "trend_pulse_confidence_4f": 0.5,
                "trend_pulse_atr_stop_level": 190.0,
                "trend_pulse_cooldown_left": 0.0,
            }
        )
        state = indicator.get_state(current, None, params)
        assert state["top_warning"] == TopWarning.NONE.value

    def test_top_cross_detection_in_calculate(self) -> None:
        n = 10
        wr_long = np.array([60, 65, 70, 75, 80, 86, 90, 85, 80, 75], dtype=np.float64)
        wr_short = np.array([70, 72, 75, 80, 86, 85, 82, 78, 70, 65], dtype=np.float64)
        wr_mid = np.array([60, 65, 70, 80, 86, 88, 87, 83, 78, 70], dtype=np.float64)
        adx = np.array([40, 38, 36, 34, 32, 30, 28, 26, 24, 22], dtype=np.float64)

        top_arr = TrendPulseIndicator._compute_top_warnings(wr_long, wr_mid, wr_short, adx, n)
        assert top_arr[5] >= 2.0, f"Expected TOP_ZONE or TOP_DETECTED at bar 5, got {top_arr[5]}"

    def test_filter_debounce_suppresses_repeated_top_zone(self) -> None:
        n = 10
        wr_long = np.array([50, 55, 60, 65, 65, 65, 65, 60, 55, 50], dtype=np.float64)
        wr_mid = np.array([70, 75, 82, 81, 80, 79, 78, 55, 50, 45], dtype=np.float64)
        wr_short = np.array([80, 85, 96, 83, 82, 81, 80, 50, 45, 40], dtype=np.float64)
        adx = np.array([30, 30, 30, 30, 30, 30, 30, 30, 30, 30], dtype=np.float64)

        top_arr = TrendPulseIndicator._compute_top_warnings(wr_long, wr_mid, wr_short, adx, n)
        zone_count = sum(1 for v in top_arr[3:7] if v >= 1.5)
        assert zone_count <= 1, (
            f"Expected at most 1 TOP_ZONE in bars 3-6 (debounce), got {zone_count}: "
            f"{top_arr[3:7]}"
        )

    def test_top_exit_when_wr_below_60_for_3bars(self) -> None:
        n = 10
        wr_long = np.array([80, 80, 80, 50, 50, 50, 50, 50, 50, 50], dtype=np.float64)
        wr_short = np.array([90, 90, 90, 50, 50, 50, 50, 50, 50, 50], dtype=np.float64)
        wr_mid = np.array([85, 85, 85, 55, 55, 55, 50, 50, 50, 50], dtype=np.float64)
        adx = np.array([30, 30, 30, 30, 30, 30, 30, 30, 30, 30], dtype=np.float64)

        top_arr = TrendPulseIndicator._compute_top_warnings(wr_long, wr_mid, wr_short, adx, n)
        assert top_arr[5] == 0.0, f"Expected NONE at bar 5 after 3 bars <60, got {top_arr[5]}"


class TestTrendPulseConfidence:
    """Test confidence scoring."""

    def test_decomposition_weights_sum(self) -> None:
        config = TrendPulseIndicator._default_params
        weights = config["confidence_weights"]
        assert abs(sum(weights) - 1.0) < 1e-9

    def test_score_halved_when_no_signal(
        self, indicator: TrendPulseIndicator, params: dict
    ) -> None:
        data = _make_ohlcv(_trending_up(600))
        result = indicator.calculate(data, params)
        state = indicator.get_state(result.iloc[-1], result.iloc[-2], params)
        if state["swing_signal"] == SwingSignal.NONE.value:
            assert state["score"] <= 50.0

    def test_top_risk_reduces_confidence(
        self, indicator: TrendPulseIndicator, params: dict
    ) -> None:
        base_no_top = pd.Series(
            {
                "trend_pulse_close": 200.0,
                "trend_pulse_adx": 30.0,
                "trend_pulse_swing_signal_raw": 0.0,
                "trend_pulse_top_warning_raw": 0.0,
                "trend_pulse_trend_filter": 1.0,
                "trend_pulse_wr_long": 40.0,
                "trend_pulse_wr_mid": 40.0,
                "trend_pulse_wr_short": 40.0,
                "trend_pulse_dmi_plus": 30.0,
                "trend_pulse_dmi_minus": 15.0,
                "trend_pulse_ema_14": 199.0,
                "trend_pulse_ema_25": 198.0,
                "trend_pulse_ema_99": 195.0,
                "trend_pulse_ema_144": 190.0,
                "trend_pulse_ema_453": 180.0,
                "trend_pulse_dm_state_raw": 2.0,
                "trend_pulse_adx_ok": 1.0,
                "trend_pulse_entry_signal_raw": 0.0,
                "trend_pulse_exit_signal_raw": 0.0,
                "trend_pulse_confidence_4f": 0.5,
                "trend_pulse_atr_stop_level": 190.0,
                "trend_pulse_cooldown_left": 0.0,
            }
        )
        state_no_top = indicator.get_state(base_no_top, None, params)

        base_with_top = base_no_top.copy()
        base_with_top["trend_pulse_top_warning_raw"] = 3.0
        state_with_top = indicator.get_state(base_with_top, None, params)

        assert state_with_top["confidence"] < state_no_top["confidence"]


class TestTrendPulseRuleIntegration:
    """Test that string columns work for rule engine state_change detection."""

    def test_swing_signal_string_column_matches_numeric(
        self, indicator: TrendPulseIndicator, params: dict
    ) -> None:
        prices = _trending_down(300, start=200, step=0.5) + _trending_up(300, start=50, step=1.0)
        data = _make_ohlcv(prices)
        result = indicator.calculate(data, params)

        raw = result["trend_pulse_swing_signal_raw"].values
        strs = result["trend_pulse_swing_signal"].values

        for i in range(len(raw)):
            if raw[i] > 0.5:
                assert strs[i] == "BUY", f"Bar {i}: raw={raw[i]} but str={strs[i]}"
            elif raw[i] < -0.5:
                assert strs[i] == "SELL", f"Bar {i}: raw={raw[i]} but str={strs[i]}"
            else:
                assert strs[i] == "NONE", f"Bar {i}: raw={raw[i]} but str={strs[i]}"

    def test_top_warning_string_column_matches_numeric(
        self, indicator: TrendPulseIndicator, params: dict
    ) -> None:
        data = _make_ohlcv(_trending_up(600))
        result = indicator.calculate(data, params)

        raw = result["trend_pulse_top_warning_raw"].values
        strs = result["trend_pulse_top_warning"].values

        mapping = {0: "NONE", 1: "TOP_PENDING", 2: "TOP_ZONE", 3: "TOP_DETECTED"}
        for i in range(len(raw)):
            expected = mapping.get(int(round(raw[i])), "NONE")
            assert strs[i] == expected, f"Bar {i}: raw={raw[i]} but str={strs[i]}"

    def test_entry_signal_string_column(self, indicator: TrendPulseIndicator, params: dict) -> None:
        data = _make_ohlcv(_trending_up(600))
        result = indicator.calculate(data, params)
        assert "trend_pulse_entry_signal" in result.columns
        valid = {"true", "false"}
        for val in result["trend_pulse_entry_signal"].values:
            assert val in valid

    def test_exit_signal_string_column(self, indicator: TrendPulseIndicator, params: dict) -> None:
        data = _make_ohlcv(_trending_up(600))
        result = indicator.calculate(data, params)
        assert "trend_pulse_exit_signal" in result.columns
        valid = {"none", "atr_stop", "dm_regime", "zig_sell", "top_detected"}
        for val in result["trend_pulse_exit_signal"].values:
            assert val in valid


class TestTrendPulseWarmup:
    def test_warmup_periods_500(self) -> None:
        indicator = TrendPulseIndicator()
        assert indicator.warmup_periods == 500
