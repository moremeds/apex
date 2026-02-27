"""Tests for equity signal synthesis."""

from src.domain.services.advisor.equity_advisor import EquityAdvisor


def _signal(rule, direction, strength):
    """Create a signal dict. Direction uses "bullish"/"bearish"/"neutral"."""
    return {"rule": rule, "direction": direction, "strength": strength}


class TestEquityAdvisor:
    def setup_method(self):
        self.advisor = EquityAdvisor()

    def test_strong_buy_many_bullish(self):
        signals = [
            _signal("trend_pulse_entry", "bullish", 85),
            _signal("dual_macd_dip_buy", "bullish", 80),
            _signal("supertrend_bullish", "bullish", 75),
            _signal("ema_golden_cross", "bullish", 65),
            _signal("volume_spike", "bullish", 60),
        ]
        advice = self.advisor.synthesize("NVDA", "Semiconductors", signals, "R0", {})
        assert advice.action in ("STRONG_BUY", "BUY")
        assert advice.confidence > 60

    def test_strong_sell_many_bearish(self):
        signals = [
            _signal("trend_pulse_sell", "bearish", 80),
            _signal("macd_bearish_cross", "bearish", 60),
            _signal("supertrend_bearish", "bearish", 75),
            _signal("ema_death_cross", "bearish", 65),
        ]
        advice = self.advisor.synthesize("GME", "Speculative", signals, "R2", {})
        assert advice.action in ("STRONG_SELL", "SELL")

    def test_hold_mixed_signals(self):
        signals = [
            _signal("supertrend_bullish", "bullish", 75),
            _signal("macd_bearish_cross", "bearish", 60),
        ]
        advice = self.advisor.synthesize("AAPL", "Technology", signals, "R1", {})
        assert advice.action == "HOLD"

    def test_hold_no_signals(self):
        advice = self.advisor.synthesize("AAPL", "Technology", [], "R0", {})
        assert advice.action == "HOLD"
        assert advice.confidence < 20

    def test_r2_blocks_buy(self):
        signals = [
            _signal("supertrend_bullish", "bullish", 75),
            _signal("ema_golden_cross", "bullish", 65),
        ]
        advice = self.advisor.synthesize("AAPL", "Technology", signals, "R2", {})
        assert advice.action in ("HOLD", "SELL", "STRONG_SELL")

    def test_r2_amplifies_sell(self):
        signals = [
            _signal("supertrend_bearish", "bearish", 75),
            _signal("macd_bearish_cross", "bearish", 60),
        ]
        r0_advice = self.advisor.synthesize("AAPL", "Technology", signals, "R0", {})
        r2_advice = self.advisor.synthesize("AAPL", "Technology", signals, "R2", {})
        assert r2_advice.confidence >= r0_advice.confidence

    def test_trend_pulse_weighted_higher(self):
        tp_signals = [_signal("trend_pulse_entry", "bullish", 70)]
        other_signals = [_signal("ema_golden_cross", "bullish", 70)]
        tp_advice = self.advisor.synthesize("AAPL", "Technology", tp_signals, "R0", {})
        other_advice = self.advisor.synthesize("AAPL", "Technology", other_signals, "R0", {})
        assert tp_advice.confidence >= other_advice.confidence

    def test_single_signal_not_strong(self):
        """Single signal should not produce STRONG_BUY/STRONG_SELL."""
        signals = [_signal("supertrend_bearish", "bearish", 80)]
        advice = self.advisor.synthesize("AAPL", "Technology", signals, "R0", {})
        assert advice.action in ("HOLD", "SELL")  # NOT STRONG_SELL

    def test_top_signals_limited_to_3(self):
        signals = [_signal(f"rule_{i}", "bullish", 80 - i) for i in range(10)]
        advice = self.advisor.synthesize("AAPL", "Technology", signals, "R0", {})
        assert len(advice.top_signals) <= 3

    def test_signal_summary_counts(self):
        signals = [
            _signal("a", "bullish", 70),
            _signal("b", "bullish", 60),
            _signal("c", "bearish", 50),
        ]
        advice = self.advisor.synthesize("AAPL", "Technology", signals, "R0", {})
        assert advice.signal_summary["bullish"] == 2
        assert advice.signal_summary["bearish"] == 1

    def test_works_for_full_universe(self):
        signals = [_signal("rsi_oversold_exit", "bullish", 70)]
        for sym in ("AAPL", "RIVN", "XOM", "NEE", "SPY"):
            advice = self.advisor.synthesize(sym, "Test", signals, "R0", {})
            assert advice.symbol == sym
