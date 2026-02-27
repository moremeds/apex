"""Tests for AdvisorService orchestrator."""

import pandas as pd

from src.domain.services.advisor.advisor_service import AdvisorService


def _make_vix_series(val=18.0, n=300):
    """Create a mock VIX pandas Series."""
    import numpy as np

    return pd.Series(np.full(n, val), index=pd.date_range("2025-06-01", periods=n, freq="B"))


def _make_underlying_series(val=500.0, n=300):
    """Create a mock underlying close pandas Series."""
    import numpy as np

    np.random.seed(42)
    returns = np.random.normal(0.0003, 0.01, n)
    prices = val * (1 + pd.Series(returns)).cumprod()
    prices.index = pd.date_range("2025-06-01", periods=n, freq="B")
    return prices


def _make_advisor_service(
    regime_states=None,
    indicator_states=None,
    vix_val=18.0,
    vix3m_val=20.0,
    recent_signals=None,
    etf_symbols=None,
    universe_symbols=None,
    sector_map=None,
):
    """Create AdvisorService with mock dependencies."""
    vix_series = _make_vix_series(vix_val) if vix_val else None
    vix3m_series = _make_vix_series(vix3m_val) if vix3m_val else None
    underlying = _make_underlying_series()

    return AdvisorService(
        get_regime_states=lambda tf="1d": regime_states or {},
        get_indicator_states=lambda sym=None, tf=None: indicator_states or {},
        get_vix_data=lambda: (vix_series, vix3m_series),
        get_underlying_close=lambda sym: underlying,
        get_recent_signals=lambda sym=None: (recent_signals or {}).get(sym, []) if sym else [],
        etf_symbols=etf_symbols or ["QQQ", "SPY"],
        universe_symbols=universe_symbols or ["AAPL", "NVDA"],
        sector_map=sector_map or {"AAPL": "Technology", "NVDA": "Semiconductors"},
    )


class TestAdvisorServiceBasic:
    def test_compute_all_returns_structure(self):
        svc = _make_advisor_service()
        result = svc.compute_all()
        assert "market_context" in result
        assert "premium" in result
        assert "equity" in result
        assert "timestamp" in result

    def test_compute_symbol_returns_structure(self):
        svc = _make_advisor_service(universe_symbols=["AAPL"])
        result = svc.compute_symbol("AAPL")
        assert "equity" in result
        assert result["equity"].symbol == "AAPL"

    def test_premium_only_for_etfs(self):
        svc = _make_advisor_service(
            etf_symbols=["QQQ"],
            universe_symbols=["AAPL"],
        )
        result = svc.compute_all()
        premium_syms = {a.symbol for a in result["premium"]}
        equity_syms = {a.symbol for a in result["equity"]}
        assert "QQQ" in premium_syms
        assert "AAPL" not in premium_syms
        assert "AAPL" in equity_syms

    def test_missing_vix_returns_defaults(self):
        """No VIX data should not crash."""
        svc = _make_advisor_service(vix_val=None, vix3m_val=None)
        result = svc.compute_all()
        assert result["market_context"].vix == 0

    def test_compute_all_with_signals(self):
        """Verify signals are passed through to equity advisor."""
        svc = _make_advisor_service(
            universe_symbols=["AAPL"],
            recent_signals={
                "AAPL": [
                    {"rule": "supertrend_bullish", "direction": "bullish", "strength": 75},
                    {"rule": "ema_golden_cross", "direction": "bullish", "strength": 65},
                ],
            },
        )
        result = svc.compute_all()
        aapl = [e for e in result["equity"] if e.symbol == "AAPL"]
        assert len(aapl) == 1
        assert aapl[0].action in ("BUY", "STRONG_BUY", "HOLD")
