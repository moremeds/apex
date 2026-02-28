"""AdvisorService — orchestrates premium and equity advisors.

Uses get_recent_signals() callable (signal buffer in pipeline) instead of
RuleEngine.get_evaluation_history() which requires trace_mode.

VIX data cached at compute_all level to avoid redundant calls.
"""

from __future__ import annotations

import logging
from datetime import datetime, timezone
from typing import Any, Callable

from src.domain.services.advisor.equity_advisor import EquityAdvisor
from src.domain.services.advisor.models import EquityAdvice, MarketContext, PremiumAdvice
from src.domain.services.advisor.premium_advisor import PremiumAdvisor
from src.domain.services.advisor.vrp import compute_term_structure, compute_vrp

logger = logging.getLogger(__name__)


class AdvisorService:
    """Orchestrates premium + equity advisors with live data feeds."""

    def __init__(
        self,
        get_regime_states: Callable,
        get_indicator_states: Callable,
        get_vix_data: Callable,
        get_underlying_close: Callable,
        get_recent_signals: Callable,
        etf_symbols: list[str],
        universe_symbols: list[str],
        sector_map: dict[str, str],
    ) -> None:
        self._get_regime_states = get_regime_states
        self._get_indicator_states = get_indicator_states
        self._get_vix_data = get_vix_data
        self._get_underlying_close = get_underlying_close
        self._get_recent_signals = get_recent_signals
        self._etf_symbols = etf_symbols
        self._universe_symbols = universe_symbols
        self._sector_map = sector_map

        self._premium_advisor = PremiumAdvisor()
        self._equity_advisor = EquityAdvisor()

    def compute_all(self) -> dict[str, Any]:
        """Compute advice for all ETFs (premium) and universe (equity)."""
        # Cache VIX data for this computation cycle
        vix_cache = self._get_vix_data()

        ctx = self._build_market_context(vix_cache)
        premium = [self._compute_premium(sym, ctx, vix_cache) for sym in self._etf_symbols]
        equity = [self._compute_equity(sym, ctx) for sym in self._universe_symbols]

        return {
            "market_context": ctx,
            "premium": [a for a in premium if a is not None],
            "equity": [a for a in equity if a is not None],
            "timestamp": datetime.now(timezone.utc).isoformat(),
        }

    def compute_symbol(self, symbol: str) -> dict[str, Any]:
        """Compute advice for a single symbol."""
        vix_cache = self._get_vix_data()
        ctx = self._build_market_context(vix_cache)
        result: dict[str, Any] = {
            "market_context": ctx,
            "timestamp": datetime.now(timezone.utc).isoformat(),
        }

        if symbol in self._etf_symbols:
            result["premium"] = self._compute_premium(symbol, ctx, vix_cache)

        result["equity"] = self._compute_equity(symbol, ctx) or EquityAdvice(
            symbol=symbol,
            sector=self._sector_map.get(symbol, "Unknown"),
            action="HOLD",
            confidence=0,
            regime=ctx.regime if ctx else "R1",
            signal_summary={"bullish": 0, "bearish": 0, "neutral": 0},
            top_signals=[],
            trend_pulse=None,
            key_levels={},
            reasoning=["Insufficient data"],
        )

        return result

    def _build_market_context(self, vix_cache: tuple) -> MarketContext:
        """Build market-level context from current data."""
        regimes = self._get_regime_states("1d")
        spy_regime = regimes.get("SPY", {})

        vix_series, vix3m_series = vix_cache

        vix_val = (
            float(vix_series.iloc[-1]) if vix_series is not None and len(vix_series) > 0 else 0
        )
        vix3m_val = (
            float(vix3m_series.iloc[-1])
            if vix3m_series is not None and len(vix3m_series) > 0
            else 0
        )

        # VRP for SPY (market-level)
        spy_close = self._get_underlying_close("SPY")
        vrp_zscore = 0.0
        iv_pctile = 50.0
        if vix_series is not None and spy_close is not None and len(vix_series) > 60:
            try:
                vrp_result = compute_vrp(vix_series, spy_close)
                vrp_zscore = vrp_result.vrp_zscore
                iv_pctile = vrp_result.iv_percentile
            except Exception:
                logger.debug("VRP computation failed — using defaults")

        ts_ratio, ts_state = (
            compute_term_structure(vix_val, vix3m_val) if vix3m_val > 0 else (1.0, "flat")
        )

        return MarketContext(
            regime=spy_regime.get("regime", "R1"),
            regime_name=spy_regime.get("regime_name", "Unknown"),
            regime_confidence=spy_regime.get("confidence", 50),
            vix=vix_val,
            vix_percentile=iv_pctile,
            vrp_zscore=vrp_zscore,
            term_structure_ratio=ts_ratio,
            term_structure_state=ts_state,
            timestamp=datetime.now(timezone.utc).isoformat(),
        )

    def _compute_premium(
        self, symbol: str, ctx: MarketContext, vix_cache: tuple
    ) -> PremiumAdvice | None:
        """Compute premium advice for an ETF symbol."""
        try:
            vix_series, _ = vix_cache
            underlying = self._get_underlying_close(symbol)

            if vix_series is None or underlying is None or len(vix_series) < 60:
                return None

            vrp = compute_vrp(vix_series, underlying)
            spot = float(underlying.iloc[-1])

            # Determine trend direction from indicators
            states = self._get_indicator_states(symbol, "1d")
            trend = self._infer_trend(states)

            return self._premium_advisor.advise(
                symbol=symbol,
                spot=spot,
                regime=ctx.regime,
                vrp=vrp,
                term_structure_ratio=ctx.term_structure_ratio,
                earnings_days_away=None,  # TODO: wire earnings service
                trend_direction=trend,
            )
        except Exception:
            logger.exception("Premium advice failed for %s", symbol)
            return None

    def _compute_equity(self, symbol: str, ctx: MarketContext) -> EquityAdvice | None:
        """Compute equity advice for a symbol."""
        try:
            regimes = self._get_regime_states("1d")
            sym_regime = regimes.get(symbol, {}).get("regime", ctx.regime)
            sector = self._sector_map.get(symbol, "Unknown")

            # Get active signals from signal buffer
            active = self._get_recent_signals(symbol)

            # Get indicator state
            states = self._get_indicator_states(symbol, "1d")
            flat_state: dict[str, Any] = {}
            if isinstance(states, dict):
                for key, val in states.items():
                    if isinstance(key, tuple) and len(key) == 3:
                        _, _, ind_name = key
                        flat_state[ind_name] = val
                    else:
                        flat_state[key] = val

            return self._equity_advisor.synthesize(symbol, sector, active, sym_regime, flat_state)
        except Exception:
            logger.exception("Equity advice failed for %s", symbol)
            return None

    def _infer_trend(self, states: dict) -> str:
        """Infer trend direction from indicator states."""
        for key, val in states.items():
            ind_name = key[2] if isinstance(key, tuple) and len(key) == 3 else key
            if ind_name == "supertrend" and isinstance(val, dict):
                direction = val.get("direction")
                if direction == "up":
                    return "up"
                elif direction == "down":
                    return "down"
        return "sideways"
