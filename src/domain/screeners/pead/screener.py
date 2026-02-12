"""PEAD screener core logic.

Pure domain service that filters and scores earnings surprises into PEAD candidates.
Uses pandas_market_calendars for trading-day arithmetic (NYSE schedule).
"""

from __future__ import annotations

import statistics
from datetime import date, datetime, timedelta
from typing import Any

import pandas_market_calendars as mcal

from src.utils.logging_setup import get_logger

from .models import (
    EarningsSurprise,
    LiquidityTier,
    PEADCandidate,
    PEADScreenResult,
)
from .scorer import classify_quality, score_pead_quality

logger = get_logger(__name__)

# NYSE calendar (module-level singleton to avoid repeated instantiation)
_NYSE = mcal.get_calendar("NYSE")


class PEADScreener:
    """Screens recent earnings for PEAD drift candidates.

    Filter pipeline (cheapest first):
        1. R2 regime block → return empty immediately
        2. Entry window: 1-5 trading days since report
        3. SUE >= min_sue
        4. Gap >= min_earnings_day_gap
        5. Volume >= min_volume_ratio
        6. 52w high hard exclude
        7. Analyst downgrade hard exclude (within N days)
        8. Forward PE filter (> max excluded)
        9. Revenue beat filter (optional)

    After filters: score quality, assign trade params, sort descending.
    """

    def __init__(self, config: dict[str, Any]) -> None:
        self._filters = config.get("filters", {})
        self._trade = config.get("trade_params", {})
        self._regime = config.get("regime_rules", {})
        self._liquidity = config.get("liquidity_tiers", {})
        self._quality = config.get("quality_thresholds", {})

    # ── Public API ────────────────────────────────────────────────────

    def generate_pead_candidates(
        self,
        recent_earnings: list[dict[str, Any]],
        regime: str,
        today: date,
        market_caps: dict[str, float],
    ) -> PEADScreenResult:
        """Screen recent earnings for PEAD candidates.

        Args:
            recent_earnings: List of dicts with keys matching EarningsSurprise fields
                plus ``historical_surprises``, ``current_price``, ``earnings_open``.
            regime: Current market regime ("R0", "R1", "R2", "R3").
            today: Current date for entry window calculation.
            market_caps: symbol -> market cap in USD.

        Returns:
            PEADScreenResult with scored, sorted candidates.
        """
        generated_at = datetime.now()

        # Gate 1: R2 block
        if regime == "R2" and self._regime.get("r2_block_entirely", True):
            logger.info("PEAD: R2 regime — blocking all new positions")
            return PEADScreenResult(
                candidates=[],
                screened_count=len(recent_earnings),
                passed_filters=0,
                regime=regime,
                generated_at=generated_at,
            )

        candidates: list[PEADCandidate] = []
        errors: dict[str, str] = {}

        for earning in recent_earnings:
            symbol = earning.get("symbol", "?")
            try:
                candidate = self._evaluate_single(earning, regime, today, market_caps)
                if candidate is not None:
                    candidates.append(candidate)
            except Exception as e:
                logger.warning(f"PEAD: error evaluating {symbol}: {e}")
                errors[symbol] = str(e)

        # Sort by quality descending
        candidates.sort(key=lambda c: c.quality_score, reverse=True)

        return PEADScreenResult(
            candidates=candidates,
            screened_count=len(recent_earnings),
            passed_filters=len(candidates),
            regime=regime,
            generated_at=generated_at,
            errors=errors,
        )

    # ── SUE Computation ───────────────────────────────────────────────

    @staticmethod
    def compute_sue(
        actual_eps: float,
        consensus_eps: float,
        historical_surprises: list[float],
    ) -> float:
        """Compute Standardized Unexpected Earnings (SUE).

        When >= 4 quarters of historical surprise data:
            SUE = raw_surprise / stdev(historical_surprises)
        Fallback (< 4 quarters):
            SUE = raw_surprise / (|consensus| * 0.05 + 0.01)

        Args:
            actual_eps: Actual reported EPS.
            consensus_eps: Analyst consensus EPS estimate.
            historical_surprises: List of past (actual - consensus) values.

        Returns:
            SUE score (positive = beat, negative = miss).
        """
        raw = actual_eps - consensus_eps

        if len(historical_surprises) >= 4:
            std = statistics.stdev(historical_surprises)
            if std > 0:
                return raw / std
            # Zero std means all identical — use fallback
        # Fallback: proxy scaling
        denom = abs(consensus_eps) * 0.05 + 0.01
        return raw / denom

    # ── Earnings Reaction ─────────────────────────────────────────────

    @staticmethod
    def check_earnings_reaction(
        prior_close: float,
        open_price: float,
        close_price: float,
        volume: float,
        avg_volume: float,
        high_52w: float,
    ) -> tuple[float, float, float, bool]:
        """Compute earnings-day reaction metrics.

        Returns:
            (gap_return, day_return, volume_ratio, at_52w_high)
            at_52w_high = prior_close >= 95% of 52w high (PRE-earnings state).
        """
        gap_return = (open_price - prior_close) / prior_close if prior_close else 0.0
        day_return = (close_price - prior_close) / prior_close if prior_close else 0.0
        volume_ratio = volume / avg_volume if avg_volume else 0.0
        at_52w_high = prior_close >= (high_52w * 0.95) if high_52w else False
        return gap_return, day_return, volume_ratio, at_52w_high

    # ── Trading Day Helpers ───────────────────────────────────────────

    @staticmethod
    def count_trading_days(start_date: date, end_date: date) -> int:
        """Count NYSE trading days between two dates (exclusive of start)."""
        if end_date <= start_date:
            return 0
        schedule = _NYSE.schedule(
            start_date=start_date + timedelta(days=1),
            end_date=end_date,
        )
        return len(schedule)

    @staticmethod
    def next_trading_day(ref_date: date, offset: int = 1) -> date:
        """Return the Nth NYSE trading day after ref_date."""
        # Look ahead generously to account for holidays
        end = ref_date + timedelta(days=offset * 3 + 10)
        schedule = _NYSE.schedule(
            start_date=ref_date + timedelta(days=1),
            end_date=end,
        )
        if len(schedule) < offset:
            return ref_date + timedelta(days=offset)
        return date.fromisoformat(schedule.index[offset - 1].strftime("%Y-%m-%d"))

    # ── Internal Filter Pipeline ──────────────────────────────────────

    def _evaluate_single(
        self,
        earning: dict[str, Any],
        regime: str,
        today: date,
        market_caps: dict[str, float],
    ) -> PEADCandidate | None:
        """Evaluate a single earning event. Returns None if filtered out."""
        symbol = earning["symbol"]
        report_date = earning["report_date"]
        if isinstance(report_date, str):
            report_date = date.fromisoformat(report_date)

        # Gate 2: Entry window (1-5 trading days)
        max_delay = self._filters.get("max_entry_delay_trading_days", 5)
        days_since = self.count_trading_days(report_date, today)
        if days_since < 1 or days_since > max_delay:
            return None

        # Compute SUE
        historical = earning.get("historical_surprises", [])
        actual = earning.get("actual_eps", 0.0)
        consensus = earning.get("consensus_eps", 0.0)
        sue = self.compute_sue(actual, consensus, historical)

        # Gate 3: SUE filter
        if sue < self._filters.get("min_sue", 2.0):
            return None

        # Earnings reaction
        gap_return = earning.get("earnings_day_gap", 0.0)
        day_return = earning.get("earnings_day_return", 0.0)
        volume_ratio = earning.get("earnings_day_volume_ratio", 0.0)

        # Gate 4: Gap filter — must be POSITIVE (long-only PEAD)
        if gap_return < self._filters.get("min_earnings_day_gap", 0.02):
            return None

        # Gate 4b: Beat-but-closed-down rejection
        # If stock beat but closed red, market says the beat was low-quality
        if day_return < 0:
            return None

        # Gate 5: Volume filter
        if volume_ratio < self._filters.get("min_volume_ratio", 2.0):
            return None

        # Gate 6: 52w high hard exclude
        at_52w_high = earning.get("at_52w_high", False)
        if at_52w_high and self._filters.get("exclude_at_52w_high", True):
            return None

        # Gate 7: Analyst downgrade hard exclude
        analyst_downgrade = earning.get("analyst_downgrade", False)
        if analyst_downgrade:
            return None

        # Gate 8: Forward PE filter
        forward_pe = earning.get("forward_pe")
        max_pe = self._filters.get("max_forward_pe", 40.0)
        if forward_pe is not None and forward_pe > max_pe:
            return None

        # Gate 9: Revenue beat (optional)
        revenue_beat = earning.get("revenue_beat", False)
        if self._filters.get("require_revenue_beat", False) and not revenue_beat:
            return None

        # ── Score ─────────────────────────────────────────────────────
        score = score_pead_quality(sue, gap_return, volume_ratio, revenue_beat)
        strong_th = self._quality.get("strong", 70)
        moderate_th = self._quality.get("moderate", 45)
        label = classify_quality(score, strong_th, moderate_th)

        # ── Liquidity tier + slippage ─────────────────────────────────
        cap = market_caps.get(symbol, 0.0)
        tier = self._classify_liquidity(cap)
        slippage = self._estimate_slippage(tier)

        # ── Trade params ──────────────────────────────────────────────
        size_factor = self._get_position_size_factor(regime)
        entry_date = self.next_trading_day(report_date, 2)
        entry_price = earning.get("current_price", 0.0)
        gap_held = entry_price >= earning.get("earnings_open", 0.0)

        surprise = EarningsSurprise(
            symbol=symbol,
            report_date=report_date,
            actual_eps=actual,
            consensus_eps=consensus,
            surprise_pct=(actual - consensus) / abs(consensus) * 100 if consensus else 0.0,
            sue_score=sue,
            earnings_day_return=day_return,
            earnings_day_gap=gap_return,
            earnings_day_volume_ratio=volume_ratio,
            revenue_beat=revenue_beat,
            at_52w_high=at_52w_high,
            analyst_downgrade=analyst_downgrade,
            liquidity_tier=tier,
            forward_pe=forward_pe,
        )

        return PEADCandidate(
            symbol=symbol,
            surprise=surprise,
            entry_date=entry_date,
            entry_price=entry_price,
            profit_target_pct=self._trade.get("default_profit_target", 0.06),
            stop_loss_pct=self._trade.get("default_stop_loss", -0.05),
            trailing_stop_atr=self._trade.get("trailing_stop_atr_multiplier", 2.0),
            trailing_activation_pct=self._trade.get("trailing_stop_activation_pct", 0.03),
            max_hold_days=self._trade.get("default_max_hold_trading_days", 25),
            position_size_factor=size_factor,
            quality_score=score,
            quality_label=label,
            regime=regime,
            gap_held=gap_held,
            estimated_slippage_bps=slippage,
        )

    def _classify_liquidity(self, market_cap: float) -> LiquidityTier:
        large_min = self._liquidity.get("large_cap_min_market_cap", 50_000_000_000)
        mid_min = self._liquidity.get("mid_cap_min_market_cap", 2_000_000_000)
        if market_cap >= large_min:
            return LiquidityTier.LARGE_CAP
        if market_cap >= mid_min:
            return LiquidityTier.MID_CAP
        return LiquidityTier.SMALL_CAP

    def _estimate_slippage(self, tier: LiquidityTier) -> int:
        if tier == LiquidityTier.LARGE_CAP:
            return int(self._liquidity.get("large_cap_slippage_bps", 10))
        if tier == LiquidityTier.MID_CAP:
            return int(self._liquidity.get("mid_cap_slippage_bps", 25))
        return int(self._liquidity.get("small_cap_slippage_bps", 50))

    def _get_position_size_factor(self, regime: str) -> float:
        if regime == "R0":
            return float(self._regime.get("r0_position_size_factor", 1.0))
        if regime == "R1":
            return float(self._regime.get("r1_position_size_factor", 0.5))
        return 1.0
