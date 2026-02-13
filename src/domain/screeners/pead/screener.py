"""PEAD screener core logic.

Pure domain service that filters and scores earnings surprises into PEAD candidates.
Uses pandas_market_calendars for trading-day arithmetic (NYSE schedule).
"""

from __future__ import annotations

from datetime import date, datetime, timedelta
from typing import Any

import pandas_market_calendars as mcal

from src.utils.logging_setup import get_logger

from .config import PEADConfig
from .models import (
    EarningsSurprise,
    LiquidityTier,
    PEADCandidate,
    PEADScreenResult,
)
from .scorer import classify_quality, score_pead_quality
from .sue import compute_sue

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

    def __init__(self, config: PEADConfig | dict[str, Any]) -> None:
        if isinstance(config, dict):
            self._config = PEADConfig.from_dict(config)
        else:
            self._config = config

    # ── Public API ────────────────────────────────────────────────────

    def generate_pead_candidates(
        self,
        recent_earnings: list[dict[str, Any]],
        regime: str,
        today: date,
        market_caps: dict[str, float],
        attention_data: dict[str, str | None] | None = None,
    ) -> PEADScreenResult:
        """Screen recent earnings for PEAD candidates.

        Args:
            recent_earnings: List of dicts with keys matching EarningsSurprise fields
                plus ``historical_surprises``, ``current_price``, ``earnings_open``.
            regime: Current market regime ("R0", "R1", "R2", "R3").
            today: Current date for entry window calculation.
            market_caps: symbol -> market cap in USD.
            attention_data: symbol -> attention level ("low"/"medium"/"high"/None).
                Optional. When provided and config.attention_filter.enabled,
                the attention modifier is applied to quality scores.

        Returns:
            PEADScreenResult with scored, sorted candidates.
        """
        generated_at = datetime.now()
        cfg = self._config

        # Gate 1: R2 block
        if regime == "R2" and cfg.regime_rules.r2_block_entirely:
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

        attn = attention_data or {}
        for earning in recent_earnings:
            symbol = earning.get("symbol", "?")
            try:
                candidate = self._evaluate_single(
                    earning, regime, today, market_caps, attn.get(symbol)
                )
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
        attention_level: str | None = None,
    ) -> PEADCandidate | None:
        """Evaluate a single earning event. Returns None if filtered out."""
        cfg = self._config
        symbol = earning["symbol"]
        report_date = earning["report_date"]
        if isinstance(report_date, str):
            report_date = date.fromisoformat(report_date)

        # Gate 2: Entry window (1-5 trading days)
        max_delay = cfg.filters.max_entry_delay_trading_days
        days_since = self.count_trading_days(report_date, today)
        if days_since < 1 or days_since > max_delay:
            return None

        # Compute SUE
        historical = earning.get("historical_surprises", [])
        actual = earning.get("actual_eps", 0.0)
        consensus = earning.get("consensus_eps", 0.0)
        sue = compute_sue(actual, consensus, historical)

        # Gate 3: SUE filter
        if sue < cfg.filters.min_sue:
            return None

        # Earnings reaction
        gap_return = earning.get("earnings_day_gap", 0.0)
        day_return = earning.get("earnings_day_return", 0.0)
        volume_ratio = earning.get("earnings_day_volume_ratio", 0.0)

        # Gate 4: Gap filter — must be POSITIVE (long-only PEAD)
        if gap_return < cfg.filters.min_earnings_day_gap:
            return None

        # Gate 4b: Beat-but-closed-down rejection
        # If stock beat but closed red, market says the beat was low-quality
        if day_return < 0:
            return None

        # Gate 5: Volume filter
        if volume_ratio < cfg.filters.min_volume_ratio:
            return None

        # Gate 6: 52w high hard exclude
        at_52w_high = earning.get("at_52w_high", False)
        if at_52w_high and cfg.filters.exclude_at_52w_high:
            return None

        # Gate 7: Analyst downgrade hard exclude
        analyst_downgrade = earning.get("analyst_downgrade", False)
        if analyst_downgrade:
            return None

        # Gate 8: Forward PE filter
        forward_pe = earning.get("forward_pe")
        if forward_pe is not None and forward_pe > cfg.filters.max_forward_pe:
            return None

        # Gate 9: Revenue beat (optional)
        revenue_beat = earning.get("revenue_beat", False)
        if cfg.filters.require_revenue_beat and not revenue_beat:
            return None

        # ── Score ─────────────────────────────────────────────────────
        score = score_pead_quality(sue, gap_return, volume_ratio, revenue_beat)

        # Multi-quarter SUE modifier
        multi_q_sue: float | None = None
        if cfg.multi_quarter_sue.enabled:
            from .sue import compute_multi_quarter_sue

            multi_q_sue = compute_multi_quarter_sue(
                historical,
                decay_lambda=cfg.multi_quarter_sue.decay_lambda,
                min_quarters=cfg.multi_quarter_sue.min_quarters,
            )
            if multi_q_sue is not None:
                from .scorer import apply_multi_quarter_modifier

                score = apply_multi_quarter_modifier(
                    score,
                    multi_q_sue,
                    max_bonus=cfg.multi_quarter_sue.max_bonus,
                    max_penalty=cfg.multi_quarter_sue.max_penalty,
                )

        # Attention modifier
        if cfg.attention_filter.enabled and attention_level is not None:
            from .scorer import apply_attention_modifier

            score = apply_attention_modifier(
                score,
                attention_level,
                low_bonus=cfg.attention_filter.low_bonus,
                high_penalty=cfg.attention_filter.high_penalty,
            )

        label = classify_quality(
            score, cfg.quality_thresholds.strong, cfg.quality_thresholds.moderate
        )

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
            multi_quarter_sue=multi_q_sue,
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
            profit_target_pct=cfg.trade_params.default_profit_target,
            stop_loss_pct=cfg.trade_params.default_stop_loss,
            trailing_stop_atr=cfg.trade_params.trailing_stop_atr_multiplier,
            trailing_activation_pct=cfg.trade_params.trailing_stop_activation_pct,
            max_hold_days=cfg.trade_params.default_max_hold_trading_days,
            position_size_factor=size_factor,
            quality_score=score,
            quality_label=label,
            regime=regime,
            gap_held=gap_held,
            estimated_slippage_bps=slippage,
        )

    def _classify_liquidity(self, market_cap: float) -> LiquidityTier:
        cfg = self._config.liquidity_tiers
        if market_cap >= cfg.large_cap_min_market_cap:
            return LiquidityTier.LARGE_CAP
        if market_cap >= cfg.mid_cap_min_market_cap:
            return LiquidityTier.MID_CAP
        return LiquidityTier.SMALL_CAP

    def _estimate_slippage(self, tier: LiquidityTier) -> int:
        cfg = self._config.liquidity_tiers
        if tier == LiquidityTier.LARGE_CAP:
            return cfg.large_cap_slippage_bps
        if tier == LiquidityTier.MID_CAP:
            return cfg.mid_cap_slippage_bps
        return cfg.small_cap_slippage_bps

    def _get_position_size_factor(self, regime: str) -> float:
        cfg = self._config.regime_rules
        if regime == "R0":
            return cfg.r0_position_size_factor
        if regime == "R1":
            return cfg.r1_position_size_factor
        return 1.0
