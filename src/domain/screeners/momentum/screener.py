"""Momentum screener filter pipeline and orchestration.

Filters universe down to top-N momentum candidates with regime-dependent
position sizing. Pure domain logic — no I/O.

Filter pipeline (cheapest first):
    1. Regime gate (R2 blocks entirely)
    2. Market cap >= min
    3. Average daily dollar volume >= min
    4. Turnover rate >= min (live-only, skipped in backtest)
    5. Price >= min
    6. Sufficient price history
    7. Compute momentum 12-1 + FIP
    8. Cross-sectional percentile ranking
    9. Composite rank >= regime threshold
   10. Top-N selection + quality classification
"""

from __future__ import annotations

from datetime import datetime
from typing import Any

import numpy as np

from src.domain.screeners.pead.models import LiquidityTier
from src.utils.logging_setup import get_logger

from .compute import compute_adaptive_momentum, compute_fip, compute_momentum_12_1
from .config import MomentumConfig
from .models import MomentumCandidate, MomentumScreenResult, MomentumSignal
from .scorer import classify_quality, compute_composite_rank, compute_percentile_ranks

logger = get_logger(__name__)


class MomentumScreener:
    """Quantitative momentum screener.

    Computes 12-1 momentum + FIP for a universe, filters by liquidity and
    regime rules, and ranks cross-sectionally to produce a top-N watchlist.
    """

    def __init__(self, config: MomentumConfig | dict[str, Any]) -> None:
        if isinstance(config, dict):
            self._config = MomentumConfig.from_dict(config)
        else:
            self._config = config

    def screen(
        self,
        price_data: dict[str, np.ndarray],
        volume_data: dict[str, np.ndarray],
        regime: str,
        market_caps: dict[str, float],
        *,
        is_backtest: bool = False,
        use_adaptive: bool = False,
    ) -> MomentumScreenResult:
        """Run the full screening pipeline.

        Args:
            price_data: Symbol -> array of daily closes (chronological).
            volume_data: Symbol -> array of daily volumes (chronological).
            regime: Current market regime ("R0", "R1", "R2", "R3").
            market_caps: Symbol -> market cap in USD.
            is_backtest: If True, skip turnover filter (avoids look-ahead bias).
            use_adaptive: If True, use adaptive momentum for stocks with < 12mo history.

        Returns:
            MomentumScreenResult with ranked candidates.
        """
        cfg = self._config
        errors: dict[str, str] = {}
        universe_size = len(price_data)

        # 1. Regime gate
        if regime == "R2" and cfg.regime_rules.r2_block_entirely:
            logger.info("R2 regime: blocking all momentum entries")
            return MomentumScreenResult(
                candidates=[],
                universe_size=universe_size,
                passed_filters=0,
                regime=regime,
                generated_at=datetime.now(),
                errors=errors,
            )

        # Determine regime-specific thresholds
        min_percentile, size_factor = self._get_regime_params(regime)

        lookback = cfg.data_source.lookback_trading_days
        skip = cfg.data_source.skip_recent_trading_days
        required_bars = lookback + skip

        # 2-6. Filter pipeline
        passed_symbols: list[str] = []
        for symbol in price_data:
            closes = price_data[symbol]
            volumes = volume_data.get(symbol, np.array([]))

            # Market cap filter
            cap = market_caps.get(symbol, 0.0)
            if cap < cfg.filters.min_market_cap:
                continue

            # Avg daily dollar volume filter
            if len(closes) > 0 and len(volumes) > 0:
                min_len = min(len(closes), len(volumes))
                recent_closes = closes[-min(20, min_len) :]
                recent_volumes = volumes[-min(20, min_len) :]
                avg_dollar_vol = float(np.mean(recent_closes * recent_volumes))
                if avg_dollar_vol < cfg.filters.min_avg_daily_dollar_volume:
                    continue
            else:
                continue

            # Turnover filter (live-only)
            if not is_backtest and cap > 0:
                avg_vol = float(np.mean(recent_volumes))
                # shares_outstanding ~ market_cap / last_close
                last_close = closes[-1] if len(closes) > 0 else 0
                if last_close > 0:
                    approx_shares = cap / last_close
                    turnover = avg_vol / approx_shares if approx_shares > 0 else 0
                    if turnover < cfg.filters.min_daily_turnover_rate:
                        continue

            # Price filter
            if len(closes) == 0 or closes[-1] < cfg.filters.min_price:
                continue

            # History length filter (relaxed for adaptive mode)
            if use_adaptive:
                if len(closes) < 126 + skip:
                    continue
            else:
                if len(closes) < required_bars:
                    continue

            passed_symbols.append(symbol)

        if not passed_symbols:
            return MomentumScreenResult(
                candidates=[],
                universe_size=universe_size,
                passed_filters=0,
                regime=regime,
                generated_at=datetime.now(),
                errors=errors,
            )

        # 7. Compute momentum + FIP
        signals: list[MomentumSignal] = []
        mom_values: list[float] = []
        fip_values: list[float] = []

        for symbol in passed_symbols:
            closes = price_data[symbol]
            volumes = volume_data.get(symbol, np.array([]))

            # Compute momentum
            if use_adaptive and len(closes) < required_bars:
                result = compute_adaptive_momentum(closes, skip=skip, target=lookback)
                if result is None:
                    errors[symbol] = "adaptive momentum computation failed"
                    continue
                mom, actual_lookback = result
            else:
                mom_result = compute_momentum_12_1(closes, skip=skip, lookback=lookback)
                if mom_result is None:
                    errors[symbol] = "momentum computation failed"
                    continue
                mom = mom_result
                actual_lookback = lookback

            # Compute FIP
            fip = compute_fip(closes, skip=skip, lookback=min(actual_lookback, lookback))
            if fip is None:
                errors[symbol] = "FIP computation failed"
                continue

            # Compute supporting data
            cap = market_caps.get(symbol, 0.0)
            min_len = min(len(closes), len(volumes))
            recent_closes = closes[-min(20, min_len) :]
            recent_volumes = volumes[-min(20, min_len) :]
            avg_dollar_vol = float(np.mean(recent_closes * recent_volumes))
            tier = self._classify_tier(cap)

            signals.append(
                MomentumSignal(
                    symbol=symbol,
                    momentum_12_1=mom,
                    fip=fip,
                    momentum_percentile=0.0,  # filled below
                    fip_percentile=0.0,
                    composite_rank=0.0,
                    last_close=float(closes[-1]),
                    market_cap=cap,
                    avg_daily_dollar_volume=avg_dollar_vol,
                    liquidity_tier=tier,
                    estimated_slippage_bps=self._estimate_slippage(tier),
                    lookback_days=actual_lookback,
                )
            )
            mom_values.append(mom)
            fip_values.append(fip)

        if not signals:
            return MomentumScreenResult(
                candidates=[],
                universe_size=universe_size,
                passed_filters=len(passed_symbols),
                regime=regime,
                generated_at=datetime.now(),
                errors=errors,
            )

        # 8. Cross-sectional percentile ranking
        mom_pcts = compute_percentile_ranks(mom_values)
        fip_pcts = compute_percentile_ranks(fip_values)

        for i, sig in enumerate(signals):
            sig.momentum_percentile = mom_pcts[i]
            sig.fip_percentile = fip_pcts[i]
            sig.composite_rank = compute_composite_rank(
                mom_pcts[i],
                fip_pcts[i],
                cfg.scoring.momentum_weight,
                cfg.scoring.fip_weight,
            )

        # 9. Regime threshold
        qualified = [s for s in signals if s.composite_rank >= min_percentile]

        # 10. Sort descending by composite, take top-N
        qualified.sort(key=lambda s: s.composite_rank, reverse=True)
        top_n = qualified[: cfg.scoring.top_n]

        candidates = []
        for rank_idx, sig in enumerate(top_n, start=1):
            quality = classify_quality(
                sig.composite_rank,
                cfg.quality_thresholds.strong,
                cfg.quality_thresholds.moderate,
            )
            candidates.append(
                MomentumCandidate(
                    signal=sig,
                    rank=rank_idx,
                    quality_label=quality,
                    position_size_factor=size_factor,
                    regime=regime,
                )
            )

        logger.info(
            f"Momentum screen: {universe_size} universe → "
            f"{len(passed_symbols)} passed filters → "
            f"{len(qualified)} above threshold → "
            f"{len(candidates)} top-N ({regime})"
        )

        return MomentumScreenResult(
            candidates=candidates,
            universe_size=universe_size,
            passed_filters=len(passed_symbols),
            regime=regime,
            generated_at=datetime.now(),
            errors=errors,
        )

    def _get_regime_params(self, regime: str) -> tuple[float, float]:
        """Return (min_composite_percentile, position_size_factor) for regime."""
        rules = self._config.regime_rules
        if regime == "R0":
            return (rules.r0_min_composite_percentile, rules.r0_position_size_factor)
        if regime == "R1":
            return (rules.r1_min_composite_percentile, rules.r1_position_size_factor)
        # R3 or unknown: use R1 settings (conservative)
        return (rules.r1_min_composite_percentile, rules.r1_position_size_factor)

    def _classify_tier(self, market_cap: float) -> LiquidityTier:
        """Classify stock into liquidity tier by market cap."""
        tiers = self._config.liquidity_tiers
        if market_cap >= tiers.large_cap_min:
            return LiquidityTier.LARGE_CAP
        if market_cap >= tiers.mid_cap_min:
            return LiquidityTier.MID_CAP
        return LiquidityTier.SMALL_CAP

    def _estimate_slippage(self, tier: LiquidityTier) -> int:
        """Estimate slippage in basis points by tier."""
        tiers = self._config.liquidity_tiers
        if tier == LiquidityTier.LARGE_CAP:
            return tiers.large_cap_slippage_bps
        if tier == LiquidityTier.MID_CAP:
            return tiers.mid_cap_slippage_bps
        return tiers.small_cap_slippage_bps
