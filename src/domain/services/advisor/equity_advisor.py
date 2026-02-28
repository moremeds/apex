"""Equity advisor — synthesizes signals into BUY/HOLD/SELL recommendations.

Signal count is factored into confidence calculation.
A single signal produces at most BUY/SELL; STRONG requires 3+ aligned signals.
"""

from __future__ import annotations

import math

from src.domain.services.advisor.models import EquityAdvice

# Minimum signal count for STRONG recommendations
_MIN_SIGNALS_FOR_STRONG = 3


class EquityAdvisor:
    """Synthesizes active signals into per-symbol equity recommendations."""

    def synthesize(
        self,
        symbol: str,
        sector: str,
        active_signals: list[dict],
        regime: str,
        indicator_state: dict,
    ) -> EquityAdvice:
        if not active_signals:
            # Extract key levels even when no signals are active
            tp = self._extract_trend_pulse(indicator_state)
            kl = self._extract_key_levels(indicator_state)
            reasoning = ["No active signals — awaiting market data"]
            if regime in ("R2", "R3"):
                reasoning.append(f"Regime {regime}: elevated caution")
            return EquityAdvice(
                symbol=symbol,
                sector=sector,
                action="HOLD",
                confidence=0,
                regime=regime,
                signal_summary={"bullish": 0, "bearish": 0, "neutral": 0},
                top_signals=[],
                trend_pulse=tp,
                key_levels=kl,
                reasoning=reasoning,
            )

        # Score each signal
        score = 0.0
        total_weight = 0.0
        for sig in active_signals:
            direction = sig.get("direction", "neutral")
            strength = sig.get("strength", 50)
            dir_weight = {"bullish": 1.0, "bearish": -1.0, "neutral": 0.0}.get(direction, 0.0)

            # TrendPulse signals get 1.5x weight
            rule = sig.get("rule", "")
            multiplier = 1.5 if rule.startswith("trend_pulse") else 1.0

            weighted = strength * dir_weight * multiplier
            score += weighted
            total_weight += abs(strength * multiplier)

        # Normalize to -100..+100
        normalized = (score / total_weight * 100) if total_weight > 0 else 0

        # Dampen score when signal count is low
        # sqrt(n)/sqrt(3) — reaches 1.0 at 3 signals, 0.58 at 1 signal
        signal_count = len([s for s in active_signals if s.get("direction") != "neutral"])
        count_factor = min(
            math.sqrt(max(signal_count, 1)) / math.sqrt(_MIN_SIGNALS_FOR_STRONG), 1.0
        )
        normalized *= count_factor

        # Regime adjustment
        regime_mult = {"R0": 1.0, "R1": 0.7, "R2": 0.0, "R3": 0.3}.get(regime, 0.5)

        if normalized > 0:
            adjusted = normalized * regime_mult
        else:
            # For sell signals, R2 amplifies
            sell_mult = {"R0": 0.7, "R1": 1.0, "R2": 1.5, "R3": 1.0}.get(regime, 1.0)
            adjusted = normalized * sell_mult

        # Map to action
        if adjusted > 60:
            action = "STRONG_BUY"
        elif adjusted > 25:
            action = "BUY"
        elif adjusted > -25:
            action = "HOLD"
        elif adjusted > -60:
            action = "SELL"
        else:
            action = "STRONG_SELL"

        confidence = min(abs(adjusted), 100)

        # Signal summary
        summary: dict[str, int] = {"bullish": 0, "bearish": 0, "neutral": 0}
        for sig in active_signals:
            d = sig.get("direction", "neutral")
            if d in summary:
                summary[d] += 1

        # Top signals
        top = sorted(active_signals, key=lambda s: s.get("strength", 0), reverse=True)[:3]

        # Extract TrendPulse state if present
        trend_pulse = self._extract_trend_pulse(indicator_state)

        # Key levels
        key_levels = self._extract_key_levels(indicator_state)

        # Reasoning
        reasoning = self._build_reasoning(action, summary, regime, confidence)

        return EquityAdvice(
            symbol=symbol,
            sector=sector,
            action=action,
            confidence=round(confidence, 1),
            regime=regime,
            signal_summary=summary,
            top_signals=top,
            trend_pulse=trend_pulse,
            key_levels=key_levels,
            reasoning=reasoning,
        )

    def _extract_trend_pulse(self, state: dict) -> dict | None:
        tp = state.get("trend_pulse")
        if not tp:
            return None
        return {
            "zig_state": tp.get("zig_state"),
            "macd_state": tp.get("macd_state"),
            "atr_trail": tp.get("atr_trail"),
            "confidence_score": tp.get("confidence"),
        }

    def _extract_key_levels(self, state: dict) -> dict:
        levels: dict[str, float] = {}
        sr = state.get("support_resistance")
        if sr:
            if "support" in sr:
                levels["support"] = sr["support"]
            if "resistance" in sr:
                levels["resistance"] = sr["resistance"]
        tp = state.get("trend_pulse")
        if tp and "atr_trail" in tp:
            levels["atr_stop"] = tp["atr_trail"]
        return levels

    def _build_reasoning(
        self, action: str, summary: dict, regime: str, confidence: float
    ) -> list[str]:
        reasons = []
        bull, bear = summary.get("bullish", 0), summary.get("bearish", 0)
        if bull > bear:
            reasons.append(f"{bull} bullish vs {bear} bearish signals")
        elif bear > bull:
            reasons.append(f"{bear} bearish vs {bull} bullish signals")
        else:
            reasons.append("Mixed signals — equal bullish and bearish")

        regime_names = {
            "R0": "Healthy Uptrend — full sizing",
            "R1": "Choppy — reduced sizing",
            "R2": "Risk-Off — no new longs",
            "R3": "Rebound — small defined-risk only",
        }
        reasons.append(f"Regime {regime}: {regime_names.get(regime, 'Unknown')}")

        return reasons
