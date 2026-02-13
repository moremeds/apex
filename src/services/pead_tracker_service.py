"""PEAD tracker service — persists candidates and resolves outcomes.

Uses OHLC first-touch logic to determine if profit target, stop loss,
or max hold timeout was hit first.

Cache file: data/cache/pead_tracker.json
"""

from __future__ import annotations

import json
from datetime import date, timedelta
from pathlib import Path
from typing import Any

import pandas_market_calendars as mcal
import yfinance as yf

from src.domain.screeners.pead.models import PEADCandidate
from src.domain.screeners.pead.tracker import TrackedCandidate, TrackerStats
from src.utils.logging_setup import get_logger

logger = get_logger(__name__)

PROJECT_ROOT = Path(__file__).resolve().parent.parent.parent
DEFAULT_TRACKER_PATH = PROJECT_ROOT / "data" / "cache" / "pead_tracker.json"

_NYSE = mcal.get_calendar("NYSE")


class PEADTrackerService:
    """Manages PEAD candidate tracking and outcome resolution."""

    def __init__(self, tracker_path: Path | None = None) -> None:
        self._path = tracker_path or DEFAULT_TRACKER_PATH
        self._candidates: list[TrackedCandidate] | None = None

    # ── Persistence ────────────────────────────────────────────────────

    def _load(self) -> list[TrackedCandidate]:
        if self._candidates is not None:
            return self._candidates

        if not self._path.exists():
            self._candidates = []
            return self._candidates

        try:
            data = json.loads(self._path.read_text())
            self._candidates = [TrackedCandidate.from_dict(d) for d in data.get("candidates", [])]
            logger.info(f"Loaded {len(self._candidates)} tracked candidates")
        except (json.JSONDecodeError, KeyError) as e:
            logger.error(f"Failed to load tracker: {e}")
            self._candidates = []

        return self._candidates

    def _save(self) -> None:
        candidates = self._load()
        self._path.parent.mkdir(parents=True, exist_ok=True)
        payload = {
            "version": "1.0",
            "candidates": [c.to_dict() for c in candidates],
        }
        self._path.write_text(json.dumps(payload, indent=2))
        logger.info(f"Saved {len(candidates)} tracked candidates")

    # ── Public API ─────────────────────────────────────────────────────

    def add_candidates(self, candidates: list[PEADCandidate]) -> int:
        """Add new candidates for tracking. Dedup by (symbol, entry_date).

        Returns count of newly added candidates.
        """
        existing = self._load()
        existing_keys = {(c.symbol, c.entry_date) for c in existing}

        added = 0
        for c in candidates:
            key = (c.symbol, c.entry_date)
            if key in existing_keys:
                continue

            tracked = TrackedCandidate(
                symbol=c.symbol,
                entry_date=c.entry_date,
                entry_price=c.entry_price,
                profit_target_pct=c.profit_target_pct,
                stop_loss_pct=c.stop_loss_pct,
                max_hold_days=c.max_hold_days,
                quality_score=c.quality_score,
                quality_label=c.quality_label,
                sue_score=c.surprise.sue_score,
                multi_quarter_sue=c.surprise.multi_quarter_sue,
                regime=c.regime,
            )
            existing.append(tracked)
            existing_keys.add(key)
            added += 1

        if added > 0:
            self._save()
            logger.info(f"Added {added} new candidates to tracker")

        return added

    def update_outcomes(self) -> int:
        """Resolve open candidates via OHLC first-touch. Returns count resolved."""
        candidates = self._load()
        open_candidates = [c for c in candidates if c.status == "open"]

        if not open_candidates:
            logger.info("No open candidates to resolve")
            return 0

        resolved = 0
        for candidate in open_candidates:
            result = self._resolve_candidate(candidate)
            if result:
                resolved += 1

        if resolved > 0:
            self._save()

        logger.info(f"Resolved {resolved}/{len(open_candidates)} open candidates")
        return resolved

    def get_stats(self) -> TrackerStats:
        """Compute aggregate performance statistics."""
        candidates = self._load()

        if not candidates:
            return TrackerStats()

        resolved = [c for c in candidates if c.status != "open"]
        won = [c for c in resolved if c.status == "won"]
        lost = [c for c in resolved if c.status == "lost"]
        timeout = [c for c in resolved if c.status == "timeout"]
        open_count = len(candidates) - len(resolved)

        win_rate = len(won) / len(resolved) if resolved else None
        avg_pnl = (
            sum(c.pnl_pct for c in resolved if c.pnl_pct is not None) / len(resolved)
            if resolved
            else None
        )

        # Average hold days
        hold_days_list = []
        for c in resolved:
            if c.exit_date and c.entry_date:
                hold_days_list.append((c.exit_date - c.entry_date).days)
        avg_hold = sum(hold_days_list) / len(hold_days_list) if hold_days_list else None

        # By quality tier
        by_quality: dict[str, dict[str, Any]] = {}
        for label in ["STRONG", "MODERATE", "MARGINAL"]:
            tier_resolved = [c for c in resolved if c.quality_label == label]
            tier_won = [c for c in tier_resolved if c.status == "won"]
            if tier_resolved:
                tier_pnls = [c.pnl_pct for c in tier_resolved if c.pnl_pct is not None]
                by_quality[label] = {
                    "total": len(tier_resolved),
                    "win_rate": round(len(tier_won) / len(tier_resolved), 4),
                    "avg_pnl_pct": (
                        round(sum(tier_pnls) / len(tier_pnls), 4) if tier_pnls else None
                    ),
                }

        return TrackerStats(
            total=len(candidates),
            open=open_count,
            won=len(won),
            lost=len(lost),
            timeout=len(timeout),
            win_rate=win_rate,
            avg_pnl_pct=avg_pnl,
            avg_hold_days=avg_hold,
            by_quality=by_quality,
        )

    def get_all_candidates(self) -> list[TrackedCandidate]:
        """Return all tracked candidates."""
        return list(self._load())

    # ── OHLC First-Touch Resolution ────────────────────────────────────

    def _resolve_candidate(self, candidate: TrackedCandidate) -> bool:
        """Resolve a single open candidate using daily OHLC first-touch logic.

        For each trading day from entry_date to today:
            1. Check if day's LOW <= stop_loss price → LOST
            2. Check if day's HIGH >= profit_target price → WON
            3. If both triggered on same bar: stop wins (conservative)
            4. If trading_days >= max_hold_days → TIMEOUT (use closing price)

        Returns True if candidate was resolved, False if data unavailable.
        """
        target_price = candidate.target_price
        stop_price = candidate.stop_price
        today = date.today()

        # Fetch OHLC data from entry date to today
        start = candidate.entry_date
        end = today + timedelta(days=1)

        try:
            ticker = yf.Ticker(candidate.symbol)
            hist = ticker.history(start=start.isoformat(), end=end.isoformat())
        except Exception as e:
            logger.warning(f"yfinance error for {candidate.symbol}: {e}")
            return False

        if hist.empty:
            return False

        # Count trading days for max hold check
        trading_day_count = 0

        for _, row in hist.iterrows():
            trading_day_count += 1
            day_date = row.name.date() if hasattr(row.name, "date") else row.name
            day_low = float(row["Low"])
            day_high = float(row["High"])
            day_close = float(row["Close"])

            stop_hit = day_low <= stop_price
            target_hit = day_high >= target_price

            # Same-bar: stop wins (conservative)
            if stop_hit and target_hit:
                candidate.exit_date = day_date
                candidate.exit_price = stop_price
                candidate.exit_reason = "stop_loss"
                candidate.pnl_pct = candidate.stop_loss_pct
                candidate.status = "lost"
                return True

            if stop_hit:
                candidate.exit_date = day_date
                candidate.exit_price = stop_price
                candidate.exit_reason = "stop_loss"
                candidate.pnl_pct = candidate.stop_loss_pct
                candidate.status = "lost"
                return True

            if target_hit:
                candidate.exit_date = day_date
                candidate.exit_price = target_price
                candidate.exit_reason = "profit_target"
                candidate.pnl_pct = candidate.profit_target_pct
                candidate.status = "won"
                return True

            # Max hold timeout
            if trading_day_count >= candidate.max_hold_days:
                pnl = (day_close - candidate.entry_price) / candidate.entry_price
                candidate.exit_date = day_date
                candidate.exit_price = day_close
                candidate.exit_reason = "timeout"
                candidate.pnl_pct = round(pnl, 6)
                candidate.status = "timeout"
                return True

        # Not enough days elapsed yet — candidate stays open
        return False
