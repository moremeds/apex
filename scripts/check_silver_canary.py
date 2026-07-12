#!/usr/bin/env python3
"""Read-only raw-versus-adjusted Livewire canary for Apex."""

from __future__ import annotations

import argparse
import asyncio
import json
import sys
from datetime import datetime, timezone
from pathlib import Path
from typing import Any

REPO_ROOT = Path(__file__).resolve().parents[1]
if str(REPO_ROOT) not in sys.path:
    sys.path.insert(0, str(REPO_ROOT))

from src.domain.events.domain_events import BarData
from src.infrastructure.adapters.livewire.ohlc_provider import LivewireOhlcProvider
from src.infrastructure.adapters.livewire.revisions import RevisionManifestReader


def _same_timestamps(raw: list[BarData], adjusted: list[BarData]) -> bool:
    return [bar.bar_start for bar in raw] == [bar.bar_start for bar in adjusted]


def _same_values(raw: list[BarData], adjusted: list[BarData]) -> bool:
    return len(raw) == len(adjusted) and all(
        (left.open, left.high, left.low, left.close, left.volume)
        == (right.open, right.high, right.low, right.close, right.volume)
        for left, right in zip(raw, adjusted, strict=True)
    )


def _return(bars: list[BarData]) -> float | None:
    if len(bars) < 2 or not bars[0].close:
        return None
    last = bars[-1].close
    if last is None:
        return None
    return float(last / bars[0].close - 1)


def _continuity_metrics(raw: list[BarData], adjusted: list[BarData]) -> dict[str, int]:
    """Count factor-change boundaries and those with a smaller local discontinuity."""
    if len(raw) != len(adjusted):
        return {"action_boundaries": 0, "improved_boundaries": 0}
    ratios: list[float | None] = []
    for raw_bar, adjusted_bar in zip(raw, adjusted, strict=True):
        if not raw_bar.close or adjusted_bar.close is None:
            ratios.append(None)
        else:
            ratios.append(float(adjusted_bar.close / raw_bar.close))

    action_boundaries = improved_boundaries = 0
    for index in range(1, len(raw)):
        previous_ratio, current_ratio = ratios[index - 1], ratios[index]
        if (
            previous_ratio is None
            or current_ratio is None
            or abs(previous_ratio - current_ratio) <= 1e-12
        ):
            continue
        previous_raw = raw[index - 1].close
        current_raw = raw[index].close
        previous_adjusted = adjusted[index - 1].close
        current_adjusted = adjusted[index].close
        if (
            not previous_raw
            or current_raw is None
            or not previous_adjusted
            or current_adjusted is None
        ):
            continue
        action_boundaries += 1
        raw_return = float(current_raw / previous_raw - 1)
        adjusted_return = float(current_adjusted / previous_adjusted - 1)
        if abs(adjusted_return) <= abs(raw_return) + 1e-12:
            improved_boundaries += 1
    return {
        "action_boundaries": action_boundaries,
        "improved_boundaries": improved_boundaries,
    }


async def check_silver_canary(
    *,
    bronze_root: Path,
    silver_root: Path,
    symbols: tuple[str, ...] = ("NVDA", "AAPL", "SPY"),
    control: str = "PLTR",
    start: datetime = datetime(2020, 1, 1, tzinfo=timezone.utc),
    end: datetime | None = None,
) -> dict[str, Any]:
    """Compare raw and adjusted daily/intraday reads without changing artifacts."""
    end = end or datetime.now(timezone.utc)
    revision = await asyncio.to_thread(RevisionManifestReader(silver_root).read_current)
    raw_provider = LivewireOhlcProvider(bronze_root, price_mode="raw")
    adjusted_provider = LivewireOhlcProvider(bronze_root, silver_root, "adjusted")
    results: dict[str, dict[str, Any]] = {}

    for symbol in (*symbols, control):
        raw_daily, adjusted_daily, raw_intraday, adjusted_intraday = await asyncio.gather(
            raw_provider.fetch_bars(symbol, "1d", start, end),
            adjusted_provider.fetch_bars(symbol, "1d", start, end),
            raw_provider.fetch_bars(symbol, "1m", start, end),
            adjusted_provider.fetch_bars(symbol, "1m", start, end),
        )
        counts_match = (
            len(raw_daily) == len(adjusted_daily) > 0
            and len(raw_intraday) == len(adjusted_intraday) > 0
        )
        timestamps_match = _same_timestamps(raw_daily, adjusted_daily) and _same_timestamps(
            raw_intraday, adjusted_intraday
        )
        identity = _same_values(raw_daily, adjusted_daily) and _same_values(
            raw_intraday, adjusted_intraday
        )
        volume_unchanged = counts_match and all(
            left.volume == right.volume
            for raw_rows, adjusted_rows in (
                (raw_daily, adjusted_daily),
                (raw_intraday, adjusted_intraday),
            )
            for left, right in zip(raw_rows, adjusted_rows, strict=True)
        )
        raw_return = _return(raw_daily)
        adjusted_return = _return(adjusted_daily)
        continuity = _continuity_metrics(raw_daily, adjusted_daily)
        continuity_improved = continuity["improved_boundaries"] > 0
        is_control = symbol == control
        symbol_passed = (
            counts_match
            and timestamps_match
            and (
                identity
                if is_control
                else (
                    not identity
                    and continuity["action_boundaries"] > 0
                    and (volume_unchanged or continuity_improved)
                )
            )
        )
        results[symbol] = {
            "passed": symbol_passed,
            "daily_count": len(raw_daily),
            "intraday_count": len(raw_intraday),
            "timestamps_match": timestamps_match,
            "raw_return": raw_return,
            "adjusted_return": adjusted_return,
            "continuity_improved": continuity_improved,
            **continuity,
            "volume_unchanged": volume_unchanged,
            "split_volume_adjusted": not volume_unchanged,
            "identity_control": identity if is_control else False,
        }

    return {
        "passed": all(item["passed"] for item in results.values()),
        "revision": revision.revision,
        "generation_id": revision.generation_id,
        "symbols": results,
    }


def _parse_timestamp(value: str) -> datetime:
    parsed = datetime.fromisoformat(value.replace("Z", "+00:00"))
    if parsed.tzinfo is None:
        parsed = parsed.replace(tzinfo=timezone.utc)
    return parsed.astimezone(timezone.utc)


def main() -> int:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--bronze-root", required=True, type=Path)
    parser.add_argument("--silver-root", required=True, type=Path)
    parser.add_argument("--tickers", nargs="+", default=["NVDA", "AAPL", "SPY"])
    parser.add_argument("--control", default="PLTR")
    parser.add_argument("--start", type=_parse_timestamp, default="2020-01-01T00:00:00Z")
    parser.add_argument("--end", type=_parse_timestamp)
    args = parser.parse_args()
    result = asyncio.run(
        check_silver_canary(
            bronze_root=args.bronze_root,
            silver_root=args.silver_root,
            symbols=tuple(args.tickers),
            control=args.control,
            start=args.start,
            end=args.end,
        )
    )
    print(json.dumps(result, indent=2, sort_keys=True))
    return 0 if result["passed"] else 1


if __name__ == "__main__":
    raise SystemExit(main())
