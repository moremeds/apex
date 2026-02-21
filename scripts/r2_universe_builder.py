#!/usr/bin/env python3
"""R2 Universe Builder — Job 0 of the R2 data pipeline.

Builds universe metadata from FMP and uploads to R2:
  - meta/universe.json   (~3000 tickers with metadata)
  - meta/sp500.json      (S&P 500 symbol list)
  - meta/nq100.json      (NQ100 proxy symbol list)
  - meta/last_updated.json

Usage:
    python scripts/r2_universe_builder.py
    python scripts/r2_universe_builder.py --dry-run   # Show counts without uploading
"""

from __future__ import annotations

import argparse
import json
import logging
import sys
from datetime import datetime, timezone
from pathlib import Path
from typing import Any

# Add project root to path
PROJECT_ROOT = Path(__file__).resolve().parent.parent
sys.path.insert(0, str(PROJECT_ROOT))

from src.infrastructure.adapters.fmp.index_constituents import FMPIndexConstituentsAdapter
from src.infrastructure.adapters.r2.client import R2Client

logging.basicConfig(
    level=logging.INFO,
    format="%(asctime)s %(levelname)-8s %(name)s — %(message)s",
    datefmt="%H:%M:%S",
)
logger = logging.getLogger(__name__)

# Tier → timeframe mapping
TIER_TIMEFRAMES: dict[str, list[str]] = {
    "sp500": ["1d", "1w", "1h", "4h"],
    "nq100": ["1d", "1w", "1h", "4h"],
    "screener": ["1d", "1w"],
}


def build_universe(fmp: FMPIndexConstituentsAdapter) -> dict[str, Any]:
    """Build the full universe with tier metadata.

    Returns dict with keys: tickers (list), sp500 (list), nq100 (list).
    """
    # Fetch index memberships
    sp500_syms = fmp.fetch_sp500()
    logger.info(f"S&P 500: {len(sp500_syms)} symbols")

    nq100_syms = fmp.fetch_nq100_proxy()
    logger.info(f"NQ100 proxy: {len(nq100_syms)} symbols")

    # Fetch full screener with metadata
    all_stocks = fmp.fetch_screener_with_metadata(cap_min=100_000_000)
    logger.info(f"Screener universe: {len(all_stocks)} stocks (>$100M cap)")

    # Build sets for tier assignment
    sp500_set = set(sp500_syms)
    nq100_set = set(nq100_syms)

    # Assign tiers and timeframes to each stock
    tickers: list[dict[str, Any]] = []
    seen: set[str] = set()

    for stock in all_stocks:
        sym = stock["symbol"]
        if sym in seen:
            continue
        seen.add(sym)

        # Determine tier
        tiers: list[str] = []
        if sym in sp500_set:
            tiers.append("sp500")
        if sym in nq100_set:
            tiers.append("nq100")
        if not tiers:
            tiers.append("screener")

        # Highest-priority tier determines timeframes
        primary_tier = tiers[0]
        timeframes = TIER_TIMEFRAMES[primary_tier]

        tickers.append(
            {
                "symbol": sym,
                "name": stock.get("name", ""),
                "sector": stock.get("sector", ""),
                "industry": stock.get("industry", ""),
                "marketCap": stock.get("marketCap", 0),
                "exchange": stock.get("exchange", ""),
                "tier": primary_tier,
                "tiers": tiers,
                "timeframes": timeframes,
            }
        )

    # Add any SP500/NQ100 symbols not found in screener
    for sym in sorted(sp500_set | nq100_set):
        if sym not in seen:
            tiers = []
            if sym in sp500_set:
                tiers.append("sp500")
            if sym in nq100_set:
                tiers.append("nq100")
            primary_tier = tiers[0]
            tickers.append(
                {
                    "symbol": sym,
                    "name": "",
                    "sector": "",
                    "industry": "",
                    "marketCap": 0,
                    "exchange": "",
                    "tier": primary_tier,
                    "tiers": tiers,
                    "timeframes": TIER_TIMEFRAMES[primary_tier],
                }
            )
            seen.add(sym)

    logger.info(
        f"Universe built: {len(tickers)} total tickers "
        f"(SP500={len(sp500_set)}, NQ100={len(nq100_set)}, "
        f"screener={len(tickers) - len(sp500_set | nq100_set)})"
    )

    return {
        "tickers": tickers,
        "sp500": sorted(sp500_set),
        "nq100": sorted(nq100_set),
    }


def upload_universe(r2: R2Client, universe: dict[str, Any]) -> None:
    """Upload universe files to R2."""
    now = datetime.now(timezone.utc).isoformat()

    # Universe JSON with full metadata
    universe_json = {
        "generated_at": now,
        "total_count": len(universe["tickers"]),
        "sp500_count": len(universe["sp500"]),
        "nq100_count": len(universe["nq100"]),
        "tickers": universe["tickers"],
    }
    r2.put_json("meta/universe.json", universe_json)
    logger.info(f"Uploaded meta/universe.json ({len(universe['tickers'])} tickers)")

    # SP500 symbol list
    r2.put_json("meta/sp500.json", universe["sp500"])
    logger.info(f"Uploaded meta/sp500.json ({len(universe['sp500'])} symbols)")

    # NQ100 symbol list
    r2.put_json("meta/nq100.json", universe["nq100"])
    logger.info(f"Uploaded meta/nq100.json ({len(universe['nq100'])} symbols)")

    # Update last_updated timestamp
    last_updated = r2.get_json("meta/last_updated.json") or {}
    last_updated["universe"] = now
    r2.put_json("meta/last_updated.json", last_updated)
    logger.info("Updated meta/last_updated.json")


def write_local(universe: dict[str, Any], output_dir: Path) -> None:
    """Write universe files locally for dashboard build."""
    output_dir.mkdir(parents=True, exist_ok=True)

    universe_json = {
        "generated_at": datetime.now(timezone.utc).isoformat(),
        "total_count": len(universe["tickers"]),
        "sp500_count": len(universe["sp500"]),
        "nq100_count": len(universe["nq100"]),
        "tickers": universe["tickers"],
    }
    (output_dir / "universe.json").write_text(
        json.dumps(universe_json, separators=(",", ":")), encoding="utf-8"
    )
    logger.info(f"Wrote local {output_dir / 'universe.json'}")


def main() -> None:
    parser = argparse.ArgumentParser(description="R2 Universe Builder (Job 0)")
    parser.add_argument("--dry-run", action="store_true", help="Show counts without uploading")
    args = parser.parse_args()

    fmp = FMPIndexConstituentsAdapter()
    universe = build_universe(fmp)

    # Summary
    tier_counts: dict[str, int] = {}
    for t in universe["tickers"]:
        tier_counts[t["tier"]] = tier_counts.get(t["tier"], 0) + 1
    logger.info(f"Tier breakdown: {tier_counts}")

    if args.dry_run:
        logger.info("Dry run — skipping upload")
        return

    r2 = R2Client()
    upload_universe(r2, universe)

    # Also write locally for dashboard build
    local_dir = PROJECT_ROOT / "out" / "signals" / "data"
    write_local(universe, local_dir)

    logger.info("Universe builder complete")


if __name__ == "__main__":
    main()
