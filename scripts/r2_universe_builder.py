#!/usr/bin/env python3
"""R2 Universe Builder — Job 0 of the R2 data pipeline.

3-stage pipeline:
  Stage 1: Data acquisition (FMP screener + Yahoo float shares with caching)
  Stage 2: Compute & filter (dollar_volume, turnover_rate, thresholds)
  Stage 3: Merge curated YAML → meta/universe.json

Usage:
    python scripts/r2_universe_builder.py
    python scripts/r2_universe_builder.py --dry-run
    python scripts/r2_universe_builder.py --force-float-refresh
    python scripts/r2_universe_builder.py --yaml config/universe.yaml
"""

from __future__ import annotations

import argparse
import json
import logging
import sys
import time
from datetime import datetime, timezone
from pathlib import Path
from typing import Any

# Add project root to path
PROJECT_ROOT = Path(__file__).resolve().parent.parent
sys.path.insert(0, str(PROJECT_ROOT))

from src.infrastructure.adapters.fmp.index_constituents import FMPIndexConstituentsAdapter
from src.infrastructure.adapters.yahoo.fundamentals_adapter import YahooFundamentalsAdapter

logging.basicConfig(
    level=logging.INFO,
    format="%(asctime)s %(levelname)-8s %(name)s — %(message)s",
    datefmt="%H:%M:%S",
)
logger = logging.getLogger(__name__)

LOCAL_CACHE_DIR = PROJECT_ROOT / "data" / "cache"
LOCAL_FLOAT_CACHE = LOCAL_CACHE_DIR / "float_shares.json"
R2_FLOAT_CACHE_KEY = "meta/float_shares_cache.json"


# ─── YAML Loader ──────────────────────────────────────────────────────────────


def load_yaml_config(yaml_path: Path) -> dict[str, Any]:
    """Load universe YAML config."""
    import yaml

    with open(yaml_path) as f:
        return yaml.safe_load(f) or {}


def load_curated_from_yaml(yaml_path: Path) -> list[dict[str, Any]]:
    """Extract curated symbols from universe YAML with tier/tier_group metadata.

    Returns list of ticker dicts with: symbol, tier, tier_group, sectors.
    """
    config = load_yaml_config(yaml_path)
    curated: list[dict[str, Any]] = []
    seen: set[str] = set()

    # Market ETFs → tier_group="market"
    for item in config.get("market", []):
        sym = item.get("symbol", "")
        if sym and sym not in seen:
            seen.add(sym)
            curated.append(
                {
                    "symbol": sym,
                    "name": item.get("name", ""),
                    "tier": "market",
                    "tier_group": "market",
                    "sectors": ["market"],
                    "timeframes": ["1d", "1w", "1h", "4h"],
                }
            )

    # Sectors → tier_group="sector"
    for sector_name, sector_data in config.get("sectors", {}).items():
        if not isinstance(sector_data, dict):
            continue
        # Sector ETF
        etf = sector_data.get("etf", "")
        if etf and etf not in seen:
            seen.add(etf)
            curated.append(
                {
                    "symbol": etf,
                    "name": f"{sector_name} ETF",
                    "tier": sector_name,
                    "tier_group": "sector",
                    "sectors": [sector_name],
                    "timeframes": ["1d", "1w", "1h", "4h"],
                }
            )
        # Sector stocks
        for sym in sector_data.get("stocks", []):
            if sym and sym not in seen:
                seen.add(sym)
                curated.append(
                    {
                        "symbol": sym,
                        "name": "",
                        "tier": sector_name,
                        "tier_group": "sector",
                        "sectors": [sector_name],
                        "timeframes": ["1d", "1w", "1h", "4h"],
                    }
                )
            elif sym and sym in seen:
                # Add this sector to existing entry's sectors list
                for c in curated:
                    if c["symbol"] == sym and sector_name not in c["sectors"]:
                        c["sectors"].append(sector_name)

    # Speculative → tier_group="speculative"
    spec = config.get("speculative", {})
    if isinstance(spec, dict):
        for sym in spec.get("stocks", []):
            if sym and sym not in seen:
                seen.add(sym)
                curated.append(
                    {
                        "symbol": sym,
                        "name": "",
                        "tier": "speculative",
                        "tier_group": "speculative",
                        "sectors": ["speculative"],
                        "timeframes": ["1d", "1w", "1h", "4h"],
                    }
                )

    # Holdout → tier_group="holdout"
    validation = config.get("validation", {})
    if isinstance(validation, dict):
        for sym in validation.get("holdout", []):
            if sym and sym not in seen:
                seen.add(sym)
                curated.append(
                    {
                        "symbol": sym,
                        "name": "",
                        "tier": "holdout",
                        "tier_group": "holdout",
                        "sectors": ["holdout"],
                        "timeframes": ["1d", "1w", "1h", "4h"],
                    }
                )

    return curated


def get_screening_config(yaml_path: Path) -> dict[str, Any]:
    """Load r2_screening config section with defaults."""
    config = load_yaml_config(yaml_path)
    defaults = {
        "min_market_cap": 500_000_000,
        "min_daily_dollar_volume": 5_000_000,
        "min_turnover_rate": 0.01,
        "sort_by": "dollar_volume",
        "float_cache_days": 7,
        "turnover_miss_threshold": 0.5,
    }
    r2_config = config.get("r2_screening", {})
    if isinstance(r2_config, dict):
        defaults.update(r2_config)
    return defaults


# ─── Float Cache ──────────────────────────────────────────────────────────────


def _load_local_float_cache() -> tuple[dict[str, float], str]:
    """Load float shares from local cache. Returns (data, timestamp)."""
    if not LOCAL_FLOAT_CACHE.exists():
        return {}, ""
    try:
        raw = json.loads(LOCAL_FLOAT_CACHE.read_text(encoding="utf-8"))
        return raw.get("data", {}), raw.get("generated_at", "")
    except (json.JSONDecodeError, KeyError):
        return {}, ""


def _save_local_float_cache(data: dict[str, float]) -> None:
    """Save float shares to local cache."""
    LOCAL_CACHE_DIR.mkdir(parents=True, exist_ok=True)
    payload = {
        "generated_at": datetime.now(timezone.utc).isoformat(),
        "count": len(data),
        "data": data,
    }
    LOCAL_FLOAT_CACHE.write_text(json.dumps(payload, separators=(",", ":")), encoding="utf-8")
    logger.info(f"Saved local float cache: {len(data)} symbols → {LOCAL_FLOAT_CACHE}")


def _load_r2_float_cache(r2: Any) -> tuple[dict[str, float], str]:
    """Load float shares from R2 cache. Returns (data, timestamp)."""
    try:
        raw = r2.get_json(R2_FLOAT_CACHE_KEY)
        if raw and isinstance(raw, dict):
            return raw.get("data", {}), raw.get("generated_at", "")
    except Exception as e:
        logger.warning(f"Failed to load R2 float cache: {e}")
    return {}, ""


def _save_r2_float_cache(r2: Any, data: dict[str, float]) -> None:
    """Save float shares to R2 cache."""
    payload = {
        "generated_at": datetime.now(timezone.utc).isoformat(),
        "count": len(data),
        "data": data,
    }
    r2.put_json(R2_FLOAT_CACHE_KEY, payload)
    logger.info(f"Saved R2 float cache: {len(data)} symbols")


def _cache_age_days(timestamp: str) -> float:
    """Return age of cache in days. Returns inf if timestamp is invalid."""
    if not timestamp:
        return float("inf")
    try:
        ts = datetime.fromisoformat(timestamp)
        if ts.tzinfo is None:
            ts = ts.replace(tzinfo=timezone.utc)
        delta = datetime.now(timezone.utc) - ts
        return delta.total_seconds() / 86400
    except (ValueError, TypeError):
        return float("inf")


def get_float_shares(
    symbols: list[str],
    r2: Any | None,
    max_cache_days: float,
    force_refresh: bool = False,
    dry_run: bool = False,
) -> dict[str, float]:
    """Get float shares with multi-level cache: local → R2 → Yahoo Finance.

    Args:
        symbols: Symbols to fetch float shares for.
        r2: R2 client (None in dry-run mode).
        max_cache_days: Maximum cache age in days before refresh.
        force_refresh: Force Yahoo fetch regardless of cache.
        dry_run: If True, never access R2.
    """
    # Level 1: Local cache
    if not force_refresh:
        local_data, local_ts = _load_local_float_cache()
        if local_data and _cache_age_days(local_ts) < max_cache_days:
            logger.info(f"Float shares: using local cache ({len(local_data)} symbols)")
            return local_data

    # Level 2: R2 cache (skip in dry-run)
    if not force_refresh and not dry_run and r2 is not None:
        r2_data, r2_ts = _load_r2_float_cache(r2)
        if r2_data and _cache_age_days(r2_ts) < max_cache_days:
            logger.info(f"Float shares: using R2 cache ({len(r2_data)} symbols)")
            _save_local_float_cache(r2_data)
            return r2_data

    # Level 3: Yahoo Finance batch fetch
    logger.info(f"Float shares: fetching from Yahoo Finance for {len(symbols)} symbols...")
    yahoo = YahooFundamentalsAdapter(max_workers=20)
    data = yahoo.fetch_float_shares(symbols)

    if data:
        _save_local_float_cache(data)
        if not dry_run and r2 is not None:
            _save_r2_float_cache(r2, data)

    return data


# ─── Stage 1: Data Acquisition ────────────────────────────────────────────────


def acquire_raw_stocks(
    fmp: FMPIndexConstituentsAdapter, min_market_cap: float
) -> list[dict[str, Any]]:
    """Fetch US stocks from FMP screener with metadata.

    Passes min_market_cap directly to the FMP screener API so the server-side
    filter reduces the response size (fewer stocks to download and process).
    """
    stocks = fmp.fetch_screener_with_metadata(cap_min=min_market_cap)
    logger.info(
        f"Stage 1: Acquired {len(stocks)} raw stocks "
        f"from FMP screener (cap >= ${min_market_cap / 1e6:.0f}M)"
    )
    return stocks


# ─── Stage 2: Compute & Filter ────────────────────────────────────────────────


def _turnover_label(rate: float) -> str:
    """Classify turnover rate into liquidity tier.

    0 (exact)  → no_data (float shares unavailable, e.g. ETFs)
    < 1%       → inactive (low liquidity)
    1% - 3%    → moderate (normal activity)
    3% - 5%    → active
    5% - 10%   → highly_active
    >= 10%     → extremely_active (potential reversal)
    """
    if rate == 0.0:
        return "no_data"
    if rate < 0.01:
        return "inactive"
    if rate < 0.03:
        return "moderate"
    if rate < 0.05:
        return "active"
    if rate < 0.10:
        return "highly_active"
    return "extremely_active"


def _is_common_stock(stock: dict[str, Any]) -> bool:
    """Filter out mutual funds and ETFs from FMP screener results.

    FMP's company-screener returns mutual fund share classes (5-letter alpha
    tickers ending in X, e.g. AAFTX, ABALX) and ETFs listed on NYSE Arca /
    CBOE. These don't have float shares and aren't suitable for backtesting.
    """
    sym = stock.get("symbol", "")
    exchange = stock.get("exchange", "")
    # Mutual fund share classes: 5+ alpha chars ending in X
    if len(sym) >= 5 and sym[-1] == "X" and sym.isalpha():
        return False
    # ETF-heavy exchanges
    if "Options Exchange" in exchange or "Arca" in exchange:
        return False
    return True


def compute_and_filter(
    raw_stocks: list[dict[str, Any]],
    float_shares: dict[str, float],
    screening: dict[str, Any],
) -> tuple[list[dict[str, Any]], dict[str, Any]]:
    """Compute dollar_volume and turnover_rate, then apply filters.

    Pre-filters mutual funds and ETFs (no float shares by design), then
    computes turnover for real stocks only.

    Returns (filtered_stocks, screening_stats).
    """
    min_cap = screening["min_market_cap"]
    min_dv = screening["min_daily_dollar_volume"]
    min_turnover = screening["min_turnover_rate"]
    turnover_miss_threshold = screening["turnover_miss_threshold"]

    # Pre-filter: exclude mutual funds and ETFs
    stocks = [s for s in raw_stocks if _is_common_stock(s)]
    excluded = len(raw_stocks) - len(stocks)
    if excluded > 0:
        logger.info(f"Excluded {excluded} mutual funds/ETFs from {len(raw_stocks)} raw entries")

    # Compute derived metrics
    float_hits = 0
    float_misses = 0

    for stock in stocks:
        price = float(stock.get("price") or 0)
        volume = float(stock.get("volume") or 0)
        stock["dollar_volume"] = price * volume

        sym = stock["symbol"]
        fs = float_shares.get(sym)
        if fs and fs > 0:
            stock["turnover_rate"] = volume / fs
            float_hits += 1
        else:
            stock["turnover_rate"] = 0.0
            float_misses += 1
        stock["turnover_label"] = _turnover_label(stock["turnover_rate"])

    # Auto-disable turnover filter if too many misses
    total = len(stocks)
    float_miss_ratio = float_misses / total if total > 0 else 1.0
    turnover_filter_active = float_miss_ratio <= turnover_miss_threshold

    if not turnover_filter_active:
        logger.warning(
            f"Float miss ratio {float_miss_ratio:.0%} > threshold "
            f"{turnover_miss_threshold:.0%}, disabling turnover filter"
        )

    # Apply filters
    filtered: list[dict[str, Any]] = []
    for stock in stocks:
        cap = float(stock.get("marketCap") or 0)
        dv = stock.get("dollar_volume", 0)
        tr = stock.get("turnover_rate", 0)

        if cap < min_cap:
            continue
        if dv < min_dv:
            continue
        if turnover_filter_active and tr < min_turnover:
            continue
        filtered.append(stock)

    # Sort by dollar_volume descending
    filtered.sort(key=lambda x: x.get("dollar_volume", 0), reverse=True)

    stats = {
        "fmp_raw": len(raw_stocks),
        "funds_etfs_excluded": excluded,
        "stocks_after_exclusion": total,
        "after_filter": len(filtered),
        "float_hits": float_hits,
        "float_misses": float_misses,
        "turnover_filter_active": turnover_filter_active,
    }
    logger.info(
        f"Stage 2: {len(raw_stocks)} raw → {total} stocks "
        f"(excl {excluded} funds/ETFs) → {len(filtered)} after filter "
        f"(float {float_hits}/{total} = {float_hits / total:.0%}, "
        f"turnover_filter={'on' if turnover_filter_active else 'off'})"
    )
    return filtered, stats


# ─── Stage 3: Merge Curated ──────────────────────────────────────────────────


def merge_curated(
    screened: list[dict[str, Any]],
    curated: list[dict[str, Any]],
    screener_metadata: dict[str, dict[str, Any]],
) -> tuple[list[dict[str, Any]], dict[str, Any]]:
    """Merge curated YAML symbols with all screened stocks.

    Curated symbols always included. All screened stocks that passed
    filters are included (no cap). Deduplication by symbol.

    Returns (final_tickers, merge_stats).
    """
    curated_set = {c["symbol"] for c in curated}

    # Enrich curated with screener metadata (marketCap, dollar_volume, etc.)
    final: list[dict[str, Any]] = []
    for c in curated:
        meta = screener_metadata.get(c["symbol"], {})
        ticker = {
            "symbol": c["symbol"],
            "name": meta.get("name") or c.get("name", ""),
            "tier": c["tier"],
            "tier_group": c["tier_group"],
            "sector": c["tier"] if c["tier_group"] == "sector" else c["tier_group"],
            "sectors": c.get("sectors", []),
            "marketCap": meta.get("marketCap", 0),
            "dollar_volume": meta.get("dollar_volume", 0),
            "turnover_rate": meta.get("turnover_rate", 0),
            "turnover_label": _turnover_label(meta.get("turnover_rate", 0)),
            "timeframes": c.get("timeframes", ["1d", "1w"]),
        }
        final.append(ticker)

    # Add all screened stocks (dedup against curated)
    screened_added = 0
    for stock in screened:
        sym = stock["symbol"]
        if sym in curated_set:
            continue

        sector = stock.get("sector", "").lower().replace(" ", "_") or "other"
        final.append(
            {
                "symbol": sym,
                "name": stock.get("name", ""),
                "tier": sector,
                "tier_group": "screened",
                "sector": sector,
                "sectors": [sector] if sector else [],
                "marketCap": stock.get("marketCap", 0),
                "dollar_volume": stock.get("dollar_volume", 0),
                "turnover_rate": stock.get("turnover_rate", 0),
                "turnover_label": stock.get("turnover_label", "no_data"),
                "timeframes": ["1d", "1w", "1h", "4h"],
            }
        )
        curated_set.add(sym)  # Prevent dupes
        screened_added += 1

    stats = {
        "curated_count": len(curated),
        "screened_added": screened_added,
        "final_count": len(final),
    }
    logger.info(
        f"Stage 3: {len(curated)} curated + {screened_added} screened " f"= {len(final)} final"
    )
    return final, stats


# ─── Upload / Write ───────────────────────────────────────────────────────────


def build_universe_json(
    tickers: list[dict[str, Any]],
    screening_stats: dict[str, Any],
    merge_stats: dict[str, Any],
    screening_config: dict[str, Any],
) -> dict[str, Any]:
    """Build the universe.json payload."""
    return {
        "generated_at": datetime.now(timezone.utc).isoformat(),
        "source": "fmp_screener + config/universe.yaml",
        "total_count": len(tickers),
        "screening": {
            "fmp_raw": screening_stats.get("fmp_raw", 0),
            "after_filter": screening_stats.get("after_filter", 0),
            "curated_added": merge_stats.get("curated_count", 0),
            "final": len(tickers),
            "float_hits": screening_stats.get("float_hits", 0),
            "float_misses": screening_stats.get("float_misses", 0),
            "turnover_filter_active": screening_stats.get("turnover_filter_active", True),
        },
        "filters_applied": {
            "min_market_cap": screening_config.get("min_market_cap"),
            "min_daily_dollar_volume": screening_config.get("min_daily_dollar_volume"),
            "min_turnover_rate": screening_config.get("min_turnover_rate"),
        },
        "tickers": tickers,
    }


def upload_universe(r2: Any, universe_json: dict[str, Any]) -> None:
    """Upload universe files to R2."""
    r2.put_json("meta/universe.json", universe_json)
    logger.info(f"Uploaded meta/universe.json ({universe_json['total_count']} tickers)")

    # Update last_updated timestamp
    last_updated = r2.get_json("meta/last_updated.json") or {}
    last_updated["universe"] = universe_json["generated_at"]
    r2.put_json("meta/last_updated.json", last_updated)
    logger.info("Updated meta/last_updated.json")


def write_local(universe_json: dict[str, Any], output_dir: Path) -> None:
    """Write universe JSON locally for dashboard build."""
    output_dir.mkdir(parents=True, exist_ok=True)
    (output_dir / "universe.json").write_text(
        json.dumps(universe_json, separators=(",", ":")), encoding="utf-8"
    )
    logger.info(f"Wrote local {output_dir / 'universe.json'}")


# ─── Main ─────────────────────────────────────────────────────────────────────


def main() -> None:
    parser = argparse.ArgumentParser(description="R2 Universe Builder (Job 0)")
    parser.add_argument("--dry-run", action="store_true", help="No R2 access, local cache only")
    parser.add_argument(
        "--force-float-refresh",
        action="store_true",
        help="Refresh float cache from Yahoo",
    )
    parser.add_argument(
        "--yaml",
        type=Path,
        default=PROJECT_ROOT / "config" / "universe.yaml",
        help="YAML path",
    )
    args = parser.parse_args()

    yaml_path: Path = args.yaml
    dry_run: bool = args.dry_run
    force_refresh: bool = args.force_float_refresh

    t0 = time.monotonic()

    # Load screening config
    screening = get_screening_config(yaml_path)
    logger.info(f"Screening config: {screening}")

    # Initialize FMP adapter
    fmp = FMPIndexConstituentsAdapter()

    # Initialize R2 client (unless dry-run)
    r2 = None
    if not dry_run:
        from src.infrastructure.adapters.r2.client import R2Client

        r2 = R2Client()

    # ── Stage 1: Data Acquisition ────────────────────────────────────────
    raw_stocks = acquire_raw_stocks(fmp, min_market_cap=screening["min_market_cap"])

    # Pre-filter to get stock symbols for float fetch
    stock_symbols = [s["symbol"] for s in raw_stocks if _is_common_stock(s)]
    logger.info(f"Stock symbols for float fetch: {len(stock_symbols)}")

    float_shares = get_float_shares(
        symbols=stock_symbols,
        r2=r2,
        max_cache_days=screening["float_cache_days"],
        force_refresh=force_refresh,
        dry_run=dry_run,
    )

    # ── Stage 2: Compute & Filter ────────────────────────────────────────
    filtered, screening_stats = compute_and_filter(raw_stocks, float_shares, screening)

    # ── Stage 3: Merge Curated ───────────────────────────────────────────
    curated = load_curated_from_yaml(yaml_path)
    logger.info(f"Curated from YAML: {len(curated)} symbols")

    # Build screener metadata lookup for enriching curated symbols
    screener_metadata: dict[str, dict[str, Any]] = {}
    for stock in raw_stocks:
        screener_metadata[stock["symbol"]] = stock

    tickers, merge_stats = merge_curated(
        screened=filtered,
        curated=curated,
        screener_metadata=screener_metadata,
    )

    # Build JSON
    universe_json = build_universe_json(tickers, screening_stats, merge_stats, screening)

    # Summary
    tier_group_counts: dict[str, int] = {}
    for t in tickers:
        tg = t.get("tier_group", "unknown")
        tier_group_counts[tg] = tier_group_counts.get(tg, 0) + 1
    logger.info(f"Tier group breakdown: {tier_group_counts}")

    elapsed = time.monotonic() - t0
    logger.info(f"Pipeline complete in {elapsed:.1f}s — {len(tickers)} tickers")

    if dry_run:
        logger.info("Dry run — skipping upload")
        # Print screening funnel
        fh = screening_stats["float_hits"]
        sa = screening_stats["stocks_after_exclusion"]
        print("\n=== Screening Funnel ===")
        print(f"  FMP raw:          {screening_stats['fmp_raw']}")
        print(f"  Funds/ETFs excl:  {screening_stats['funds_etfs_excluded']}")
        print(f"  Stocks only:      {sa}")
        print(f"  After filter:     {screening_stats['after_filter']}")
        print(f"  Curated (YAML):   {merge_stats['curated_count']}")
        print(f"  Screened added:   {merge_stats['screened_added']}")
        print(f"  Final:            {len(tickers)}")
        print(f"  Float coverage:   {fh}/{sa} ({fh / sa:.0%})" if sa else "  Float coverage: N/A")
        print(
            f"  Turnover filter:  "
            f"{'on' if screening_stats['turnover_filter_active'] else 'off'}"
        )
        print(f"  Tier groups:      {tier_group_counts}")
        return

    # Upload to R2
    upload_universe(r2, universe_json)

    # Write locally for dashboard build
    local_dir = PROJECT_ROOT / "out" / "signals" / "data"
    write_local(universe_json, local_dir)

    logger.info("Universe builder complete")


if __name__ == "__main__":
    main()
