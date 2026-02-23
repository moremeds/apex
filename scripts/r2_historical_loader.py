#!/usr/bin/env python3
"""R2 Historical Data Loader — Job 1 of the R2 data pipeline.

Loads OHLCV data from FMP/Yahoo waterfall and uploads to R2 as Parquet.
Generates data_quality.json for the monitor page.

Usage:
    python scripts/r2_historical_loader.py --backfill --symbols AAPL MSFT SPY
    python scripts/r2_historical_loader.py --delta
    python scripts/r2_historical_loader.py --validate-only
"""

from __future__ import annotations

import argparse
import json
import logging
import sys
import time
from datetime import date, datetime, timedelta, timezone
from pathlib import Path
from typing import Any

import pandas as pd

# Add project root to path
PROJECT_ROOT = Path(__file__).resolve().parent.parent
sys.path.insert(0, str(PROJECT_ROOT))

from src.infrastructure.adapters.r2.client import R2Client
from src.services.bar_loader import load_bars

logging.basicConfig(
    level=logging.INFO,
    format="%(asctime)s %(levelname)-8s %(name)s — %(message)s",
    datefmt="%H:%M:%S",
)
logger = logging.getLogger(__name__)

# Coverage status thresholds
STATUS_THRESHOLDS = {
    "PASS": 0.98,
    "WARN": 0.95,
    "CAUTION": 0.90,
}

# Batch size for processing symbols
BATCH_SIZE = 50

# Backfill start date
BACKFILL_START = date(2019, 1, 1)

# Delta overlap: re-fetch this many extra days to catch corrections
DELTA_OVERLAP_DAYS = 3


def normalize_timestamps(df: pd.DataFrame, timeframe: str) -> pd.DataFrame:
    """Apply canonical normalization matching parquet_historical_store merge logic.

    1. Convert timestamps to UTC
    2. Floor to seconds (truncate microseconds)
    3. For daily/weekly: normalize to midnight UTC
    """
    if df.empty:
        return df

    idx = df.index
    if hasattr(idx, "tz") and idx.tz is not None:
        idx = idx.tz_convert("UTC")
    else:
        idx = idx.tz_localize("UTC")

    # Floor to seconds
    idx = idx.floor("s")

    # Daily/weekly: normalize to midnight
    if timeframe in ("1d", "1w"):
        idx = idx.normalize()

    df.index = idx
    return df


def dedupe_and_sort(df: pd.DataFrame) -> pd.DataFrame:
    """Drop duplicate timestamps keeping last, sort by index."""
    if df.empty:
        return df
    df = df[~df.index.duplicated(keep="last")]
    return df.sort_index()


def load_universe_from_r2(r2: R2Client) -> list[dict[str, Any]]:
    """Load universe from R2 meta/universe.json."""
    data = r2.get_json("meta/universe.json")
    if not data:
        logger.error("No universe found in R2. Run r2_universe_builder.py first.")
        sys.exit(1)
    tickers: list[dict[str, Any]] = data.get("tickers", [])
    logger.info(f"Loaded universe: {len(tickers)} tickers from R2")
    return tickers


def get_symbols_for_timeframe(
    tickers: list[dict[str, Any]],
    timeframe: str,
    symbol_filter: list[str] | None = None,
) -> list[str]:
    """Get symbols that need a given timeframe based on tier rules."""
    symbols = []
    for t in tickers:
        if timeframe in t.get("timeframes", []):
            symbols.append(t["symbol"])

    if symbol_filter:
        symbols = [s for s in symbols if s in set(symbol_filter)]

    return symbols


def _compute_delta_days(
    r2: R2Client, symbols: list[str], timeframe: str, overlap: int
) -> dict[str, int]:
    """Compute per-symbol days_back for delta mode based on last bar in R2.

    Returns dict of symbol → days_back. Symbols without R2 data get max backfill.
    """
    today = date.today()
    max_days = (today - BACKFILL_START).days
    result: dict[str, int] = {}

    for sym in symbols:
        lm = r2.get_last_modified(f"parquet/historical/{timeframe}/{sym}.parquet")
        if lm is None:
            # No existing data — full backfill for this symbol
            result[sym] = max_days
            continue

        # Use last_modified as proxy; for more precision, could read parquet and check index.max()
        # but that's expensive for 3000 symbols. last_modified is a good heuristic.
        days_since = (today - lm.date()).days + overlap
        result[sym] = max(days_since, overlap + 1)

    return result


def fetch_and_upload_timeframe(
    r2: R2Client,
    symbols: list[str],
    timeframe: str,
    days: int,
    is_delta: bool = False,
) -> dict[str, int]:
    """Fetch bars for symbols×timeframe and upload to R2.

    Derived timeframes (never fetched directly):
    - 1w: resampled from 1d bars
    - 4h: resampled from 1h bars

    Returns dict of symbol → bar_count for quality validation (not full DataFrames).
    """
    if timeframe == "1w":
        return _handle_1w(r2, symbols, is_delta)
    if timeframe == "4h":
        return _handle_4h(r2, symbols, days, is_delta)

    # For delta, compute per-symbol days_back from R2 last-modified
    per_symbol_days: dict[str, int] | None = None
    if is_delta:
        per_symbol_days = _compute_delta_days(r2, symbols, timeframe, DELTA_OVERLAP_DAYS)

    bar_counts: dict[str, int] = {}

    # Process in batches
    for batch_start in range(0, len(symbols), BATCH_SIZE):
        batch = symbols[batch_start : batch_start + BATCH_SIZE]
        logger.info(
            f"[{timeframe}] Batch {batch_start // BATCH_SIZE + 1}: "
            f"{len(batch)} symbols ({batch_start+1}-{batch_start+len(batch)}/{len(symbols)})"
        )

        # For delta, use the max days_back in this batch for the load_bars call
        # (load_bars takes a single days param for the whole batch)
        batch_days = days
        if per_symbol_days:
            batch_days = max(per_symbol_days.get(s, days) for s in batch)

        # Fetch via waterfall (FMP → Yahoo)
        bars = load_bars(batch, timeframe=timeframe, days=batch_days)

        # Normalize, merge with existing if delta, upload
        upload_items: list[tuple[str, pd.DataFrame]] = []
        for sym, df in bars.items():
            if df.empty:
                continue

            df = normalize_timestamps(df, timeframe)

            if is_delta:
                existing = r2.get_parquet(f"parquet/historical/{timeframe}/{sym}.parquet")
                if existing is not None and not existing.empty:
                    existing = normalize_timestamps(existing, timeframe)
                    df = pd.concat([existing, df])

            df = dedupe_and_sort(df)
            bar_counts[sym] = len(df)
            upload_items.append((f"parquet/historical/{timeframe}/{sym}.parquet", df))

        if upload_items:
            failed = r2.put_parquet_batch(upload_items, workers=20)
            if failed:
                logger.warning(f"[{timeframe}] Failed uploads: {failed}")

            logger.info(f"[{timeframe}] Uploaded {len(upload_items) - len(failed)} Parquet files")

        # Brief pause between batches
        if batch_start + BATCH_SIZE < len(symbols):
            time.sleep(1)

    return bar_counts


def _handle_1w(
    r2: R2Client,
    symbols: list[str],
    is_delta: bool,
) -> dict[str, int]:
    """Handle 1w timeframe by resampling from 1d bars.

    Always derives 1w from 1d — never fetches weekly directly.
    Returns dict of symbol -> bar_count.
    """
    logger.info(f"[1w] Deriving from 1d bars for {len(symbols)} symbols")

    bar_counts: dict[str, int] = {}

    for batch_start in range(0, len(symbols), BATCH_SIZE):
        batch = symbols[batch_start : batch_start + BATCH_SIZE]
        upload_items: list[tuple[str, pd.DataFrame]] = []

        for sym in batch:
            df_1d = r2.get_parquet(f"parquet/historical/1d/{sym}.parquet")
            if df_1d is None or df_1d.empty:
                continue

            df_1d = normalize_timestamps(df_1d, "1d")
            df_1w = _resample_df_to_1w(df_1d)

            if df_1w.empty:
                continue

            if is_delta:
                existing = r2.get_parquet(f"parquet/historical/1w/{sym}.parquet")
                if existing is not None and not existing.empty:
                    existing = normalize_timestamps(existing, "1w")
                    df_1w = pd.concat([existing, df_1w])

            df_1w = dedupe_and_sort(df_1w)
            bar_counts[sym] = len(df_1w)
            upload_items.append((f"parquet/historical/1w/{sym}.parquet", df_1w))

        if upload_items:
            failed = r2.put_parquet_batch(upload_items, workers=20)
            logger.info(f"[1w] Uploaded {len(upload_items) - len(failed)} Parquet files")

    return bar_counts


def _resample_df_to_1w(df_1d: pd.DataFrame) -> pd.DataFrame:
    """Resample 1d DataFrame to 1w (weekly) bars."""
    if df_1d.empty:
        return df_1d

    ohlcv_cols = {"open": "first", "high": "max", "low": "min", "close": "last", "volume": "sum"}
    available_agg = {k: v for k, v in ohlcv_cols.items() if k in df_1d.columns}

    if not available_agg:
        return pd.DataFrame()

    # Resample to weekly, anchored on Monday (W-FRI = week ending Friday)
    df_1w = df_1d.resample("W-FRI").agg(available_agg).dropna(subset=["close"])
    return df_1w


def _handle_4h(
    r2: R2Client,
    symbols: list[str],
    days: int,
    is_delta: bool,
) -> dict[str, int]:
    """Handle 4h timeframe by resampling from 1h bars.

    Always derives 4h from 1h — never fetches 4h directly.
    Returns dict of symbol → bar_count.
    """
    logger.info(f"[4h] Deriving from 1h bars for {len(symbols)} symbols")

    bar_counts: dict[str, int] = {}

    for batch_start in range(0, len(symbols), BATCH_SIZE):
        batch = symbols[batch_start : batch_start + BATCH_SIZE]
        upload_items: list[tuple[str, pd.DataFrame]] = []

        for sym in batch:
            # Get 1h bars (may already be in R2 from 1h upload)
            df_1h = r2.get_parquet(f"parquet/historical/1h/{sym}.parquet")
            if df_1h is None or df_1h.empty:
                continue

            # Resample to 4h using pandas
            df_1h = normalize_timestamps(df_1h, "1h")
            df_4h = _resample_df_to_4h(df_1h)

            if df_4h.empty:
                continue

            if is_delta:
                existing = r2.get_parquet(f"parquet/historical/4h/{sym}.parquet")
                if existing is not None and not existing.empty:
                    existing = normalize_timestamps(existing, "4h")
                    df_4h = pd.concat([existing, df_4h])

            df_4h = dedupe_and_sort(df_4h)
            bar_counts[sym] = len(df_4h)
            upload_items.append((f"parquet/historical/4h/{sym}.parquet", df_4h))

        if upload_items:
            failed = r2.put_parquet_batch(upload_items, workers=20)
            logger.info(f"[4h] Uploaded {len(upload_items) - len(failed)} Parquet files")

    return bar_counts


def _resample_df_to_4h(df_1h: pd.DataFrame) -> pd.DataFrame:
    """Resample 1h DataFrame to 4h bars."""
    if df_1h.empty:
        return df_1h

    # Simple 4h resample
    ohlcv_cols = {"open": "first", "high": "max", "low": "min", "close": "last", "volume": "sum"}
    available_agg = {k: v for k, v in ohlcv_cols.items() if k in df_1h.columns}

    if not available_agg:
        return pd.DataFrame()

    df_4h = df_1h.resample("4h").agg(available_agg).dropna(subset=["close"])
    return df_4h


def compute_coverage(actual: int, timeframe: str, start: date, end: date) -> float:
    """Compute coverage percentage based on expected trading days/bars."""
    cal_days = (end - start).days
    if cal_days <= 0:
        return 100.0

    # Rough expected bar estimates
    if timeframe == "1d":
        expected = int(cal_days * 252 / 365)  # ~252 trading days/year
    elif timeframe == "1w":
        expected = int(cal_days / 7)
    elif timeframe == "1h":
        expected = int(cal_days * 252 / 365 * 7)  # ~7 bars/trading day
    elif timeframe == "4h":
        expected = int(cal_days * 252 / 365 * 2)  # ~2 bars/trading day
    else:
        expected = actual  # Unknown timeframe

    if expected <= 0:
        return 100.0

    return min(100.0, (actual / expected) * 100)


def coverage_status(pct: float) -> str:
    """Map coverage percentage to status string."""
    if pct >= STATUS_THRESHOLDS["PASS"] * 100:
        return "PASS"
    elif pct >= STATUS_THRESHOLDS["WARN"] * 100:
        return "WARN"
    elif pct >= STATUS_THRESHOLDS["CAUTION"] * 100:
        return "CAUTION"
    else:
        return "FAIL"


def _detect_gaps(df: pd.DataFrame, timeframe: str) -> list[dict[str, str]]:
    """Detect gaps in a bar DataFrame using expected frequency.

    Returns list of gap dicts with start/end timestamps.
    """
    if df.empty or len(df) < 2:
        return []

    # Map timeframe to expected frequency
    freq_map = {"1d": "B", "1w": "W", "1h": "h", "4h": "4h"}
    freq = freq_map.get(timeframe)
    if not freq:
        return []

    gaps: list[dict[str, str]] = []
    idx = df.index.sort_values()

    if timeframe in ("1d", "1w"):
        # For daily: business day gaps > 1 business day
        expected = pd.bdate_range(start=idx.min(), end=idx.max())
        missing = expected.difference(idx.normalize())
        # Group consecutive missing dates into gap ranges
        if len(missing) > 0:
            gap_start = missing[0]
            prev = missing[0]
            for dt in missing[1:]:
                if (dt - prev).days > 3:  # Non-consecutive gap
                    gaps.append({"start": str(gap_start), "end": str(prev)})
                    gap_start = dt
                prev = dt
            gaps.append({"start": str(gap_start), "end": str(prev)})
    else:
        # For intraday: detect gaps larger than 2x expected interval
        diffs = idx.to_series().diff()
        if timeframe == "1h":
            threshold = pd.Timedelta(hours=3)  # >3h gap in 1h data
        else:
            threshold = pd.Timedelta(hours=10)  # >10h gap in 4h data
        gap_mask = diffs > threshold
        for i in range(len(gap_mask)):
            if gap_mask.iloc[i]:
                gaps.append({"start": str(idx[i - 1]), "end": str(idx[i])})

    return gaps


def _detect_anomalies(df: pd.DataFrame) -> int:
    """Count anomalies in bar data (zero volume, negative prices, extreme moves)."""
    if df.empty:
        return 0

    count = 0
    if "volume" in df.columns:
        count += int((df["volume"] == 0).sum())
    for col in ("open", "high", "low", "close"):
        if col in df.columns:
            count += int((df[col] <= 0).sum())
    # Extreme daily moves (>50%)
    if "close" in df.columns and len(df) > 1:
        pct_change = df["close"].pct_change().abs()
        count += int((pct_change > 0.5).sum())

    return count


def generate_data_quality(
    r2: R2Client,
    tickers: list[dict[str, Any]],
    bar_counts: dict[tuple[str, str], int],
    symbol_filter: list[str] | None = None,
) -> dict[str, Any]:
    """Generate data_quality.json with real gap/anomaly detection.

    Downloads Parquet from R2 per symbol to run quality checks.
    Processes one symbol at a time to keep memory bounded.

    Args:
        r2: R2Client instance.
        tickers: Universe ticker list.
        bar_counts: Dict of (symbol, timeframe) → bar count from just-completed load.
        symbol_filter: Optional filter to only validate specific symbols.
    """
    today = date.today()
    quality_entries: list[dict[str, Any]] = []
    filter_set = set(symbol_filter) if symbol_filter else None

    for ticker in tickers:
        sym = ticker["symbol"]
        if filter_set and sym not in filter_set:
            continue

        for tf in ticker.get("timeframes", ["1d"]):
            key = (sym, tf)
            known_count = bar_counts.get(key)

            # Download from R2 to run quality checks
            df = r2.get_parquet(f"parquet/historical/{tf}/{sym}.parquet")

            if df is None or df.empty:
                quality_entries.append(
                    {
                        "symbol": sym,
                        "timeframe": tf,
                        "bars": 0,
                        "first_bar": None,
                        "last_bar": None,
                        "coverage_pct": 0.0,
                        "gaps": 0,
                        "anomalies": 0,
                        "status": "EMPTY",
                    }
                )
                continue

            bar_count = known_count if known_count is not None else len(df)
            first_bar = str(df.index.min())
            last_bar = str(df.index.max())

            # Compute coverage
            start = df.index.min().date() if hasattr(df.index.min(), "date") else BACKFILL_START
            cov_pct = compute_coverage(bar_count, tf, start, today)
            status = coverage_status(cov_pct)

            # Real gap and anomaly detection
            gap_list = _detect_gaps(df, tf)
            anomaly_count = _detect_anomalies(df)

            quality_entries.append(
                {
                    "symbol": sym,
                    "timeframe": tf,
                    "bars": bar_count,
                    "first_bar": first_bar,
                    "last_bar": last_bar,
                    "coverage_pct": round(cov_pct, 1),
                    "gaps": len(gap_list),
                    "gap_details": gap_list[:20],  # Cap at 20 to limit JSON size
                    "anomalies": anomaly_count,
                    "status": status,
                }
            )

            # DataFrame goes out of scope here — memory freed per symbol

    # Summary stats
    total = len(quality_entries)
    by_status: dict[str, int] = {}
    for e in quality_entries:
        s = e["status"]
        by_status[s] = by_status.get(s, 0) + 1

    quality_json = {
        "generated_at": datetime.now(timezone.utc).isoformat(),
        "total_entries": total,
        "by_status": by_status,
        "entries": quality_entries,
    }

    return quality_json


def main() -> None:
    parser = argparse.ArgumentParser(description="R2 Historical Data Loader (Job 1)")
    mode = parser.add_mutually_exclusive_group(required=True)
    mode.add_argument("--backfill", action="store_true", help="Full 2019-present backfill")
    mode.add_argument("--delta", action="store_true", help="Incremental (last-bar + overlap)")
    mode.add_argument(
        "--validate-only", action="store_true", help="Generate data_quality.json only"
    )
    parser.add_argument("--symbols", nargs="*", help="Specific symbols (default: all)")
    parser.add_argument(
        "--timeframes",
        nargs="*",
        default=["1d", "1w", "1h", "4h"],
        help="Timeframes to process",
    )
    args = parser.parse_args()

    r2 = R2Client()
    tickers = load_universe_from_r2(r2)

    if args.validate_only:
        logger.info("Validate-only mode: generating data_quality.json")
        quality = generate_data_quality(r2, tickers, {}, symbol_filter=args.symbols)
        _write_quality(r2, quality)
        return

    # Compute days to fetch
    today = date.today()
    if args.backfill:
        days = (today - BACKFILL_START).days
        logger.info(f"Backfill mode: {days} days back to {BACKFILL_START}")
    else:
        days = 7 + DELTA_OVERLAP_DAYS  # Delta: last week + overlap
        logger.info(f"Delta mode: {days} days with {DELTA_OVERLAP_DAYS}d overlap")

    # Process each timeframe (1w derived from 1d, 4h derived from 1h)
    timeframe_order = [tf for tf in ["1d", "1h", "1w", "4h"] if tf in args.timeframes]
    all_bar_counts: dict[tuple[str, str], int] = {}

    for tf in timeframe_order:
        symbols = get_symbols_for_timeframe(tickers, tf, args.symbols)
        if not symbols:
            logger.info(f"[{tf}] No symbols to process")
            continue

        logger.info(f"[{tf}] Processing {len(symbols)} symbols")

        # For 1h backfill: use more days (Yahoo returns max 730d of 1h)
        tf_days = days
        if tf == "1h" and args.backfill:
            tf_days = 730  # Yahoo 1h max

        counts = fetch_and_upload_timeframe(r2, symbols, tf, tf_days, is_delta=args.delta)

        for sym, count in counts.items():
            all_bar_counts[(sym, tf)] = count

        logger.info(f"[{tf}] Complete: {len(counts)} symbols loaded")

    # Generate data quality (downloads from R2 per-symbol, memory-bounded)
    quality = generate_data_quality(r2, tickers, all_bar_counts, symbol_filter=args.symbols)
    _write_quality(r2, quality)

    # Update last_updated
    last_updated = r2.get_json("meta/last_updated.json") or {}
    last_updated["historical"] = datetime.now(timezone.utc).isoformat()
    r2.put_json("meta/last_updated.json", last_updated)

    logger.info("Historical loader complete")


def _write_quality(r2: R2Client, quality: dict[str, Any]) -> None:
    """Write data_quality.json to both R2 and local output."""
    # Upload to R2
    r2.put_json("meta/data_quality.json", quality)
    logger.info(f"Uploaded meta/data_quality.json ({quality['total_entries']} entries)")

    # Write locally for dashboard build
    local_path = PROJECT_ROOT / "out" / "signals" / "data" / "data_quality.json"
    local_path.parent.mkdir(parents=True, exist_ok=True)
    local_path.write_text(json.dumps(quality, separators=(",", ":")), encoding="utf-8")
    logger.info(f"Wrote local {local_path}")


if __name__ == "__main__":
    main()
