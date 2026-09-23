"""P1.6 performance experiment: identical real-lake workloads against several servers.

Each target is a lake-only server (``serve.py``) on loopback: the baseline tree, and
the candidate in each LakeDb mode. Requests are interleaved per repetition in a
shuffled target order so filesystem-cache warmth and ordering bias fall on every
target alike; the first request per (workload, target) is reported separately from
the warm distribution. A result hash per response checks that targets agree.

    uv run python scripts/lake_verify/bench.py --out DIR \
        --target base=http://127.0.0.1:8341=PID --target per_call=...=PID ...

Writes ``performance.json`` (and prints a table). Never writes to the lake.
"""

from __future__ import annotations

import argparse
import hashlib
import json
import random
import statistics
import subprocess
import time
from dataclasses import dataclass
from pathlib import Path
from typing import Any, Dict, List, Optional

import httpx

REPS = 20


@dataclass(frozen=True)
class Workload:
    name: str
    path: str
    params: Dict[str, Any]
    baseline: bool  # the route and parameters exist on the baseline tree


def _bulk_symbols() -> str:
    # 50 large-cap US tickers chosen for the workload; a ticker without live daily
    # Bronze would land in the bulk response's `missing` map, visible in the result.
    return ",".join(
        "AAPL MSFT AMZN GOOGL META NVDA JPM JNJ V PG XOM UNH HD MA CVX ABBV PFE KO PEP "
        "MRK COST WMT BAC DIS CSCO ADBE CRM NFLX TMO ABT ORCL ACN MCD LIN DHR NKE TXN "
        "WFC NEE PM UPS BMY RTX HON QCOM LOW AMGN IBM SBUX CAT".split()
    )


WORKLOADS: List[Workload] = [
    Workload(
        "equity_1d_raw",
        "/v1/equity/SPY/bars",
        {"timeframe": "1d", "price_mode": "raw"},
        True,
    ),
    Workload(
        "equity_1d_adjusted",
        "/v1/equity/SPY/bars",
        {"timeframe": "1d", "price_mode": "adjusted"},
        True,
    ),
    Workload(
        "equity_1m_raw",
        "/v1/equity/SPY/bars",
        {"timeframe": "1m", "price_mode": "raw"},
        True,
    ),
    Workload(
        "equity_1m_adjusted",
        "/v1/equity/SPY/bars",
        {"timeframe": "1m", "price_mode": "adjusted"},
        True,
    ),
    Workload(
        "equity_5m_raw",
        "/v1/equity/AAPL/bars",
        {"timeframe": "5m", "price_mode": "raw"},
        True,
    ),
    Workload(
        "equity_30m_raw",
        "/v1/equity/AAPL/bars",
        {"timeframe": "30m", "price_mode": "raw"},
        True,
    ),
    Workload(
        "equity_1h_raw",
        "/v1/equity/QQQ/bars",
        {"timeframe": "1h", "price_mode": "raw"},
        True,
    ),
    Workload("volatility_5m", "/v1/volatility/VIX/bars", {"timeframe": "5m"}, True),
    Workload("volatility_1d", "/v1/volatility/VIX/bars", {"timeframe": "1d"}, True),
    Workload("fx_1m", "/v1/fx/EURUSD/bars", {"timeframe": "1m"}, True),
    Workload("fx_1d", "/v1/fx/EURUSD/bars", {"timeframe": "1d"}, True),
    Workload("cmdty_1d", "/v1/cmdty/XAUUSD/bars", {"timeframe": "1d"}, True),
    Workload("futures_1d", "/v1/futures/OJ_202611/bars", {"timeframe": "1d"}, True),
    Workload("rates_series", "/v1/rates/DGS10/series", {}, True),
    Workload(
        "dual_any_1d",
        "/v1/equity/VSCO/bars",
        {"listing": "any", "price_mode": "raw"},
        True,
    ),
    Workload(
        "archive_1d",
        "/v1/equity/FEUL/bars",
        {"listing": "delisted", "price_mode": "raw"},
        True,
    ),
    Workload("bulk_50_1d", "/v1/equity/bars", {"symbols": _bulk_symbols(), "limit": 50}, True),
    Workload("catalog_search", "/v1/instruments", {"q": "AA", "limit": 200}, True),
    Workload("silver_pin_1d", "/v1/equity/SPY/bars", {"silver_revision": 76}, False),
    Workload(
        "pit_1d",
        "/v1/equity/FSLR/bars",
        {"pit_revision": 1, "start": "2023-01-03T00:00:00Z"},
        False,
    ),
    Workload(
        "coverage_page",
        "/v1/lake/coverage",
        {"asset_class": "equity", "limit": 500},
        False,
    ),
    Workload(
        "gaps_equity_1d",
        "/v1/equity/SPY/gaps",
        {"start": "2025-09-01", "end": "2026-09-18"},
        False,
    ),
    Workload(
        "gaps_equity_5m",
        "/v1/equity/AAPL/gaps",
        {"timeframe": "5m", "start": "2026-08-01", "end": "2026-09-18"},
        False,
    ),
    Workload("lake_status", "/v1/lake/status", {}, False),
]


def _normalized_hash(body: Any) -> str:
    """Hash of the substantive payload: generated_at and the time-anchored default
    window are dropped, bar rows are kept as time/OHLCV so the baseline (which lacks
    the additive provenance fields) is comparable."""

    def scrub(value: Any) -> Any:
        if isinstance(value, dict):
            return {
                k: scrub(v)
                for k, v in sorted(value.items())
                if k
                not in (
                    "generated_at",
                    "window",
                    "provenance",
                    "truncated",
                    "source_price_basis",
                    "silver_revision",
                )
            }
        if isinstance(value, list):
            return [scrub(v) for v in value]
        return value

    return hashlib.sha256(json.dumps(scrub(body), sort_keys=True, default=str).encode()).hexdigest()


def _rss_kb(pid: Optional[int]) -> Optional[int]:
    if pid is None:
        return None
    out = subprocess.run(["ps", "-o", "rss=", "-p", str(pid)], capture_output=True, text=True)
    return int(out.stdout.strip()) if out.stdout.strip() else None


def main() -> None:
    parser = argparse.ArgumentParser()
    parser.add_argument("--target", action="append", required=True, help="name=url[=pid]")
    parser.add_argument("--out", type=Path, required=True)
    parser.add_argument("--reps", type=int, default=REPS)
    parser.add_argument("--seed", type=int, default=20260923)
    args = parser.parse_args()
    targets = {}
    for spec in args.target:
        name, url, *pid = spec.split("=", 2)
        targets[name] = (url, int(pid[0]) if pid else None)
    rng = random.Random(args.seed)
    samples: Dict[str, Dict[str, List[float]]] = {}
    first: Dict[str, Dict[str, float]] = {}
    hashes: Dict[str, Dict[str, set]] = {}
    failures: Dict[str, Dict[str, List[str]]] = {}
    peak: Dict[str, int] = {}
    started = time.time()
    with httpx.Client(timeout=120) as client:
        for rep in range(args.reps):
            for workload in WORKLOADS:
                names = [n for n in targets if workload.baseline or n != "base"]
                rng.shuffle(names)
                for name in names:
                    url, _ = targets[name]
                    t0 = time.perf_counter()
                    response = client.get(url + workload.path, params=workload.params)
                    elapsed = (time.perf_counter() - t0) * 1000
                    key = workload.name
                    if response.status_code != 200:
                        failures.setdefault(key, {}).setdefault(name, []).append(
                            f"{response.status_code} {response.text[:200]}"
                        )
                        continue
                    hashes.setdefault(key, {}).setdefault(name, set()).add(
                        _normalized_hash(response.json())
                    )
                    if rep == 0:
                        first.setdefault(key, {})[name] = elapsed
                    else:
                        samples.setdefault(key, {}).setdefault(name, []).append(elapsed)
            for name, (_, pid) in targets.items():
                rss = _rss_kb(pid)
                if rss is not None:
                    peak[name] = max(peak.get(name, 0), rss)
            print(
                f"rep {rep + 1}/{args.reps} done at {time.time() - started:.0f}s",
                flush=True,
            )

    report: Dict[str, Any] = {
        "reps": args.reps,
        "seed": args.seed,
        "peak_rss_kb": peak,
        "workloads": {},
    }
    for workload in WORKLOADS:
        key = workload.name
        entry: Dict[str, Any] = {
            "baseline_comparable": workload.baseline,
            "targets": {},
        }
        for name, values in samples.get(key, {}).items():
            ordered = sorted(values)
            entry["targets"][name] = {
                "first_ms": round(first.get(key, {}).get(name, float("nan")), 2),
                "warm_median_ms": round(statistics.median(ordered), 2),
                "warm_p95_ms": round(ordered[max(0, int(len(ordered) * 0.95) - 1)], 2),
                "n": len(ordered),
                "distinct_hashes": len(hashes.get(key, {}).get(name, ())),
            }
        all_hashes = [h for per in hashes.get(key, {}).values() for h in per]
        entry["results_equal_across_targets"] = len(set(all_hashes)) <= 1
        entry["failures"] = failures.get(key, {})
        report["workloads"][key] = entry
    args.out.mkdir(parents=True, exist_ok=True)
    (args.out / "performance.json").write_text(json.dumps(report, indent=1, default=list))
    for key, entry in report["workloads"].items():
        cells = "  ".join(
            f"{n}: {t['warm_median_ms']:.0f}/{t['warm_p95_ms']:.0f}ms"
            for n, t in entry["targets"].items()
        )
        print(f"{key:22s} equal={entry['results_equal_across_targets']!s:5s} {cells}")


if __name__ == "__main__":
    main()
