"""Commit the actual Silver manifest contract over disposable test artifacts."""

from __future__ import annotations

import hashlib
import json
import os
from datetime import date, datetime, timezone
from pathlib import Path
from urllib.parse import unquote

import pyarrow as pa
import pyarrow.parquet as pq

from src.infrastructure.adapters.livewire.paths import encode_symbol


def write_generation(root: Path, attempt: str, value: float, symbol: str = "TEST") -> list[Path]:
    """A daily/factor pair whose values identify its immutable attempt."""
    generation = root / "generations" / attempt
    partition = Path("asset_class=equity") / f"symbol={encode_symbol(symbol)}"
    daily = generation / partition / "1d.parquet"
    factors = generation / "adjustments" / partition / "factors.parquet"
    daily.parent.mkdir(parents=True, exist_ok=True)
    factors.parent.mkdir(parents=True, exist_ok=True)
    pq.write_table(
        pa.Table.from_pylist(
            [
                {
                    "trade_date": date(2026, 1, 2),
                    "open": value,
                    "high": value + 1,
                    "low": value - 1,
                    "close": value,
                    "volume": 100,
                }
            ]
        ),
        daily,
    )
    pq.write_table(
        pa.Table.from_pylist(
            [
                {
                    "effective_start": date(2026, 1, 2),
                    "effective_end": date(2026, 1, 2),
                    "price_adjustment_factor": value / 10,
                    "split_volume_factor": 1.0,
                    "adjustment_revision": 1,
                }
            ]
        ),
        factors,
    )
    return [daily, factors]


def write_bronze_intraday(root: Path, symbol: str = "TEST") -> None:
    path = root / "asset_class=equity" / f"symbol={encode_symbol(symbol)}" / "1m.parquet"
    path.parent.mkdir(parents=True, exist_ok=True)
    pq.write_table(
        pa.Table.from_pylist(
            [
                {
                    "bar_timestamp": datetime(2026, 1, 2, 14, 30, tzinfo=timezone.utc),
                    "open": 10.0,
                    "high": 11.0,
                    "low": 9.0,
                    "close": 10.0,
                    "volume": 100,
                }
            ]
        ),
        path,
    )


def publish_manifest(root: Path, revision: int = 1, artifacts: list[Path] | None = None) -> None:
    paths = list(sorted(root.rglob("*.parquet")) if artifacts is None else artifacts)
    # Most adapter tests exercise only one half of the pair. Add an inert peer so
    # their manifests still satisfy the real schema-1 membership contract.
    by_symbol: dict[str, dict[str, Path]] = {}
    for path in paths:
        symbol = unquote(path.parent.name.removeprefix("symbol="))
        kind = "factors" if path.name == "factors.parquet" else "daily"
        by_symbol.setdefault(symbol, {})[kind] = path
    for symbol, pair in by_symbol.items():
        exemplar = next(iter(pair.values()))
        relative = exemplar.relative_to(root)
        parts = list(relative.parts)
        asset_index = parts.index("asset_class=equity")
        prefix = parts[:asset_index]
        if prefix and prefix[-1] == "adjustments":
            prefix.pop()
        partition = Path("asset_class=equity") / f"symbol={encode_symbol(symbol)}"
        if "daily" not in pair:
            daily = root.joinpath(*prefix) / partition / "1d.parquet"
            daily.parent.mkdir(parents=True, exist_ok=True)
            pq.write_table(
                pa.Table.from_pylist(
                    [
                        {
                            "trade_date": date(2026, 1, 2),
                            "open": 1.0,
                            "high": 1.0,
                            "low": 1.0,
                            "close": 1.0,
                            "volume": 1,
                        }
                    ]
                ),
                daily,
            )
            paths.append(daily)
        if "factors" not in pair:
            factors = root.joinpath(*prefix) / "adjustments" / partition / "factors.parquet"
            factors.parent.mkdir(parents=True, exist_ok=True)
            pq.write_table(
                pa.Table.from_pylist(
                    [
                        {
                            "effective_start": date(2026, 1, 2),
                            "effective_end": None,
                            "price_adjustment_factor": 1.0,
                            "split_volume_factor": 1.0,
                            "adjustment_revision": revision,
                        }
                    ]
                ),
                factors,
            )
            paths.append(factors)
    symbols = sorted({unquote(path.parent.name.removeprefix("symbol=")) for path in paths})
    payload = {
        "schema_version": 1,
        "revision": revision,
        "generation_id": f"test-{revision}",
        "published_at": "2026-01-31T00:00:00Z",
        "corporate_actions_as_of": "2026-01-31T00:00:00Z",
        "affected": [
            {"symbol": symbol, "earliest_date": "2026-01-01", "timeframes": ["1d", "1m"]}
            for symbol in symbols
        ],
        "artifacts": [
            {
                "path": path.relative_to(root).as_posix(),
                "sha256": hashlib.sha256(path.read_bytes()).hexdigest(),
            }
            for path in paths
        ],
    }
    revisions = root / "revisions"
    revisions.mkdir(parents=True, exist_ok=True)
    encoded = json.dumps(payload).encode()
    (revisions / f"revision={revision}.json").write_bytes(encoded)
    temporary = revisions / "current.tmp"
    temporary.write_bytes(encoded)
    os.replace(temporary, revisions / "current.json")
