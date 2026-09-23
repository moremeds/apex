"""Independent, read-only access to the Livewire lake for the verification oracle.

Deliberately imports nothing from ``src``: the oracle re-derives paths, manifests and
rows from Livewire's published contracts so a bug in the candidate cannot also be in
its check. Everything here reads; nothing writes.
"""

from __future__ import annotations

import json
import os
from dataclasses import dataclass
from datetime import date, datetime, timezone
from functools import lru_cache
from pathlib import Path
from typing import Any, Dict, List, Optional, Tuple

import duckdb

_CASE_SAFE = frozenset("ABCDEFGHIJKLMNOPQRSTUVWXYZ0123456789._-")
LADDERS = {
    "equity": ("1m", "5m", "30m", "1h", "1d"),
    "volatility": ("5m", "30m", "1h", "1d"),
    "fx": ("1m", "5m", "30m", "1h", "1d"),
    "cmdty": ("1d",),
    "futures": ("1d",),
    "rates": ("1d",),
}


def encode(symbol: str) -> str:
    """Livewire clients/symbol_paths.py: keep [A-Z0-9._-], percent-encode the rest."""
    return "".join(
        ch if ch in _CASE_SAFE else "".join(f"%{b:02X}" for b in ch.encode("utf-8"))
        for ch in symbol
    )


@dataclass(frozen=True)
class Lake:
    root: Path

    @classmethod
    def from_env(cls) -> "Lake":
        return cls(Path(os.environ["APEX_VERIFY_LAKE_ROOT"]))

    @property
    def silver(self) -> Path:
        return self.root / "silver"

    def bronze(self, asset_class: str, symbol: str, timeframe: str) -> Path:
        return (
            self.root
            / "bronze"
            / f"asset_class={asset_class}"
            / f"symbol={encode(symbol)}"
            / f"{timeframe}.parquet"
        )

    def archive(self, asset_class: str, symbol: str, timeframe: str) -> Path:
        return (
            self.root
            / "bronze-delisted"
            / f"asset_class={asset_class}"
            / f"symbol={encode(symbol)}"
            / f"{timeframe}.parquet"
        )

    # -- Silver manifests (revisions/revision={n}.json; current.json is a byte copy)

    def silver_current_number(self) -> int:
        return int(_json(self.silver / "revisions" / "current.json")["revision"])

    def silver_artifact(self, revision: int, symbol: str, kind: str) -> Optional[Path]:
        relative = _silver_index(self.silver, revision).get((encode(symbol), kind))
        return None if relative is None else self.silver / relative

    # -- PIT manifests (pit-revisions/revision={n}.json)

    def pit_numbers(self) -> List[int]:
        directory = self.silver / "pit-revisions"
        if not directory.is_dir():
            return []
        out = []
        for name in os.listdir(directory):
            if name.startswith("revision=") and name.endswith(".json"):
                out.append(int(name[len("revision=") : -len(".json")]))
        return sorted(out, reverse=True)

    def pit(self, revision: int) -> Dict[str, Any]:
        return _json(self.silver / "pit-revisions" / f"revision={revision}.json")


@lru_cache(maxsize=8)
def _json(path: Path) -> Any:
    return json.loads(path.read_bytes())


@lru_cache(maxsize=4)
def _silver_index(silver: Path, revision: int) -> Dict[Tuple[str, str], str]:
    """(encoded symbol, daily|factors) -> Silver-relative path, from the numbered manifest."""
    index: Dict[Tuple[str, str], str] = {}
    for entry in _json(silver / "revisions" / f"revision={revision}.json")["artifacts"]:
        parts = entry["path"].split("/")
        kind = "daily" if parts[-1] == "1d.parquet" else "factors"
        index[(parts[-2][len("symbol=") :], kind)] = entry["path"]
    return index


def identity(path: Path) -> Optional[Tuple[int, int, int, int]]:
    """(device, inode, size, mtime_ns) of a mutable source, or None when absent."""
    try:
        st = path.stat()
    except FileNotFoundError:
        return None
    return (st.st_dev, st.st_ino, st.st_size, st.st_mtime_ns)


def read_rows(
    path: Path, columns: str, where: str = "true", params: Optional[List[Any]] = None
) -> List[Dict[str, Any]]:
    con = duckdb.connect()
    try:
        con.execute("SET TimeZone='UTC'")
        return (
            con.execute(
                f"SELECT {columns} FROM read_parquet(?) WHERE {where}",
                [path.as_posix(), *(params or [])],
            )
            .fetch_arrow_table()
            .to_pylist()
        )
    finally:
        con.close()


def as_utc(value: Any) -> datetime:
    if isinstance(value, datetime):
        return (
            value.astimezone(timezone.utc) if value.tzinfo else value.replace(tzinfo=timezone.utc)
        )
    if isinstance(value, date):
        return datetime(value.year, value.month, value.day, tzinfo=timezone.utc)
    raise TypeError(type(value))
