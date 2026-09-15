"""Builders for small livewire-shaped membership / security-master parquet logs.

These are labelled **test doubles of log shape**, not observed market data. The two
schemas are copied verbatim from livewire's `clients/index_membership_store.py` and
`clients/security_master.py`, so apex reads a log built here exactly as it reads the
lake. Tickers are real and dates plausible; no row is a market fact, and none is
presented as one. They exist because the checked-in real fixture cannot reach the
cases that matter -- it has zero verified membership events and one master row, so
superseding corrections, a knowledge-gated master and ticker collisions have nothing
to act on there.
"""

from __future__ import annotations

from datetime import datetime, timezone
from pathlib import Path

import pyarrow as pa
import pyarrow.parquet as pq

from src.infrastructure.adapters.livewire.membership import MembershipReader

MEMBERSHIP_SCHEMA = pa.schema(
    [
        pa.field("event_id", pa.string(), nullable=False),
        pa.field("index_id", pa.string(), nullable=False),
        pa.field("security_id", pa.string(), nullable=False),
        pa.field("action", pa.string(), nullable=False),
        pa.field("announced_at", pa.timestamp("us", tz="UTC"), nullable=True),
        pa.field("effective_at", pa.timestamp("us", tz="UTC"), nullable=False),
        pa.field("known_at", pa.timestamp("us", tz="UTC"), nullable=False),
        pa.field("source_refs", pa.list_(pa.string()), nullable=False),
        pa.field("source_hashes", pa.list_(pa.string()), nullable=False),
        pa.field("revision", pa.int64(), nullable=False),
        pa.field("supersedes", pa.string(), nullable=True),
        pa.field("status", pa.string(), nullable=False),
    ]
)

MASTER_SCHEMA = pa.schema(
    [
        pa.field("event_id", pa.string(), nullable=False),
        pa.field("security_id", pa.string(), nullable=False),
        pa.field("revision", pa.int64(), nullable=False),
        pa.field("symbol", pa.string(), nullable=False),
        pa.field("provider", pa.string(), nullable=False),
        pa.field("exchange_mic", pa.string(), nullable=False),
        pa.field("currency", pa.string(), nullable=False),
        pa.field("effective_from", pa.timestamp("us", tz="UTC"), nullable=False),
        pa.field("effective_to", pa.timestamp("us", tz="UTC"), nullable=True),
        pa.field("known_at", pa.timestamp("us", tz="UTC"), nullable=False),
        pa.field("issuer_name", pa.string(), nullable=False),
        pa.field("cik", pa.string(), nullable=True),
        pa.field("composite_figi", pa.string(), nullable=True),
        pa.field("share_class_figi", pa.string(), nullable=True),
        pa.field("continuity_basis", pa.string(), nullable=False),
        pa.field("relationship_type", pa.string(), nullable=True),
        pa.field("related_security_id", pa.string(), nullable=True),
        pa.field("source_refs", pa.list_(pa.string()), nullable=False),
        pa.field("source_hashes", pa.list_(pa.string()), nullable=False),
        pa.field("status", pa.string(), nullable=False),
        pa.field("supersedes", pa.string(), nullable=True),
    ]
)

HASH = "b" * 64
REFS = [f"artifact://sha256/{HASH}"]
HASHES = [HASH]

AAPL_SID = "sec_" + "aa11" * 8
MSFT_SID = "sec_" + "cc22" * 8


def _dt(value: str) -> datetime:
    return datetime.fromisoformat(value).replace(tzinfo=timezone.utc)


def _membership_row(
    event_id: str,
    security_id: str,
    action: str,
    effective_at: str,
    known_at: str,
    *,
    index_id: str = "sp500",
    revision: int = 1,
    supersedes: str | None = None,
    status: str = "verified",
) -> dict[str, object]:
    return {
        "event_id": event_id,
        "index_id": index_id,
        "security_id": security_id,
        "action": action,
        "announced_at": None,
        "effective_at": _dt(effective_at),
        "known_at": _dt(known_at),
        "source_refs": REFS,
        "source_hashes": HASHES,
        "revision": revision,
        "supersedes": supersedes,
        "status": status,
    }


def _identity_row(
    event_id: str,
    security_id: str,
    symbol: str,
    known_at: str,
    *,
    effective_from: str = "2000-01-01",
    effective_to: str | None = None,
    exchange_mic: str = "XNAS",
    revision: int = 1,
    supersedes: str | None = None,
    status: str = "verified",
) -> dict[str, object]:
    return {
        "event_id": event_id,
        "security_id": security_id,
        "revision": revision,
        "symbol": symbol,
        "provider": "massive",
        "exchange_mic": exchange_mic,
        "currency": "USD",
        "effective_from": _dt(effective_from),
        "effective_to": None if effective_to is None else _dt(effective_to),
        "known_at": _dt(known_at),
        "issuer_name": f"{symbol} issuer",
        "cik": "0000000001",
        "composite_figi": None,
        "share_class_figi": None,
        "continuity_basis": "provider_figi",
        "relationship_type": None,
        "related_security_id": None,
        "source_refs": REFS,
        "source_hashes": HASHES,
        "status": status,
        "supersedes": supersedes,
    }


def _shaped_lake(
    tmp_path: Path,
    *,
    events: list[dict[str, object]],
    identities: list[dict[str, object]],
    index_id: str = "sp500",
) -> MembershipReader:
    events_path = tmp_path / "index_membership" / index_id / "events.parquet"
    events_path.parent.mkdir(parents=True, exist_ok=True)
    pq.write_table(pa.Table.from_pylist(events, schema=MEMBERSHIP_SCHEMA), events_path)
    master_path = tmp_path / "security_master" / "events.parquet"
    master_path.parent.mkdir(parents=True, exist_ok=True)
    pq.write_table(pa.Table.from_pylist(identities, schema=MASTER_SCHEMA), master_path)
    return MembershipReader(tmp_path)
