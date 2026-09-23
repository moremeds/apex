"""Build a PIT Silver manifest over a disposable Silver tree, from frozen real values.

Serialization follows Livewire ``clients/pit_silver_revision.py`` ``_build_core`` at
Livewire 1fad842 (key set and nesting), checked against production
``silver/pit-revisions/revision=1.json`` (sp500, PARTIAL) on 2026-09-23.

Frozen real data (read-only from the mini lake on 2026-09-23):
- FSLR member scopes from PIT revision 1 (sp500): two scopes, one security_id.
- FSLR adjusted daily rows 2026-09-15..2026-09-21 from Silver revision 77
  (``generations/20260922T093620.708794Z-.../symbol=FSLR/1d.parquet``).
- BIIB: scope from 2003-11-13 (the earliest sp500 scope in revision 1), and its Silver
  revision 77 rows for the same sessions.

Manifests this builder writes are test fixtures, not published production facts.
``silver_artifacts`` hashes are computed over the test-written artifacts; the lineage
hashes (input_hash, receipt, membership, security master, silver_manifest) are
repeated-digit placeholders because apex never checks them (design §3.4).
"""

from __future__ import annotations

import copy
import hashlib
import json
from datetime import date
from pathlib import Path
from typing import Any

import pyarrow as pa
import pyarrow.parquet as pq

from src.infrastructure.adapters.livewire.paths import encode_symbol

GENERATION = "generations/20260922T093620.708794Z-108ced3daee8416288907099cf7af6b5"
# Silver revision 76's generation; its FSLR artifact ends 2026-09-18 (published before
# the 09-21 session) and carries adjustment_revision 75 on every row.
GENERATION_R76 = "generations/20260919T075732.226646Z-6d55d99fb9dd47359cb2d36762e1b5af"

# (trade_date, open, high, low, close, volume) -- Silver revision 77, factors 1.0.
FSLR_ROWS = [
    (date(2026, 9, 15), 209.48, 210.3, 202.19, 202.34, 1448020),
    (date(2026, 9, 16), 205.52, 205.7299, 189.4, 191.07, 3096981),
    (date(2026, 9, 17), 193.15, 203.52, 192.72, 201.16, 2660857),
    (date(2026, 9, 18), 202.11, 202.4074, 192.42, 195.96, 2763462),
    (date(2026, 9, 21), 198.22, 201.155, 193.73, 199.84, 1838283),
]
BIIB_ROWS = [
    (date(2026, 9, 15), 216.71, 216.71, 213.54, 216.17, 756048),
    (date(2026, 9, 16), 216.41, 218.29, 214.81, 216.01, 606494),
    (date(2026, 9, 17), 217.68, 219.41, 215.33, 219.05, 836708),
    (date(2026, 9, 18), 219.05, 220.17, 215.26, 215.5, 2636088),
    (date(2026, 9, 21), 216.14, 221.92, 214.59, 219.07, 928543),
]

FSLR_SECURITY = "sec_745989483e9341fe9506d29edebc801d"
BIIB_SECURITY = "sec_93490e5e8936417c83edd874afdacc3f"

FSLR_SCOPES: list[dict[str, Any]] = [
    {
        "effective_from": "2009-10-16T00:00:00+00:00",
        "effective_to": "2017-03-20T00:00:00+00:00",
        "identity_event_id": "e3368571008431264f00b6063619f18f1d418e051a08c7a4b9b0959676bcdf8e",
        "membership_event_id": "df4a4786f0c689712e0d39cf8b61d54331fb8b2eb6f89cf5351f06bdb5b57171",
        "security_id": FSLR_SECURITY,
        "session_from": "2009-10-16",
        "session_to": "2017-03-20",
        "symbol": "FSLR",
    },
    {
        "effective_from": "2022-12-19T00:00:00+00:00",
        "effective_to": None,
        "identity_event_id": "e3368571008431264f00b6063619f18f1d418e051a08c7a4b9b0959676bcdf8e",
        "membership_event_id": "cb42aa78db755f39931586eb9fe4c89bcab035082e19c4747e0e24a3b29c59e6",
        "security_id": FSLR_SECURITY,
        "session_from": "2022-12-19",
        "session_to": None,
        "symbol": "FSLR",
    },
]
BIIB_SCOPE: dict[str, Any] = {
    "effective_from": "2003-11-13T00:00:00+00:00",
    "effective_to": None,
    "identity_event_id": "2c9b5018f0e74f46673c0e9c4610b64fe6d5fde29a7e6dcda1517f1a29cfba3d",
    "membership_event_id": "ac7e30eba255b5b39d2c2d506798f231bfe226a8e07b9aeebb298b0453f0c2d7",
    "security_id": BIIB_SECURITY,
    "session_from": "2003-11-13",
    "session_to": None,
    "symbol": "BIIB",
}


def write_daily(
    silver_root: Path,
    symbol: str,
    rows: list[tuple],
    generation: str = GENERATION,
    adjustment_revision: int = 77,
) -> str:
    """Write a Silver-shaped daily artifact; return its Silver-relative path."""
    relative = f"{generation}/asset_class=equity/symbol={encode_symbol(symbol)}/1d.parquet"
    path = silver_root / relative
    path.parent.mkdir(parents=True, exist_ok=True)
    pq.write_table(
        pa.Table.from_pylist(
            [
                {
                    "trade_date": d,
                    "open": o,
                    "high": h,
                    "low": lo,
                    "close": c,
                    "adj_close": c,
                    "volume": v,
                    "price_adjustment_factor": 1.0,
                    "split_volume_factor": 1.0,
                    "adjustment_revision": adjustment_revision,
                    "asset_class": "equity",
                    "symbol": symbol,
                }
                for d, o, h, lo, c, v in rows
            ]
        ),
        path,
    )
    return relative


def pit_payload(
    silver_root: Path,
    revision: int = 1,
    *,
    index_id: str = "sp500",
    status: str = "PARTIAL",
    members: list[dict[str, Any]] | None = None,
    artifacts: dict[str, str] | None = None,
) -> dict[str, Any]:
    """A manifest dict; ``artifacts`` maps symbol -> Silver-relative daily path."""
    if artifacts is None:
        artifacts = {
            "FSLR": write_daily(silver_root, "FSLR", FSLR_ROWS),
            "BIIB": write_daily(silver_root, "BIIB", BIIB_ROWS),
        }
    entries = [
        {
            "path": rel,
            "sha256": hashlib.sha256((silver_root / rel).read_bytes()).hexdigest(),
        }
        for rel in artifacts.values()
    ]
    return {
        "as_of": "2026-09-23T00:00:00+00:00",
        "corporate_actions_as_of": "2026-09-22T08:37:03.890538+00:00",
        "daily_bar_cutoff": "2026-09-22",
        "generation_id": "20260923T042544Z-1",
        "index_id": index_id,
        "input_hash": "sha256:" + "0" * 64,
        "inputs": {
            "corporate_action_receipt": {
                "path": "pit-revisions/evidence/actions-fixture.json",
                "receipt_hash": "sha256:" + "1" * 64,
                "sha256": "1" * 64,
            },
            "corporate_action_receipt_hash": "sha256:" + "1" * 64,
            "membership": {
                "path": f"index_membership/{index_id}/events.parquet",
                "revision": 4896,
                "revision_semantics": "append-order-prefix",
                "sha256": "2" * 64,
            },
            "security_master": {
                "path": "security_master/events.parquet",
                "revision": 3579,
                "revision_semantics": "append-order-prefix",
                "sha256": "3" * 64,
            },
            "silver_artifacts": entries,
            "silver_manifest": {
                "path": "revisions/revision=77.json",
                "sha256": "4" * 64,
            },
        },
        # Deep copies: callers mutate payloads, and the frozen scopes are module-level.
        "members": copy.deepcopy(members if members is not None else [*FSLR_SCOPES, BIIB_SCOPE]),
        "membership_revision": 4896,
        "policy_version": "pit-silver-v1",
        "published_at": "2026-09-23T04:25:44.553025+00:00",
        "revision": revision,
        "schema_version": 1,
        "session_policy": "XNYS-close-and-early-close-v2",
        "silver_revision": 77,
        "status": status,
    }


def publish_pit(silver_root: Path, payload: dict[str, Any], revision: int | None = None) -> Path:
    """Write ``revision={n}.json`` and mirror it to the shared ``current.json``."""
    number = payload["revision"] if revision is None else revision
    directory = silver_root / "pit-revisions"
    directory.mkdir(parents=True, exist_ok=True)
    encoded = json.dumps(payload, sort_keys=True).encode()
    path = directory / f"revision={number}.json"
    path.write_bytes(encoded)
    (directory / "current.json").write_bytes(encoded)
    return path
