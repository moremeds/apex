"""Controlled cases (plan §4.3): corruption, replacement, unset sources and concurrency,
run against an isolated SCRATCH copy of real lake files, never the production lake.

    python scripts/lake_verify/controlled.py --lake /path/to/lake --out RUN_DIR

Builds ``RUN_DIR/scratch`` from real bytes (a handful of symbols' Bronze/archive files,
their Silver artifacts, and subset manifests written in Livewire's schema), starts
lake-only candidate servers (``serve.py``) against it on loopback, mutates the scratch
copy scenario by scenario, and records every result in ``RUN_DIR/controlled.jsonl``
labelled CONTROLLED_CASE. A subset manifest is a test construction over real artifacts,
not a published revision; that is what the label says.
"""

from __future__ import annotations

import argparse
import asyncio
import hashlib
import json
import os
import shutil
import socket
import subprocess
import sys
import time
from pathlib import Path
from typing import Any, Callable, Dict, List, Optional, Tuple

import httpx

HERE = Path(__file__).resolve().parent
SYMBOLS_1D = ("SPY", "AAPL", "FSLR", "BIIB")
DUAL = "VSCO"


def _free_port() -> int:
    with socket.socket() as probe:
        probe.bind(("127.0.0.1", 0))
        return probe.getsockname()[1]


def _sha(path: Path) -> str:
    return hashlib.sha256(path.read_bytes()).hexdigest()


def _copy(src: Path, dst: Path) -> None:
    dst.parent.mkdir(parents=True, exist_ok=True)
    shutil.copy2(src, dst)


# -- scratch lake -------------------------------------------------------------------


def build_scratch(lake: Path, scratch: Path) -> Dict[str, Any]:
    """Copy real files for a few symbols and write subset manifests over them."""
    if scratch.exists():
        shutil.rmtree(scratch)
    bronze = scratch / "bronze" / "asset_class=equity"
    for symbol in (*SYMBOLS_1D, DUAL):
        for tf in ("1d", "5m"):
            src = lake / "bronze" / "asset_class=equity" / f"symbol={symbol}" / f"{tf}.parquet"
            if src.exists():
                _copy(src, bronze / f"symbol={symbol}" / f"{tf}.parquet")
    for tf in ("1d",):
        src = lake / "bronze-delisted" / "asset_class=equity" / f"symbol={DUAL}" / f"{tf}.parquet"
        _copy(
            src,
            scratch / "bronze-delisted" / "asset_class=equity" / f"symbol={DUAL}" / f"{tf}.parquet",
        )
    silver = lake / "silver"
    current = json.loads((silver / "revisions" / "current.json").read_bytes())
    wanted = {s for s in SYMBOLS_1D}
    artifacts, affected = [], []
    for entry in current["artifacts"]:
        symbol = entry["path"].split("/")[-2][len("symbol=") :]
        if symbol in wanted:
            _copy(silver / entry["path"], scratch / "silver" / entry["path"])
            artifacts.append(entry)
    for item in current["affected"]:
        if item["symbol"] in wanted:
            affected.append(item)
    subset = {**current, "affected": affected, "artifacts": artifacts}
    _publish_silver(scratch, subset)
    pit_numbers = sorted(
        (int(n[len("revision=") : -5]) for n in os.listdir(silver / "pit-revisions") if n.startswith("revision=") and n.endswith(".json")),
        reverse=True,
    )  # fmt: skip
    pit_source = next(json.loads((silver / "pit-revisions" / f"revision={n}.json").read_bytes()) for n in pit_numbers if json.loads((silver / "pit-revisions" / f"revision={n}.json").read_bytes())["index_id"] == "sp500")  # fmt: skip
    pit = dict(pit_source)
    pit["members"] = [m for m in pit_source["members"] if m["symbol"] in wanted]
    pit["inputs"] = {
        **pit_source["inputs"],
        "silver_artifacts": [a for a in pit_source["inputs"]["silver_artifacts"] if a["path"].split("/")[-2][len("symbol=") :] in wanted],
    }  # fmt: skip
    for entry in pit["inputs"]["silver_artifacts"]:
        target = scratch / "silver" / entry["path"]
        if not target.exists():
            _copy(silver / entry["path"], target)
    pit["revision"] = 1
    _publish_pit(scratch, pit)
    catalog = lake / "catalog" / "analytics.duckdb"
    _copy(catalog, scratch / "catalog" / "analytics.duckdb")
    (scratch / "repairs").mkdir(parents=True)
    for name in sorted(os.listdir(lake / "repairs")):
        if name.startswith(("tier_a_", "decisions_")) and name.endswith(".json"):
            _copy(lake / "repairs" / name, scratch / "repairs" / name)
    return {"silver_revision": current["revision"], "pit_revision": 1}


def _publish_silver(scratch: Path, payload: Dict[str, Any]) -> None:
    revisions = scratch / "silver" / "revisions"
    revisions.mkdir(parents=True, exist_ok=True)
    raw = json.dumps(payload, sort_keys=True).encode()
    (revisions / f"revision={payload['revision']}.json").write_bytes(raw)
    tmp = revisions / "current.tmp"
    tmp.write_bytes(raw)
    os.replace(tmp, revisions / "current.json")


def _publish_pit(scratch: Path, payload: Dict[str, Any]) -> Path:
    directory = scratch / "silver" / "pit-revisions"
    directory.mkdir(parents=True, exist_ok=True)
    path = directory / f"revision={payload['revision']}.json"
    path.write_bytes(json.dumps(payload, sort_keys=True).encode())
    return path


# -- servers ------------------------------------------------------------------------


class Server:
    def __init__(self, tree: Path, python: str, env: Dict[str, str], log: Path) -> None:
        self.port = _free_port()
        full = {
            "HOME": os.environ.get("HOME", ""),
            "PATH": os.environ.get("PATH", ""),
            **env,
        }
        self.proc = subprocess.Popen(
            [
                python,
                str(tree / "scripts" / "lake_verify" / "serve.py"),
                "--port",
                str(self.port),
            ],
            cwd=tree,
            env=full,
            stdout=log.open("a"),
            stderr=subprocess.STDOUT,
        )
        self.url = f"http://127.0.0.1:{self.port}"
        deadline = time.time() + 120
        while time.time() < deadline:
            try:
                httpx.get(self.url + "/v1/lake/asset-classes", timeout=2)
                return
            except httpx.HTTPError:
                time.sleep(1)
        raise RuntimeError("scratch server did not start")

    def stop(self) -> None:
        self.proc.terminate()
        self.proc.wait(timeout=30)


def lake_env(scratch: Path, **overrides: Optional[str]) -> Dict[str, str]:
    env = {
        "APEX_LIVEWIRE_ROOT": str(scratch / "bronze"),
        "APEX_LIVEWIRE_SILVER_ROOT": str(scratch / "silver"),
        "APEX_LIVEWIRE_DELISTED_ROOT": str(scratch / "bronze-delisted"),
        "APEX_LIVEWIRE_COVERAGE_DB": str(scratch / "catalog" / "analytics.duckdb"),
        "APEX_LIVEWIRE_REPAIRS_ROOT": str(scratch / "repairs"),
        "APEX_LIVEWIRE_PRICE_MODE": "adjusted",
    }
    for key, value in overrides.items():
        if value is None:
            env.pop(key, None)
        else:
            env[key] = value
    return env


# -- scenarios ----------------------------------------------------------------------


Check = Callable[[httpx.Client], Tuple[bool, str]]


def expect(
    path: str,
    status: int,
    code: Optional[str] = None,
    params: Optional[Dict[str, Any]] = None,
    predicate: Optional[Callable[[Any], bool]] = None,
) -> Check:
    def check(client: httpx.Client) -> Tuple[bool, str]:
        response = client.get(path, params=params or {})
        body = response.json()
        got_code = body.get("error", {}).get("code") if isinstance(body, dict) else None
        ok = response.status_code == status and (code is None or got_code == code)
        if ok and predicate is not None:
            ok = bool(predicate(body))
        return (
            ok,
            f"{path} {params or ''} -> {response.status_code} {got_code or ''} {str(body)[:200] if not ok else ''}",
        )

    return check


def run_all(lake: Path, out: Path, tree: Path, python: str) -> List[Dict[str, Any]]:
    scratch = out / "scratch"
    results: List[Dict[str, Any]] = []
    log = out / "controlled-servers.log"

    def record(name: str, ok: bool, detail: str) -> None:
        results.append(
            {
                "id": f"controlled:{name}",
                "label": "CONTROLLED_CASE",
                "status": "PASS" if ok else "FAIL",
                "detail": detail,
            }
        )
        print(
            ("PASS " if ok else "FAIL ") + name + ("" if ok else f"  {detail}"),
            flush=True,
        )

    def scenario(
        name: str,
        env_overrides: Dict[str, Optional[str]],
        setup: Callable[[], None],
        checks: List[Check],
    ) -> None:
        meta = build_scratch(lake, scratch)
        setup_ctx.update(meta)
        setup()
        server = Server(tree, python, lake_env(scratch, **env_overrides), log)
        try:
            with httpx.Client(base_url=server.url, timeout=120) as client:
                for index, check in enumerate(checks):
                    ok, detail = check(client)
                    record(f"{name}#{index}", ok, detail)
        finally:
            server.stop()

    setup_ctx: Dict[str, Any] = {}
    silver = scratch / "silver"

    def silver_manifest() -> Dict[str, Any]:
        return json.loads((silver / "revisions" / "current.json").read_bytes())

    def daily_path(symbol: str) -> Path:
        entry = next(
            a
            for a in silver_manifest()["artifacts"]
            if a["path"].endswith(f"symbol={symbol}/1d.parquet")
        )
        return silver / entry["path"]

    # 1. baseline scratch reads work
    scenario("baseline", {}, lambda: None, [
        expect("/v1/equity/SPY/bars", 200, predicate=lambda b: b["price_mode"] == "adjusted" and b["count"] > 0),
        expect("/v1/equity/FSLR/bars", 200, params={"pit_revision": 1, "start": "2023-01-03T00:00:00Z"}, predicate=lambda b: b["provenance"]["pit"]["publisher_status"] in ("PROVEN", "PARTIAL")),
        expect("/v1/lake/status", 200, predicate=lambda b: b["sources"]["silver"]["available"]),
    ])  # fmt: skip
    # 2. corrupt current manifest: adjusted fails closed, raw unaffected, list reports it
    scenario("corrupt_current_manifest", {}, lambda: (silver / "revisions" / "current.json").write_bytes(b'{"schema_version": 1, "revision'), [
        expect("/v1/equity/SPY/bars", 503, "adjusted_unavailable"),
        expect("/v1/equity/SPY/bars", 200, params={"price_mode": "raw"}),
        expect("/v1/lake/silver-revisions", 200, predicate=lambda b: b["current"] is None and b["current_error"]),
    ])  # fmt: skip
    # 3. artifact hash mismatch and missing artifact
    scenario("artifact_hash_mismatch", {}, lambda: daily_path("AAPL").write_bytes(daily_path("AAPL").read_bytes() + b"\0"), [
        expect("/v1/equity/AAPL/bars", 503, "adjusted_unavailable"),
        expect("/v1/equity/SPY/bars", 200),
    ])  # fmt: skip
    scenario("artifact_missing", {}, lambda: daily_path("AAPL").unlink(), [
        expect("/v1/equity/AAPL/bars", 503, "adjusted_unavailable"),
    ])  # fmt: skip

    # 4. escaped path and symlink escape in the Silver manifest
    def escape_path() -> None:
        manifest = silver_manifest()
        manifest["artifacts"][0]["path"] = "../outside/asset_class=equity/symbol=SPY/1d.parquet"
        _publish_silver(scratch, manifest)

    scenario(
        "manifest_escaped_path",
        {},
        escape_path,
        [expect("/v1/equity/SPY/bars", 503, "adjusted_unavailable")],
    )

    def symlink_escape() -> None:
        target = daily_path("SPY")
        outside = scratch / "outside.parquet"
        shutil.copy2(target, outside)
        target.unlink()
        target.symlink_to(outside)

    scenario(
        "artifact_symlink_escape",
        {},
        symlink_escape,
        [expect("/v1/equity/SPY/bars", 503, "adjusted_unavailable")],
    )

    # 5. PIT: malformed, hash mismatch, evicted, two security ids, unknown
    pit_path = silver / "pit-revisions" / "revision=1.json"
    scenario("pit_malformed", {}, lambda: pit_path.write_bytes(b"{"), [
        expect("/v1/equity/FSLR/bars", 503, "pit_unavailable", params={"pit_revision": 1}),
        expect("/v1/lake/pit-revisions", 503, "pit_unavailable"),
    ])  # fmt: skip

    def pit_artifact(symbol: str) -> Path:
        manifest = json.loads(pit_path.read_bytes())
        entry = next(
            a
            for a in manifest["inputs"]["silver_artifacts"]
            if a["path"].endswith(f"symbol={symbol}/1d.parquet")
        )
        return silver / entry["path"]

    scenario("pit_hash_mismatch", {}, lambda: pit_artifact("BIIB").write_bytes(pit_artifact("BIIB").read_bytes() + b"\0"), [
        expect("/v1/equity/BIIB/bars", 503, "pit_unavailable", params={"pit_revision": 1}),
        expect("/v1/equity/FSLR/bars", 200, params={"pit_revision": 1, "start": "2023-01-03T00:00:00Z"}),
    ])  # fmt: skip
    scenario("pit_evicted_artifact", {}, lambda: pit_artifact("BIIB").unlink(), [
        expect("/v1/equity/BIIB/bars", 503, "pit_unavailable", params={"pit_revision": 1}),
    ])  # fmt: skip

    def two_ids() -> None:
        manifest = json.loads(pit_path.read_bytes())
        spells = [m for m in manifest["members"] if m["symbol"] == "FSLR"]
        if len(spells) >= 2:
            spells[0]["security_id"] = "sec_controlled_other_issuer"
        pit_path.write_bytes(json.dumps(manifest, sort_keys=True).encode())

    scenario("pit_two_security_ids", {}, two_ids, [
        expect("/v1/equity/FSLR/bars", 409, "ambiguous_symbol", params={"pit_revision": 1, "start": "2016-01-04T00:00:00Z", "end": "2026-09-18T00:00:00Z"}),
    ])  # fmt: skip

    # 6. argument boundaries (need no mutation)
    scenario("argument_boundaries", {}, lambda: None, [
        expect("/v1/equity/SPY/bars", 400, "invalid_parameter", params={"start": "2026-09-01T00:00:00"}),
        expect("/v1/equity/SPY/bars", 400, "invalid_parameter", params={"start": "2026-09-10T00:00:00Z", "end": "2026-09-01T00:00:00Z"}),
        expect("/v1/equity/SPY/bars", 400, "invalid_parameter", params={"silver_revision": 1, "price_mode": "raw"}),
        expect("/v1/equity/SPY/bars", 404, "unknown_revision", params={"silver_revision": 99999}),
        expect("/v1/equity/SPY/bars", 400, "unsupported_timeframe", params={"timeframe": "4h"}),
        expect("/v1/equity/SPY/bars", 422, "invalid_parameter", params={"limit": "many"}),
        expect("/v1/equity/bars", 400, "invalid_parameter", params={"symbols": ",".join(f"S{i}" for i in range(201))}),
        expect("/v1/equity/bars", 200, params={"symbols": "SPY,NOPE_XYZ"}, predicate=lambda b: "SPY" in b["symbols"] and "NOPE_XYZ" in b["missing"]),
        expect("/v1/equity/NOPE_XYZ/bars", 404, "unknown_symbol"),
    ])  # fmt: skip

    # 7. unset sources degrade per source, never crash
    scenario("unset_silver", {"APEX_LIVEWIRE_SILVER_ROOT": None, "APEX_LIVEWIRE_PRICE_MODE": "raw"}, lambda: None, [
        expect("/v1/equity/SPY/bars", 200),
        expect("/v1/lake/silver-revisions", 503, "provider_not_configured"),
        expect("/v1/lake/pit-revisions", 503, "provider_not_configured"),
    ])  # fmt: skip
    scenario("unset_catalog_and_repairs", {"APEX_LIVEWIRE_COVERAGE_DB": None, "APEX_LIVEWIRE_REPAIRS_ROOT": None}, lambda: None, [
        expect("/v1/lake/coverage", 503, "provider_not_configured"),
        expect("/v1/equity/SPY/gaps", 200, params={"start": "2026-09-01", "end": "2026-09-18"}, predicate=lambda b: b["repairs"]["state"] == "not_configured"),
        expect("/v1/lake/status", 200, predicate=lambda b: b["sources"]["catalog"]["configured"] is False),
    ])  # fmt: skip
    scenario("unset_lake_root", {}, lambda: None, [
        expect("/v1/security/SPY", 503, "provider_not_configured"),
        expect("/v1/membership/indices", 503, "provider_not_configured"),
    ])  # fmt: skip
    # 8. partial repairs
    scenario("repairs_degraded", {}, lambda: (scratch / "repairs" / "decisions_2026-09-30.json").write_bytes(b"{truncated"), [
        expect("/v1/equity/SPY/gaps", 200, params={"start": "2026-09-01", "end": "2026-09-18"}, predicate=lambda b: b["repairs"]["state"] == "degraded" and b["repairs"]["warnings"]),
    ])  # fmt: skip

    # 9. replacement with identical size and mtime is still seen (content-keyed cache)
    build_scratch(lake, scratch)
    server = Server(tree, python, lake_env(scratch), log)
    try:
        with httpx.Client(base_url=server.url, timeout=120) as client:
            first = client.get("/v1/lake/silver-revisions").json()["current"]
            current = silver / "revisions" / "current.json"
            stat = current.stat()
            manifest = json.loads(current.read_bytes())
            manifest["generation_id"] = manifest["generation_id"][:-1] + (
                "x" if manifest["generation_id"][-1] != "x" else "y"
            )
            raw = json.dumps(manifest, sort_keys=True).encode()
            ok_size = len(raw) == stat.st_size
            (silver / "revisions" / f"revision={manifest['revision']}.json").write_bytes(raw)
            tmp = current.with_suffix(".tmp")
            tmp.write_bytes(raw)
            os.utime(tmp, ns=(stat.st_atime_ns, stat.st_mtime_ns))
            os.replace(tmp, current)
            detail = client.get(f"/v1/lake/silver-revisions/{first}").json()
            record(
                "cache_same_size_mtime",
                ok_size and detail["generation_id"] == manifest["generation_id"],
                f"size_equal={ok_size} served={detail.get('generation_id')} want={manifest['generation_id']}",
            )

            # 10. concurrency: two pins + default read at once; bulk while current moves
            async def burst() -> List[Any]:
                async with httpx.AsyncClient(base_url=server.url, timeout=120) as aclient:
                    calls = [
                        aclient.get("/v1/equity/SPY/bars", params={"silver_revision": first}),
                        aclient.get(
                            "/v1/equity/FSLR/bars",
                            params={"pit_revision": 1, "start": "2023-01-03T00:00:00Z"},
                        ),
                        aclient.get("/v1/equity/SPY/bars", params={"price_mode": "raw"}),
                    ]
                    return [r.json() for r in await asyncio.gather(*calls)]

            pinned, pit_read, raw_read = asyncio.run(burst())
            record(
                "concurrent_pins_and_default",
                pinned.get("adjustment_revision") == first
                and pit_read.get("provenance", {}).get("pit") is not None
                and raw_read.get("price_mode") == "raw",
                f"{pinned.get('adjustment_revision')} {bool(pit_read.get('provenance'))} {raw_read.get('price_mode')}",
            )

            async def bulk_while_publishing() -> Any:
                async with httpx.AsyncClient(base_url=server.url, timeout=120) as aclient:
                    task = asyncio.create_task(
                        aclient.get(
                            "/v1/equity/bars",
                            params={"symbols": ",".join(SYMBOLS_1D), "limit": 0},
                        )
                    )
                    await asyncio.sleep(0.05)
                    newer = json.loads(current.read_bytes())
                    newer["revision"] = first + 1
                    _publish_silver(scratch, newer)
                    return (await task).json()

            bulk = asyncio.run(bulk_while_publishing())
            revisions_seen = {bulk.get("adjustment_revision")}
            record(
                "bulk_single_revision_during_publish",
                len(revisions_seen) == 1 and bulk.get("adjustment_revision") in (first, first + 1),
                f"adjustment_revision={bulk.get('adjustment_revision')}",
            )

            # 11. catalog replacement: identity changes and rows follow the new file
            before = client.get("/v1/lake/coverage", params={"limit": 1}).json()["catalog"]
            import duckdb

            copy = scratch / "catalog" / "replacement.duckdb"
            shutil.copy2(scratch / "catalog" / "analytics.duckdb", copy)
            con = duckdb.connect(str(copy))
            con.execute("DELETE FROM coverage WHERE symbol = 'AAPL'")
            con.close()
            os.replace(copy, scratch / "catalog" / "analytics.duckdb")
            after = client.get("/v1/lake/coverage", params={"symbol": "AAPL"}).json()
            record(
                "catalog_atomic_replacement",
                after["returned"] == 0 and after["catalog"] != before,
                f"returned={after['returned']} identity_changed={after['catalog'] != before}",
            )
    finally:
        server.stop()
    return results


def main() -> None:
    parser = argparse.ArgumentParser()
    parser.add_argument("--lake", type=Path, required=True)
    parser.add_argument("--out", type=Path, required=True)
    args = parser.parse_args()
    if str(args.out.resolve()).startswith(str(args.lake.resolve())):
        raise SystemExit("the scratch copy must live outside the lake")
    args.out.mkdir(parents=True, exist_ok=True)
    tree = Path.cwd()
    results = run_all(args.lake, args.out, tree, sys.executable)
    with (args.out / "controlled.jsonl").open("w") as sink:
        for record in results:
            sink.write(json.dumps(record) + "\n")
    failed = [r for r in results if r["status"] != "PASS"]
    print(f"{len(results)} controlled cases, {len(failed)} failed")


if __name__ == "__main__":
    main()
