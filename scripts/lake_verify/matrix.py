"""PR1 real-lake verification matrix (plan §4): generate, run, summarize.

    python scripts/lake_verify/matrix.py generate --out RUN --inventory INVENTORY.json
    python scripts/lake_verify/matrix.py run --out RUN --target raw=URL --target adjusted=URL
    python scripts/lake_verify/matrix.py summarize --out RUN

``generate`` writes ``matrix.json`` (every case with its dimensions, request and
expectation) and ``counts.json`` before anything runs. ``run`` appends one line per
settled case to ``results.jsonl`` as it goes and skips ids already recorded, so an
interrupted run resumes. ``summarize`` writes ``summary.md``.

Statuses (plan §4.4): PASS, EXPECTED_REJECTION, BLOCKED_DATA, BLOCKED_DEPENDENCY, FAIL,
NOT_RUN. A value case is checked against an oracle computed independently from the same
files (scripts/lake_verify/lake.py never imports the candidate); mutable sources are
stat-identified before and after, and a comparison that straddles a change is retried
and otherwise left NOT_RUN.

Environment: APEX_VERIFY_LAKE_ROOT (oracle), plus the APEX_LIVEWIRE_* lake variables
for the in-process bounded-policy cases. Nothing here writes to the lake.
"""

from __future__ import annotations

import argparse
import asyncio
import collections
import importlib
import json
import sys
import time
import traceback
from dataclasses import asdict
from pathlib import Path
from typing import Any, Dict, List, Optional, Tuple

import httpx

sys.path.insert(0, str(Path(__file__).resolve().parent))
sys.path.insert(0, str(Path.cwd()))

from lake import Lake, identity  # noqa: E402
from model import Case, Invalidated, NotApplicable, Outcome  # noqa: E402

STATUSES = (
    "PASS",
    "EXPECTED_REJECTION",
    "BLOCKED_DATA",
    "BLOCKED_DEPENDENCY",
    "FAIL",
    "NOT_RUN",
    "NOT_APPLICABLE",  # mcp target only: a REST-only cell (mcp_exec.NotApplicable)
)
OPERATION_MODULES = ("bars_cases", "surface_cases")
RETRIES = 2


MODULES = list(OPERATION_MODULES)


def _modules() -> List[Any]:
    return [importlib.import_module(name) for name in MODULES]


# -- generate ---------------------------------------------------------------------


def generate(out: Path, inventory: Path) -> None:
    lake = Lake.from_env()
    inv = json.loads(inventory.read_text())
    cases: List[Case] = []
    for module in _modules():
        cases.extend(module.generate(lake, inv))
    ids = [c.id for c in cases]
    duplicates = [i for i, n in collections.Counter(ids).items() if n > 1]
    if duplicates:
        raise SystemExit(f"duplicate case ids: {duplicates[:5]}")
    out.mkdir(parents=True, exist_ok=True)
    (out / "matrix.json").write_text(json.dumps([asdict(c) for c in cases]))
    counts: Dict[str, Any] = {
        "total": len(cases),
        "by_operation": {},
        "by_dimension": {},
    }
    for case in cases:
        counts["by_operation"][case.operation] = counts["by_operation"].get(case.operation, 0) + 1
        for dim, value in case.dims.items():
            bucket = counts["by_dimension"].setdefault(f"{case.operation}.{dim}", {})
            bucket[str(value)] = bucket.get(str(value), 0) + 1
        kind = case.expect["kind"]
        counts.setdefault("by_expect", {})[kind] = (
            counts.setdefault("by_expect", {}).get(kind, 0) + 1
        )
    (out / "counts.json").write_text(json.dumps(counts, indent=1))
    print(
        json.dumps(
            {
                "total": counts["total"],
                "by_operation": counts["by_operation"],
                "by_expect": counts["by_expect"],
            }
        )
    )


# -- run --------------------------------------------------------------------------


def _load_cases(out: Path) -> List[Case]:
    return [Case(**c) for c in json.loads((out / "matrix.json").read_text())]


def _done(out: Path) -> set:
    path = out / "results.jsonl"
    if not path.exists():
        return set()
    done = set()
    for line in path.read_text().splitlines():
        if line.strip():
            record = json.loads(line)
            if record["status"] != "NOT_RUN":
                done.add(record["id"])
    return done


_APPS: Dict[str, Any] = {}


def _app(process: str) -> Any:
    """One real REST app per process configuration, lifespan never run."""
    if process not in _APPS:
        from serve import build_app

        _APPS[process] = build_app(price_mode=process)
    return _APPS[process]


class Executor:
    """``http`` requests go to a loopback server URL, or -- with target ``asgi`` -- to
    the real app in this process (FastAPI TestClient: real routing, middleware and
    serialization, no sockets). Loopback servers answer ``Connection: close`` under
    uvicorn here and a fast run exhausts macOS ephemeral ports (Errno 49, measured
    2026-09-23), so the matrix uses ``asgi``; the benchmark keeps real sockets."""

    def __init__(self, targets: Dict[str, str]) -> None:
        self.targets = targets
        self.client = httpx.Client(timeout=180)
        self._asgi: Dict[str, Any] = {}
        self._mcp: Dict[str, Any] = {}

    def _client(self, process: str) -> Tuple[Any, str]:
        target = self.targets[process]
        if target != "asgi":
            return self.client, target
        if process not in self._asgi:
            from fastapi.testclient import TestClient

            self._asgi[process] = TestClient(_app(process), raise_server_exceptions=False)
        return self._asgi[process], ""

    def http(self, request: Dict[str, Any]) -> Tuple[int, Any]:
        client, base = self._client(request["process"])
        response = client.get(base + request["path"], params=request.get("params") or {})
        try:
            body = response.json()
        except ValueError:
            body = {"_raw": response.text[:500]}
        return response.status_code, body

    def inproc(self, request: Dict[str, Any]) -> Tuple[int, Any]:
        """Call the shared query in this process with the bounded policy, rendered by
        the same payload builders REST uses; LakeError maps to its REST status."""
        from inproc import call  # candidate-side adapter, imported only when needed

        return asyncio.run(call(request))

    def mcp(self, request: Dict[str, Any]) -> Tuple[int, Any]:
        """Target ``mcp``: the case as an MCP tool call through the real MCP app."""
        from mcp_exec import McpTarget

        process = request["process"]
        if process not in self._mcp:
            self._mcp[process] = McpTarget(process)
        return self._mcp[process].execute(request)

    def execute(self, request: Dict[str, Any]) -> Tuple[int, Any]:
        if self.targets[request["process"]] == "mcp":
            return self.mcp(request)
        return self.http(request) if request["transport"] == "http" else self.inproc(request)


def _settle(case: Case, executor: Executor, lake: Lake, checkers: Dict[str, Any]) -> Outcome:
    expect = case.expect
    if expect["kind"] == "blocked_data":
        return Outcome("BLOCKED_DATA", expect["reason"])
    if expect["kind"] == "blocked_dependency":
        return Outcome("BLOCKED_DEPENDENCY", expect["reason"])
    if expect["kind"] == "rejection":
        status, body = executor.execute(case.request)
        code = body.get("error", {}).get("code") if isinstance(body, dict) else None
        # An MCP tool error carries a code but no HTTP status (mcp_exec derives one),
        # so the mcp target is judged on the code alone.
        mcp = executor.targets[case.request["process"]] == "mcp"
        if code == expect["code"] and (mcp or status == expect["status"]):
            return Outcome("EXPECTED_REJECTION", f"{status} {code}")
        return Outcome(
            "FAIL",
            f"expected {expect['status']} {expect['code']}, got {status} {code}: {str(body)[:300]}",
        )
    checker = checkers[expect["check"]]
    last = Outcome("NOT_RUN", "source changed during every attempt")
    for _ in range(RETRIES + 1):
        sources = checker.sources(case, lake)
        before = [identity(p) for p in sources]
        try:
            outcome = checker.run(case, lake, executor)
        except Invalidated as exc:
            last = Outcome("NOT_RUN", str(exc))
            continue
        after = [identity(p) for p in sources]
        if before != after:
            last = Outcome("NOT_RUN", "mutable source changed between candidate and oracle")
            continue
        return outcome
    return last


def run(
    out: Path,
    targets: Dict[str, str],
    only: Optional[str],
    limit: Optional[int],
    workers: int = 1,
) -> None:
    """Settle pending cases; ``workers`` (<= 2, plan 4.4) run value cases in parallel,
    each thread with its own HTTP client."""
    import threading
    from concurrent.futures import ThreadPoolExecutor

    if not 1 <= workers <= 2:
        raise SystemExit("workers must be 1 or 2 (plan 4.4 caps parallel reads at two)")
    lake = Lake.from_env()
    cases = _load_cases(out)
    done = _done(out)
    checkers: Dict[str, Any] = {}
    for module in _modules():
        checkers.update(module.CHECKERS)
    local = threading.local()
    pending = [c for c in cases if c.id not in done and (only is None or c.operation == only)]
    if limit is not None:
        pending = pending[:limit]
    started = time.time()
    lock = threading.Lock()
    settled = [0]

    def settle(case: Case) -> None:
        if not hasattr(local, "executor"):
            local.executor = Executor(targets)
        t0 = time.perf_counter()
        try:
            outcome = _settle(case, local.executor, lake, checkers)
        except NotApplicable as exc:
            outcome = Outcome("NOT_APPLICABLE", str(exc))
        except httpx.TransportError as exc:
            # The harness could not reach the target: nothing about the candidate was
            # observed. NOT_RUN keeps the case pending for a resumed run.
            outcome = Outcome("NOT_RUN", f"transport error: {exc!r}")
        except Exception as exc:  # a crashed check is a FAIL, never a silent skip
            outcome = Outcome("FAIL", f"runner error: {exc!r} {traceback.format_exc()[-600:]}")
        record = {
            "id": case.id,
            "operation": case.operation,
            "dims": case.dims,
            "status": outcome.status,
            "detail": outcome.detail,
            "facts": outcome.facts,
            "elapsed_ms": round((time.perf_counter() - t0) * 1000, 1),
            "at": time.strftime("%Y-%m-%dT%H:%M:%SZ", time.gmtime()),
        }
        with lock:
            sink.write(json.dumps(record, default=str) + "\n")
            sink.flush()
            settled[0] += 1
            if settled[0] % 1000 == 0:
                rate = settled[0] / (time.time() - started)
                print(f"{settled[0]}/{len(pending)} settled, {rate:.1f}/s", flush=True)

    with (out / "results.jsonl").open("a") as sink:
        with ThreadPoolExecutor(max_workers=workers) as pool:
            list(pool.map(settle, pending))


# -- summarize --------------------------------------------------------------------


def summarize(out: Path) -> None:
    cases = _load_cases(out)
    latest: Dict[str, Dict[str, Any]] = {}
    path = out / "results.jsonl"
    if path.exists():
        for line in path.read_text().splitlines():
            if line.strip():
                record = json.loads(line)
                latest[record["id"]] = record
    status_of = {c.id: latest.get(c.id, {}).get("status", "NOT_RUN") for c in cases}
    totals = collections.Counter(status_of.values())
    by_class: Dict[str, collections.Counter] = collections.defaultdict(collections.Counter)
    by_op: Dict[str, collections.Counter] = collections.defaultdict(collections.Counter)
    for case in cases:
        by_op[case.operation][status_of[case.id]] += 1
        by_class[str(case.dims.get("asset_class", "-"))][status_of[case.id]] += 1
    gate = all(
        totals.get(s, 0) == 0 for s in ("FAIL", "NOT_RUN", "BLOCKED_DATA", "BLOCKED_DEPENDENCY")
    )
    lines = [
        "# PR1 real-lake matrix summary",
        "",
        f"Planned cases: {len(cases)}; settled: {sum(1 for s in status_of.values() if s != 'NOT_RUN')}",
        f"Gate (zero FAIL/NOT_RUN/BLOCKED_*): {'PASS' if gate else 'OPEN'}",
        "",
        "| status | count |",
        "|---|---|",
        *[f"| {s} | {totals.get(s, 0)} |" for s in STATUSES],
        "",
        "## By operation",
        "",
        "| operation | " + " | ".join(STATUSES) + " |",
        "|---|" + "---|" * len(STATUSES),
        *[
            f"| {op} | " + " | ".join(str(c.get(s, 0)) for s in STATUSES) + " |"
            for op, c in sorted(by_op.items())
        ],
        "",
        "## By asset class",
        "",
        "| asset_class | " + " | ".join(STATUSES) + " |",
        "|---|" + "---|" * len(STATUSES),
        *[
            f"| {ac} | " + " | ".join(str(c.get(s, 0)) for s in STATUSES) + " |"
            for ac, c in sorted(by_class.items())
        ],
        "",
        "## Non-passing cases (first 200 per status)",
    ]
    for status in ("FAIL", "BLOCKED_DATA", "BLOCKED_DEPENDENCY", "NOT_RUN"):
        ids = [cid for cid, s in status_of.items() if s == status]
        if not ids:
            continue
        lines += ["", f"### {status} ({len(ids)})", ""]
        reasons = collections.Counter(
            latest.get(i, {}).get("detail", "not settled")[:160] for i in ids
        )
        lines += [f"- {n} x {reason}" for reason, n in reasons.most_common(40)]
        lines += ["", "ids: " + ", ".join(ids[:200])]
    (out / "summary.md").write_text("\n".join(lines) + "\n")
    print("\n".join(lines[:30]))


def main() -> None:
    parser = argparse.ArgumentParser()
    sub = parser.add_subparsers(dest="command", required=True)
    g = sub.add_parser("generate")
    g.add_argument("--out", type=Path, required=True)
    g.add_argument("--inventory", type=Path, required=True)
    r = sub.add_parser("run")
    r.add_argument("--out", type=Path, required=True)
    r.add_argument("--target", action="append", required=True, help="process=url")
    r.add_argument("--only")
    r.add_argument("--limit", type=int)
    r.add_argument("--workers", type=int, default=1)
    s = sub.add_parser("summarize")
    s.add_argument("--out", type=Path, required=True)
    parser.add_argument("--modules", default=",".join(OPERATION_MODULES))
    args = parser.parse_args()
    MODULES[:] = args.modules.split(",")
    if args.command == "generate":
        generate(args.out, args.inventory)
    elif args.command == "run":
        run(
            args.out,
            dict(t.split("=", 1) for t in args.target),
            args.only,
            args.limit,
            args.workers,
        )
    else:
        summarize(args.out)


if __name__ == "__main__":
    main()
