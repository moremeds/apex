"""Discovery, revision, identity, membership and gap cases for the PR1 matrix.

Covers every lake REST route that is not bars/bulk/rates (those are ``bars_cases.py``):
asset-classes, lake status, coverage, instruments, instrument detail, futures
contracts, corporate actions, delisting, security resolution, index membership,
Silver/PIT revisions and session gaps. Plugs into ``matrix.py`` exactly like
``bars_cases.py``: ``generate(lake, inventory) -> List[Case]`` and ``CHECKERS``.

The oracle re-derives every answer from the lake files and livewire's documented
contracts (``src/infrastructure/CLAUDE.md``, the route/application modules read only
to learn field names and SQL semantics) and never imports ``src``. Where the contract
lets a request be legal but the lake hold no matching data, the case is generated as
``blocked_data``; where the contract itself makes the request data-dependent (unknown
symbol, ambiguous security, empty membership log, ...) the oracle raises ``Reject`` at
run time and the checker treats a matching candidate rejection as a pass, exactly the
pattern ``bars_cases.BarsChecker`` uses.
"""

from __future__ import annotations

import json
import re
from datetime import date, datetime, time, timedelta, timezone
from pathlib import Path
from typing import Any, Dict, List, Optional, Sequence, Tuple
from urllib.parse import unquote

import duckdb
import pandas_market_calendars as mcal
from lake import LADDERS, Lake, encode
from model import Case, Outcome

UTC = timezone.utc
CLASSES = ("equity", "volatility", "fx", "cmdty", "futures", "rates")
GAP_LADDERS = {**LADDERS, "rates": ("1d",)}  # gaps allows rates; bars does not
RESIDENCIES = ("live_only", "archive_only", "dual")
_SYMBOL_PROVIDER = "massive"
_SYMBOL_EXCHANGES = ("XNAS", "XNYS", "ARCX")
_CLOSURES = (
    "2018-12-05",
    "2025-01-09",
)  # Bush / Carter national-day-of-mourning XNYS closures


class Reject(Exception):
    """The oracle's own prediction that the candidate must answer with an error."""

    def __init__(self, status: int, code: str) -> None:
        super().__init__(f"{status} {code}")
        self.status, self.code = status, code


# -- generic helpers ---------------------------------------------------------------


def _req(path: str, params: Optional[Dict[str, Any]] = None) -> Dict[str, Any]:
    return {
        "transport": "http",
        "process": "adjusted",
        "path": path,
        "params": params or {},
    }


def _rejection(status: int, code: str) -> Dict[str, Any]:
    return {"kind": "rejection", "status": status, "code": code}


def _value(check: str, **args: Any) -> Dict[str, Any]:
    return {"kind": "value", "check": check, "args": args}


def _blocked(reason: str) -> Dict[str, Any]:
    return {"kind": "blocked_data", "reason": reason}


def _mismatches(expected: Dict[str, Any], got: Any, fields: Sequence[str]) -> List[str]:
    if not isinstance(got, dict):
        return [f"body is not an object: {str(got)[:200]}"]
    out = []
    for f in fields:
        e, g = expected.get(f), got.get(f)
        if e != g:
            out.append(f"{f}: expected {e!r} got {g!r}")
    return out


def _page_slice(items: Sequence[Any], limit: int, offset: int) -> Tuple[List[Any], bool]:
    window = list(items[offset : offset + limit + 1])
    return window[:limit], len(window) > limit


def _page_fields(items: Sequence[Any], limit: int, offset: int) -> Dict[str, Any]:
    page, truncated = _page_slice(items, limit, offset)
    return {
        "limit": limit,
        "offset": offset,
        "returned": len(page),
        "truncated": truncated,
        "next_offset": offset + len(page) if truncated else None,
    }


def _bad_page_cases(op: str, path: str, base: Dict[str, Any]) -> List[Case]:
    """limit=0 / limit=2001 / offset=-1 -> 400 invalid_parameter (check_page routes)."""
    cases = []
    for label, extra in (
        ("limit_zero", {"limit": 0}),
        ("limit_over", {"limit": 2001}),
        ("offset_neg", {"offset": -1}),
    ):
        cases.append(
            Case(
                f"{op}:reject:{label}",
                op,
                {"reject": label},
                _req(path, {**base, **extra}),
                _rejection(400, "invalid_parameter"),
            )
        )
    return cases


def _find(items: Sequence[Any], key: str, target: Any) -> Optional[Any]:
    for item in items:
        if item.get(key) == target:
            return item
    return None


def _no_absolute_paths(body: Any) -> Optional[str]:
    text = json.dumps(body)
    for needle in ("/Volumes", "/Users"):
        if needle in text:
            return f"body leaks an absolute path containing {needle!r}"
    return None


def _duck(sql: str, params: Sequence[Any]) -> List[Tuple[Any, ...]]:
    con = duckdb.connect()
    try:
        con.execute("SET TimeZone='UTC'")
        return con.execute(sql, list(params)).fetchall()
    finally:
        con.close()


def _catalog_path(lake: Lake) -> Path:
    return lake.root / "catalog" / "analytics.duckdb"


def _catalog(lake: Lake) -> Any:
    return duckdb.connect(str(_catalog_path(lake)), read_only=True)


def _security_master_path(lake: Lake) -> Path:
    return lake.root / "security_master" / "events.parquet"


def _corp_action_path(lake: Lake, symbol: str) -> Path:
    return (
        lake.root
        / "bronze"
        / "asset_class=corporate_action"
        / f"symbol={encode(symbol)}"
        / "events.parquet"
    )


def _membership_events_path(lake: Lake, index_id: str) -> Path:
    return lake.root / "index_membership" / index_id / "events.parquet"


def _repairs_root(lake: Lake) -> Path:
    return lake.root / "repairs"


def _end_of_day(day: date) -> datetime:
    return datetime.combine(day, time.max, tzinfo=UTC)


def _start_of_day(day: date) -> datetime:
    return datetime.combine(day, time.min, tzinfo=UTC)


def _today() -> date:
    return datetime.now(UTC).date()


def _as_utc(value: Any) -> Optional[datetime]:
    """Mirrors ``membership.py::_as_utc``: attach UTC to a naive timestamp, else convert."""
    if value is None or not isinstance(value, datetime):
        return None
    return value.replace(tzinfo=UTC) if value.tzinfo is None else value.astimezone(UTC)


def _iso_or_none(value: Optional[datetime]) -> Optional[str]:
    return None if value is None else value.isoformat()


# -- inventory sampling --------------------------------------------------------------


def _samples(inventory: Dict[str, Any]) -> Dict[Tuple[str, str, str], Dict[str, Any]]:
    """(asset_class, timeframe, residency) -> {"symbol", "first": date, "last": date}."""
    out: Dict[Tuple[str, str, str], Dict[str, Any]] = {}
    for cell in inventory["cells"]:
        if not cell["samples"]:
            continue
        sample = cell["samples"][0]
        facts = [sample[k] for k in ("live", "archive") if k in sample]
        firsts = [datetime.fromisoformat(f["first"]).date() for f in facts]
        lasts = [datetime.fromisoformat(f["last"]).date() for f in facts]
        out[(cell["asset_class"], cell["timeframe"], cell["residency"])] = {
            "symbol": unquote(sample["symbol_dir"]),
            "first": min(firsts),
            "last": max(lasts),
        }
    return out


def _best_sample(
    samples: Dict[Tuple[str, str, str], Any], ac: str, tf: str
) -> Optional[Tuple[str, Dict]]:
    for residency in ("dual", "live_only", "archive_only"):
        s = samples.get((ac, tf, residency))
        if s is not None:
            return residency, s
    return None


# ==== 1. GET /v1/lake/asset-classes =================================================

_EXPECTED_ASSET_CLASSES = [
    {
        "asset_class": "equity",
        "payload": "bars",
        "timeframes": ["1m", "5m", "30m", "1h", "1d"],
        "supports_adjusted": True,
        "extra_bar_fields": [],
    },
    {
        "asset_class": "volatility",
        "payload": "bars",
        "timeframes": ["5m", "30m", "1h", "1d"],
        "supports_adjusted": False,
        "extra_bar_fields": [],
    },
    {
        "asset_class": "fx",
        "payload": "bars",
        "timeframes": ["1m", "5m", "30m", "1h", "1d"],
        "supports_adjusted": False,
        "extra_bar_fields": [],
    },
    {
        "asset_class": "cmdty",
        "payload": "bars",
        "timeframes": ["1d"],
        "supports_adjusted": False,
        "extra_bar_fields": [],
    },
    {
        "asset_class": "futures",
        "payload": "bars",
        "timeframes": ["1d"],
        "supports_adjusted": False,
        "extra_bar_fields": ["settlement", "open_interest"],
    },
    {
        "asset_class": "rates",
        "payload": "rates_series",
        "timeframes": ["1d"],
        "supports_adjusted": False,
        "extra_bar_fields": [],
    },
]


class AssetClassesChecker:
    def sources(self, case: Case, lake: Lake) -> List[Path]:
        return []

    def run(self, case: Case, lake: Lake, executor: Any) -> Outcome:
        status, body = executor.execute(case.request)
        if status != 200:
            return Outcome("FAIL", f"expected 200, got {status}: {str(body)[:300]}")
        got = {row.get("asset_class"): row for row in body.get("asset_classes", [])}
        problems = []
        for expected in _EXPECTED_ASSET_CLASSES:
            row = got.get(expected["asset_class"])
            if row is None:
                problems.append(f"missing asset class {expected['asset_class']!r}")
                continue
            bad = _mismatches(expected, row, ("payload", "supports_adjusted", "extra_bar_fields"))
            if row.get("timeframes") != expected["timeframes"]:
                bad.append(
                    f"timeframes: expected {expected['timeframes']} got {row.get('timeframes')}"
                )
            if bad:
                problems.append(f"{expected['asset_class']}: " + "; ".join(bad))
        extra = set(got) - {e["asset_class"] for e in _EXPECTED_ASSET_CLASSES}
        if extra:
            problems.append(f"unexpected asset classes: {sorted(extra)}")
        if problems:
            return Outcome("FAIL", "; ".join(problems))
        return Outcome("PASS")


# ==== 2. GET /v1/lake/status =========================================================


class StatusChecker:
    def sources(self, case: Case, lake: Lake) -> List[Path]:
        return [_catalog_path(lake), lake.silver / "revisions" / "current.json"]

    def run(self, case: Case, lake: Lake, executor: Any) -> Outcome:
        status, body = executor.execute(case.request)
        if status != 200:
            return Outcome("FAIL", f"expected 200, got {status}: {str(body)[:300]}")
        leak = _no_absolute_paths(body)
        if leak:
            return Outcome("FAIL", leak)
        sources = body.get("sources", {})
        problems = []

        silver_current = lake.silver_current_number()
        retained = len(list((lake.silver / "revisions").glob("revision=*.json")))
        bad = _mismatches(
            {"current_revision": silver_current, "retained_revisions": retained},
            sources.get("silver", {}),
            ("current_revision", "retained_revisions"),
        )
        if bad:
            problems.append("silver: " + "; ".join(bad))

        pit_numbers = lake.pit_numbers()
        latest_per_index: Dict[str, Dict[str, Any]] = {}
        for n in sorted(pit_numbers, reverse=True):
            manifest = lake.pit(n)
            latest_per_index.setdefault(
                manifest["index_id"],
                {"revision": n, "publisher_status": manifest["status"]},
            )
        got_pit = sources.get("pit", {})
        if got_pit.get("revisions") != len(pit_numbers):
            problems.append(
                f"pit.revisions: expected {len(pit_numbers)} got {got_pit.get('revisions')}"
            )
        if got_pit.get("latest_per_index") != latest_per_index:
            problems.append(
                f"pit.latest_per_index: expected {latest_per_index} got {got_pit.get('latest_per_index')}"
            )

        cat_path = _catalog_path(lake)
        got_catalog = sources.get("catalog", {})
        if cat_path.is_file():
            st = cat_path.stat()
            if got_catalog.get("size_bytes") != st.st_size:
                problems.append(
                    f"catalog.size_bytes: expected {st.st_size} got {got_catalog.get('size_bytes')}"
                )

        membership_root = lake.root / "index_membership"
        indices = sorted(
            p.name
            for p in (membership_root.iterdir() if membership_root.is_dir() else [])
            if p.is_dir() and (p / "events.parquet").is_file()
        )
        got_membership = sources.get("membership", {})
        if got_membership.get("indices") != indices:
            problems.append(
                f"membership.indices: expected {indices} got {got_membership.get('indices')}"
            )

        repairs_root = _repairs_root(lake)
        expected_reports = 0
        if repairs_root.is_dir():
            for p in repairs_root.iterdir():
                if p.is_file() and (_REPORT_RE.match(p.name) or p.name == "unresolved.json"):
                    expected_reports += 1
        got_repairs = sources.get("repairs", {})
        if got_repairs.get("reports_read") != expected_reports:
            problems.append(
                f"repairs.reports_read: expected {expected_reports} got {got_repairs.get('reports_read')}"
            )

        if problems:
            return Outcome("FAIL", "; ".join(problems))
        return Outcome("PASS")


# ==== 3. GET /v1/lake/coverage =======================================================

_VIEW_RE = re.compile(r"^(bronze|silver)_([a-z_]+)_(1m|5m|30m|1h|1d)$")


def _classify_view(
    view_name: str,
) -> Tuple[Optional[str], Optional[str], Optional[str]]:
    m = _VIEW_RE.match(view_name)
    return (None, None, None) if m is None else (m.group(1), m.group(2), m.group(3))


def _coverage_rows(
    lake: Lake,
    *,
    symbol: Optional[str],
    asset_class: Optional[str],
    include_silver: bool,
) -> List[Dict[str, Any]]:
    con = _catalog(lake)
    try:
        views = [v for (v,) in con.execute("SELECT DISTINCT view_name FROM coverage").fetchall()]
        selected = []
        for v in views:
            tier, cls, _tf = _classify_view(v)
            if tier == "silver" and not include_silver:
                continue
            if asset_class is not None and cls != asset_class:
                continue
            selected.append(v)
        if not selected:
            return []
        placeholders = ", ".join("?" for _ in selected)
        sql = (
            "SELECT view_name, symbol, n_rows, first_date, last_date FROM coverage "
            f"WHERE view_name IN ({placeholders}) AND (? IS NULL OR symbol = ?) "
            "ORDER BY symbol, view_name"
        )
        rows = con.execute(sql, [*selected, symbol, symbol]).fetchall()
    finally:
        con.close()
    out = []
    for view_name, sym, n_rows, first_date, last_date in rows:
        tier, cls, tf = _classify_view(view_name)
        out.append(
            {
                "view_name": view_name,
                "tier": tier,
                "asset_class": cls,
                "timeframe": tf,
                "symbol": sym,
                "n_rows": int(n_rows),
                "first_date": None if first_date is None else str(first_date),
                "last_date": None if last_date is None else str(last_date),
            }
        )
    return out


class CoverageChecker:
    def sources(self, case: Case, lake: Lake) -> List[Path]:
        return [_catalog_path(lake)]

    def run(self, case: Case, lake: Lake, executor: Any) -> Outcome:
        args = case.expect["args"]
        params = case.request["params"]
        limit, offset = params.get("limit", 100), params.get("offset", 0)
        rows = _coverage_rows(
            lake,
            symbol=args.get("symbol"),
            asset_class=args.get("asset_class"),
            include_silver=args["include_silver"],
        )
        status, body = executor.execute(case.request)
        if status != 200:
            return Outcome(
                "FAIL",
                f"expected 200 with {len(rows)} rows, got {status}: {str(body)[:300]}",
            )
        page, truncated = _page_slice(rows, limit, offset)
        problems = _mismatches(
            _page_fields(rows, limit, offset),
            body,
            ("limit", "offset", "returned", "truncated", "next_offset"),
        )
        got_rows = body.get("rows", [])
        if len(got_rows) != len(page):
            problems.append(f"rows: expected {len(page)} got {len(got_rows)}")
        else:
            for i, (e, g) in enumerate(zip(page, got_rows)):
                bad = _mismatches(e, g, tuple(e))
                if bad:
                    problems.append(f"row {i}: " + "; ".join(bad))
        cat_path = _catalog_path(lake)
        if cat_path.is_file():
            st = cat_path.stat()
            got_cat = body.get("catalog", {})
            if got_cat.get("size_bytes") != st.st_size:
                problems.append(
                    f"catalog.size_bytes: expected {st.st_size} got {got_cat.get('size_bytes')}"
                )
        if problems:
            return Outcome("FAIL", "; ".join(problems))
        return Outcome("PASS")


def _gen_coverage(lake: Lake) -> List[Case]:
    cases: List[Case] = []
    cases += _bad_page_cases("coverage", "/v1/lake/coverage", {})
    cases.append(
        Case(
            "coverage:bad_asset_class",
            "coverage",
            {},
            _req("/v1/lake/coverage", {"asset_class": "bonds"}),
            _rejection(400, "unsupported_asset_class"),
        )
    )
    for ac in (*CLASSES, None):
        for include_silver in (True, False):
            dims = {"asset_class": str(ac), "include_silver": include_silver}
            params: Dict[str, Any] = {"include_silver": include_silver, "limit": 50}
            if ac is not None:
                params["asset_class"] = ac
            cases.append(
                Case(
                    f"coverage:page1:{ac}:{include_silver}",
                    "coverage",
                    dims,
                    _req("/v1/lake/coverage", params),
                    _value(
                        "coverage",
                        symbol=None,
                        asset_class=ac,
                        include_silver=include_silver,
                    ),
                )
            )
    # paging edges + exact-symbol + unknown-symbol, on the unfiltered view
    total = len(_coverage_rows(lake, symbol=None, asset_class=None, include_silver=True))
    edges = [("first", 0), ("empty", total + 1000)]
    if total > 4:
        edges.append(("middle", total // 2))
    for label, offset in edges:
        cases.append(
            Case(
                f"coverage:edge:{label}",
                "coverage",
                {"offset": offset},
                _req("/v1/lake/coverage", {"limit": 2, "offset": offset}),
                _value("coverage", symbol=None, asset_class=None, include_silver=True),
            )
        )
    return cases


# ==== 4. GET /v1/instruments =========================================================


def _instrument_rows(
    lake: Lake, *, q: Optional[str], asset_class: Optional[str], limit: int
) -> List[Dict[str, Any]]:
    view_by_class = {c: f"bronze_{c}_1d" for c in CLASSES}
    views = [view_by_class[asset_class]] if asset_class else list(view_by_class.values())
    con = _catalog(lake)
    try:
        placeholders = ", ".join("?" for _ in views)
        like = None
        if q:
            like = q.upper().replace("\\", "\\\\").replace("%", "\\%").replace("_", "\\_") + "%"
        sql = (
            "SELECT b.view_name, b.symbol, b.first_date, b.last_date, (s.symbol IS NOT NULL) AS has_silver "
            "FROM coverage b LEFT JOIN coverage s ON s.view_name = 'silver_equity_1d' AND s.symbol = b.symbol "
            f"WHERE b.view_name IN ({placeholders}) AND (? IS NULL OR b.symbol LIKE ? ESCAPE '\\') "
            "ORDER BY b.symbol LIMIT ?"
        )
        rows = con.execute(sql, [*views, like, like, limit]).fetchall()
    finally:
        con.close()
    view_to_class = {v: c for c, v in view_by_class.items()}
    out = []
    for view_name, symbol, first_date, last_date, has_silver in rows:
        cls = view_to_class[view_name]
        silver_available = bool(has_silver) and cls == "equity"
        out.append(
            {
                "symbol": symbol,
                "asset_class": cls,
                "listing_status": "listed",
                "first_date": None if first_date is None else str(first_date),
                "last_date": None if last_date is None else str(last_date),
                "silver_available": silver_available,
                "price_mode": "adjusted" if silver_available else "raw",
            }
        )
    return out


class InstrumentsChecker:
    def sources(self, case: Case, lake: Lake) -> List[Path]:
        return [_catalog_path(lake)]

    def run(self, case: Case, lake: Lake, executor: Any) -> Outcome:
        args = case.expect["args"]
        expected = _instrument_rows(
            lake,
            q=args.get("q"),
            asset_class=args.get("asset_class"),
            limit=args["limit"],
        )
        status, body = executor.execute(case.request)
        if status != 200:
            return Outcome(
                "FAIL",
                f"expected 200 with {len(expected)} rows, got {status}: {str(body)[:300]}",
            )
        got = body.get("instruments", [])
        if len(got) != len(expected):
            return Outcome(
                "FAIL",
                f"count: expected {len(expected)} got {len(got)} (body count={body.get('count')})",
            )
        for i, (e, g) in enumerate(zip(expected, got)):
            bad = _mismatches(e, g, tuple(e))
            if bad:
                return Outcome("FAIL", f"row {i}: " + "; ".join(bad))
        if body.get("count") != len(expected):
            return Outcome("FAIL", f"count field: expected {len(expected)} got {body.get('count')}")
        return Outcome("PASS")


def _gen_instruments(lake: Lake, samples: Dict[Tuple[str, str, str], Any]) -> List[Case]:
    cases: List[Case] = [
        Case(
            "instruments:reject:limit_zero",
            "instruments",
            {},
            _req("/v1/instruments", {"limit": 0}),
            _rejection(422, "invalid_parameter"),
        ),
        Case(
            "instruments:reject:limit_over",
            "instruments",
            {},
            _req("/v1/instruments", {"limit": 5001}),
            _rejection(422, "invalid_parameter"),
        ),
        Case(
            "instruments:reject:delisted",
            "instruments",
            {},
            _req("/v1/instruments", {"listing": "delisted"}),
            _rejection(501, "not_yet_available"),
        ),
        Case(
            "instruments:reject:any",
            "instruments",
            {},
            _req("/v1/instruments", {"listing": "any"}),
            _rejection(501, "not_yet_available"),
        ),
        Case(
            "instruments:reject:bad_listing",
            "instruments",
            {},
            _req("/v1/instruments", {"listing": "bogus"}),
            _rejection(400, "invalid_parameter"),
        ),
    ]
    for ac in (*CLASSES, None):
        dims = {"asset_class": str(ac)}
        params = {"limit": 200}
        if ac:
            params["asset_class"] = ac
        cases.append(
            Case(
                f"instruments:list:{ac}",
                "instruments",
                dims,
                _req("/v1/instruments", params),
                _value("instruments", q=None, asset_class=ac, limit=200),
            )
        )
    equity = samples.get(("equity", "1d", "live_only")) or samples.get(("equity", "1d", "dual"))
    if equity is not None:
        prefix = equity["symbol"][:2]
        cases.append(
            Case(
                "instruments:query_prefix",
                "instruments",
                {"q": prefix},
                _req("/v1/instruments", {"q": prefix, "limit": 500}),
                _value("instruments", q=prefix, asset_class=None, limit=500),
            )
        )
    return cases


# ==== 5. GET /v1/{asset_class}/{symbol} (instrument detail) =========================


class InstrumentDetailChecker:
    def sources(self, case: Case, lake: Lake) -> List[Path]:
        return [_catalog_path(lake), lake.silver / "revisions" / "current.json"]

    def run(self, case: Case, lake: Lake, executor: Any) -> Outcome:
        args = case.expect["args"]
        ac, symbol = args["asset_class"], args["symbol"]
        ladder = LADDERS[ac]
        current_silver = lake.silver_current_number()
        silver_daily = ac == "equity" and (
            lake.silver_artifact(current_silver, symbol, "daily") is not None
        )
        residency: Dict[str, str] = {}
        timeframes: List[str] = []
        for tf in ladder:
            live = lake.bronze(ac, symbol, tf).exists()
            arch = lake.archive(ac, symbol, tf).exists()
            if live or (tf == "1d" and silver_daily):
                timeframes.append(tf)
            if live or arch:
                residency[tf] = "dual" if live and arch else ("live" if live else "archive")
            elif tf == "1d" and silver_daily:
                residency[tf] = "silver"  # Silver-only daily (contract since a63d2b97)
        if not timeframes:
            status, body = executor.execute(case.request)
            if (
                status == 404
                and isinstance(body, dict)
                and body.get("error", {}).get("code") == "unknown_symbol"
            ):
                return Outcome("EXPECTED_REJECTION", "404 unknown_symbol")
            return Outcome("FAIL", f"expected 404 unknown_symbol, got {status}: {str(body)[:300]}")
        row = _find(_instrument_rows(lake, q=symbol, asset_class=ac, limit=50), "symbol", symbol)
        expected = {
            "symbol": symbol,
            "asset_class": ac,
            "timeframes": timeframes,
            "residency": residency,
            "first_date": row["first_date"] if row else None,
            "last_date": row["last_date"] if row else None,
            "silver_available": silver_daily,
            "price_mode": "adjusted" if ac == "equity" else "raw",
            "adjustment_revision": current_silver if silver_daily else None,
        }
        status, body = executor.execute(case.request)
        if status != 200:
            return Outcome("FAIL", f"expected 200, got {status}: {str(body)[:300]}")
        bad = _mismatches(
            expected,
            body,
            (
                "symbol",
                "asset_class",
                "timeframes",
                "residency",
                "first_date",
                "last_date",
                "silver_available",
                "price_mode",
                "adjustment_revision",
            ),
        )
        if bad:
            return Outcome("FAIL", "; ".join(bad))
        return Outcome("PASS")


def _gen_instrument_detail(samples: Dict[Tuple[str, str, str], Any]) -> List[Case]:
    cases: List[Case] = []
    for ac in CLASSES:
        picked = _best_sample(
            samples, ac, LADDERS[ac][-1]
        )  # coarsest timeframe (all ladders end in 1d)
        if picked is None:
            cases.append(
                Case(
                    f"instrument_detail:{ac}",
                    "instrument_detail",
                    {"asset_class": ac},
                    _req(f"/v1/{ac}/NOSAMPLE"),
                    _blocked(f"no inventory sample for {ac}"),
                )
            )
            continue
        _residency, sample = picked
        cases.append(
            Case(
                f"instrument_detail:{ac}",
                "instrument_detail",
                {"asset_class": ac},
                _req(f"/v1/{ac}/{sample['symbol']}"),
                _value("instrument_detail", asset_class=ac, symbol=sample["symbol"]),
            )
        )
    cases.append(
        Case(
            "instrument_detail:unknown",
            "instrument_detail",
            {"asset_class": "equity"},
            _req("/v1/equity/ZZZZNOSUCHSYMBOL"),
            _value("instrument_detail", asset_class="equity", symbol="ZZZZNOSUCHSYMBOL"),
        )
    )
    return cases


# ==== 6. GET /v1/futures/{root}/contracts ============================================


def _futures_roots(lake: Lake) -> List[str]:
    partition = lake.root / "bronze" / "asset_class=futures"
    if not partition.is_dir():
        return []
    roots = set()
    for entry in partition.iterdir():
        if entry.name.startswith("symbol="):
            sym = entry.name[len("symbol=") :]
            if "_" in sym:
                roots.add(sym.split("_", 1)[0])
    return sorted(roots)


def _futures_contracts(lake: Lake, root: str) -> List[Dict[str, Any]]:
    partition = lake.root / "bronze" / "asset_class=futures"
    prefix = f"symbol={root}_"
    symbols = sorted(
        e.name[len("symbol=") :] for e in partition.iterdir() if e.name.startswith(prefix)
    )
    catalogued: set = set()
    con = _catalog(lake)
    try:
        like = root.replace("\\", "\\\\").replace("%", "\\%").replace("_", "\\_") + "\\_%"
        rows = con.execute(
            "SELECT DISTINCT symbol FROM coverage WHERE view_name = 'bronze_futures_1d' AND symbol LIKE ? ESCAPE '\\'",
            [like],
        ).fetchall()
        catalogued = {r[0] for r in rows}
    finally:
        con.close()
    out = []
    for symbol in symbols:
        path = lake.bronze("futures", symbol, "1d")
        rows = _duck(
            "SELECT any_value(contract_id), any_value(root_symbol), any_value(expiry_date), "
            "min(trade_date), max(trade_date), count(*) FROM read_parquet(?)",
            [path.as_posix()],
        )
        contract_id, root_symbol, expiry_date, first_date, last_date, n = rows[0]
        out.append(
            {
                "symbol": symbol,
                "contract_id": contract_id,
                "root_symbol": root_symbol,
                "expiry_date": None if expiry_date is None else str(expiry_date),
                "first_date": None if first_date is None else str(first_date),
                "last_date": None if last_date is None else str(last_date),
                "rows": int(n),
                "in_catalog": symbol in catalogued,
            }
        )
    return out


class FuturesChecker:
    def sources(self, case: Case, lake: Lake) -> List[Path]:
        return [_catalog_path(lake), lake.root / "bronze" / "asset_class=futures"]

    def run(self, case: Case, lake: Lake, executor: Any) -> Outcome:
        args = case.expect["args"]
        root = args["root"]
        params = case.request["params"]
        limit, offset = params.get("limit", 100), params.get("offset", 0)
        contracts = _futures_contracts(lake, root)
        status, body = executor.execute(case.request)
        if not contracts:
            if (
                status == 404
                and isinstance(body, dict)
                and body.get("error", {}).get("code") == "unknown_symbol"
            ):
                return Outcome("EXPECTED_REJECTION", "404 unknown_symbol")
            return Outcome("FAIL", f"expected 404 unknown_symbol, got {status}: {str(body)[:300]}")
        if status != 200:
            return Outcome(
                "FAIL",
                f"expected 200 with {len(contracts)} contracts, got {status}: {str(body)[:300]}",
            )
        page, _ = _page_slice(contracts, limit, offset)
        problems = _mismatches(
            _page_fields(contracts, limit, offset),
            body,
            ("limit", "offset", "returned", "truncated", "next_offset"),
        )
        got = body.get("contracts", [])
        if len(got) != len(page):
            problems.append(f"contracts: expected {len(page)} got {len(got)}")
        else:
            for i, (e, g) in enumerate(zip(page, got)):
                bad = _mismatches(e, g, tuple(e))
                if bad:
                    problems.append(f"contract {i}: " + "; ".join(bad))
        if problems:
            return Outcome("FAIL", "; ".join(problems))
        return Outcome("PASS")


def _gen_futures(lake: Lake) -> List[Case]:
    cases: List[Case] = [
        Case(
            "futures:unknown_root",
            "futures",
            {},
            _req("/v1/futures/ZZZZ/contracts"),
            _value("futures", root="ZZZZ"),
        ),
    ]
    roots = _futures_roots(lake)
    biggest = None
    biggest_n = -1
    for root in roots:
        n = len(list((lake.root / "bronze" / "asset_class=futures").glob(f"symbol={root}_*")))
        if n > biggest_n:
            biggest, biggest_n = root, n
        cases.append(
            Case(
                f"futures:{root}",
                "futures",
                {"root": root},
                _req(f"/v1/futures/{root}/contracts", {"limit": 100}),
                _value("futures", root=root),
            )
        )
    if biggest and biggest_n > 3:
        cases.append(
            Case(
                f"futures:page:middle:{biggest}",
                "futures",
                {"root": biggest},
                _req(
                    f"/v1/futures/{biggest}/contracts",
                    {"limit": 1, "offset": biggest_n // 2},
                ),
                _value("futures", root=biggest),
            )
        )
    if biggest:
        cases.append(
            Case(
                f"futures:page:empty:{biggest}",
                "futures",
                {"root": biggest},
                _req(
                    f"/v1/futures/{biggest}/contracts",
                    {"limit": 5, "offset": biggest_n + 500},
                ),
                _value("futures", root=biggest),
            )
        )
    cases += _bad_page_cases(
        "futures",
        f"/v1/futures/{roots[0]}/contracts" if roots else "/v1/futures/ZZZZ/contracts",
        {},
    )
    return cases


# ==== 7. GET /v1/equity/{symbol}/actions ============================================

_ACTION_TYPES = ("split", "cash_dividend")
_ACTION_FIELDS = (
    "action_type",
    "ex_date",
    "split_from",
    "split_to",
    "cash_amount",
    "currency",
    "declaration_date",
    "record_date",
    "pay_date",
)


def _actions(
    lake: Lake,
    symbol: str,
    *,
    action_type: Optional[str],
    start: Optional[str],
    end: Optional[str],
) -> Optional[List[Dict[str, Any]]]:
    path = _corp_action_path(lake, symbol.upper())
    if not path.is_file():
        return None
    sql = (
        "WITH ranked AS (SELECT *, row_number() OVER (PARTITION BY action_id ORDER BY event_revision DESC) AS rn "
        "FROM read_parquet(?) WHERE status = 'active') "
        "SELECT action_type, ex_date, split_from, split_to, cash_amount, currency, declaration_date, record_date, pay_date "
        "FROM ranked WHERE rn = 1"
    )
    params: List[Any] = [path.as_posix()]
    if action_type is not None:
        sql += " AND action_type = ?"
        params.append(action_type)
    if start is not None:
        sql += " AND ex_date >= ?"
        params.append(date.fromisoformat(start))
    if end is not None:
        sql += " AND ex_date <= ?"
        params.append(date.fromisoformat(end))
    sql += " ORDER BY ex_date ASC, action_type ASC"
    rows = _duck(sql, params)
    out = []
    for r in rows:
        d = dict(
            zip(
                (
                    "action_type",
                    "ex_date",
                    "split_from",
                    "split_to",
                    "cash_amount",
                    "currency",
                    "declaration_date",
                    "record_date",
                    "pay_date",
                ),
                r,
            )
        )
        for k in ("ex_date", "declaration_date", "record_date", "pay_date"):
            if d[k] is not None:
                d[k] = str(d[k])
        for k in ("split_from", "split_to", "cash_amount"):
            if d[k] is not None:
                d[k] = float(d[k])
        out.append(d)
    return out


class ActionsChecker:
    def sources(self, case: Case, lake: Lake) -> List[Path]:
        return [_corp_action_path(lake, case.expect["args"]["symbol"].upper())]

    def run(self, case: Case, lake: Lake, executor: Any) -> Outcome:
        args = case.expect["args"]
        symbol = args["symbol"]
        try:
            if args.get("start") and args.get("end") and args["start"] > args["end"]:
                raise Reject(400, "invalid_parameter")
            if args.get("action_type") and args["action_type"] not in _ACTION_TYPES:
                raise Reject(400, "invalid_parameter")
            rows = _actions(
                lake,
                symbol,
                action_type=args.get("action_type"),
                start=args.get("start"),
                end=args.get("end"),
            )
            if rows is None:
                raise Reject(404, "unknown_symbol")
        except Reject as rej:
            status, body = executor.execute(case.request)
            code = body.get("error", {}).get("code") if isinstance(body, dict) else None
            if status == rej.status and code == rej.code:
                return Outcome("EXPECTED_REJECTION", f"{status} {code}")
            return Outcome(
                "FAIL",
                f"expected {rej.status} {rej.code}, got {status} {code}: {str(body)[:300]}",
            )
        params = case.request["params"]
        paged = "limit" in params or "offset" in params
        status, body = executor.execute(case.request)
        if status != 200:
            return Outcome(
                "FAIL",
                f"expected 200 with {len(rows)} actions, got {status}: {str(body)[:300]}",
            )
        problems = []
        if paged:
            limit, offset = params.get("limit", 100), params.get("offset", 0)
            page, _ = _page_slice(rows, limit, offset)
            problems += _mismatches(
                _page_fields(rows, limit, offset),
                body,
                ("limit", "offset", "returned", "truncated", "next_offset"),
            )
        else:
            page = rows
        got = body.get("actions", [])
        if len(got) != len(page):
            problems.append(f"actions: expected {len(page)} got {len(got)}")
        else:
            for i, (e, g) in enumerate(zip(page, got)):
                bad = _mismatches(e, g, _ACTION_FIELDS)
                if bad:
                    problems.append(f"action {i}: " + "; ".join(bad))
        if body.get("symbol") != symbol.upper() or body.get("identity") != "ticker":
            problems.append(f"symbol/identity: got {body.get('symbol')!r}/{body.get('identity')!r}")
        if body.get("count") != len(page):
            problems.append(f"count: expected {len(page)} got {body.get('count')}")
        if problems:
            return Outcome("FAIL", "; ".join(problems))
        return Outcome("PASS")


def _gen_actions(samples: Dict[Tuple[str, str, str], Any]) -> List[Case]:
    cases: List[Case] = []
    equity_symbols = []
    for residency in RESIDENCIES:
        s = samples.get(("equity", "1d", residency))
        if s is not None:
            equity_symbols.append((residency, s))
    if not equity_symbols:
        return [
            Case(
                "actions:no_sample",
                "actions",
                {},
                _req("/v1/equity/NOSAMPLE/actions"),
                _blocked("no equity 1d sample in inventory"),
            )
        ]
    for residency, sample in equity_symbols:
        symbol = sample["symbol"]
        cases.append(
            Case(
                f"actions:{residency}:all",
                "actions",
                {"residency": residency},
                _req(f"/v1/equity/{symbol}/actions"),
                _value("actions", symbol=symbol, action_type=None, start=None, end=None),
            )
        )
        for at in _ACTION_TYPES:
            cases.append(
                Case(
                    f"actions:{residency}:{at}",
                    "actions",
                    {"residency": residency, "type": at},
                    _req(f"/v1/equity/{symbol}/actions", {"type": at}),
                    _value("actions", symbol=symbol, action_type=at, start=None, end=None),
                )
            )
        mid = sample["first"] + (sample["last"] - sample["first"]) // 2
        cases.append(
            Case(
                f"actions:{residency}:window",
                "actions",
                {"residency": residency, "window": "bounded"},
                _req(
                    f"/v1/equity/{symbol}/actions",
                    {
                        "start": sample["first"].isoformat(),
                        "end": mid.isoformat(),
                        "limit": 20,
                    },
                ),
                _value(
                    "actions",
                    symbol=symbol,
                    action_type=None,
                    start=sample["first"].isoformat(),
                    end=mid.isoformat(),
                ),
            )
        )
    first_symbol = equity_symbols[0][1]["symbol"]
    cases.append(
        Case(
            "actions:bad_type",
            "actions",
            {},
            _req(f"/v1/equity/{first_symbol}/actions", {"type": "bogus"}),
            _rejection(400, "invalid_parameter"),
        )
    )
    cases.append(
        Case(
            "actions:reversed",
            "actions",
            {},
            _req(
                f"/v1/equity/{first_symbol}/actions",
                {"start": "2020-06-01", "end": "2020-01-01"},
            ),
            _rejection(400, "invalid_parameter"),
        )
    )
    cases.append(
        Case(
            "actions:unknown",
            "actions",
            {},
            _req("/v1/equity/ZZZZNOSUCHSYMBOL/actions"),
            _value(
                "actions",
                symbol="ZZZZNOSUCHSYMBOL",
                action_type=None,
                start=None,
                end=None,
            ),
        )
    )
    return cases


# ==== 8. GET /v1/equity/{symbol}/delisting ==========================================

_INTERVAL_FIELDS = (
    "security_id",
    "symbol",
    "issuer_name",
    "exchange_mic",
    "currency",
    "effective_from",
    "effective_to",
    "status",
    "continuity_basis",
    "relationship_type",
    "related_security_id",
)


def _identity_intervals(lake: Lake, symbol: str) -> Optional[List[Dict[str, Any]]]:
    path = _security_master_path(lake)
    if not path.is_file():
        return None
    sql = (
        "WITH src AS (SELECT * FROM read_parquet(?)), "
        "superseded AS (SELECT DISTINCT CAST(supersedes AS VARCHAR) AS s FROM src WHERE supersedes IS NOT NULL) "
        "SELECT security_id, symbol, issuer_name, exchange_mic, currency, effective_from, effective_to, "
        "status, continuity_basis, relationship_type, related_security_id FROM src "
        "WHERE symbol = ? AND status = 'verified' AND CAST(event_id AS VARCHAR) NOT IN (SELECT s FROM superseded) "
        "ORDER BY effective_from ASC, security_id ASC"
    )
    rows = _duck(sql, [path.as_posix(), symbol.upper()])
    out = []
    for r in rows:
        d = dict(zip(_INTERVAL_FIELDS, r))
        d["security_id"] = str(d["security_id"])
        d["symbol"] = str(d["symbol"])
        for k in ("effective_from", "effective_to"):
            if d[k] is not None:
                d[k] = str(d[k])
        out.append(d)
    return out


class DelistingChecker:
    def sources(self, case: Case, lake: Lake) -> List[Path]:
        return [_security_master_path(lake)]

    def run(self, case: Case, lake: Lake, executor: Any) -> Outcome:
        symbol = case.expect["args"]["symbol"]
        intervals = _identity_intervals(lake, symbol)
        status, body = executor.execute(case.request)
        if intervals is None:
            if (
                status == 503
                and isinstance(body, dict)
                and body.get("error", {}).get("code") == "provider_not_configured"
            ):
                return Outcome("EXPECTED_REJECTION", "503 provider_not_configured")
            return Outcome(
                "FAIL",
                f"security master not configured; expected 503, got {status}: {str(body)[:300]}",
            )
        if not intervals:
            if (
                status == 404
                and isinstance(body, dict)
                and body.get("error", {}).get("code") == "unknown_symbol"
            ):
                return Outcome("EXPECTED_REJECTION", "404 unknown_symbol")
            return Outcome("FAIL", f"expected 404 unknown_symbol, got {status}: {str(body)[:300]}")
        if status != 200:
            return Outcome(
                "FAIL",
                f"expected 200 with {len(intervals)} intervals, got {status}: {str(body)[:300]}",
            )
        params = case.request["params"]
        paged = "limit" in params or "offset" in params
        problems = []
        if paged:
            limit, offset = params.get("limit", 100), params.get("offset", 0)
            page, _ = _page_slice(intervals, limit, offset)
            problems += _mismatches(
                _page_fields(intervals, limit, offset),
                body,
                ("limit", "offset", "returned", "truncated", "next_offset"),
            )
        else:
            page = intervals
        got = body.get("intervals", [])
        if len(got) != len(page):
            problems.append(f"intervals: expected {len(page)} got {len(got)}")
        else:
            for i, (e, g) in enumerate(zip(page, got)):
                bad = _mismatches(e, g, _INTERVAL_FIELDS)
                if bad:
                    problems.append(f"interval {i}: " + "; ".join(bad))
        if body.get("delisting_reason_available") is not False:
            problems.append(
                f"delisting_reason_available: expected False got {body.get('delisting_reason_available')!r}"
            )
        if body.get("count") != len(page):
            problems.append(f"count: expected {len(page)} got {body.get('count')}")
        if problems:
            return Outcome("FAIL", "; ".join(problems))
        return Outcome("PASS")


def _gen_delisting(samples: Dict[Tuple[str, str, str], Any]) -> List[Case]:
    cases: List[Case] = [
        Case(
            "delisting:unknown",
            "delisting",
            {},
            _req("/v1/equity/ZZZZNOSUCHSYMBOL/delisting"),
            _value("delisting", symbol="ZZZZNOSUCHSYMBOL"),
        ),
    ]
    live = samples.get(("equity", "1d", "live_only"))
    dual = samples.get(("equity", "1d", "dual"))
    archive = samples.get(("equity", "1d", "archive_only"))
    for label, sample in (("live", live), ("dual", dual), ("archive", archive)):
        if sample is None:
            continue
        symbol = sample["symbol"]
        cases.append(
            Case(
                f"delisting:{label}",
                "delisting",
                {"residency": label},
                _req(f"/v1/equity/{symbol}/delisting"),
                _value("delisting", symbol=symbol),
            )
        )
        cases.append(
            Case(
                f"delisting:{label}:paged",
                "delisting",
                {"residency": label},
                _req(f"/v1/equity/{symbol}/delisting", {"limit": 1, "offset": 0}),
                _value("delisting", symbol=symbol),
            )
        )
    return cases


# ==== 9. GET /v1/security/{symbol} ===================================================


def _security_rows(
    lake: Lake, symbol: str, as_of: date, known_at: Optional[date]
) -> Optional[List[str]]:
    path = _security_master_path(lake)
    if not path.is_file():
        return None
    source = "SELECT * FROM read_parquet(?)"
    params: List[Any] = [path.as_posix()]
    if known_at is not None:
        source += " WHERE known_at <= ?"
        params.append(_end_of_day(known_at))
    venues = ", ".join("?" for _ in _SYMBOL_EXCHANGES)
    sql = (
        f"WITH src AS ({source}), "
        "superseded AS (SELECT DISTINCT CAST(supersedes AS VARCHAR) AS s FROM src WHERE supersedes IS NOT NULL), "
        "current_rows AS (SELECT * FROM src WHERE status = 'verified' AND CAST(event_id AS VARCHAR) NOT IN (SELECT s FROM superseded)) "
        f"SELECT DISTINCT security_id FROM current_rows WHERE symbol = ? AND provider = ? AND exchange_mic IN ({venues}) "
        "AND effective_from <= ? AND (effective_to IS NULL OR effective_to > ?)"
    )
    instant = _start_of_day(as_of)
    params += [symbol.upper(), _SYMBOL_PROVIDER, *_SYMBOL_EXCHANGES, instant, instant]
    rows = _duck(sql, params)
    return [str(r[0]) for r in rows]


class SecurityChecker:
    def sources(self, case: Case, lake: Lake) -> List[Path]:
        return [_security_master_path(lake)]

    def run(self, case: Case, lake: Lake, executor: Any) -> Outcome:
        args = case.expect["args"]
        symbol, as_of, known_at = (
            args["symbol"],
            date.fromisoformat(args["as_of"]),
            (date.fromisoformat(args["known_at"]) if args.get("known_at") else None),
        )
        ids = _security_rows(lake, symbol, as_of, known_at)
        status, body = executor.execute(case.request)
        if ids is None:
            if (
                status == 503
                and isinstance(body, dict)
                and body.get("error", {}).get("code") == "membership_unavailable"
            ):
                return Outcome("EXPECTED_REJECTION", "503 membership_unavailable")
            return Outcome(
                "FAIL",
                f"security master unavailable; expected 503, got {status}: {str(body)[:300]}",
            )
        distinct = sorted(set(ids))
        if len(distinct) > 1:
            if (
                status == 404
                and isinstance(body, dict)
                and body.get("error", {}).get("code") == "ambiguous_security"
            ):
                return Outcome("EXPECTED_REJECTION", "404 ambiguous_security")
            return Outcome(
                "FAIL",
                f"expected 404 ambiguous_security ({distinct}), got {status}: {str(body)[:300]}",
            )
        if not distinct:
            if (
                status == 404
                and isinstance(body, dict)
                and body.get("error", {}).get("code") == "unknown_symbol"
            ):
                return Outcome("EXPECTED_REJECTION", "404 unknown_symbol")
            return Outcome("FAIL", f"expected 404 unknown_symbol, got {status}: {str(body)[:300]}")
        if status != 200:
            return Outcome(
                "FAIL",
                f"expected 200 security_id={distinct[0]}, got {status}: {str(body)[:300]}",
            )
        expected = {
            "symbol": symbol.upper(),
            "security_id": distinct[0],
            "as_of": as_of.isoformat(),
            "known_at": None if known_at is None else known_at.isoformat(),
            "knowledge": "today" if known_at is None else "as_known_at",
        }
        bad = _mismatches(expected, body, tuple(expected))
        if bad:
            return Outcome("FAIL", "; ".join(bad))
        return Outcome("PASS")


def _gen_security(lake: Lake, samples: Dict[Tuple[str, str, str], Any]) -> List[Case]:
    cases: List[Case] = [
        Case(
            "security:unknown",
            "security",
            {},
            _req("/v1/security/ZZZZNOSUCHSYMBOL"),
            _value("security", symbol="ZZZZNOSUCHSYMBOL", as_of=_today().isoformat()),
        ),
    ]
    equity_symbols = [s["symbol"] for k, s in samples.items() if k[0] == "equity" and k[1] == "1d"]
    probes = [
        ("today", _today().isoformat(), None),
        ("far_past", "1990-01-01", None),
        ("far_future", "2999-01-01", None),
        ("known_at_past", _today().isoformat(), "2015-01-01"),
        ("known_at_today", _today().isoformat(), _today().isoformat()),
    ]
    for symbol in equity_symbols[:3]:
        for label, as_of, known_at in probes:
            dims = {"symbol": symbol, "probe": label}
            params = {"as_of": as_of}
            if known_at:
                params["known_at"] = known_at
            args = {"symbol": symbol, "as_of": as_of}
            if known_at:
                args["known_at"] = known_at
            cases.append(
                Case(
                    f"security:{symbol}:{label}",
                    "security",
                    dims,
                    _req(f"/v1/security/{symbol}", params),
                    _value("security", **args),
                )
            )
    # A resolvable probe on the PASS path: none of the inventory equity samples need
    # carry a provider='massive' security_master row (VSCO/SPY/FEUL don't), so pick a
    # real one directly to exercise a successful resolution, not just unknown_symbol.
    path = _security_master_path(lake)
    resolvable = None
    if path.is_file():
        instant = _start_of_day(_today())
        rows = _duck(
            "SELECT DISTINCT symbol FROM read_parquet(?) WHERE provider = ? AND status = 'verified' "
            "AND effective_from <= ? AND (effective_to IS NULL OR effective_to > ?) LIMIT 1",
            [path.as_posix(), _SYMBOL_PROVIDER, instant, instant],
        )
        resolvable = rows[0][0] if rows else None
    if resolvable:
        cases.append(
            Case(
                f"security:resolvable:{resolvable}",
                "security",
                {"symbol": resolvable},
                _req(f"/v1/security/{resolvable}"),
                _value("security", symbol=resolvable, as_of=_today().isoformat()),
            )
        )
    # identity-seam probe: a symbol with >=2 distinct verified non-superseded security_ids today
    seam = None
    if path.is_file():
        sql = (
            "WITH src AS (SELECT * FROM read_parquet(?)), "
            "superseded AS (SELECT DISTINCT CAST(supersedes AS VARCHAR) AS s FROM src WHERE supersedes IS NOT NULL), "
            "cur AS (SELECT * FROM src WHERE status='verified' AND CAST(event_id AS VARCHAR) NOT IN (SELECT s FROM superseded) "
            "  AND provider = ? AND effective_from <= ? AND (effective_to IS NULL OR effective_to > ?)) "
            "SELECT symbol FROM cur GROUP BY symbol HAVING count(DISTINCT security_id) > 1 LIMIT 1"
        )
        instant = _start_of_day(_today())
        rows = _duck(sql, [path.as_posix(), _SYMBOL_PROVIDER, instant, instant])
        seam = rows[0][0] if rows else None
    if seam:
        cases.append(
            Case(
                f"security:seam:{seam}",
                "security",
                {"symbol": seam},
                _req(f"/v1/security/{seam}"),
                _value("security", symbol=seam, as_of=_today().isoformat()),
            )
        )
    else:
        cases.append(
            Case(
                "security:seam",
                "security",
                {},
                _req("/v1/security/ZZZZSEAMPROBE"),
                _blocked(
                    "no identity-seam ticker (>=2 distinct verified security_ids today) found in security_master"
                ),
            )
        )
    return cases


# ==== 10. Membership ==================================================================


def _membership_indices(lake: Lake) -> List[str]:
    root = lake.root / "index_membership"
    if not root.is_dir():
        return []
    return sorted(p.name for p in root.iterdir() if p.is_dir() and (p / "events.parquet").is_file())


def _membership_events(
    lake: Lake,
    index_id: str,
    *,
    known_at: Optional[date] = None,
    security_ids: Sequence[str] = (),
) -> List[Dict[str, Any]]:
    path = _membership_events_path(lake, index_id)
    clauses = ["TRUE"]
    params: List[Any] = [path.as_posix()]
    if known_at is not None:
        clauses.append("known_at <= ?")
        params.append(_end_of_day(known_at))
    if security_ids:
        clauses.append("security_id IN (" + ", ".join("?" for _ in security_ids) + ")")
        params.extend(security_ids)
    sql = (
        "SELECT index_id, security_id, action, announced_at, effective_at, known_at, revision, status, event_id, supersedes "
        f"FROM read_parquet(?) WHERE {' AND '.join(clauses)} ORDER BY effective_at, known_at, revision, event_id"
    )
    rows = _duck(sql, params)
    out = []
    for r in rows:
        keys = (
            "index_id",
            "security_id",
            "action",
            "announced_at",
            "effective_at",
            "known_at",
            "revision",
            "status",
            "event_id",
            "supersedes",
        )
        d = dict(zip(keys, r))
        d["effective_at"] = _as_utc(d["effective_at"])
        d["known_at"] = _as_utc(d["known_at"])
        d["announced_at"] = _as_utc(d["announced_at"])
        if d["effective_at"] is None or d["known_at"] is None:
            continue
        d["index_id"] = str(d["index_id"] or index_id)
        d["security_id"] = str(d["security_id"])
        d["revision"] = int(d["revision"] or 0)
        d["status"] = str(d["status"])
        d["event_id"] = str(d["event_id"])
        d["supersedes"] = None if d["supersedes"] is None else str(d["supersedes"])
        out.append(d)
    return out


def _members_as_of(
    lake: Lake,
    index_id: str,
    as_of: date,
    *,
    known_at: Optional[date],
    include_candidates: bool,
) -> List[str]:
    events = _membership_events(lake, index_id, known_at=known_at)
    superseded = {e["supersedes"] for e in events if e["supersedes"] is not None}
    cutoff = _end_of_day(as_of)
    applicable = [
        e
        for e in events
        if e["event_id"] not in superseded
        and e["effective_at"] <= cutoff
        and ((e["status"] != "rejected") if include_candidates else (e["status"] == "verified"))
    ]
    members: set = set()
    for e in applicable:
        if e["action"] == "add":
            members.add(e["security_id"])
        elif e["action"] == "remove":
            members.discard(e["security_id"])
    if not include_candidates and members:
        members &= _verified_ids(lake, sorted(members), as_of, known_at)
    return sorted(members)


def _verified_ids(lake: Lake, ids: Sequence[str], as_of: date, known_at: Optional[date]) -> set:
    path = _security_master_path(lake)
    if not ids or not path.is_file():
        return set()
    source = "SELECT * FROM read_parquet(?)"
    params: List[Any] = [path.as_posix()]
    if known_at is not None:
        source += " WHERE known_at <= ?"
        params.append(_end_of_day(known_at))
    id_ph = ", ".join("?" for _ in ids)
    sql = (
        f"WITH src AS ({source}), "
        "superseded AS (SELECT DISTINCT CAST(supersedes AS VARCHAR) AS s FROM src WHERE supersedes IS NOT NULL), "
        "cur AS (SELECT * FROM src WHERE status='verified' AND CAST(event_id AS VARCHAR) NOT IN (SELECT s FROM superseded)) "
        f"SELECT DISTINCT security_id FROM cur WHERE security_id IN ({id_ph}) "
        "AND effective_from <= ? AND (effective_to IS NULL OR effective_to > ?)"
    )
    instant = _start_of_day(as_of)
    params += [*ids, instant, instant]
    return {str(r[0]) for r in _duck(sql, params)}


def _has_events_for_status(lake: Lake, index_id: str, include_candidates: bool) -> bool:
    path = _membership_events_path(lake, index_id)
    if include_candidates:
        sql = "SELECT 1 FROM read_parquet(?) WHERE status IS DISTINCT FROM 'rejected' LIMIT 1"
    else:
        sql = "SELECT 1 FROM read_parquet(?) WHERE status = 'verified' LIMIT 1"
    return bool(_duck(sql, [path.as_posix()]))


class MembershipIndicesChecker:
    def sources(self, case: Case, lake: Lake) -> List[Path]:
        return [lake.root / "index_membership"]

    def run(self, case: Case, lake: Lake, executor: Any) -> Outcome:
        expected = _membership_indices(lake)
        status, body = executor.execute(case.request)
        if status != 200:
            return Outcome("FAIL", f"expected 200, got {status}: {str(body)[:300]}")
        if body.get("indices") != expected:
            return Outcome("FAIL", f"indices: expected {expected} got {body.get('indices')}")
        return Outcome("PASS")


class MembershipMembersChecker:
    def sources(self, case: Case, lake: Lake) -> List[Path]:
        index_id = case.expect["args"]["index_id"]
        return [_membership_events_path(lake, index_id), _security_master_path(lake)]

    def run(self, case: Case, lake: Lake, executor: Any) -> Outcome:
        args = case.expect["args"]
        index_id, include_candidates = args["index_id"], args["include_candidates"]
        as_of = date.fromisoformat(args["as_of"])
        known_at = date.fromisoformat(args["known_at"]) if args.get("known_at") else None
        if not _membership_events_path(lake, index_id).is_file():
            status, body = executor.execute(case.request)
            if (
                status == 404
                and isinstance(body, dict)
                and body.get("error", {}).get("code") == "unknown_index"
            ):
                return Outcome("EXPECTED_REJECTION", "404 unknown_index")
            return Outcome("FAIL", f"expected 404 unknown_index, got {status}: {str(body)[:300]}")
        members = _members_as_of(
            lake,
            index_id,
            as_of,
            known_at=known_at,
            include_candidates=include_candidates,
        )
        status, body = executor.execute(case.request)
        if not members and not _has_events_for_status(lake, index_id, include_candidates):
            if (
                status == 503
                and isinstance(body, dict)
                and body.get("error", {}).get("code") == "membership_unavailable"
            ):
                return Outcome("EXPECTED_REJECTION", "503 membership_unavailable")
            return Outcome(
                "FAIL",
                f"expected 503 membership_unavailable, got {status}: {str(body)[:300]}",
            )
        if status != 200:
            return Outcome(
                "FAIL",
                f"expected 200 with {len(members)} members, got {status}: {str(body)[:300]}",
            )
        got_ids = sorted(m["security_id"] for m in body.get("members", []))
        problems = []
        if (
            got_ids != members
            and "limit" not in case.request["params"]
            and "offset" not in case.request["params"]
        ):
            problems.append(f"member ids: expected {members} got {got_ids}")
        if body.get("unresolved_count") != sum(1 for m in members if m.startswith("unresolved:")):
            problems.append(
                f"unresolved_count: expected {sum(1 for m in members if m.startswith('unresolved:'))} got {body.get('unresolved_count')}"
            )
        params = case.request["params"]
        if "limit" in params or "offset" in params:
            limit, offset = params.get("limit", 100), params.get("offset", 0)
            page, _ = _page_slice(members, limit, offset)
            if body.get("total") != len(members):
                problems.append(f"total: expected {len(members)} got {body.get('total')}")
            problems += _mismatches(
                _page_fields(members, limit, offset),
                body,
                ("limit", "offset", "returned", "truncated", "next_offset"),
            )
            got_page_ids = [m["security_id"] for m in body.get("members", [])]
            if got_page_ids != page:
                problems.append(f"page ids: expected {page} got {got_page_ids}")
        if problems:
            return Outcome("FAIL", "; ".join(problems))
        return Outcome("PASS")


class MembershipHistoryChecker:
    def sources(self, case: Case, lake: Lake) -> List[Path]:
        return [_security_master_path(lake), lake.root / "index_membership"]

    def run(self, case: Case, lake: Lake, executor: Any) -> Outcome:
        args = case.expect["args"]
        symbol, as_of = args["symbol"], date.fromisoformat(args["as_of"])
        index_id = args.get("index_id")
        ids = _security_rows(lake, symbol, as_of, None)
        status, body = executor.execute(case.request)
        if ids is None:
            if status == 503:
                return Outcome("EXPECTED_REJECTION", "503")
            return Outcome("FAIL", f"expected 503, got {status}: {str(body)[:300]}")
        distinct = sorted(set(ids))
        if len(distinct) > 1:
            if (
                status == 404
                and isinstance(body, dict)
                and body.get("error", {}).get("code") == "ambiguous_security"
            ):
                return Outcome("EXPECTED_REJECTION", "404 ambiguous_security")
            return Outcome(
                "FAIL",
                f"expected 404 ambiguous_security, got {status}: {str(body)[:300]}",
            )
        placeholder = f"unresolved:{symbol.upper()}"
        probe_ids = [placeholder] if not distinct else [distinct[0], placeholder]
        indices = [index_id] if index_id else _membership_indices(lake)
        events: List[Dict[str, Any]] = []
        for idx in indices:
            if not _membership_events_path(lake, idx).is_file():
                continue
            events += _membership_events(lake, idx, security_ids=probe_ids)
        superseded = {e["supersedes"] for e in events if e["supersedes"] is not None}
        events = [
            e for e in events if e["event_id"] not in superseded and e["status"] != "rejected"
        ]
        events.sort(
            key=lambda e: (
                e["effective_at"],
                e["known_at"],
                e["revision"],
                e["event_id"],
            )
        )
        security_id = distinct[0] if distinct else (placeholder if events else None)
        if security_id is None:
            if (
                status == 404
                and isinstance(body, dict)
                and body.get("error", {}).get("code") == "unknown_symbol"
            ):
                return Outcome("EXPECTED_REJECTION", "404 unknown_symbol")
            return Outcome("FAIL", f"expected 404 unknown_symbol, got {status}: {str(body)[:300]}")
        if status != 200:
            return Outcome(
                "FAIL",
                f"expected 200 with {len(events)} events, got {status}: {str(body)[:300]}",
            )
        if body.get("security_id") != security_id or body.get("symbol") != symbol.upper():
            return Outcome(
                "FAIL",
                f"symbol/security_id: expected {symbol.upper()}/{security_id} got {body.get('symbol')}/{body.get('security_id')}",
            )
        got = body.get("events", [])
        if len(got) != len(events):
            return Outcome("FAIL", f"events: expected {len(events)} got {len(got)}")
        problems = []
        for i, (e, g) in enumerate(zip(events, got)):
            expected_event = {
                "index_id": e["index_id"],
                "security_id": e["security_id"],
                "action": e["action"],
                "effective_at": _iso_or_none(e["effective_at"]),
                "announced_at": _iso_or_none(e["announced_at"]),
                "known_at": _iso_or_none(e["known_at"]),
                "status": e["status"],
                "event_id": e["event_id"],
                "supersedes": e["supersedes"],
            }
            bad = _mismatches(expected_event, g, tuple(expected_event))
            if bad:
                problems.append(f"event {i}: " + "; ".join(bad))
        if problems:
            return Outcome("FAIL", "; ".join(problems))
        return Outcome("PASS")


def _gen_membership(lake: Lake, samples: Dict[Tuple[str, str, str], Any]) -> List[Case]:
    cases: List[Case] = [
        Case(
            "membership_indices:list",
            "membership_indices",
            {},
            _req("/v1/membership/indices"),
            _value("membership_indices"),
        ),
        Case(
            "membership_members:unknown_index",
            "membership_members",
            {},
            _req("/v1/membership/ZZZZNOSUCHINDEX"),
            _value(
                "membership_members",
                index_id="ZZZZNOSUCHINDEX",
                as_of=_today().isoformat(),
                include_candidates=False,
            ),
        ),
    ]
    cases += [
        Case(
            f"membership_members:reject:{lbl}",
            "membership_members",
            {},
            _req("/v1/membership/sp500", {**extra}),
            _rejection(400, "invalid_parameter"),
        )
        for lbl, extra in (
            ("limit_zero", {"limit": 0}),
            ("limit_over", {"limit": 2001}),
            ("offset_neg", {"offset": -1}),
        )
    ]
    indices = _membership_indices(lake)
    if not indices:
        cases.append(
            Case(
                "membership_members:no_indices",
                "membership_members",
                {},
                _req("/v1/membership/none"),
                _blocked("no index_membership directories found on disk"),
            )
        )
    biggest_index, biggest_count = None, -1
    for idx in indices:
        for include_candidates in (False, True):
            cases.append(
                Case(
                    f"membership_members:{idx}:{include_candidates}",
                    "membership_members",
                    {"index_id": idx, "include_candidates": include_candidates},
                    _req(
                        f"/v1/membership/{idx}",
                        {
                            "as_of": _today().isoformat(),
                            "include_candidates": include_candidates,
                        },
                    ),
                    _value(
                        "membership_members",
                        index_id=idx,
                        as_of=_today().isoformat(),
                        include_candidates=include_candidates,
                    ),
                )
            )
        n = len(_members_as_of(lake, idx, _today(), known_at=None, include_candidates=False))
        if n > biggest_count:
            biggest_index, biggest_count = idx, n
    if biggest_index and biggest_count > 2:
        cases.append(
            Case(
                f"membership_members:{biggest_index}:paged",
                "membership_members",
                {"index_id": biggest_index},
                _req(
                    f"/v1/membership/{biggest_index}",
                    {
                        "as_of": _today().isoformat(),
                        "limit": 2,
                        "offset": biggest_count // 2,
                    },
                ),
                _value(
                    "membership_members",
                    index_id=biggest_index,
                    as_of=_today().isoformat(),
                    include_candidates=False,
                ),
            )
        )
    equity_symbols = [s["symbol"] for k, s in samples.items() if k[0] == "equity" and k[1] == "1d"][
        :2
    ]
    # None of the inventory samples need resolve through the security master (see
    # _gen_security); add one real symbol that does, to exercise the PASS path too.
    sm_path = _security_master_path(lake)
    if sm_path.is_file():
        instant = _start_of_day(_today())
        rows = _duck(
            "SELECT DISTINCT symbol FROM read_parquet(?) WHERE provider = ? AND status = 'verified' "
            "AND effective_from <= ? AND (effective_to IS NULL OR effective_to > ?) LIMIT 1",
            [sm_path.as_posix(), _SYMBOL_PROVIDER, instant, instant],
        )
        if rows and rows[0][0] not in equity_symbols:
            equity_symbols.append(rows[0][0])
    for symbol in equity_symbols:
        cases.append(
            Case(
                f"membership_history:{symbol}",
                "membership_history",
                {"symbol": symbol},
                _req(
                    "/v1/membership/history",
                    {"symbol": symbol, "as_of": _today().isoformat()},
                ),
                _value("membership_history", symbol=symbol, as_of=_today().isoformat()),
            )
        )
        if indices:
            cases.append(
                Case(
                    f"membership_history:{symbol}:{indices[0]}",
                    "membership_history",
                    {"symbol": symbol, "index_id": indices[0]},
                    _req(
                        "/v1/membership/history",
                        {
                            "symbol": symbol,
                            "as_of": _today().isoformat(),
                            "index_id": indices[0],
                        },
                    ),
                    _value(
                        "membership_history",
                        symbol=symbol,
                        as_of=_today().isoformat(),
                        index_id=indices[0],
                    ),
                )
            )
    return cases


# ==== 11. Revisions ====================================================================


def _read_json(path: Path) -> Any:
    return json.loads(path.read_bytes())


class SilverRevisionsChecker:
    def sources(self, case: Case, lake: Lake) -> List[Path]:
        return [lake.silver / "revisions" / "current.json"]

    def run(self, case: Case, lake: Lake, executor: Any) -> Outcome:
        numbers = sorted(
            (
                int(p.name[len("revision=") : -len(".json")])
                for p in (lake.silver / "revisions").glob("revision=*.json")
            ),
            reverse=True,
        )
        current = lake.silver_current_number()
        params = case.request["params"]
        limit, offset = params.get("limit", 100), params.get("offset", 0)
        status, body = executor.execute(case.request)
        if status != 200:
            return Outcome("FAIL", f"expected 200, got {status}: {str(body)[:300]}")
        page, _ = _page_slice(numbers, limit, offset)
        problems = _mismatches(
            _page_fields(numbers, limit, offset),
            body,
            ("limit", "offset", "returned", "truncated", "next_offset"),
        )
        if body.get("current") != current:
            problems.append(f"current: expected {current} got {body.get('current')}")
        got = [(r["revision"], r["is_current"]) for r in body.get("revisions", [])]
        exp = [(n, n == current) for n in page]
        if got != exp:
            problems.append(f"revisions: expected {exp} got {got}")
        if problems:
            return Outcome("FAIL", "; ".join(problems))
        return Outcome("PASS")


class SilverRevisionDetailChecker:
    def sources(self, case: Case, lake: Lake) -> List[Path]:
        rev = case.expect["args"].get("revision")
        paths = [lake.silver / "revisions" / "current.json"]
        if rev is not None:
            paths.append(lake.silver / "revisions" / f"revision={rev}.json")
        return paths

    def run(self, case: Case, lake: Lake, executor: Any) -> Outcome:
        revision = case.expect["args"]["revision"]
        path = lake.silver / "revisions" / f"revision={revision}.json"
        if not path.is_file():
            status, body = executor.execute(case.request)
            if (
                status == 404
                and isinstance(body, dict)
                and body.get("error", {}).get("code") == "unknown_revision"
            ):
                return Outcome("EXPECTED_REJECTION", "404 unknown_revision")
            return Outcome(
                "FAIL",
                f"expected 404 unknown_revision, got {status}: {str(body)[:300]}",
            )
        manifest = _read_json(path)
        current = lake.silver_current_number()
        affected = manifest["affected"]
        params = case.request["params"]
        limit, offset = params.get("limit", 100), params.get("offset", 0)
        page, _ = _page_slice(affected, limit, offset)
        status, body = executor.execute(case.request)
        if status != 200:
            return Outcome("FAIL", f"expected 200, got {status}: {str(body)[:300]}")
        problems = _mismatches(
            {
                "revision": manifest["revision"],
                "is_current": manifest["revision"] == current,
                "generation_id": manifest["generation_id"],
                "affected_count": len(affected),
                "artifact_count": len(manifest["artifacts"]),
            },
            body,
            (
                "revision",
                "is_current",
                "generation_id",
                "affected_count",
                "artifact_count",
            ),
        )
        problems += _mismatches(
            _page_fields(affected, limit, offset),
            body,
            ("limit", "offset", "returned", "truncated", "next_offset"),
        )
        got = [(a["symbol"], a["earliest_date"], a["timeframes"]) for a in body.get("affected", [])]
        exp = [(a["symbol"], a["earliest_date"], a["timeframes"]) for a in page]
        if got != exp:
            problems.append(f"affected: expected {exp[:3]}... got {got[:3]}...")
        if problems:
            return Outcome("FAIL", "; ".join(problems))
        return Outcome("PASS")


def _pit_daily_artifact_count(manifest: Dict[str, Any]) -> int:
    entries = manifest.get("inputs", {}).get("silver_artifacts", [])
    return sum(1 for e in entries if str(e.get("path", "")).endswith("/1d.parquet"))


class PitRevisionsChecker:
    def sources(self, case: Case, lake: Lake) -> List[Path]:
        return [lake.silver / "pit-revisions"]

    def run(self, case: Case, lake: Lake, executor: Any) -> Outcome:
        index_id = case.expect["args"].get("index_id")
        numbers = lake.pit_numbers()
        summaries = [lake.pit(n) for n in numbers]
        latest: Dict[str, int] = {}
        for m in summaries:
            latest.setdefault(m["index_id"], m["revision"])
        if index_id is not None:
            summaries = [m for m in summaries if m["index_id"] == index_id]
            latest = {k: v for k, v in latest.items() if k == index_id}
        params = case.request["params"]
        limit, offset = params.get("limit", 100), params.get("offset", 0)
        page, _ = _page_slice(summaries, limit, offset)
        status, body = executor.execute(case.request)
        if status != 200:
            return Outcome("FAIL", f"expected 200, got {status}: {str(body)[:300]}")
        problems = []
        if body.get("latest_per_index") != latest:
            problems.append(
                f"latest_per_index: expected {latest} got {body.get('latest_per_index')}"
            )
        problems += _mismatches(
            _page_fields(summaries, limit, offset),
            body,
            ("limit", "offset", "returned", "truncated", "next_offset"),
        )
        got_revs = [r["revision"] for r in body.get("revisions", [])]
        exp_revs = [m["revision"] for m in page]
        if got_revs != exp_revs:
            problems.append(f"revisions: expected {exp_revs} got {got_revs}")
        if problems:
            return Outcome("FAIL", "; ".join(problems))
        return Outcome("PASS")


class PitRevisionDetailChecker:
    def sources(self, case: Case, lake: Lake) -> List[Path]:
        return [lake.silver / "pit-revisions"]

    def run(self, case: Case, lake: Lake, executor: Any) -> Outcome:
        revision = case.expect["args"]["revision"]
        path = lake.silver / "pit-revisions" / f"revision={revision}.json"
        if not path.is_file():
            status, body = executor.execute(case.request)
            if (
                status == 404
                and isinstance(body, dict)
                and body.get("error", {}).get("code") == "unknown_revision"
            ):
                return Outcome("EXPECTED_REJECTION", "404 unknown_revision")
            return Outcome(
                "FAIL",
                f"expected 404 unknown_revision, got {status}: {str(body)[:300]}",
            )
        manifest = _read_json(path)
        members = manifest["members"]
        params = case.request["params"]
        limit, offset = params.get("limit", 100), params.get("offset", 0)
        page, _ = _page_slice(members, limit, offset)
        status, body = executor.execute(case.request)
        if status != 200:
            return Outcome("FAIL", f"expected 200, got {status}: {str(body)[:300]}")
        problems = _mismatches(
            {
                "revision": manifest["revision"],
                "index_id": manifest["index_id"],
                "publisher_status": manifest["status"],
                "as_of": manifest["as_of"],
                "published_at": manifest["published_at"],
                "daily_bar_cutoff": manifest["daily_bar_cutoff"],
                "silver_revision": manifest["silver_revision"],
                "membership_revision": manifest["membership_revision"],
                "member_count": len(members),
                "daily_artifact_count": _pit_daily_artifact_count(manifest),
            },
            body,
            (
                "revision",
                "index_id",
                "publisher_status",
                "silver_revision",
                "membership_revision",
                "member_count",
                "daily_artifact_count",
            ),
        )
        problems += _mismatches(
            _page_fields(members, limit, offset),
            body,
            ("limit", "offset", "returned", "truncated", "next_offset"),
        )
        got_syms = [m["symbol"] for m in body.get("members", [])]
        exp_syms = [m["symbol"] for m in page]
        if got_syms != exp_syms:
            problems.append(f"members order: expected {exp_syms[:5]}... got {got_syms[:5]}...")
        if problems:
            return Outcome("FAIL", "; ".join(problems))
        return Outcome("PASS")


def _gen_revisions(lake: Lake) -> List[Case]:
    cases: List[Case] = []
    cases += _bad_page_cases("silver_revisions", "/v1/lake/silver-revisions", {})
    cases.append(
        Case(
            "silver_revisions:list",
            "silver_revisions",
            {},
            _req("/v1/lake/silver-revisions", {"limit": 10}),
            _value("silver_revisions"),
        )
    )
    numbers = sorted(
        (
            int(p.name[len("revision=") : -len(".json")])
            for p in (lake.silver / "revisions").glob("revision=*.json")
        ),
        reverse=True,
    )
    current = lake.silver_current_number()
    for label, rev in (
        ("current", current),
        ("older", numbers[1] if len(numbers) > 1 else None),
    ):
        if rev is None:
            continue
        cases.append(
            Case(
                f"silver_revision_detail:{label}",
                "silver_revision_detail",
                {"revision": rev},
                _req(f"/v1/lake/silver-revisions/{rev}", {"limit": 10}),
                _value("silver_revision_detail", revision=rev),
            )
        )
    cases.append(
        Case(
            "silver_revision_detail:unknown",
            "silver_revision_detail",
            {},
            _req("/v1/lake/silver-revisions/999999"),
            _value("silver_revision_detail", revision=999999),
        )
    )
    cases += _bad_page_cases("silver_revision_detail", f"/v1/lake/silver-revisions/{current}", {})

    cases += _bad_page_cases("pit_revisions", "/v1/lake/pit-revisions", {})
    cases.append(
        Case(
            "pit_revisions:all",
            "pit_revisions",
            {},
            _req("/v1/lake/pit-revisions", {"limit": 50}),
            _value("pit_revisions", index_id=None),
        )
    )
    seen_indices = sorted({lake.pit(n)["index_id"] for n in lake.pit_numbers()})
    for idx in seen_indices:
        cases.append(
            Case(
                f"pit_revisions:{idx}",
                "pit_revisions",
                {"index_id": idx},
                _req("/v1/lake/pit-revisions", {"index_id": idx, "limit": 50}),
                _value("pit_revisions", index_id=idx),
            )
        )
    for n in lake.pit_numbers():
        cases.append(
            Case(
                f"pit_revision_detail:{n}",
                "pit_revision_detail",
                {"revision": n},
                _req(f"/v1/lake/pit-revisions/{n}", {"limit": 20}),
                _value("pit_revision_detail", revision=n),
            )
        )
    cases.append(
        Case(
            "pit_revision_detail:current",
            "pit_revision_detail",
            {},
            _req("/v1/lake/pit-revisions/current"),
            _rejection(422, "invalid_parameter"),
        )
    )
    cases.append(
        Case(
            "pit_revision_detail:unknown",
            "pit_revision_detail",
            {},
            _req("/v1/lake/pit-revisions/999999"),
            _value("pit_revision_detail", revision=999999),
        )
    )
    if lake.pit_numbers():
        cases += _bad_page_cases(
            "pit_revision_detail", f"/v1/lake/pit-revisions/{lake.pit_numbers()[0]}", {}
        )
    return cases


# ==== 12. Gaps ==========================================================================

_REPORT_RE = re.compile(r"^(tier_a|decisions)_(\d{4}-\d{2}-\d{2})\.json$")
_KNOWN_REPAIR_FIELDS = ("symbol", "asset_class", "timeframe", "sessions", "session")


def _read_repairs(lake: Lake) -> Tuple[str, List[Dict[str, Any]], List[str], int]:
    root = _repairs_root(lake)
    if not root.is_dir():
        return "absent", [], [f"repairs root is not a directory: {root.name}"], 0
    entries: List[Dict[str, Any]] = []
    warnings: List[str] = []
    count = 0
    for path in sorted(root.iterdir()):
        m = _REPORT_RE.match(path.name)
        if m is None and path.name != "unresolved.json":
            continue
        if not path.is_file():
            continue
        kind = m.group(1) if m else "unresolved"
        report_date = m.group(2) if m else None
        try:
            payload = _read_json(path)
            rows = (
                payload.get("repairs")
                if kind == "tier_a" and isinstance(payload, dict)
                else payload
            )
            if not isinstance(rows, list):
                raise ValueError("report must hold a list")
            for row in rows:
                sessions = row["sessions"] if "sessions" in row else [row["session"]]
                entries.append(
                    {
                        "report_kind": kind,
                        "reports": (path.name,),
                        "report_date": report_date,
                        "symbol": row["symbol"],
                        "asset_class": row["asset_class"],
                        "timeframe": row["timeframe"],
                        "sessions": tuple(sessions),
                        "details": {k: v for k, v in row.items() if k not in _KNOWN_REPAIR_FIELDS},
                    }
                )
            count += 1
        except (OSError, ValueError, TypeError, KeyError) as exc:
            warnings.append(f"{path.name}: {exc}")
    if warnings:
        return "degraded", entries, warnings, count
    if count == 0:
        return "absent", [], ["no repair reports in the repairs root"], 0
    return "available", entries, warnings, count


def _repairs_evidence(
    lake: Lake, symbol: str, asset_class: str, timeframe: str, start: date, end: date
) -> Tuple[str, List[Dict[str, Any]], int]:
    state, raw, _warnings, count = _read_repairs(lake)
    lo, hi = start.isoformat(), end.isoformat()
    merged: Dict[str, Dict[str, Any]] = {}
    for e in raw:
        if (e["symbol"], e["asset_class"], e["timeframe"]) != (
            symbol,
            asset_class,
            timeframe,
        ):
            continue
        if not any(lo <= s <= hi for s in e["sessions"]):
            continue
        key = json.dumps(
            [e["report_kind"], e["sessions"], e["details"]], sort_keys=True, default=str
        )
        prev = merged.get(key)
        merged[key] = (
            e
            if prev is None
            else {
                **e,
                "reports": prev["reports"] + e["reports"],
                "report_date": max(prev["report_date"] or "", e["report_date"] or "") or None,
            }
        )
    return state, list(merged.values()), count


def _xnys_sessions(start: date, end: date) -> List[date]:
    cal = mcal.get_calendar("XNYS")
    return [
        ts.date() for ts in cal.valid_days(start_date=start.isoformat(), end_date=end.isoformat())
    ]


def _weekday_sessions(start: date, end: date) -> List[date]:
    out = []
    d = start
    while d <= end:
        if d.weekday() < 5:
            out.append(d)
        d += timedelta(days=1)
    return out


def _session_presence(
    lake: Lake, paths: List[Path], timeframe: str, start: date, end: date
) -> Tuple[List[date], Optional[date], Optional[date]]:
    day_expr = (
        "trade_date"
        if timeframe == "1d"
        else "CAST(timezone('America/New_York', bar_timestamp) AS DATE)"
    )
    union = " UNION ALL ".join(f"SELECT {day_expr} AS day FROM read_parquet(?)" for _ in paths)
    params = [p.as_posix() for p in paths]
    con = duckdb.connect()
    try:
        con.execute("SET TimeZone='UTC'")
        bounds = con.execute(f"SELECT min(day), max(day) FROM ({union})", params).fetchone()
        rows = con.execute(
            f"SELECT DISTINCT day FROM ({union}) WHERE day >= ? AND day <= ? ORDER BY day",
            [*params, start, end],
        ).fetchall()
    finally:
        con.close()
    first, last = bounds if bounds is not None else (None, None)
    return [r[0] for r in rows], first, last


def _gaps_listing_status(lake: Lake, ac: str, symbol: str, tf: str, listing: str) -> str:
    if listing == "listed":
        return "listed"
    if listing == "delisted":
        return "delisted"
    live = lake.bronze(ac, symbol, tf).exists()
    arch = lake.archive(ac, symbol, tf).exists()
    if live and arch:
        return "dual"
    if arch:
        return "delisted"
    return "listed"


def _gaps_lifetime(lake: Lake, ac: str, symbol: str) -> Tuple[Dict[str, Any], Any]:
    def everything(_d: date) -> bool:
        return True

    if ac != "equity":
        return {"state": "unknown"}, everything
    intervals = _identity_intervals(lake, symbol)
    if not intervals:
        return {"state": "unknown"}, everything
    spans = []
    for i in intervals:
        lo = date.fromisoformat(i["effective_from"][:10]) if i["effective_from"] else date.min
        hi = date.fromisoformat(i["effective_to"][:10]) if i["effective_to"] else date.max
        spans.append((lo, hi))

    def in_life(d: date) -> bool:
        return any(lo <= d < hi for lo, hi in spans)

    return {"state": "known"}, in_life


def _gaps_runs(missing: List[date], order: Dict[date, int]) -> List[Tuple[date, date, int]]:
    runs: List[List[Any]] = []
    for day in missing:
        if runs and order[day] == order[runs[-1][1]] + 1:
            runs[-1][1] = day
            runs[-1][2] += 1
        else:
            runs.append([day, day, 1])
    return [tuple(r) for r in runs]


class GapsChecker:
    def sources(self, case: Case, lake: Lake) -> List[Path]:
        args = case.expect["args"]
        ac, symbol, tf = args["asset_class"], args["symbol"], args["timeframe"]
        return [
            lake.bronze(ac, symbol, tf),
            lake.archive(ac, symbol, tf),
            _security_master_path(lake),
            _repairs_root(lake),
        ]

    def run(self, case: Case, lake: Lake, executor: Any) -> Outcome:
        args = case.expect["args"]
        ac, symbol, tf = args["asset_class"], args["symbol"], args["timeframe"]
        params = case.request["params"]
        listing = params.get("listing", "listed")
        max_gaps = params.get("max_gaps", 100)
        start = (
            date.fromisoformat(params["start"])
            if "start" in params
            else _today() - timedelta(days=365)
        )
        end = date.fromisoformat(params["end"]) if "end" in params else _today()
        try:
            if start > end:
                raise Reject(400, "invalid_parameter")
            if not 1 <= max_gaps <= 2000:
                raise Reject(400, "invalid_parameter")
            status_label = _gaps_listing_status(lake, ac, symbol, tf, listing)
            paths = []
            if status_label in ("listed", "dual"):
                paths.append(lake.bronze(ac, symbol, tf))
            if status_label in ("delisted", "dual"):
                paths.append(lake.archive(ac, symbol, tf))
            paths = [p for p in paths if p.exists()]
            if not paths:
                raise Reject(404, "unknown_symbol")
        except Reject as rej:
            status, body = executor.execute(case.request)
            code = body.get("error", {}).get("code") if isinstance(body, dict) else None
            if status == rej.status and code == rej.code:
                return Outcome("EXPECTED_REJECTION", f"{status} {code}")
            return Outcome(
                "FAIL",
                f"expected {rej.status} {rej.code}, got {status} {code}: {str(body)[:300]}",
            )

        calendar = params.get("calendar", "auto")
        policy = calendar if calendar != "auto" else ("weekdays" if ac == "fx" else "xnys")
        expected_days = (
            _weekday_sessions(start, end) if policy == "weekdays" else _xnys_sessions(start, end)
        )
        _lifetime, in_life = _gaps_lifetime(lake, ac, symbol)
        present, file_first, file_last = _session_presence(lake, paths, tf, start, end)
        present_set = set(present)
        considered = [d for d in expected_days if in_life(d)]
        order = {d: i for i, d in enumerate(considered)}
        observed = [d for d in considered if d in present_set]

        status, body = executor.execute(case.request)
        if status != 200:
            return Outcome("FAIL", f"expected 200, got {status}: {str(body)[:300]}")
        problems = []
        if body.get("listing_status") != status_label:
            problems.append(
                f"listing_status: expected {status_label} got {body.get('listing_status')}"
            )
        if body.get("expected_sessions") != len(considered):
            problems.append(
                f"expected_sessions: expected {len(considered)} got {body.get('expected_sessions')}"
            )
        if body.get("present_sessions") != len(observed):
            problems.append(
                f"present_sessions: expected {len(observed)} got {body.get('present_sessions')}"
            )
        got_bounds = body.get("file_bounds", {})
        exp_bounds = {
            "first": None if file_first is None else file_first.isoformat(),
            "last": None if file_last is None else file_last.isoformat(),
        }
        if got_bounds != exp_bounds:
            problems.append(f"file_bounds: expected {exp_bounds} got {got_bounds}")
        if not observed:
            if body.get("status") != "no_data":
                problems.append(f"status: expected no_data got {body.get('status')}")
            if body.get("gaps"):
                problems.append(f"gaps: expected [] got {body.get('gaps')}")
        else:
            first_obs, last_obs = observed[0], observed[-1]
            leading = [d for d in considered if d < first_obs]
            trailing = [d for d in considered if d > last_obs]
            interior = [d for d in considered if first_obs < d < last_obs and d not in present_set]
            runs = _gaps_runs(interior, order)
            # Contract since a63d2b97 (design §4: unknown lifetime is never "complete").
            if runs:
                expected_status = "gaps"
            elif leading or trailing:
                expected_status = "edges_unobserved"
            elif _lifetime.get("state") == "known":
                expected_status = "complete_sessions"
            else:
                expected_status = "lifetime_unknown"
            if body.get("status") != expected_status:
                problems.append(f"status: expected {expected_status} got {body.get('status')}")
            exp_gaps_total = len(runs)
            if body.get("gaps_total") != exp_gaps_total:
                problems.append(
                    f"gaps_total: expected {exp_gaps_total} got {body.get('gaps_total')}"
                )
            if body.get("truncated") != (exp_gaps_total > max_gaps):
                problems.append(
                    f"truncated: expected {exp_gaps_total > max_gaps} got {body.get('truncated')}"
                )
            got_gaps = body.get("gaps", [])
            exp_gaps_page = runs[:max_gaps]
            if len(got_gaps) != len(exp_gaps_page):
                problems.append(f"gaps count: expected {len(exp_gaps_page)} got {len(got_gaps)}")
            else:
                for (glo, ghi, gn), got_g in zip(exp_gaps_page, got_gaps):
                    if (
                        got_g.get("start") != glo.isoformat()
                        or got_g.get("end") != ghi.isoformat()
                        or got_g.get("sessions") != gn
                    ):
                        problems.append(f"gap mismatch: expected {glo}/{ghi}/{gn} got {got_g}")
                        break
            exp_leading = (
                None
                if not leading
                else {
                    "start": leading[0].isoformat(),
                    "end": leading[-1].isoformat(),
                    "sessions": len(leading),
                }
            )
            exp_trailing = (
                None
                if not trailing
                else {
                    "start": trailing[0].isoformat(),
                    "end": trailing[-1].isoformat(),
                    "sessions": len(trailing),
                }
            )
            if body.get("leading_unobserved") != exp_leading:
                problems.append(
                    f"leading_unobserved: expected {exp_leading} got {body.get('leading_unobserved')}"
                )
            if body.get("trailing_unobserved") != exp_trailing:
                problems.append(
                    f"trailing_unobserved: expected {exp_trailing} got {body.get('trailing_unobserved')}"
                )
            for closure in _CLOSURES:
                cd = date.fromisoformat(closure)
                if start <= cd <= end and cd in considered and cd not in present_set:
                    if any(glo <= cd <= ghi for glo, ghi, _ in runs):
                        problems.append(f"national closure {closure} reported as a gap")
        rep_state, rep_entries, rep_count = _repairs_evidence(lake, symbol, ac, tf, start, end)
        got_repairs = body.get("repairs", {})
        if got_repairs.get("reports_read") != rep_count:
            problems.append(
                f"repairs.reports_read: expected {rep_count} got {got_repairs.get('reports_read')}"
            )
        exp_entry_keys = sorted(
            json.dumps(
                {
                    "report_kind": e["report_kind"],
                    "reports": sorted(e["reports"]),
                    "sessions": sorted(e["sessions"]),
                    **e["details"],
                },
                sort_keys=True,
                default=str,
            )
            for e in rep_entries
        )
        got_entry_keys = sorted(
            json.dumps(
                {k: v for k, v in ge.items() if k != "report_date"},
                sort_keys=True,
                default=str,
            )
            for ge in got_repairs.get("entries", [])
        )
        if len(exp_entry_keys) != len(got_entry_keys):
            problems.append(
                f"repairs.entries count: expected {len(exp_entry_keys)} got {len(got_entry_keys)}"
            )
        if problems:
            return Outcome("FAIL", "; ".join(problems))
        return Outcome("PASS")


def _gaps_window(kind: str, sample: Dict[str, Any]) -> Dict[str, str]:
    last, first = sample["last"], sample["first"]
    if kind == "normal":
        start = last - timedelta(days=30)
        return {"start": start.isoformat(), "end": last.isoformat()}
    if kind == "no_data":
        day = last - timedelta(days=7)
        while day.weekday() != 5:  # Saturday
            day -= timedelta(days=1)
        return {"start": day.isoformat(), "end": day.isoformat()}
    # straddling: crosses the first observed session
    return {
        "start": (first - timedelta(days=400)).isoformat(),
        "end": (first + timedelta(days=45)).isoformat(),
    }


_GAP_LISTINGS = ("listed", "delisted", "any")
_GAP_WINDOWS = ("normal", "no_data", "straddling")
_GAP_MAX_GAPS = (1, 100)


def _any_sample_for(
    samples: Dict[Tuple[str, str, str], Any], ac: str, residency: str
) -> Optional[Dict[str, Any]]:
    """Any real inventory sample for (ac, residency), at any timeframe -- used only to
    build a well-formed date window for an unsupported-timeframe (contract) rejection,
    where the window's content cannot affect the outcome (check_timeframe fires first)."""
    for tf in GAP_LADDERS[ac]:
        s = samples.get((ac, tf, residency))
        if s is not None:
            return s
    for (a, _tf, r), s in samples.items():
        if a == ac:
            return s
    return None


def _gen_gaps(samples: Dict[Tuple[str, str, str], Any]) -> List[Case]:
    """Full cross product (no sampling/pairwise reduction): every asset_class x (its
    ladder timeframes + "4h") x listing x residency x window x max_gaps, calendar=auto.
    An unsupported timeframe ("4h", never in any ladder) is a pure argument rejection
    -- ``check_timeframe`` in gaps.py fires before any lake read, so it is generated as
    a rejection case regardless of whether a sample exists for that cell. A cell whose
    timeframe IS supported but has no inventory sample for that residency is BLOCKED_DATA.
    """
    cases: List[Case] = []
    generic_window_sample = {"first": date(2015, 1, 1), "last": _today()}
    for ac, ladder in GAP_LADDERS.items():
        for tf in (*ladder, "4h"):
            valid_tf = tf in ladder
            for residency in RESIDENCIES:
                sample = samples.get((ac, tf, residency)) if valid_tf else None
                for listing in _GAP_LISTINGS:
                    for window_kind in _GAP_WINDOWS:
                        for max_gaps in _GAP_MAX_GAPS:
                            case_id = (
                                f"gaps:{ac}:{tf}:{listing}:{residency}:" f"{window_kind}:{max_gaps}"
                            )
                            dims = {
                                "asset_class": ac,
                                "timeframe": tf,
                                "listing": listing,
                                "residency": residency,
                                "window": window_kind,
                                "max_gaps": max_gaps,
                                "calendar": "auto",
                            }
                            if not valid_tf:
                                probe = _any_sample_for(samples, ac, residency)
                                symbol = probe["symbol"] if probe else "PLACEHOLDER"
                                win = _gaps_window(window_kind, probe or generic_window_sample)
                                cases.append(
                                    Case(
                                        case_id,
                                        "gaps",
                                        dims,
                                        _req(
                                            f"/v1/{ac}/{symbol}/gaps",
                                            {
                                                "timeframe": tf,
                                                "listing": listing,
                                                "max_gaps": max_gaps,
                                                **win,
                                            },
                                        ),
                                        _rejection(400, "unsupported_timeframe"),
                                    )
                                )
                                continue
                            if sample is None:
                                cases.append(
                                    Case(
                                        case_id,
                                        "gaps",
                                        dims,
                                        _req(f"/v1/{ac}/NOSAMPLE/gaps"),
                                        _blocked(f"no {residency} {ac} {tf} sample in inventory"),
                                    )
                                )
                                continue
                            symbol = sample["symbol"]
                            win = _gaps_window(window_kind, sample)
                            cases.append(
                                Case(
                                    case_id,
                                    "gaps",
                                    dims,
                                    _req(
                                        f"/v1/{ac}/{symbol}/gaps",
                                        {
                                            "timeframe": tf,
                                            "listing": listing,
                                            "max_gaps": max_gaps,
                                            **win,
                                        },
                                    ),
                                    _value(
                                        "gaps",
                                        asset_class=ac,
                                        symbol=symbol,
                                        timeframe=tf,
                                    ),
                                )
                            )

    equity_1d = samples.get(("equity", "1d", "live_only")) or samples.get(("equity", "1d", "dual"))
    if equity_1d:
        symbol = equity_1d["symbol"]
        cases.append(
            Case(
                "gaps:reject:reversed",
                "gaps",
                {},
                _req(
                    f"/v1/equity/{symbol}/gaps",
                    {"start": "2020-06-01", "end": "2020-01-01"},
                ),
                _rejection(400, "invalid_parameter"),
            )
        )
        cases.append(
            Case(
                "gaps:reject:max_gaps_zero",
                "gaps",
                {},
                _req(f"/v1/equity/{symbol}/gaps", {"max_gaps": 0}),
                _rejection(400, "invalid_parameter"),
            )
        )
        for closure in _CLOSURES:
            cd = date.fromisoformat(closure)
            lo, hi = (
                (cd - timedelta(days=20)).isoformat(),
                (cd + timedelta(days=20)).isoformat(),
            )
            if equity_1d["first"].isoformat() <= lo and hi <= equity_1d["last"].isoformat():
                cases.append(
                    Case(
                        f"gaps:closure:{closure}",
                        "gaps",
                        {"closure": closure},
                        _req(
                            f"/v1/equity/{symbol}/gaps",
                            {
                                "timeframe": "1d",
                                "start": lo,
                                "end": hi,
                                "listing": "listed",
                            },
                        ),
                        _value("gaps", asset_class="equity", symbol=symbol, timeframe="1d"),
                    )
                )
            else:
                cases.append(
                    Case(
                        f"gaps:closure:{closure}",
                        "gaps",
                        {"closure": closure},
                        _req(f"/v1/equity/{symbol}/gaps"),
                        _blocked(f"sample window does not cover the {closure} closure"),
                    )
                )

    return cases


# ==== generate / CHECKERS ==============================================================


def generate(lake: Lake, inventory: Dict[str, Any]) -> List[Case]:
    samples = _samples(inventory)
    cases: List[Case] = [
        Case(
            "asset_classes:list",
            "asset_classes",
            {},
            _req("/v1/lake/asset-classes"),
            _value("asset_classes"),
        )
    ]
    cases.append(
        Case(
            "lake_status:get",
            "lake_status",
            {},
            _req("/v1/lake/status"),
            _value("lake_status"),
        )
    )
    cases += _gen_coverage(lake)
    cases += _gen_instruments(lake, samples)
    cases += _gen_instrument_detail(samples)
    cases += _gen_futures(lake)
    cases += _gen_actions(samples)
    cases += _gen_delisting(samples)
    cases += _gen_security(lake, samples)
    cases += _gen_membership(lake, samples)
    cases += _gen_revisions(lake)
    cases += _gen_gaps(samples)
    return cases


CHECKERS = {
    "asset_classes": AssetClassesChecker(),
    "lake_status": StatusChecker(),
    "coverage": CoverageChecker(),
    "instruments": InstrumentsChecker(),
    "instrument_detail": InstrumentDetailChecker(),
    "futures": FuturesChecker(),
    "actions": ActionsChecker(),
    "delisting": DelistingChecker(),
    "security": SecurityChecker(),
    "membership_indices": MembershipIndicesChecker(),
    "membership_members": MembershipMembersChecker(),
    "membership_history": MembershipHistoryChecker(),
    "silver_revisions": SilverRevisionsChecker(),
    "silver_revision_detail": SilverRevisionDetailChecker(),
    "pit_revisions": PitRevisionsChecker(),
    "pit_revision_detail": PitRevisionDetailChecker(),
    "gaps": GapsChecker(),
}
