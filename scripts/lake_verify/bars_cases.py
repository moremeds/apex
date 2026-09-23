"""Bars, bulk-bars and rates cases for the PR1 matrix (plan §4.2-4.3).

Single bars are the full cross product of the §4.2 dimensions. A combination is
settled one of three ways:

1. ``contract_rejection`` -- the contract rejects it from its arguments alone (design
   §3.4/§6 guard order), so the expected status/code is fixed at generation;
2. ``blocked_data`` -- legal, but the lake holds no sample for that
   (class, timeframe, residency) cell (P0 inventory);
3. a value case -- the oracle below decides at run time, from the files themselves,
   whether the answer is rows (compared value by value) or a data-dependent rejection
   (unknown symbol, not a PIT member, evicted artifact, ...).

The oracle re-derives everything from the files through ``lake.py`` and never calls
the candidate. Dimension collapses, each justified by the route contract:
bulk has no asset_class/timeframe-class/PIT dimension (equity-only, no pit param);
rates series has no price_mode/listing/revision/timeframe (the route takes none).
"""

from __future__ import annotations

import hashlib
import itertools
from dataclasses import dataclass
from datetime import date, datetime, time, timedelta, timezone
from decimal import ROUND_HALF_UP, Decimal
from typing import Any, Dict, List, Optional, Tuple
from urllib.parse import unquote
from zoneinfo import ZoneInfo

from lake import LADDERS, Lake, as_utc, read_rows
from model import Case, Outcome

UTC = timezone.utc
NY = ZoneInfo("America/New_York")
EPOCH = datetime(1970, 1, 1, tzinfo=UTC)
CLASSES = ("equity", "volatility", "fx", "cmdty", "futures", "rates")
TIMEFRAMES = ("1m", "5m", "30m", "1h", "1d", "4h")
PROCESSES = ("raw", "adjusted")
PRICE_MODES = ("omitted", "raw", "adjusted")
POLICIES = ("legacy", "bounded")
LISTINGS = ("listed", "delisted", "any")
RESIDENCIES = ("live_only", "archive_only", "dual")
REVISIONS = ("none", "current_silver", "older_silver", "pit", "both")
WINDOWS = (
    "default",
    "start_only",
    "end_only",
    "bounded_nonempty",
    "bounded_empty",
    "outside_edges",
)
LIMITS = ("default", "one", "exact", "below", "maximum", "nonpositive")
DELTA = {
    "1m": timedelta(minutes=1),
    "5m": timedelta(minutes=5),
    "30m": timedelta(minutes=30),
    "1h": timedelta(hours=1),
    "1d": timedelta(days=1),
    "4h": timedelta(hours=4),
}
LEGACY_DEFAULT, BOUNDED_DEFAULT, BOUNDED_MAX = 2000, 250, 5000
FLOAT_TOL = 1e-9
BULK_BOUNDED_DEFAULT = 50
RATES_BOUNDED_DEFAULT = 500
PIT_INDEX = "sp500"


# -- data selection (generation time) ---------------------------------------------


@dataclass(frozen=True)
class Sample:
    symbol: str
    first: datetime
    last: datetime


def _samples(inventory: Dict[str, Any]) -> Dict[Tuple[str, str, str], Sample]:
    out = {}
    for cell in inventory["cells"]:
        if not cell["samples"]:
            continue
        sample = cell["samples"][0]
        facts = [sample[k] for k in ("live", "archive") if k in sample]
        firsts = [datetime.fromisoformat(f["first"]) for f in facts]
        lasts = [datetime.fromisoformat(f["last"]) for f in facts]
        out[(cell["asset_class"], cell["timeframe"], cell["residency"])] = Sample(
            unquote(sample["symbol_dir"]),
            as_utc(min(firsts)),
            as_utc(max(lasts)),
        )
    return out


def _pit_member(lake: Lake) -> Tuple[Optional[int], Optional[str]]:
    """Latest PIT revision of PIT_INDEX and its member with the longest open scope."""
    for number in lake.pit_numbers():
        manifest = lake.pit(number)
        if manifest["index_id"] != PIT_INDEX:
            continue
        open_scopes = [m for m in manifest["members"] if m["session_to"] is None]
        best = min(open_scopes, key=lambda m: m["session_from"])
        return number, best["symbol"]
    return None, None


def _iso(value: datetime) -> str:
    return value.astimezone(UTC).isoformat().replace("+00:00", "Z")


def _window_params(kind: str, tf: str, sample: Sample) -> Dict[str, str]:
    span = timedelta(days=45) if tf == "1d" else timedelta(days=3)
    last, first = sample.last, sample.first
    if kind == "default":
        return {}
    if kind == "start_only":
        return {"start": _iso(last - span)}
    if kind == "end_only":
        return {"end": _iso(last - span / 3)}
    if kind == "bounded_nonempty":
        return {"start": _iso(last - span), "end": _iso(last)}
    if kind == "bounded_empty":
        # A Saturday 12:00-13:00 UTC hour before the last bar: every class in the lake is
        # closed then. The oracle confirms the emptiness rather than assuming it.
        day = last - timedelta(days=7)
        while day.weekday() != 5:
            day -= timedelta(days=1)
        noon = day.replace(hour=12, minute=0, second=0, microsecond=0)
        return {"start": _iso(noon), "end": _iso(noon + timedelta(hours=1))}
    # outside_edges: a window straddling the artifact's first bar.
    return {"start": _iso(first - timedelta(days=400)), "end": _iso(first + span)}


# -- the contract (argument-only rejections, in guard order) ----------------------


def contract_rejection(d: Dict[str, Any]) -> Optional[Tuple[int, str]]:
    ac, tf, pm, rev, listing, policy, limit = (
        d["asset_class"],
        d["timeframe"],
        d["price_mode"],
        d["revision"],
        d["listing"],
        d["output_policy"],
        d["limit"],
    )
    if ac == "rates":
        return 400, "unsupported_asset_class"
    if tf not in LADDERS[ac]:
        return 400, "unsupported_timeframe"
    if pm == "adjusted" and ac != "equity":
        return 400, "adjusted_not_supported"
    if rev == "both":
        return 400, "invalid_parameter"
    if rev != "none":
        if pm == "raw":
            return 400, "invalid_parameter"
        if ac != "equity" or tf != "1d" or listing != "listed":
            return 400, "revision_not_supported"
    if policy == "bounded" and limit in ("nonpositive",):
        return 400, "invalid_parameter"
    return None


# -- generation -------------------------------------------------------------------


def _limit_value(limit: str, policy: str) -> Optional[Any]:
    return {
        "default": None,
        "one": 1,
        "exact": "exact",
        "below": "below",
        # The contract's maximum (bounded policy). Legacy REST has no ceiling of its own;
        # its unbounded path is exercised by "nonpositive" (<=0 means all).
        "maximum": BOUNDED_MAX,
        "nonpositive": 0,
    }[limit]


def _request(
    d: Dict[str, Any],
    symbol: str,
    window: Dict[str, str],
    silver_pin: Optional[int],
    pit_pin: Optional[int],
) -> Dict[str, Any]:
    limit = _limit_value(d["limit"], d["output_policy"])
    if d["output_policy"] == "legacy":
        params: Dict[str, Any] = {
            "timeframe": d["timeframe"],
            "listing": d["listing"],
            **window,
        }
        if d["price_mode"] != "omitted":
            params["price_mode"] = d["price_mode"]
        if limit is not None:
            params["limit"] = limit
        if silver_pin is not None:
            params["silver_revision"] = silver_pin
        if pit_pin is not None:
            params["pit_revision"] = pit_pin
        return {
            "transport": "http",
            "process": d["process_config"],
            "path": f"/v1/{d['asset_class']}/{symbol}/bars",
            "params": params,
        }
    kwargs: Dict[str, Any] = {
        "symbol": symbol,
        "asset_class": d["asset_class"],
        "timeframe": d["timeframe"],
        "listing": d["listing"],
        **window,
    }
    if d["price_mode"] != "omitted":
        kwargs["price_mode"] = d["price_mode"]
    if limit is not None:
        kwargs["limit"] = limit
    if silver_pin is not None:
        kwargs["silver_revision_pin"] = silver_pin
    if pit_pin is not None:
        kwargs["pit_revision"] = pit_pin
    return {
        "transport": "inproc",
        "process": d["process_config"],
        "call": "bars",
        "kwargs": kwargs,
    }


def generate(lake: Lake, inventory: Dict[str, Any]) -> List[Case]:
    samples = _samples(inventory)
    current = lake.silver_current_number()
    pit_number, pit_symbol = _pit_member(lake)
    cases: List[Case] = []
    for combo in itertools.product(
        CLASSES,
        TIMEFRAMES,
        PROCESSES,
        PRICE_MODES,
        POLICIES,
        LISTINGS,
        RESIDENCIES,
        REVISIONS,
        WINDOWS,
        LIMITS,
    ):
        d = dict(
            zip(
                (
                    "asset_class",
                    "timeframe",
                    "process_config",
                    "price_mode",
                    "output_policy",
                    "listing",
                    "residency",
                    "revision",
                    "window",
                    "limit",
                ),
                combo,
            )
        )
        case_id = "bars:" + ":".join(str(v) for v in combo)
        rejection = contract_rejection(d)
        sample = samples.get((d["asset_class"], d["timeframe"], d["residency"]))
        fallback = (
            samples.get((d["asset_class"], "1d", "live_only"))
            or samples[("equity", "1d", "live_only")]
        )
        silver_pin = {"current_silver": current, "older_silver": current - 1}.get(d["revision"])
        pit_pin = pit_number if d["revision"] in ("pit", "both") else None
        if d["revision"] == "both":
            silver_pin = current
        chosen = sample or fallback
        if d["revision"] == "pit" and d["residency"] == "live_only" and pit_symbol:
            chosen = Sample(pit_symbol, chosen.first, chosen.last)
        window = _window_params(
            d["window"], d["timeframe"] if d["timeframe"] in DELTA else "1d", chosen
        )
        request = _request(d, chosen.symbol, window, silver_pin, pit_pin)
        if rejection is not None:
            for holder in (request.get("params"), request.get("kwargs")):
                if holder is not None and holder.get("limit") in ("exact", "below"):
                    holder["limit"] = 7  # any positive value: the argument check fires first
            expect = {"kind": "rejection", "status": rejection[0], "code": rejection[1]}
        elif sample is None:
            expect = {
                "kind": "blocked_data",
                "reason": f"no {d['residency']} {d['asset_class']} {d['timeframe']} artifact in the lake (P0 inventory)",
            }
        elif d["revision"] == "pit" and pit_number is None:
            expect = {
                "kind": "blocked_data",
                "reason": f"no published {PIT_INDEX} PIT revision",
            }
        else:
            expect = {
                "kind": "value",
                "check": "bars",
                "args": {"symbol": chosen.symbol},
            }
        cases.append(Case(case_id, "bars", d, request, expect))
    cases.extend(_bulk_cases(samples, current))
    cases.extend(_rates_cases(samples))
    return cases


def _bulk_cases(samples: Dict[Tuple[str, str, str], Sample], current: int) -> List[Case]:
    cases = []
    for tf, proc, pm, policy, listing, rev, win, limit in itertools.product(
        TIMEFRAMES, PROCESSES, PRICE_MODES, POLICIES, LISTINGS,
        ("none", "current_silver", "older_silver"), WINDOWS, LIMITS,
    ):  # fmt: skip
        d = {
            "asset_class": "equity",
            "timeframe": tf,
            "process_config": proc,
            "price_mode": pm,
            "output_policy": policy,
            "listing": listing,
            "revision": rev,
            "window": win,
            "limit": limit,
        }
        # One request covers every residency: the symbol list mixes them.
        symbols = [
            samples[k].symbol for k in (("equity", tf, r) for r in RESIDENCIES) if k in samples
        ] or [samples[("equity", "1d", "live_only")].symbol]
        anchor = samples.get(("equity", tf, "live_only")) or samples[("equity", "1d", "live_only")]
        window = _window_params(win, tf if tf in DELTA else "1d", anchor)
        rejection = contract_rejection({**d, "asset_class": "equity"})
        if rejection is None and policy == "bounded" and limit == "maximum":
            rejection = None  # 5000 x 3 symbols stays inside the 10000-row budget? checked below
            if BOUNDED_MAX * len(symbols) > 10000:
                rejection = (400, "invalid_parameter")
        silver_pin = {"current_silver": current, "older_silver": current - 1}.get(rev)
        lim = _limit_value(limit, policy)
        if rejection is not None and lim in ("exact", "below"):
            lim = 7
        if policy == "legacy":
            params: Dict[str, Any] = {
                "symbols": ",".join(symbols),
                "timeframe": tf,
                "listing": listing,
                **window,
            }
            if pm != "omitted":
                params["price_mode"] = pm
            if lim is not None:
                params["limit"] = lim
            if silver_pin is not None:
                params["silver_revision"] = silver_pin
            request = {
                "transport": "http",
                "process": proc,
                "path": "/v1/equity/bars",
                "params": params,
            }
        else:
            kwargs: Dict[str, Any] = {
                "symbols": symbols,
                "timeframe": tf,
                "listing": listing,
                **window,
            }
            if pm != "omitted":
                kwargs["price_mode"] = pm
            if lim is not None:
                kwargs["limit"] = lim
            if silver_pin is not None:
                kwargs["silver_revision_pin"] = silver_pin
            request = {
                "transport": "inproc",
                "process": proc,
                "call": "bulk",
                "kwargs": kwargs,
            }
        if rejection is not None:
            expect = {"kind": "rejection", "status": rejection[0], "code": rejection[1]}
        else:
            expect = {"kind": "value", "check": "bulk", "args": {"symbols": symbols}}
        case_id = "bulk:" + ":".join((tf, proc, pm, policy, listing, rev, win, limit))
        cases.append(Case(case_id, "bulk", d, request, expect))
    return cases


def _rates_cases(samples: Dict[Tuple[str, str, str], Sample]) -> List[Case]:
    cases = []
    rates = samples[("rates", "1d", "live_only")]
    for policy, win, limit in itertools.product(POLICIES, WINDOWS, LIMITS):
        d = {
            "asset_class": "rates",
            "output_policy": policy,
            "window": win,
            "limit": limit,
        }
        window = _window_params(win, "1d", rates)
        lim = _limit_value(limit, "bounded")
        rejection = (400, "invalid_parameter") if limit == "nonpositive" else None
        if policy == "legacy":
            params: Dict[str, Any] = dict(window)
            if lim is not None:
                params["limit"] = 7 if rejection and lim in ("exact", "below") else lim
            request = {
                "transport": "http",
                "process": "raw",
                "path": f"/v1/rates/{rates.symbol}/series",
                "params": params,
            }
        else:
            kwargs: Dict[str, Any] = {"symbol": rates.symbol, **window}
            if lim is not None:
                kwargs["limit"] = lim
            request = {
                "transport": "inproc",
                "process": "raw",
                "call": "rates",
                "kwargs": kwargs,
            }
        expect = (
            {"kind": "rejection", "status": 400, "code": "invalid_parameter"}
            if rejection
            else {"kind": "value", "check": "rates", "args": {"symbol": rates.symbol}}
        )
        cases.append(Case("rates:" + ":".join((policy, win, limit)), "rates", d, request, expect))
    return cases


# -- oracle -----------------------------------------------------------------------


class Reject(Exception):
    def __init__(self, status: int, code: str) -> None:
        super().__init__(f"{status} {code}")
        self.status, self.code = status, code


def _ts_col(tf: str) -> str:
    return "trade_date" if tf == "1d" else "bar_timestamp"


def _read(path, tf: str, start: datetime, end: datetime, extra: str = "") -> List[Dict[str, Any]]:
    col = _ts_col(tf)
    bounds: List[Any] = [start.date(), end.date()] if tf == "1d" else [start, end]
    rows = read_rows(path, "*", f"{col} >= ? AND {col} <= ?{extra}", bounds)
    return sorted(rows, key=lambda r: as_utc(r[col]))


def _ny_day(row: Dict[str, Any], tf: str) -> date:
    value = row[_ts_col(tf)]
    return value if tf == "1d" else as_utc(value).astimezone(NY).date()


def _duckdb_round(value: float) -> int:
    """DuckDB ROUND rounds half away from zero; Python's round() is half-even."""
    return int(Decimal(repr(value)).quantize(Decimal("1"), rounding=ROUND_HALF_UP))


class BarsChecker:
    """Oracle + comparison for one bars case."""

    def sources(self, case: Case, lake: Lake) -> List[Any]:
        d, symbol = case.dims, case.expect["args"]["symbol"]
        ac, tf = d["asset_class"], d["timeframe"]
        return [lake.bronze(ac, symbol, tf), lake.archive(ac, symbol, tf)]

    def run(self, case: Case, lake: Lake, executor: Any) -> Outcome:
        d, request = case.dims, _copy(case.request)
        symbol = case.expect["args"]["symbol"]
        try:
            expected = self.expected(case, lake, datetime.now(UTC))
        except Reject as rej:
            return self._compare_rejection(case, executor, request, rej)
        holder = request.get("params") if request["transport"] == "http" else request["kwargs"]
        limit = holder.get("limit")
        if limit in ("exact", "below"):
            count = len(expected["window_rows"])
            if limit == "below" and count < 2:
                return Outcome("BLOCKED_DATA", f"window holds {count} row(s); 'below' needs >= 2")
            holder["limit"] = count if limit == "exact" else count - 1
            if holder["limit"] < 1:
                return Outcome(
                    "BLOCKED_DATA",
                    "window holds no rows; 'exact' limit is not positive",
                )
            expected = self.expected(_with_request(case, request), lake, datetime.now(UTC))
        status, body = executor.execute(request)
        if status != 200:
            return Outcome(
                "FAIL",
                f"expected 200 with {len(expected['rows'])} rows, got {status}: {str(body)[:300]}",
            )
        echoed = body.get("window") or {}
        if "end" in echoed and holder.get("end") is None:
            # The server anchored "now" a moment before the oracle did: re-derive the
            # default window at the server's instant, after checking they agree.
            served_end = datetime.fromisoformat(echoed["end"])
            if abs((served_end - expected["end"]).total_seconds()) > 120:
                return Outcome("FAIL", f"window end {served_end} vs oracle {expected['end']}")
            expected = self.expected(_with_request(case, request), lake, served_end)
        if "start" in echoed and datetime.fromisoformat(echoed["start"]) != expected["start"]:
            return Outcome(
                "FAIL",
                f"window start {echoed['start']} != contract {expected['start'].isoformat()}",
            )
        return self._compare_rows(d, symbol, expected, body)

    # expected() re-derives the contract from the files.
    def expected(
        self,
        case: Case,
        lake: Lake,
        now: datetime,
        bounded_default: int = BOUNDED_DEFAULT,
        from_epoch_listing: Optional[str] = None,
    ) -> Dict[str, Any]:
        d = case.dims
        req = case.request
        args = req.get("params") if req["transport"] == "http" else req["kwargs"]
        symbol = case.expect["args"]["symbol"]
        ac, tf, policy = d["asset_class"], d["timeframe"], d["output_policy"]
        pin = args.get("silver_revision", args.get("silver_revision_pin"))
        pit = args.get("pit_revision")
        pm = args.get("price_mode")
        effective = (
            "adjusted"
            if (pin or pit)
            else (
                pm
                or ("adjusted" if d["process_config"] == "adjusted" and ac == "equity" else "raw")
            )
        )
        live, arch = lake.bronze(ac, symbol, tf), lake.archive(ac, symbol, tf)
        listing = d["listing"]
        status = (
            listing
            if listing != "any"
            else (
                "dual"
                if live.exists() and arch.exists()
                else "delisted" if arch.exists() else "listed"
            )
        )
        if effective == "adjusted" and status != "listed":
            raise Reject(400, "adjusted_not_supported")
        limit = args.get("limit")
        if limit in ("exact", "below"):
            limit = None  # first pass: count rows under the default limit's window
        start = _parse(args.get("start"))
        end = _parse(args.get("end")) or now
        # Bulk resolves one window for the table from the REQUESTED listing.
        basis = from_epoch_listing if from_epoch_listing is not None else status
        from_epoch = basis != "listed" or pit is not None
        if policy == "bounded":
            tail = bounded_default if limit is None else limit
        else:
            tail = None if start is not None else (LEGACY_DEFAULT if limit is None else limit)
            if tail is not None and tail <= 0:
                tail, from_epoch = None, True
        if start is None:
            if policy == "bounded":
                window_limit = bounded_default if limit is None else limit
            else:
                window_limit = limit if isinstance(limit, int) else LEGACY_DEFAULT
            start = EPOCH if from_epoch else end - DELTA[tf] * window_limit * 10
        meta: Dict[str, Any] = {
            "price_mode": effective,
            "listing_status": status,
            "revision": None,
            "publisher_status": None,
        }
        if pit is not None:
            rows, start, end = self._pit_rows(lake, pit, symbol, start, end, meta)
        elif effective == "adjusted":
            rows = self._adjusted_rows(lake, symbol, tf, start, end, pin, meta)
        else:
            rows = self._raw_rows(lake, ac, symbol, tf, status, start, end)
        return {
            "window_rows": rows,
            "rows": rows if tail is None else rows[-tail:],
            "truncated": tail is not None and len(rows) > tail,
            "start": start,
            "end": end,
            "meta": meta,
            "tf": tf,
            "ac": ac,
        }

    def _raw_rows(
        self,
        lake: Lake,
        ac: str,
        symbol: str,
        tf: str,
        status: str,
        start: datetime,
        end: datetime,
    ) -> List[Dict[str, Any]]:
        live, arch = lake.bronze(ac, symbol, tf), lake.archive(ac, symbol, tf)
        use_live = status in ("listed", "dual") and live.exists()
        use_arch = status in ("delisted", "dual") and arch.exists()
        if not use_live and not use_arch:
            raise Reject(404, "unknown_symbol")
        live_rows = _read(live, tf, start, end) if use_live else []
        arch_rows = _read(arch, tf, start, end) if use_arch else []
        taken = {_ny_day(r, tf) for r in live_rows}
        for row in arch_rows:
            row.pop("price_basis", None)
        merged = live_rows + [r for r in arch_rows if _ny_day(r, tf) not in taken]
        merged.sort(key=lambda r: as_utc(r[_ts_col(tf)]))
        return merged

    def _adjusted_rows(
        self,
        lake: Lake,
        symbol: str,
        tf: str,
        start: datetime,
        end: datetime,
        pin: Optional[int],
        meta: Dict[str, Any],
    ) -> List[Dict[str, Any]]:
        revision = pin or lake.silver_current_number()
        if not (lake.silver / "revisions" / f"revision={revision}.json").exists():
            raise Reject(404, "unknown_revision")
        meta["revision"] = revision
        bronze = lake.bronze("equity", symbol, tf)
        if tf == "1d":
            daily = lake.silver_artifact(revision, symbol, "daily")
            if daily is None:
                raise Reject(
                    503 if bronze.exists() else 404,
                    "adjusted_unavailable" if bronze.exists() else "unknown_symbol",
                )
            if not daily.exists():
                raise Reject(503, "adjusted_unavailable")
            return _read(daily, "1d", start, end)
        if not bronze.exists():
            raise Reject(404, "unknown_symbol")
        factors_path = lake.silver_artifact(revision, symbol, "factors")
        if factors_path is None or not factors_path.exists():
            raise Reject(503, "adjusted_unavailable")
        factors = read_rows(factors_path, "*")
        out = []
        for bar in _read(bronze, tf, start, end):
            day = as_utc(bar["bar_timestamp"]).astimezone(NY).date()
            hits = [
                f for f in factors
                if (f["effective_start"] or date.min) <= day <= (f["effective_end"] or date.max)
            ]  # fmt: skip
            if len(hits) != 1 or hits[0]["adjustment_revision"] is None:
                raise Reject(503, "adjusted_unavailable")
            f = hits[0]
            p = f["price_adjustment_factor"]
            out.append({
                "bar_timestamp": bar["bar_timestamp"],
                "open": bar["open"] * p, "high": bar["high"] * p, "low": bar["low"] * p, "close": bar["close"] * p,
                "volume": _duckdb_round(bar["volume"] * f["split_volume_factor"]),
            })  # fmt: skip
        return out

    def _pit_rows(
        self,
        lake: Lake,
        pit: int,
        symbol: str,
        start: datetime,
        end: datetime,
        meta: Dict[str, Any],
    ) -> Tuple[List[Dict[str, Any]], datetime, datetime]:
        if pit not in lake.pit_numbers():
            raise Reject(404, "unknown_revision")
        manifest = lake.pit(pit)
        meta.update(revision=manifest["silver_revision"], publisher_status=manifest["status"])
        scopes = [m for m in manifest["members"] if m["symbol"] == symbol]
        if not scopes:
            raise Reject(400, "invalid_parameter")
        cutoff = date.fromisoformat(manifest["daily_bar_cutoff"])
        first, last = start.date(), min(end.date(), cutoff)
        hits = [
            s for s in scopes
            if first <= last and date.fromisoformat(s["session_from"]) <= last
            and (s["session_to"] is None or first < date.fromisoformat(s["session_to"]))
        ]  # fmt: skip
        if not hits:
            raise Reject(400, "invalid_parameter")
        if len({s["security_id"] for s in hits}) > 1:
            raise Reject(409, "ambiguous_symbol")
        entry = next(
            (
                a
                for a in manifest["inputs"]["silver_artifacts"]
                if a["path"].endswith(f"symbol={_enc(symbol)}/1d.parquet")
            ),
            None,
        )
        if entry is None:
            raise Reject(503, "pit_unavailable")
        path = lake.silver / entry["path"]
        try:
            digest = hashlib.sha256(path.read_bytes()).hexdigest()
        except OSError:
            raise Reject(503, "pit_unavailable") from None
        if digest != entry["sha256"]:
            raise Reject(503, "pit_unavailable")
        end = min(end, datetime.combine(last, time.max, tzinfo=UTC))
        rows = [
            r for r in _read(path, "1d", start, end)
            if any(
                date.fromisoformat(s["session_from"]) <= r["trade_date"]
                and (s["session_to"] is None or r["trade_date"] < date.fromisoformat(s["session_to"]))
                for s in hits
            )
        ]  # fmt: skip
        return rows, start, end

    def _compare_rejection(
        self, case: Case, executor: Any, request: Dict[str, Any], rej: Reject
    ) -> Outcome:
        holder = request.get("params") if request["transport"] == "http" else request["kwargs"]
        if holder.get("limit") in ("exact", "below"):
            holder["limit"] = 7
        status, body = executor.execute(request)
        code = body.get("error", {}).get("code") if isinstance(body, dict) else None
        if (status, code) == (rej.status, rej.code):
            return Outcome("EXPECTED_REJECTION", f"data-dependent {status} {code}")
        return Outcome(
            "FAIL",
            f"oracle expects {rej.status} {rej.code}, got {status} {code}: {str(body)[:300]}",
        )

    def _compare_rows(
        self, d: Dict[str, Any], symbol: str, exp: Dict[str, Any], body: Dict[str, Any]
    ) -> Outcome:
        tf, ac, meta = exp["tf"], exp["ac"], exp["meta"]
        got = body.get("bars", [])
        problems = []
        for key in ("price_mode", "listing_status"):
            if body.get(key) != meta[key]:
                problems.append(f"{key} {body.get(key)!r} != {meta[key]!r}")
        if meta["price_mode"] == "adjusted" and body.get("adjustment_revision") != meta["revision"]:
            problems.append(
                f"adjustment_revision {body.get('adjustment_revision')} != {meta['revision']}"
            )
        pit = (body.get("provenance") or {}).get("pit")
        if (
            meta["publisher_status"] is not None
            and (pit or {}).get("publisher_status") != meta["publisher_status"]
        ):
            problems.append(
                f"publisher_status {(pit or {}).get('publisher_status')} != {meta['publisher_status']}"
            )
        if bool(body.get("truncated")) != exp["truncated"]:
            problems.append(f"truncated {body.get('truncated')} != {exp['truncated']}")
        rows = exp["rows"]
        if len(got) != len(rows):
            problems.append(f"row count {len(got)} != {len(rows)}")
        for mine, theirs in zip(rows, got):
            when = as_utc(mine[_ts_col(tf)])
            if datetime.fromisoformat(theirs["time"]) != when:
                problems.append(f"time {theirs['time']} != {when.isoformat()}")
                break
            for field in ("open", "high", "low", "close", "settlement"):
                a, b = mine.get(field), theirs.get(field)
                if field == "settlement" and a is None and b is None:
                    continue
                if (a is None) != (b is None) or (
                    a is not None and abs(a - b) > FLOAT_TOL * max(1.0, abs(a))
                ):
                    problems.append(f"{field} at {when.date()}: {b} != {a}")
            for field in ("volume", "open_interest"):
                a, b = mine.get(field), theirs.get(field)
                if a is not None and b is not None and int(a) != int(b):
                    problems.append(f"{field} at {when.date()}: {b} != {a}")
            if ac == "equity" and meta["price_mode"] == "raw":
                want = mine.get("price_basis") or "unknown"
                if theirs.get("source_price_basis") != want:
                    problems.append(
                        f"source_price_basis {theirs.get('source_price_basis')} != {want}"
                    )
            if len(problems) > 5:
                break
        facts = {
            "rows": len(got),
            "first": got[0]["time"] if got else None,
            "last": got[-1]["time"] if got else None,
            "hash": hashlib.sha256(
                repr([(r["time"], r["close"]) for r in got]).encode()
            ).hexdigest()[:16],
        }
        if problems:
            return Outcome("FAIL", "; ".join(problems[:6]), facts)
        return Outcome("PASS", f"{len(got)} rows match", facts)


class BulkChecker:
    def sources(self, case: Case, lake: Lake) -> List[Any]:
        tf = case.dims["timeframe"]
        return [
            p
            for s in case.expect["args"]["symbols"]
            for p in (lake.bronze("equity", s, tf), lake.archive("equity", s, tf))
        ]

    def run(self, case: Case, lake: Lake, executor: Any) -> Outcome:
        # Each symbol is the single-bars oracle under the same arguments, with bulk's
        # own default (50) and one table window from the requested listing; the table
        # must agree symbol by symbol, missing ones included.
        request = _copy(case.request)
        holder = request.get("params") if request["transport"] == "http" else request["kwargs"]
        symbols = case.expect["args"]["symbols"]
        bounded = case.dims["output_policy"] == "bounded"
        default = BULK_BOUNDED_DEFAULT if bounded else LEGACY_DEFAULT

        def oracle(symbol: str, now: datetime) -> Dict[str, Any]:
            args = {k: v for k, v in holder.items() if k != "symbols"}
            one = Case(
                "x",
                "bars",
                {**case.dims, "residency": None},
                {**request, "params": args, "kwargs": args},
                {"args": {"symbol": symbol}},
            )
            return BarsChecker().expected(
                one, lake, now, bounded_default=default, from_epoch_listing=holder["listing"]
            )

        if holder.get("limit") in ("exact", "below"):
            try:
                count = len(oracle(symbols[0], datetime.now(UTC))["window_rows"])
            except Reject:
                count = 0
            if count < 2:
                return Outcome("BLOCKED_DATA", f"anchor {symbols[0]} window holds {count} row(s)")
            holder["limit"] = count if holder["limit"] == "exact" else count - 1
        status, body = executor.execute(request)
        if status != 200:
            return Outcome("FAIL", f"bulk returned {status}: {str(body)[:300]}")
        served_end = datetime.fromisoformat(body["window"]["end"])
        problems, served = [], 0
        for symbol in symbols:
            try:
                exp = oracle(symbol, served_end)
            except Reject as rej:
                if symbol not in body.get("missing", {}):
                    problems.append(f"{symbol}: oracle {rej.status} {rej.code} but served")
                continue
            if datetime.fromisoformat(body["window"]["start"]) != exp["start"]:
                problems.append(
                    f"window start {body['window']['start']} != {exp['start'].isoformat()}"
                )
                break
            series = body.get("symbols", {}).get(symbol)
            if series is None:
                problems.append(
                    f"{symbol}: oracle has {len(exp['rows'])} rows, reported missing: {body.get('missing', {}).get(symbol)}"
                )
                continue
            served += 1
            outcome = BarsChecker()._compare_rows(
                case.dims,
                symbol,
                exp,
                {"bars": series["bars"], "price_mode": body["price_mode"], "listing_status": series["listing_status"],
                 "adjustment_revision": body["adjustment_revision"], "truncated": series["truncated"]},
            )  # fmt: skip
            if outcome.status != "PASS":
                problems.append(f"{symbol}: {outcome.detail}")
        if problems:
            return Outcome("FAIL", "; ".join(problems[:4]))
        return Outcome(
            "PASS", f"{served} series match, {len(body.get('missing', {}))} missing as expected"
        )


class RatesChecker:
    def sources(self, case: Case, lake: Lake) -> List[Any]:
        return [lake.bronze("rates", case.expect["args"]["symbol"], "1d")]

    def run(self, case: Case, lake: Lake, executor: Any) -> Outcome:
        request = _copy(case.request)
        holder = request.get("params") if request["transport"] == "http" else request["kwargs"]
        symbol = case.expect["args"]["symbol"]
        start = _parse(holder.get("start")) or EPOCH
        end = _parse(holder.get("end")) or datetime.now(UTC)
        rows = _read(lake.bronze("rates", symbol, "1d"), "1d", start, end)
        bounded = case.dims["output_policy"] == "bounded" or holder.get("limit") is not None
        limit = holder.get("limit")
        if limit in ("exact", "below"):
            if len(rows) < 2:
                return Outcome("BLOCKED_DATA", f"window holds {len(rows)} rate point(s)")
            holder["limit"] = limit = len(rows) if limit == "exact" else len(rows) - 1
        if bounded and limit is None:
            tail: Optional[int] = RATES_BOUNDED_DEFAULT
        else:
            tail = limit if bounded else None
        want = rows if tail is None else rows[-tail:]
        status, body = executor.execute(request)
        if status != 200:
            return Outcome("FAIL", f"rates returned {status}: {str(body)[:300]}")
        got = body.get("points", [])
        if len(got) != len(want):
            return Outcome("FAIL", f"points {len(got)} != {len(want)}")
        for mine, theirs in zip(want, got):
            if (
                datetime.fromisoformat(theirs["time"]) != as_utc(mine["trade_date"])
                or abs(theirs["yield_pct"] - mine["yield_pct"]) > FLOAT_TOL
            ):
                return Outcome(
                    "FAIL",
                    f"point {theirs} != {mine['trade_date']} {mine['yield_pct']}",
                )
        if bounded and bool(body.get("truncated")) != (tail is not None and len(rows) > tail):
            return Outcome("FAIL", f"truncated {body.get('truncated')}")
        return Outcome("PASS", f"{len(got)} points match")


def _enc(symbol: str) -> str:
    from lake import encode

    return encode(symbol)


def _parse(value: Optional[str]) -> Optional[datetime]:
    return None if value is None else datetime.fromisoformat(value.replace("Z", "+00:00"))


def _copy(request: Dict[str, Any]) -> Dict[str, Any]:
    import copy

    return copy.deepcopy(request)


def _with_request(case: Case, request: Dict[str, Any]) -> Case:
    return Case(case.id, case.operation, case.dims, request, case.expect)


CHECKERS = {"bars": BarsChecker(), "bulk": BulkChecker(), "rates": RatesChecker()}
