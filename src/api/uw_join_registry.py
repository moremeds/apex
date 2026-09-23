"""Static, parameter-bound SQL for the supported ``uw_scan`` joins."""

from __future__ import annotations

import re
from dataclasses import dataclass
from datetime import date
from typing import Any

from src.api.errors import ApiError, ApiErrorCode

_PARAM_TYPES = {
    "ticker": "text",
    "tier": "text",
    "start": "date",
    "end": "date",
    "run_id": "bigint",
    "expiry": "date",
}
_TOKEN = re.compile(r":(ticker|tier|start|end|run_id|expiry)\b")


@dataclass(frozen=True)
class JoinSpec:
    sql: str
    filters: tuple[str, ...]
    required: frozenset[str]
    coverage: tuple[tuple[str, str], ...]


@dataclass(frozen=True)
class JoinQuery:
    name: str
    sql: str
    params: tuple[Any, ...]
    limit: int
    coverage: tuple[tuple[str, str], ...]


def _spec(
    sql: str,
    filters: tuple[str, ...],
    coverage: tuple[tuple[str, str], ...],
    *,
    required: tuple[str, ...] = (),
) -> JoinSpec:
    return JoinSpec(sql.strip(), filters, frozenset(required), coverage)


JOIN_REGISTRY = {
    "watchlist_active_card": _spec(
        """
SELECT w.ticker, w.sector, wc.run_id, wc.scanned_at, q.price, q.quoted_at,
       r.status AS run_status,
       (wc.ticker IS NOT NULL) AS has_card, (q.ticker IS NOT NULL) AS has_quote,
       (r.run_id IS NOT NULL) AS has_run
FROM uw_scan.watchlist w
LEFT JOIN uw_scan.watchlist_card wc ON wc.ticker = w.ticker
LEFT JOIN uw_scan.intraday_quote q ON q.ticker = w.ticker
LEFT JOIN uw_scan.scan_runs r ON r.run_id = wc.run_id
WHERE w.removed_at IS NULL AND (:ticker IS NULL OR w.ticker = :ticker)
ORDER BY w.ticker, wc.run_id DESC NULLS LAST
""",
        ("ticker",),
        (("watchlist_card", "has_card"), ("intraday_quote", "has_quote"), ("scan_runs", "has_run")),
    ),
    "scan_run_signal_bundle": _spec(
        """
WITH picked AS (
  SELECT COALESCE(:run_id, max(run_id)) AS run_id
  FROM uw_scan.signal_gates WHERE ticker = :ticker
), hits AS (
  SELECT h.run_id, h.ticker,
         jsonb_agg(jsonb_build_object('signal_type',h.signal_type,'tier',h.tier,
           'score',h.score,'freshness',h.freshness) ORDER BY h.signal_type) AS signal_hits
  FROM uw_scan.signal_hits h JOIN picked p ON p.run_id = h.run_id
  WHERE h.ticker = :ticker GROUP BY h.run_id, h.ticker
), flags AS (
  SELECT f.run_id, f.ticker,
         jsonb_agg(jsonb_build_object('layer',f.layer,'label',f.label,'value',f.value)
           ORDER BY f.layer, f.label) AS context_flags
  FROM uw_scan.signal_context_flags f JOIN picked p ON p.run_id = f.run_id
  WHERE f.ticker = :ticker GROUP BY f.run_id, f.ticker
)
SELECT g.*, h.signal_hits, f.context_flags, (h.run_id IS NOT NULL) AS has_hits,
       (f.run_id IS NOT NULL) AS has_context_flags
FROM picked p JOIN uw_scan.signal_gates g ON g.run_id=p.run_id AND g.ticker=:ticker
LEFT JOIN hits h ON h.run_id=g.run_id AND h.ticker=g.ticker
LEFT JOIN flags f ON f.run_id=g.run_id AND f.ticker=g.ticker
ORDER BY g.run_id, g.ticker
""",
        ("ticker", "run_id"),
        (("signal_hits", "has_hits"), ("signal_context_flags", "has_context_flags")),
        required=("ticker",),
    ),
    "oi_change_with_quote": _spec(
        """
WITH picked AS (
  SELECT COALESCE(:run_id, max(run_id)) AS run_id
  FROM uw_scan.oi_change_events WHERE underlying_symbol = :ticker
)
SELECT o.*, q.last_price, q.nbbo_bid, q.nbbo_ask, q.implied_volatility,
       btrim(substring(o.option_symbol from '^(.+)[0-9]{6}[CP][0-9]{8}$')) AS root,
       (q.option_symbol IS NOT NULL) AS has_quote
FROM picked p JOIN uw_scan.oi_change_events o ON o.run_id=p.run_id
LEFT JOIN uw_scan.option_contract_snapshots q
  ON q.run_id=o.run_id AND q.option_symbol=o.option_symbol
WHERE o.underlying_symbol=:ticker
ORDER BY o.run_id, o.option_symbol
""",
        ("ticker", "run_id"),
        (("option_contract_snapshots", "has_quote"),),
        required=("ticker",),
    ),
    "strike_grid": _spec(
        """
WITH picked AS (
  SELECT COALESCE(:run_id, max(run_id)) AS run_id
  FROM uw_scan.greeks_by_expiry_strike WHERE ticker=:ticker
), grid AS (
  SELECT g.*, e.dte, e.call_gex, e.put_gex, e.call_vanna AS exposure_call_vanna,
         e.put_vanna AS exposure_put_vanna, s.call_iv AS surface_call_iv,
         s.put_iv AS surface_put_iv, s.underlying_spot,
         btrim(substring(g.call_option_symbol from '^(.+)[0-9]{6}[CP][0-9]{8}$')) AS call_root,
         btrim(substring(g.put_option_symbol from '^(.+)[0-9]{6}[CP][0-9]{8}$')) AS put_root,
         (e.run_id IS NOT NULL) AS has_exposure, (s.ticker IS NOT NULL) AS has_surface
  FROM picked p JOIN uw_scan.greeks_by_expiry_strike g ON g.run_id=p.run_id
  LEFT JOIN uw_scan.exposures_by_expiry_strike e
    ON (e.run_id,e.ticker,e.market_date,e.expiry,e.strike)=
       (g.run_id,g.ticker,g.market_date,g.expiry,g.strike)
  LEFT JOIN uw_scan.option_surface_grid_daily s
    ON (s.ticker,s.market_date,s.expiry,s.strike)=
       (g.ticker,g.market_date,g.expiry,g.strike)
  WHERE g.ticker=:ticker AND (:expiry IS NULL OR g.expiry=:expiry)
    AND (:start IS NULL OR g.market_date>=:start)
    AND (:end IS NULL OR g.market_date<=:end)
)
SELECT grid.*, CASE WHEN call_root=put_root THEN call_root END AS root
FROM grid ORDER BY run_id, market_date, expiry, strike
""",
        ("ticker", "start", "end", "run_id", "expiry"),
        (
            ("exposures_by_expiry_strike", "has_exposure"),
            ("option_surface_grid_daily", "has_surface"),
        ),
        required=("ticker",),
    ),
    "trade_insight_thread": _spec(
        """
WITH picked AS (
  SELECT COALESCE(:run_id, max(run_id)) AS run_id
  FROM uw_scan.trade_insight_snapshots WHERE ticker=:ticker
)
SELECT s.snapshot_id, s.run_id, s.ticker, s.as_of, s.preferred_idea_id,
       c.candidates, a.analyses,
       (c.candidates IS NOT NULL) AS has_candidate,
       (a.analyses IS NOT NULL) AS has_analysis,
       COALESCE(a.has_outcome, false) AS has_outcome
FROM picked p JOIN uw_scan.trade_insight_snapshots s
  ON s.run_id=p.run_id AND s.ticker=:ticker
LEFT JOIN LATERAL (
  SELECT jsonb_agg(to_jsonb(c) ORDER BY c.rank,c.idea_id) AS candidates
  FROM uw_scan.trade_insight_candidates c WHERE c.snapshot_id=s.snapshot_id
) c ON true
LEFT JOIN LATERAL (
  SELECT jsonb_agg(
           to_jsonb(a) || jsonb_build_object('outcome', to_jsonb(o))
           ORDER BY a.analysis_id
         ) AS analyses,
         bool_or(o.analysis_id IS NOT NULL) AS has_outcome
  FROM uw_scan.trade_insight_ai_analyses a
  LEFT JOIN uw_scan.trade_insight_outcomes o ON o.analysis_id=a.analysis_id
  WHERE a.snapshot_id=s.snapshot_id
) a ON true
WHERE (:start IS NULL OR (s.as_of AT TIME ZONE 'UTC')::date>=:start)
  AND (:end IS NULL OR (s.as_of AT TIME ZONE 'UTC')::date<=:end)
ORDER BY s.as_of DESC, s.snapshot_id
""",
        ("ticker", "start", "end", "run_id"),
        (
            ("trade_insight_candidates", "has_candidate"),
            ("trade_insight_ai_analyses", "has_analysis"),
            ("trade_insight_outcomes", "has_outcome"),
        ),
        required=("ticker",),
    ),
    "chain_exposure": _spec(
        """
SELECT m.taxonomy_version,rc.domain,m.chain,m.layer,m.ticker,m.evidence_class,
       e.exposure_id,e.role,e.direction,e.magnitude,e.status,
       (rc.chain IS NOT NULL) AS has_chain,(e.exposure_id IS NOT NULL) AS has_exposure
FROM uw_scan.chain_membership m
LEFT JOIN uw_scan.research_chains rc
  ON (rc.taxonomy_version,rc.chain,rc.layer)=(m.taxonomy_version,m.chain,m.layer)
LEFT JOIN uw_scan.company_exposure e
  ON (e.taxonomy_version,e.chain,e.ticker)=(m.taxonomy_version,m.chain,m.ticker)
 AND e.valid_to IS NULL
WHERE m.valid_to IS NULL AND m.ticker=:ticker
ORDER BY m.taxonomy_version,m.chain,m.layer,e.exposure_id
""",
        ("ticker",),
        (("research_chains", "has_chain"), ("company_exposure", "has_exposure")),
        required=("ticker",),
    ),
    "universe_identity_sector": _spec(
        """
SELECT u.tier,u.ticker,u.layer,u.reason,ci.identity_id,ci.company_type,
       ci.sector AS identity_sector,ci.status AS identity_status,
       cs.sector AS vendor_sector,(ci.identity_id IS NOT NULL) AS has_identity,
       (cs.ticker IS NOT NULL) AS has_sector
FROM uw_scan.fundamental_universe u
LEFT JOIN uw_scan.company_identity ci ON ci.ticker=u.ticker AND ci.valid_to IS NULL
LEFT JOIN uw_scan.company_sector cs ON cs.ticker=u.ticker
WHERE u.removed_at IS NULL AND (:ticker IS NULL OR u.ticker=:ticker)
  AND (:tier IS NULL OR u.tier=:tier)
ORDER BY u.tier,u.ticker
""",
        ("ticker", "tier"),
        (("company_identity", "has_identity"), ("company_sector", "has_sector")),
    ),
    "daily_ohlc_technical": _spec(
        """
SELECT d.*,t.sma20,t.sma50,t.sma200,t.rsi14,t.macd_hist_atr,t.bars_n,
       (t.ticker IS NOT NULL) AS has_technical
FROM uw_scan.daily_ohlc d
LEFT JOIN uw_scan.technical_daily t ON t.ticker=d.ticker AND t.as_of=d.date
WHERE d.ticker=:ticker AND (:start IS NULL OR d.date>=:start)
  AND (:end IS NULL OR d.date<=:end)
ORDER BY d.date,d.ticker
""",
        ("ticker", "start", "end"),
        (("technical_daily", "has_technical"),),
        required=("ticker",),
    ),
    "macro_evidence_chain": _spec(
        """
SELECT s.state_id,s.domain,s.as_of,s.status,e.causal_role,e.ordinal,
       o.obs_id,o.series_id,o.period_end,o.available_at,a.artifact_id,a.source,
       a.source_record_id,a.available_at AS artifact_available_at,
       (e.state_id IS NOT NULL) AS has_evidence,(o.obs_id IS NOT NULL) AS has_observation,
       (a.artifact_id IS NOT NULL) AS has_artifact
FROM uw_scan.macro_domain_states s
LEFT JOIN uw_scan.macro_domain_state_evidence e ON e.state_id=s.state_id
LEFT JOIN uw_scan.macro_observations o ON o.obs_id=e.obs_id
LEFT JOIN uw_scan.macro_source_artifacts a ON a.artifact_id=o.artifact_id
WHERE s.status='published' AND (:start IS NULL OR (s.as_of AT TIME ZONE 'UTC')::date>=:start)
  AND (:end IS NULL OR (s.as_of AT TIME ZONE 'UTC')::date<=:end)
ORDER BY s.as_of,s.state_id,e.ordinal
""",
        ("start", "end"),
        (
            ("macro_domain_state_evidence", "has_evidence"),
            ("macro_observations", "has_observation"),
            ("macro_source_artifacts", "has_artifact"),
        ),
    ),
    "fundamental_evidence_chain": _spec(
        """
SELECT s.result_id,s.ticker,s.as_of,s.engine_version,p.provenance_id,p.role,p.stage,
       o.obs_id,o.period_end,o.statement,(p.provenance_id IS NOT NULL) AS has_provenance,
       (o.obs_id IS NOT NULL) AS has_observation
FROM uw_scan.fundamental_scores s
LEFT JOIN uw_scan.fundamental_result_provenance p ON p.result_id=s.result_id
LEFT JOIN uw_scan.fundamental_statement_obs o ON o.obs_id=p.obs_id
WHERE s.ticker=:ticker AND (:start IS NULL OR s.as_of>=:start)
  AND (:end IS NULL OR s.as_of<=:end)
ORDER BY s.as_of,s.result_id,p.provenance_id
""",
        ("ticker", "start", "end"),
        (
            ("fundamental_result_provenance", "has_provenance"),
            ("fundamental_statement_obs", "has_observation"),
        ),
        required=("ticker",),
    ),
    "daily_signal_panel": _spec(
        """
SELECT v.*,g.call_wall,g.put_wall,g.gamma_flip,g.gamma_magnet,
       s.rr_25d,s.rr_z_180d,s.directional_lean,i.iv_rank_1y,
       (g.ticker IS NOT NULL) AS has_gex,(s.ticker IS NOT NULL) AS has_skew,
       (i.ticker IS NOT NULL) AS has_iv_rank
FROM uw_scan.vrp_daily v
LEFT JOIN uw_scan.uw_gex_levels_daily g
  ON (g.ticker,g.market_date)=(v.ticker,v.market_date)
LEFT JOIN uw_scan.skew_analytics_snapshot s
  ON (s.ticker,s.market_date)=(v.ticker,v.market_date) AND s.basis='eod'
LEFT JOIN uw_scan.iv_rank_history i
  ON (i.ticker,i.market_date)=(v.ticker,v.market_date)
WHERE v.ticker=:ticker AND (:start IS NULL OR v.market_date>=:start)
  AND (:end IS NULL OR v.market_date<=:end)
ORDER BY v.market_date,v.ticker
""",
        ("ticker", "start", "end"),
        (
            ("uw_gex_levels_daily", "has_gex"),
            ("skew_analytics_snapshot", "has_skew"),
            ("iv_rank_history", "has_iv_rank"),
        ),
        required=("ticker",),
    ),
}


def _invalid(message: str) -> ApiError:
    return ApiError(ApiErrorCode.INVALID_PARAMETER, message)


def _one(query: Any, name: str) -> str | None:
    values = query.getlist(name)
    if len(values) > 1:
        raise _invalid(f"{name} may only be specified once")
    return values[0] if values else None


def build_join_query(name: str, query: Any) -> JoinQuery:
    spec = JOIN_REGISTRY.get(name)
    if spec is None:
        raise _invalid(f"unknown join {name!r}")
    accepted = set(spec.filters) | {"limit", "offset"}
    unknown = set(query) - accepted
    if unknown:
        raise _invalid(f"unknown query parameter {min(unknown)!r}")

    raw = {key: _one(query, key) for key in spec.filters}
    missing = sorted(key for key in spec.required if not raw.get(key))
    if missing:
        raise _invalid(f"{missing[0]} is required")
    values: dict[str, Any] = {}
    for key, value in raw.items():
        if value is None:
            values[key] = None
        elif key in {"ticker", "tier"}:
            if not value.strip():
                raise _invalid(f"{key} must not be empty")
            values[key] = value
        elif key == "run_id":
            try:
                values[key] = int(value)
            except ValueError as exc:
                raise _invalid("run_id must be an integer") from exc
            if values[key] < 1:
                raise _invalid("run_id must be at least 1")
            if values[key] > 9_223_372_036_854_775_807:
                raise _invalid("run_id is too large")
        else:
            try:
                values[key] = date.fromisoformat(value)
            except ValueError as exc:
                raise _invalid(f"{key} must be an ISO date") from exc
    if values.get("start") and values.get("end") and values["start"] > values["end"]:
        raise _invalid("start must not be after end")

    def parse_page(key: str, default: int, minimum: int) -> int:
        value = _one(query, key)
        try:
            parsed = default if value is None else int(value)
        except ValueError as exc:
            raise _invalid(f"{key} must be an integer") from exc
        if parsed < minimum:
            raise _invalid(f"{key} must be at least {minimum}")
        if parsed > 2_147_483_647:
            raise _invalid(f"{key} is too large")
        return parsed

    limit = min(parse_page("limit", 500, 1), 5000)
    offset = parse_page("offset", 0, 0)
    params = [values[key] for key in spec.filters]
    positions = {key: index + 1 for index, key in enumerate(spec.filters)}
    sql = _TOKEN.sub(
        lambda match: f"${positions[match.group(1)]}::{_PARAM_TYPES[match.group(1)]}", spec.sql
    )
    params.extend((limit + 1, offset))
    sql += f"\nLIMIT ${len(params)-1}::int OFFSET ${len(params)}::int"
    return JoinQuery(name, sql, tuple(params), limit, spec.coverage)
