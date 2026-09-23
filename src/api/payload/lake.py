"""Payloads for the lake discovery, revision, identity and gap routes.

Shared by REST and (PR2) MCP so both transports serialize the same fields. Pages carry
``limit``, ``offset``, ``returned``, ``truncated`` and ``next_offset``.
"""

from __future__ import annotations

from dataclasses import asdict
from datetime import datetime, timezone
from typing import Any, Dict, Optional

from src.application.lake.catalog import (
    CoverageResult,
    FuturesContract,
    InstrumentDetail,
)
from src.application.lake.gaps import GapsResult, SessionRange
from src.application.lake.identity import SecurityResolution
from src.application.lake.revisions import (
    PitRevisionDetail,
    PitRevisionList,
    SilverRevisionDetail,
    SilverRevisionList,
    pit_summary_dict,
    scope_dict,
)
from src.application.lake.services import Page


def _now() -> str:
    return datetime.now(timezone.utc).isoformat()


def page_fields(page: Page[Any]) -> Dict[str, Any]:
    return {
        "limit": page.limit,
        "offset": page.offset,
        "returned": len(page.items),
        "truncated": page.truncated,
        "next_offset": page.next_offset,
    }


def instrument_payload(detail: InstrumentDetail) -> Dict[str, Any]:
    return {
        "symbol": detail.symbol,
        "asset_class": detail.asset_class,
        "listing_status": "listed",
        "timeframes": detail.timeframes,
        # Additive: per-timeframe residency, including the archived tree.
        "residency": detail.residency,
        "coverage_source": detail.coverage_source,
        "first_date": detail.first_date,
        "last_date": detail.last_date,
        "silver_available": detail.silver_available,
        "price_mode": detail.price_mode,
        "adjustment_revision": detail.adjustment_revision,
        "generated_at": _now(),
    }


def coverage_payload(result: CoverageResult) -> Dict[str, Any]:
    return {
        "source": "livewire_coverage_snapshot",
        "catalog": asdict(result.catalog),
        "rows": [asdict(row) for row in result.page.items],
        **page_fields(result.page),
        "generated_at": _now(),
    }


def futures_payload(root: str, page: Page[FuturesContract]) -> Dict[str, Any]:
    return {
        "root": root.upper(),
        "contracts": [asdict(contract) for contract in page.items],
        **page_fields(page),
        "generated_at": _now(),
    }


def silver_list_payload(result: SilverRevisionList) -> Dict[str, Any]:
    return {
        "current": result.current,
        "current_error": result.current_error,
        "revisions": [
            {"revision": number, "is_current": number == result.current}
            for number in result.page.items
        ],
        **page_fields(result.page),
        "generated_at": _now(),
    }


def silver_detail_payload(detail: SilverRevisionDetail) -> Dict[str, Any]:
    revision = detail.revision
    return {
        "revision": revision.revision,
        "is_current": detail.is_current,
        "generation_id": revision.generation_id,
        "published_at": revision.published_at.isoformat(),
        "corporate_actions_as_of": revision.corporate_actions_as_of.isoformat(),
        "affected_count": len(revision.affected),
        "artifact_count": len(revision.artifacts),
        "affected": [
            {
                "symbol": item.symbol,
                "earliest_date": item.earliest_date.isoformat(),
                "timeframes": list(item.timeframes),
            }
            for item in detail.affected.items
        ],
        **page_fields(detail.affected),
        "generated_at": _now(),
    }


def pit_list_payload(result: PitRevisionList) -> Dict[str, Any]:
    return {
        "available": bool(result.page.items) or bool(result.latest_per_index),
        "latest_per_index": result.latest_per_index,
        "revisions": [pit_summary_dict(summary) for summary in result.page.items],
        **page_fields(result.page),
        "generated_at": _now(),
    }


def pit_detail_payload(detail: PitRevisionDetail) -> Dict[str, Any]:
    manifest = detail.manifest
    return {
        **pit_summary_dict(manifest.summary),
        "policy_version": manifest.policy_version,
        "session_policy": manifest.session_policy,
        "corporate_actions_as_of": manifest.corporate_actions_as_of,
        "generation_id": manifest.generation_id,
        "input_hash": manifest.input_hash,
        # Livewire's lineage references, echoed as metadata; apex does not replay them.
        "inputs": dict(manifest.input_references),
        "daily_artifact_count": detail.daily_artifact_count,
        "members": [scope_dict(scope) for scope in detail.members.items],
        **page_fields(detail.members),
        "generated_at": _now(),
    }


def security_payload(resolution: SecurityResolution) -> Dict[str, Any]:
    return {
        "symbol": resolution.symbol,
        "security_id": resolution.security_id,
        "as_of": resolution.as_of.isoformat(),
        "known_at": None if resolution.known_at is None else resolution.known_at.isoformat(),
        # Dates evaluate at UTC end-of-day; no known_at is today's reconstruction.
        "knowledge": "today" if resolution.known_at is None else "as_known_at",
        "source": "livewire_security_master",
        "generated_at": _now(),
    }


def _range(value: Optional[SessionRange]) -> Optional[Dict[str, Any]]:
    if value is None:
        return None
    return {
        "start": value.start.isoformat(),
        "end": value.end.isoformat(),
        "sessions": value.sessions,
    }


def gaps_payload(result: GapsResult) -> Dict[str, Any]:
    repairs = result.repairs
    return {
        "symbol": result.symbol,
        "asset_class": result.asset_class,
        "timeframe": result.timeframe,
        "listing_status": result.listing_status,
        "assessment": "session_presence",
        "window": {"start": result.start.isoformat(), "end": result.end.isoformat()},
        "calendar": result.calendar,
        "lifetime": result.lifetime,
        "file_bounds": {
            "first": None if result.file_first is None else result.file_first.isoformat(),
            "last": None if result.file_last is None else result.file_last.isoformat(),
        },
        "status": result.status,
        "expected_sessions": result.expected_sessions,
        "present_sessions": result.present_sessions,
        "not_expected_sessions": result.not_expected_sessions,
        "leading_unobserved": _range(result.leading_unobserved),
        "trailing_unobserved": _range(result.trailing_unobserved),
        "gaps": [_range(gap) for gap in result.gaps],
        "gaps_total": result.gaps_total,
        "truncated": result.truncated,
        "repairs": {
            "state": repairs.state,
            "warnings": list(repairs.warnings),
            "reports_read": repairs.reports_read,
            # Historical gap-engine evidence, not current truth.
            "entries": [
                {
                    "report_kind": entry.report_kind,
                    "reports": list(entry.reports),
                    "report_date": entry.report_date,
                    "sessions": list(entry.sessions),
                    **entry.details,
                }
                for entry in repairs.entries
            ],
        },
        "generated_at": _now(),
    }
