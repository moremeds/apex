"""Fake infra/application modules for isolated keep-set imports.

Each stub is registered into sys.modules BEFORE the keep-set is imported, so the
real infrastructure never loads. This proves the cores depend only on the shape
of these ports, not their implementations. Stubs are added one-per-uncovered
import as the harness surfaces them (each addition is recorded in the carve doc).
"""

from __future__ import annotations

import logging
import sys
import types
from typing import List


def _module(name: str) -> types.ModuleType:
    mod = types.ModuleType(name)
    sys.modules[name] = mod
    return mod


def install() -> List[str]:
    """Register stub modules; return the list of names installed (for teardown)."""
    installed: List[str] = []

    # --- infrastructure.observability: no-op logger/metrics -----------------
    obs = _module("src.infrastructure.observability")

    class _NoOp:
        """Permissive no-op: any attribute access returns a no-op callable.

        Covers SignalMetrics' full record_*/set_*/time_* surface without having
        to enumerate it — the cores only need the calls to not fail.
        """

        def __init__(self, *a, **k) -> None: ...

        def __getattr__(self, _name: str):
            return lambda *a, **k: None

    def _noop_decorator(*a, **k):
        # Works as @dec and as @dec(...).
        if len(a) == 1 and callable(a[0]) and not k:
            return a[0]
        return lambda f: f

    obs.SignalMetrics = _NoOp  # type: ignore[attr-defined]
    obs.get_logger = lambda *a, **k: logging.getLogger("carve-stub")  # type: ignore[attr-defined]
    obs.record_metric = lambda *a, **k: None  # type: ignore[attr-defined]
    obs.trace = lambda *a, **k: (lambda f: f)  # type: ignore[attr-defined]
    for _name in (
        "time_confluence_calculation",
        "time_alignment_calculation",
        "time_indicator_computation",
        "time_rule_evaluation",
    ):
        setattr(obs, _name, _noop_decorator)
    installed.append("src.infrastructure.observability")

    # --- services.historical_data_manager: fake bar source ------------------
    hdm = _module("src.services.historical_data_manager")

    class HistoricalDataManager:  # minimal shape used by domain/signals
        def __init__(self, *a, **k) -> None: ...

        def get_bars(self, *a, **k) -> list:
            return []

    hdm.HistoricalDataManager = HistoricalDataManager  # type: ignore[attr-defined]
    installed.append("src.services.historical_data_manager")

    return installed


def uninstall(names: List[str]) -> None:
    for name in names:
        sys.modules.pop(name, None)
