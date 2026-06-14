# Phase 0 — Carve / Scoping Pass Implementation Plan

> **For agentic workers:** REQUIRED SUB-SKILL: Use superpowers:subagent-driven-development (recommended) or superpowers:executing-plans to implement this plan task-by-task. Steps use checkbox (`- [ ]`) syntax for tracking.

**Goal:** Prove the TA-signal cores detach cleanly from apex's infrastructure/application layers, and produce the authoritative extraction manifest that Phases 1–3 execute against — without building any feature or deleting any code.

**Architecture:** A small AST-based import-graph tool classifies every cross-layer import in the keep-set as `clean` / `cut` / `follow`. Its output seeds an extraction manifest (markdown). A stub harness then imports the keep-set with all infra/application dependencies replaced by fakes registered in `sys.modules`, and runs the existing TA-signal core unit tests against those stubs. Green proves separability.

**Tech Stack:** Python 3.13, `ast` (stdlib), pytest, existing apex test suite.

**Spec:** `docs/superpowers/specs/2026-06-14-apex-adaptation-design.md` §4.

---

## File Structure

| File | Responsibility |
|---|---|
| `scripts/carve/import_graph.py` | Walk keep-set modules, parse imports via `ast`, classify each cross-layer edge, emit JSON report. |
| `scripts/carve/__init__.py` | Package marker. |
| `tests/carve/test_import_graph.py` | Verify the classifier against known-clean and known-coupled modules. |
| `tests/carve/test_keepset_imports_isolated.py` | Import the keep-set with infra stubbed; assert no real infra import escapes. |
| `tests/carve/stubs/__init__.py` | Fake infra/application modules registered into `sys.modules`. |
| `tests/carve/conftest.py` | Fixture that installs/uninstalls the stubs around isolated-import tests. |
| `docs/superpowers/specs/2026-06-14-apex-phase0-carve.md` | Audit report + extraction manifest + ranked coupling-cut list (the deliverable). |

**Keep-set under audit** (TA-signal path only; backtest excluded per spec §2):
`src/domain/indicators`, `src/domain/signals`, `src/domain/strategy`,
`src/application/services/ta_signal_service.py`,
`src/application/orchestrator/signal_pipeline`,
plus configs `config/signals/*.yaml`, `config/regime_*.yaml`.

**Known coupling (verified 2026-06-14), the classifier must reproduce:**
- `domain/indicators`, `domain/strategy`: zero infra/app imports (clean).
- `domain/signals` → `src.services.historical_data_manager` (×4), `src.infrastructure.observability` (×3), `src.application.services.turning_point.*` (×2+), `src.application.orchestrator.signal_pipeline` (×2), `src.application.services.ta_signal_service` (×1).
- `application/services/ta_signal_service` → `domain.events.event_types.EventType`, `domain.signals.signal_state_tracker.SignalStateTracker` (domain-only).

---

## Task 1: Import-graph classifier tool

**Files:**
- Create: `scripts/carve/__init__.py`
- Create: `scripts/carve/import_graph.py`
- Test: `tests/carve/test_import_graph.py`

- [ ] **Step 1: Create the package marker**

Create `scripts/carve/__init__.py`:

```python
"""Phase 0 carve tooling: import-graph audit + extraction manifest support."""
```

- [ ] **Step 2: Write the failing test**

Create `tests/carve/test_import_graph.py`:

```python
"""Tests for the Phase 0 import-graph classifier."""

from __future__ import annotations

from pathlib import Path

from scripts.carve.import_graph import EdgeClass, classify_module, scan_keepset

REPO = Path(__file__).resolve().parents[2]


def test_clean_core_has_no_cut_edges() -> None:
    """domain/indicators imports nothing from infra/application (CUT).

    Intra-keepset domain imports are allowed (classified FOLLOW); only CUT
    edges (→ infrastructure/services/application) would break separability.
    """
    edges = classify_module(REPO / "src/domain/indicators")
    cuts = [e for e in edges if e.kind == EdgeClass.CUT]
    assert cuts == [], cuts


def test_signals_core_surfaces_known_cuts() -> None:
    """domain/signals must surface its infra/application coupling as CUT edges."""
    edges = classify_module(REPO / "src/domain/signals")
    targets = {e.target for e in edges if e.kind == EdgeClass.CUT}
    assert any("infrastructure.observability" in t for t in targets), targets
    assert any("services.historical_data_manager" in t for t in targets), targets


def test_scan_keepset_returns_all_modules() -> None:
    """The keepset scan classifies every module path it is given."""
    report = scan_keepset(REPO)
    assert set(report.keys()) >= {"domain/indicators", "domain/signals", "domain/strategy"}
    assert all("edges" in v for v in report.values())
```

- [ ] **Step 3: Run test to verify it fails**

Run: `uv run pytest tests/carve/test_import_graph.py -v`
Expected: FAIL with `ModuleNotFoundError: No module named 'scripts.carve.import_graph'`

- [ ] **Step 4: Write minimal implementation**

Create `scripts/carve/import_graph.py`:

```python
"""AST-based import-graph classifier for the Phase 0 carve.

Classifies every cross-layer import in a keep-set module as:
  - CLEAN:  stays inside src/domain (no infra/app dependency)
  - CUT:    imports src.infrastructure / src.services / src.application (or tui/api)
  - FOLLOW: imports another keep-set module (must also be carved)
"""

from __future__ import annotations

import ast
import enum
from dataclasses import dataclass
from pathlib import Path
from typing import Dict, List

# Every internal top-level package under src/ EXCEPT domain. An edge from the
# keep-set to any of these must be handled by the carve (replace-with-port, move,
# or stub) — so they are all "cut candidates". Listing them explicitly means
# edges to src.utils / src.models are classified, never silently skipped.
_CUT_PREFIXES = (
    "infrastructure", "services", "application", "tui", "api",
    "utils", "models", "runners", "verification", "backtest",
)
# Keep-set modules: a FOLLOW edge points at one of these.
_KEEPSET = (
    "domain.indicators",
    "domain.signals",
    "domain.strategy",
    "application.services.ta_signal_service",
    "application.orchestrator.signal_pipeline",
)
_KEEPSET_DIRS = {
    "domain/indicators": "src/domain/indicators",
    "domain/signals": "src/domain/signals",
    "domain/strategy": "src/domain/strategy",
    "application/services/ta_signal_service": "src/application/services/ta_signal_service.py",
    "application/orchestrator/signal_pipeline": "src/application/orchestrator/signal_pipeline",
}


class EdgeClass(enum.Enum):
    CLEAN = "clean"
    CUT = "cut"
    FOLLOW = "follow"


@dataclass(frozen=True)
class ImportEdge:
    source_file: str
    target: str  # dotted module, "src." stripped
    kind: EdgeClass


def _normalize(module: str) -> str:
    """Strip a leading 'src.' so prefixes compare uniformly."""
    return module[4:] if module.startswith("src.") else module


def _classify_target(target: str) -> EdgeClass:
    norm = _normalize(target)
    if any(norm == k or norm.startswith(k + ".") for k in _KEEPSET):
        return EdgeClass.FOLLOW
    head = norm.split(".", 1)[0]
    if head in _CUT_PREFIXES:
        # An application.* import that is itself a keep-set module is FOLLOW (handled above).
        return EdgeClass.CUT
    return EdgeClass.CLEAN


def _iter_py_files(path: Path) -> List[Path]:
    if path.is_file():
        return [path]
    return sorted(p for p in path.rglob("*.py"))


def classify_module(path: Path) -> List[ImportEdge]:
    """Classify every import statement found under `path`."""
    edges: List[ImportEdge] = []
    for py in _iter_py_files(path):
        tree = ast.parse(py.read_text(encoding="utf-8"), filename=str(py))
        for node in ast.walk(tree):
            modules: List[str] = []
            if isinstance(node, ast.ImportFrom) and node.module and node.level == 0:
                modules.append(node.module)
            elif isinstance(node, ast.Import):
                modules.extend(alias.name for alias in node.names)
            for mod in modules:
                norm = _normalize(mod)
                # Only cross-layer edges matter; ignore stdlib/third-party.
                head = norm.split(".", 1)[0]
                if head not in ("domain", *_CUT_PREFIXES):
                    continue
                edges.append(ImportEdge(str(py), norm, _classify_target(mod)))
    return edges


def scan_keepset(repo: Path) -> Dict[str, dict]:
    """Classify every keep-set module; return a serializable report."""
    report: Dict[str, dict] = {}
    for name, rel in _KEEPSET_DIRS.items():
        edges = classify_module(repo / rel)
        report[name] = {
            "path": rel,
            "edges": [
                {"file": e.source_file, "target": e.target, "kind": e.kind.value}
                for e in edges
            ],
            "cut_targets": sorted({e.target for e in edges if e.kind == EdgeClass.CUT}),
            "follow_targets": sorted({e.target for e in edges if e.kind == EdgeClass.FOLLOW}),
        }
    return report


def _main() -> None:
    import json

    repo = Path(__file__).resolve().parents[2]
    print(json.dumps(scan_keepset(repo), indent=2, sort_keys=True))


if __name__ == "__main__":
    _main()
```

- [ ] **Step 5: Run test to verify it passes**

Run: `uv run pytest tests/carve/test_import_graph.py -v`
Expected: PASS (3 passed)

If `test_signals_core_surfaces_known_cuts` fails because relative imports (`from ...services`) are used instead of absolute, extend `classify_module` to resolve `node.level > 0` relative imports against the file's package path. (Note for implementer: apex domain code uses `from ...infrastructure` relative style — handle `ast.ImportFrom` with `level > 0` by reconstructing the absolute module from the file's position under `src/`.)

- [ ] **Step 6: Commit**

```bash
git add scripts/carve/ tests/carve/test_import_graph.py
git commit -m "feat(carve): import-graph classifier for Phase 0 audit"
```

---

## Task 2: Handle relative imports in the classifier

Apex domain modules use relative imports (`from ...infrastructure.observability import ...`). Task 1's classifier only catches absolute imports; this task makes it complete so the manifest is trustworthy.

**Files:**
- Modify: `scripts/carve/import_graph.py`
- Test: `tests/carve/test_import_graph.py`

- [ ] **Step 1: Write the failing test**

Append to `tests/carve/test_import_graph.py`. The test builds a synthetic
package under a temp dir so it is deterministic regardless of which import
style the real `domain/signals` happens to use (it mixes absolute `from
src.infrastructure...` and relative `from ...infrastructure...`):

```python
def test_relative_imports_are_resolved(tmp_path) -> None:
    """A `from ...infrastructure.observability import X` edge resolves to CUT."""
    src = tmp_path / "src"
    pkg = src / "domain" / "signals"
    pkg.mkdir(parents=True)
    (src / "__init__.py").write_text("")
    (src / "domain" / "__init__.py").write_text("")
    (pkg / "__init__.py").write_text("")
    # 3-dot relative: from src.domain.signals -> up 3 -> src, then infrastructure...
    (pkg / "mod.py").write_text(
        "from ...infrastructure.observability import get_logger\n"
    )
    edges = classify_module(pkg)
    rel = [e for e in edges if "infrastructure.observability" in e.target]
    assert rel, f"relative infra import not resolved; got {[e.target for e in edges]}"
    assert all(e.kind == EdgeClass.CUT for e in rel)
```

- [ ] **Step 2: Run test to verify it fails**

Run: `uv run pytest tests/carve/test_import_graph.py::test_relative_imports_are_resolved -v`
Expected: FAIL (relative imports currently skipped because Task 1 only handled `node.level == 0`)

- [ ] **Step 3: Implement relative-import resolution**

In `scripts/carve/import_graph.py`, replace the `classify_module` import-collection block with one that resolves relative imports against the file's package path:

```python
def _package_parts(py: Path, repo_src: Path) -> List[str]:
    """Dotted parts of the PACKAGE containing `py`, relative to src/.

    For both regular modules and __init__.py this is the parent directory:
      src/domain/signals/foo.py     -> ['domain', 'signals']
      src/domain/signals/__init__.py -> ['domain', 'signals']
    Python resolves `from ...x` (level L) relative to the package, so using the
    package parts with (level - 1) is correct for both file kinds.
    """
    return list(py.relative_to(repo_src).parent.parts)


def classify_module(path: Path) -> List[ImportEdge]:
    edges: List[ImportEdge] = []
    repo_src = path
    while repo_src.name != "src" and repo_src.parent != repo_src:
        repo_src = repo_src.parent
    for py in _iter_py_files(path):
        pkg_parts = _package_parts(py, repo_src)
        tree = ast.parse(py.read_text(encoding="utf-8"), filename=str(py))
        for node in ast.walk(tree):
            modules: List[str] = []
            if isinstance(node, ast.ImportFrom):
                if node.level and node.level > 0:
                    # level 1 = current package; each extra dot strips one parent.
                    base = pkg_parts[: len(pkg_parts) - (node.level - 1)]
                    suffix = [node.module] if node.module else []
                    modules.append(".".join(base + suffix))
                elif node.module:
                    modules.append(node.module)
            elif isinstance(node, ast.Import):
                modules.extend(alias.name for alias in node.names)
            for mod in modules:
                norm = _normalize(mod)
                head = norm.split(".", 1)[0]
                if head not in ("domain", *_CUT_PREFIXES):
                    continue
                edges.append(ImportEdge(str(py), norm, _classify_target(mod)))
    return edges
```

- [ ] **Step 4: Run all classifier tests**

Run: `uv run pytest tests/carve/test_import_graph.py -v`
Expected: PASS (4 passed)

- [ ] **Step 5: Commit**

```bash
git add scripts/carve/import_graph.py tests/carve/test_import_graph.py
git commit -m "feat(carve): resolve relative imports in classifier"
```

---

## Task 3: Generate the audit report + extraction manifest

**Files:**
- Modify: `scripts/carve/import_graph.py` (add manifest renderer)
- Create: `docs/superpowers/specs/2026-06-14-apex-phase0-carve.md` (generated + hand-finished)

- [ ] **Step 1: Add a manifest renderer**

Append to `scripts/carve/import_graph.py`:

```python
def render_manifest(report: Dict[str, dict]) -> str:
    """Render the cut/follow findings as a markdown table for the manifest doc."""
    lines = ["| Module | Clean | Cut targets | Follow targets |", "|---|---|---|---|"]
    for name, data in sorted(report.items()):
        cuts = data["cut_targets"]
        follows = data["follow_targets"]
        clean = "✅" if not cuts and not follows else ""
        lines.append(
            f"| `{name}` | {clean} | "
            f"{'<br>'.join(f'`{c}`' for c in cuts) or '—'} | "
            f"{'<br>'.join(f'`{f}`' for f in follows) or '—'} |"
        )
    return "\n".join(lines)
```

- [ ] **Step 2: Run the tool and capture output**

Run: `uv run python -m scripts.carve.import_graph > /tmp/carve_report.json`
Run: `uv run python -c "import json,scripts.carve.import_graph as g; print(g.render_manifest(json.load(open('/tmp/carve_report.json'))))"`
Expected: a markdown table listing each keep-set module with its cut/follow targets.

- [ ] **Step 3: Write the audit/manifest deliverable**

Create `docs/superpowers/specs/2026-06-14-apex-phase0-carve.md` with these sections (paste the generated table into "Extraction manifest"):

```markdown
# Apex Phase 0 — Carve Audit & Extraction Manifest

Date: 2026-06-14 · Branch: feat/apex-adaptation
Generated by: scripts/carve/import_graph.py

## 1. Import-graph summary
<one paragraph: which cores are clean, which are coupled>

## 2. Extraction manifest
<paste render_manifest() table>

## 3. Per-module disposition
For every top-level module under src/, tag KEEP / DROP / MOVE / STUB:
| Module | Disposition | Reason |
|---|---|---|
<fill from the tree; KEEP = TA-signal path, DROP = adapters/tui/worker-assets/risk, etc.>

## 4. Coupling-cut list (ranked, seeds Phase 1)
1. domain/signals → services.historical_data_manager  ⇒ replace with HistoricalSourcePort wiring (Phase 1)
2. domain/signals → infrastructure.observability        ⇒ no-op shim / structured-logging port
3. domain/signals → application.services.turning_point  ⇒ DECIDE: pull into keep-set or stub
...

## 4b. Observed-but-NOT-acted-on (out of scope)
- src/backtest → infrastructure.adapters.ib.historical_adapter — RECORDED ONLY.
  Backtest is out of scope (spec §2); this collision is **deferred decision D5**
  for Phase 6. Do NOT cut, rewire, or modify backtest in Phases 0–3.

## 5. Risk notes
<anything surprising the scan revealed>
```

- [ ] **Step 4: Commit**

```bash
git add -f docs/superpowers/specs/2026-06-14-apex-phase0-carve.md
git add scripts/carve/import_graph.py
git commit -m "docs(carve): Phase 0 audit report + extraction manifest"
```

---

## Task 4: Stub harness — keep-set imports with infra faked

**Files:**
- Create: `tests/carve/stubs/__init__.py`
- Create: `tests/carve/conftest.py`
- Test: `tests/carve/test_keepset_imports_isolated.py`

- [ ] **Step 1: Write the failing test**

Create `tests/carve/test_keepset_imports_isolated.py`:

```python
"""Prove the ENTIRE TA-signal keep-set imports with infrastructure stubbed out.

Importing a package's __init__ does not import its submodules, so we enumerate
every .py file under the keep-set dirs (plus the two single-file keep-set
modules) and import each one individually under the stub harness.
"""

from __future__ import annotations

import importlib
from pathlib import Path

import pytest

REPO = Path(__file__).resolve().parents[2]
SRC = REPO / "src"

# Directories whose every submodule must import; plus explicit single-file modules.
_KEEPSET_DIRS = [
    SRC / "domain" / "indicators",
    SRC / "domain" / "signals",
    SRC / "domain" / "strategy",
    SRC / "application" / "orchestrator" / "signal_pipeline",
]
_KEEPSET_FILES = [SRC / "application" / "services" / "ta_signal_service.py"]


def _module_name(py: Path) -> str:
    rel = py.relative_to(REPO).with_suffix("")
    parts = [p for p in rel.parts if p != "__init__"]
    return ".".join(parts)


def _all_keepset_modules() -> list[str]:
    mods: list[str] = []
    for d in _KEEPSET_DIRS:
        mods += [_module_name(p) for p in sorted(d.rglob("*.py"))]
    mods += [_module_name(p) for p in _KEEPSET_FILES]
    return sorted(set(mods))


@pytest.mark.usefixtures("install_carve_stubs")
@pytest.mark.parametrize("modname", _all_keepset_modules())
def test_keepset_module_imports_with_stubs(modname: str) -> None:
    mod = importlib.import_module(modname)
    assert mod is not None
```

- [ ] **Step 2: Run test to verify it fails**

Run: `uv run pytest tests/carve/test_keepset_imports_isolated.py -v`
Expected: FAIL with `fixture 'install_carve_stubs' not found` (or an ImportError pulling in real infra).

- [ ] **Step 3: Write the stub package**

Create `tests/carve/stubs/__init__.py`:

```python
"""Fake infra/application modules for isolated keep-set imports.

Each stub is registered into sys.modules BEFORE the keep-set is imported, so the
real infrastructure never loads. This proves the cores depend only on the shape
of these ports, not their implementations.
"""

from __future__ import annotations

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

    # observability: no-op logger/metrics
    obs = _module("src.infrastructure.observability")
    obs.get_logger = lambda *a, **k: __import__("logging").getLogger("carve-stub")  # type: ignore[attr-defined]
    obs.record_metric = lambda *a, **k: None  # type: ignore[attr-defined]
    installed.append("src.infrastructure.observability")

    # historical_data_manager: fake bar source returning empty lists
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
```

- [ ] **Step 4: Write the conftest fixture**

Create `tests/carve/conftest.py`:

```python
"""Fixtures for Phase 0 carve isolation tests."""

from __future__ import annotations

import sys
from typing import Iterator

import pytest

from tests.carve.stubs import install, uninstall


@pytest.fixture
def install_carve_stubs() -> Iterator[None]:
    # Drop any already-imported keep-set/infra modules so stubs take effect.
    for name in list(sys.modules):
        if name.startswith(("src.domain", "src.infrastructure", "src.services", "src.application")):
            del sys.modules[name]
    installed = install()
    try:
        yield
    finally:
        uninstall(installed)
        for name in list(sys.modules):
            if name.startswith("src.domain"):
                del sys.modules[name]
```

- [ ] **Step 5: Run test to verify it passes**

Run: `uv run pytest tests/carve/test_keepset_imports_isolated.py -v`
Expected: PASS for `domain.indicators` and `domain.strategy`. `domain.signals` may still FAIL if it pulls additional uncovered modules — that is a **finding**, not a failure of the plan: add the missing stub to `install()` and record the coupling in the manifest §4. Iterate until all three pass, adding one stub per uncovered import.

- [ ] **Step 6: Commit**

```bash
git add tests/carve/
git commit -m "test(carve): isolated keep-set import harness with infra stubs"
```

---

## Task 5: Confirm TA-signal core unit tests pass

**Note on what this proves:** the *isolation* claim rests on Task 4 (every keep-set
module imports with infra stubbed). This task is the complementary check — that the
core suites are green in the normal environment (the cores are functional, not just
importable). Do not assume directory names; discover them first.

**Files:**
- Test: the real core suites discovered in Step 1
- Modify: `docs/superpowers/specs/2026-06-14-apex-phase0-carve.md` (record results)

- [ ] **Step 1: Discover the real core test paths**

Run: `find tests -type d | grep -iE 'indicator|signal|strateg' ; find tests -name 'test_*.py' | grep -iE 'indicator|signal|strateg' | head -30`
Expected: the actual test files/dirs. Use the paths that exist — do not guess `tests/unit/domain/...` if the tree differs.

- [ ] **Step 2: Run the discovered indicator + strategy suites**

Run: `uv run pytest <discovered indicator/strategy paths> -v`
Expected: PASS (these cores have zero CUT edges, so they run standalone).

- [ ] **Step 3: Run the discovered signals suite, note any infra-import errors**

Run: `uv run pytest <discovered signals paths> -v`
Expected: PASS, OR collection errors that name a real infra import. Each such error is a coupling-cut to add to manifest §4 and a stub to add to `tests/carve/stubs`.

- [ ] **Step 4: Record results in the carve doc**

Edit `docs/superpowers/specs/2026-06-14-apex-phase0-carve.md` §5: list which suites pass, which (if any) needed a stub, and the exact import that forced each stub. If a suite cannot be collected at all without real infra, that is the single most important Phase-0 finding — record it prominently.

- [ ] **Step 5: Commit**

```bash
git add -f docs/superpowers/specs/2026-06-14-apex-phase0-carve.md
git commit -m "docs(carve): record core-test isolation results"
```

---

## Task 6: Finalize manifest — every src/ module classified

**Files:**
- Modify: `docs/superpowers/specs/2026-06-14-apex-phase0-carve.md`

- [ ] **Step 1: Enumerate every top-level src/ module**

Run: `ls -d src/*/ src/*.py`
Expected: the full top-level module list.

- [ ] **Step 2: Tag each KEEP / DROP / MOVE / STUB in §3**

Fill the §3 table so **every** entry from Step 1 has a disposition and reason. Cross-check against spec §3 "Remove" list (broker adapters, tui, worker-assets, risk monitor, event bus, orchestrator → DROP) and keep-set (→ KEEP).

- [ ] **Step 3: Verify success criteria**

Confirm against spec §4.5: (a) every module classified; (b) keep-set imports with stubs; (c) core tests pass isolated; (d) zero ambiguous FOLLOW edges. Add a "Success criteria" checklist to the doc with each box ticked or an explicit follow-up noted.

- [ ] **Step 4: Commit**

```bash
git add -f docs/superpowers/specs/2026-06-14-apex-phase0-carve.md
git commit -m "docs(carve): complete extraction manifest (all modules classified)"
```

---

## Self-Review (completed during planning)

- **Spec coverage:** §4.3 step 1 (audit)→Tasks 1–3; step 2 (manifest)→Tasks 3,6; step 3 (proof)→Tasks 4,5. §4.4 deliverables→Tasks 3,5,6. §4.5 criteria→Task 6 step 3. ✅
- **Out-of-scope guard:** no task deletes drop-set or touches backtest (spec §4.6). ✅
- **Type consistency:** `EdgeClass`, `ImportEdge`, `classify_module`, `scan_keepset`, `render_manifest`, `install`/`uninstall`/`install_carve_stubs` used consistently across tasks. ✅
