"""The lake query layer must not reach REST, PostgreSQL or streaming startup.

MCP (PR2) imports ``src.application.lake`` directly; if that package pulled in the
REST app, the PG pools or the subscription pipeline -- even transitively -- the MCP
process would start them. This walks the real import graph from the package's
source files (module-level imports only, as executed at import time).

``src.mcp_server`` itself is walked separately below: it is allowed to reach the pure
REST payload builders it reuses (``src.api.payload.chart`` / ``.lake``, plus their
parent packages), and nothing else under FORBIDDEN -- not the FastAPI app, not the
routes, not PG.
"""

from __future__ import annotations

import ast
from pathlib import Path

REPO = Path(__file__).resolve().parents[2]
ROOTS = ("src.application.lake",)
FORBIDDEN = (
    "src.api",
    "src.application.subscriptions",
    "src.application.orchestrator",
    "src.infrastructure.persistence",
    "src.services",
    "asyncpg",
    "fastapi",
)

MCP_ROOTS = ("src.mcp_server",)
# Exact packages (reached only as parent-package side effects, never suffixed with an
# imported name) vs. the two payload modules MCP actually imports names from (which
# the walk also records as "<module>.<name>" pseudo-entries -- see ``_imports``).
MCP_ALLOWED_EXACT = frozenset({"src.api", "src.api.payload"})
MCP_ALLOWED_PREFIX = frozenset({"src.api.payload.chart", "src.api.payload.lake"})


def _mcp_allowed(module: str) -> bool:
    return module in MCP_ALLOWED_EXACT or any(
        module == prefix or module.startswith(prefix + ".") for prefix in MCP_ALLOWED_PREFIX
    )


def _module_file(module: str) -> Path | None:
    base = REPO / Path(*module.split("."))
    for candidate in (base.with_suffix(".py"), base / "__init__.py"):
        if candidate.is_file():
            return candidate
    return None


def _imports(module: str, path: Path) -> set[str]:
    package = module if path.name == "__init__.py" else module.rsplit(".", 1)[0]
    found: set[str] = set()
    for node in ast.parse(path.read_text()).body:  # module level: what import executes
        if isinstance(node, ast.Import):
            found.update(alias.name for alias in node.names)
        elif isinstance(node, ast.ImportFrom):
            if node.level:
                parts = package.split(".")
                base = ".".join(parts[: len(parts) - node.level + 1])
                target = f"{base}.{node.module}" if node.module else base
            else:
                target = node.module or ""
            found.add(target)
            found.update(f"{target}.{alias.name}" for alias in node.names)
    return found


def _closure(roots: tuple[str, ...] = ROOTS) -> dict[str, str]:
    """Every reachable module mapped to the module that first imported it."""
    seen: dict[str, str] = {}
    queue = []
    for root in roots:
        for path in sorted((REPO / Path(*root.split("."))).glob("*.py")):
            name = f"{root}.{path.stem}" if path.stem != "__init__" else root
            seen[name] = "<root>"
            queue.append(name)
    while queue:
        module = queue.pop()
        path = _module_file(module)
        if path is None:
            continue
        # Importing a.b.c first executes a/__init__.py and a/b/__init__.py: a parent
        # package's re-exports are as much a part of the import as the module itself.
        parents = [".".join(module.split(".")[:i]) for i in range(1, module.count(".") + 1)]
        for target in [*parents, *_imports(module, path)]:
            if target not in seen:
                seen[target] = module
                if target.startswith("src.") and _module_file(target) is not None:
                    queue.append(target)
    return seen


def test_lake_queries_do_not_reach_rest_pg_or_streaming() -> None:
    reached = _closure()
    violations = {
        module: via
        for module, via in reached.items()
        if any(module == bad or module.startswith(bad + ".") for bad in FORBIDDEN)
    }
    assert violations == {}, violations


def test_the_walk_actually_follows_imports() -> None:
    """Guard against a walk that finds nothing and passes vacuously."""
    reached = _closure()
    assert "src.infrastructure.adapters.livewire.pit_revisions" in reached
    assert "duckdb" in reached


def test_mcp_server_reaches_forbidden_modules_only_through_the_payload_seam() -> None:
    """MCP may import the pure REST payload builders (``chart.py`` / ``lake.py`` under
    ``src.api.payload``, plus their parent packages) and nothing else FORBIDDEN reaches
    -- not the FastAPI app, not the routes, not PG, not the subscription pipeline."""
    reached = _closure(MCP_ROOTS)
    violations = {
        module: via
        for module, via in reached.items()
        if not _mcp_allowed(module)
        and any(module == bad or module.startswith(bad + ".") for bad in FORBIDDEN)
    }
    assert violations == {}, violations


def test_mcp_server_walk_actually_follows_imports_and_reaches_the_allowed_seam() -> None:
    """Guard against a walk that finds nothing, and against an allowlist so broad it
    would let the guard above pass vacuously."""
    reached = _closure(MCP_ROOTS)
    assert "src.mcp_server.tools_bars" in reached
    assert "src.application.lake.bars" in reached
    # The allowed seam is really exercised, not just permitted...
    assert "src.api.payload.chart" in reached
    assert "src.api.payload.lake" in reached
    # ...and nothing outside it under "src.api" is allowed to sneak in unnoticed.
    assert not _mcp_allowed("src.api.routes.chart")
    assert not _mcp_allowed("src.api.server")
