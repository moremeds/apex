"""AST-based import-graph classifier for the Phase 0 carve.

Classifies every cross-layer import in a keep-set module as:
  - CLEAN:  stays inside src/domain (no infra/app dependency)
  - CUT:    imports src.infrastructure / src.services / src.application (or tui/api/...)
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
        return EdgeClass.CUT
    return EdgeClass.CLEAN


def _iter_py_files(path: Path) -> List[Path]:
    if path.is_file():
        return [path]
    return sorted(p for p in path.rglob("*.py"))


def _package_parts(py: Path, repo_src: Path) -> List[str]:
    """Dotted parts of the PACKAGE containing `py`, relative to src/.

    For both regular modules and __init__.py this is the parent directory:
      src/domain/signals/foo.py      -> ['domain', 'signals']
      src/domain/signals/__init__.py -> ['domain', 'signals']
    Python resolves `from ...x` (level L) relative to the package, so using the
    package parts with (level - 1) is correct for both file kinds.
    """
    return list(py.relative_to(repo_src).parent.parts)


def classify_module(path: Path) -> List[ImportEdge]:
    """Classify every import statement found under `path`."""
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


def render_manifest(report: Dict[str, dict]) -> str:
    """Render the cut/follow findings as a markdown table for the manifest doc."""
    lines = ["| Module | Clean | Cut targets | Follow targets |", "|---|---|---|---|"]
    for name, data in sorted(report.items()):
        cuts = data["cut_targets"]
        follows = data["follow_targets"]
        clean = "yes" if not cuts and not follows else ""
        lines.append(
            f"| `{name}` | {clean} | "
            f"{'<br>'.join(f'`{c}`' for c in cuts) or '-'} | "
            f"{'<br>'.join(f'`{f}`' for f in follows) or '-'} |"
        )
    return "\n".join(lines)


def _main() -> None:
    import json

    repo = Path(__file__).resolve().parents[2]
    print(json.dumps(scan_keepset(repo), indent=2, sort_keys=True))


if __name__ == "__main__":
    _main()
