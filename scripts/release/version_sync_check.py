"""Fail if VERSION and pyproject.toml [project].version disagree.

VERSION (repo root) is the source of truth. apex is Python-only — there is no
package.json — so the single tracked artifact is pyproject.toml's [project].version.
cut.sh rewrites both in lockstep; CI runs this to lock the invariant in.
"""

from __future__ import annotations

import argparse
import sys
import tomllib
from pathlib import Path


def main() -> int:
    ap = argparse.ArgumentParser()
    ap.add_argument("--root", default=".", type=Path)
    args = ap.parse_args()

    version = (args.root / "VERSION").read_text().strip()

    pyproject = tomllib.loads((args.root / "pyproject.toml").read_text())
    proj_version = pyproject.get("project", {}).get("version", "")

    if proj_version != version:
        print(
            f"version mismatch: VERSION={version!r} "
            f"pyproject.toml[project.version]={proj_version!r}",
            file=sys.stderr,
        )
        return 1

    print(f"OK: {version}")
    return 0


if __name__ == "__main__":
    sys.exit(main())
