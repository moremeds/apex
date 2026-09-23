"""``scripts/lake_verify`` is a standalone script tree (``matrix.py`` adds itself to
``sys.path`` at import time so its siblings -- ``model``, ``mcp_exec`` -- resolve as
bare top-level imports, never as a package). Tests here import those same modules the
same way, so this adds the directory once, before collection.
"""

import sys
from pathlib import Path

SCRIPTS_LAKE_VERIFY = Path(__file__).resolve().parents[3] / "scripts" / "lake_verify"
if str(SCRIPTS_LAKE_VERIFY) not in sys.path:
    sys.path.insert(0, str(SCRIPTS_LAKE_VERIFY))
