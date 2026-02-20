"""DashboardBuilder — Orchestrator for static SPA dashboard generation."""

import logging
import shutil
import subprocess
import time
from dataclasses import dataclass
from pathlib import Path

from src.infrastructure.reporting.dashboard.data_transformer import (
    DataTransformer,
    TransformConfig,
)

logger = logging.getLogger(__name__)

# Static assets live alongside this module
STATIC_DIR = Path(__file__).parent / "static"

# CF cache headers: short TTL for data (refreshes hourly), long for assets
CF_HEADERS = """\
/data/*
  Cache-Control: public, max-age=300
  X-Content-Type-Options: nosniff

/assets/*
  Cache-Control: public, max-age=86400, immutable
  X-Content-Type-Options: nosniff

/index.html
  Cache-Control: public, max-age=300
  X-Content-Type-Options: nosniff
  X-Frame-Options: DENY
  Referrer-Policy: strict-origin-when-cross-origin
"""


@dataclass
class DashboardManifest:
    """Result summary from a dashboard build."""

    symbols: list[str]
    file_count: int
    total_size_bytes: int
    elapsed_seconds: float


class DashboardBuilder:
    """Build a static SPA dashboard from pipeline output.

    Usage:
        builder = DashboardBuilder()
        manifest = builder.build()
    """

    def __init__(
        self,
        source_dir: Path = Path("out/signals"),
        max_bars: int = 500,
        momentum_path: Path = Path("out/momentum/data/momentum_watchlist.json"),
        pead_path: Path = Path("out/pead/data/pead_candidates.json"),
    ) -> None:
        self.source_dir = source_dir
        self.max_bars = max_bars
        self.momentum_path = momentum_path
        self.pead_path = pead_path

    def build(self, output_dir: Path = Path("out/site")) -> DashboardManifest:
        """Build the dashboard into output_dir.

        Steps:
        1. Validate source exists
        2. Clean output dir
        3. Copy static assets
        4. Compile TypeScript → JS
        5. Transform data
        6. Write CF headers + .nojekyll
        """
        t0 = time.monotonic()

        # 0. Validate source
        if not self.source_dir.exists():
            raise FileNotFoundError(
                f"Source directory not found: {self.source_dir}. "
                "Run the signal pipeline first (make signals-test)."
            )

        source_data = self.source_dir / "data"
        if not source_data.exists():
            raise FileNotFoundError(
                f"Source data directory not found: {source_data}. "
                "Signal pipeline output appears incomplete."
            )

        # 1. Clean output
        if output_dir.exists():
            shutil.rmtree(output_dir)
        output_dir.mkdir(parents=True)

        # 2. Copy static assets
        self._copy_static(output_dir)

        # 3. Compile TypeScript → JS (in output dir, removes .ts sources)
        self._compile_typescript(output_dir)

        # 4. Transform data
        config = TransformConfig(
            max_bars=self.max_bars,
            source_dir=self.source_dir,
            momentum_path=self.momentum_path,
            pead_path=self.pead_path,
        )
        transformer = DataTransformer(config)
        result = transformer.transform(output_dir)

        # 5. Write CF headers + .nojekyll
        (output_dir / "_headers").write_text(CF_HEADERS, encoding="utf-8")
        (output_dir / ".nojekyll").touch()

        # Count total files
        all_files = list(output_dir.rglob("*"))
        total_files = sum(1 for f in all_files if f.is_file())
        total_size = sum(f.stat().st_size for f in all_files if f.is_file())
        elapsed = time.monotonic() - t0

        logger.info(
            "Dashboard built: %d files, %.1f MB, %.1fs",
            total_files,
            total_size / (1024 * 1024),
            elapsed,
        )

        return DashboardManifest(
            symbols=result.symbols,
            file_count=total_files,
            total_size_bytes=total_size,
            elapsed_seconds=elapsed,
        )

    def _copy_static(self, output_dir: Path) -> None:
        """Copy static/ directory contents to output, preserving structure."""
        if not STATIC_DIR.exists():
            raise FileNotFoundError(f"Static assets not found at: {STATIC_DIR}")

        for src_file in STATIC_DIR.rglob("*"):
            if not src_file.is_file():
                continue
            rel = src_file.relative_to(STATIC_DIR)

            # index.html goes to root, everything else under assets/
            if rel.name == "index.html":
                dst = output_dir / "index.html"
            else:
                dst = output_dir / "assets" / rel
            dst.parent.mkdir(parents=True, exist_ok=True)
            shutil.copy2(src_file, dst)

    def _compile_typescript(self, output_dir: Path) -> None:
        """Compile .ts → .js via esbuild (transpile-only, no bundling)."""
        assets_dir = output_dir / "assets"
        ts_files = list(assets_dir.rglob("*.ts"))
        if not ts_files:
            return

        try:
            subprocess.run(
                [
                    "npx",
                    "--yes",
                    "esbuild@0.24",
                    *[str(f) for f in ts_files],
                    f"--outdir={assets_dir}",
                    "--format=esm",
                    "--target=es2022",
                ],
                check=True,
                capture_output=True,
            )
        except FileNotFoundError:
            logger.warning(
                "esbuild not available (npx not found). "
                "Falling back to direct .ts → .js rename (strip types manually)."
            )
            # Fallback: just rename .ts → .js (works if code is valid JS + type annotations)
            for f in ts_files:
                js_path = f.with_suffix(".js")
                if not js_path.exists():
                    shutil.copy2(f, js_path)
        except subprocess.CalledProcessError as e:
            logger.warning("esbuild compilation failed: %s", e.stderr.decode()[:500])

        # Remove .ts sources from output (keep only compiled .js)
        for f in assets_dir.rglob("*.ts"):
            f.unlink()

        # Remove tsconfig.json and types/ from output (dev-only files)
        types_dir = assets_dir / "types"
        if types_dir.exists():
            shutil.rmtree(types_dir)
        tsconfig = assets_dir / "tsconfig.json"
        if tsconfig.exists():
            tsconfig.unlink()
