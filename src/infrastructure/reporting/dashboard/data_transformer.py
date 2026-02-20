"""Transform pipeline output into trimmed dashboard data."""

import json
import logging
import shutil
from dataclasses import dataclass, field
from pathlib import Path

logger = logging.getLogger(__name__)


@dataclass
class TransformConfig:
    """Configuration for data transformation."""

    max_bars: int = 500
    max_history_entries: int = 60
    source_dir: Path = field(default_factory=lambda: Path("out/signals"))
    momentum_path: Path = field(
        default_factory=lambda: Path("out/momentum/data/momentum_watchlist.json")
    )
    pead_path: Path = field(default_factory=lambda: Path("out/pead/data/pead_candidates.json"))
    market_caps_path: Path = field(default_factory=lambda: Path("data/cache/market_caps.json"))


@dataclass
class TransformResult:
    """Result of a data transformation run."""

    symbols: list[str]
    file_count: int
    total_size_bytes: int


def _copy_json_if_exists(src: Path, dst: Path) -> bool:
    """Copy a JSON file if the source exists. Returns True if copied."""
    if src.exists():
        shutil.copy2(src, dst)
        return True
    return False


def _trim_symbol_data(data: dict, config: TransformConfig) -> dict:
    """Trim a per-symbol JSON to max_bars and cap strategy histories.

    Trimming logic:
    1. chart_data arrays: slice [-max_bars:] for ALL parallel arrays
    2. signals: filter by timestamp — keep only signals within trimmed window
    3. Strategy histories: slice [:max_history_entries] (already newest-first)
    4. Update bar_count
    """
    max_bars = config.max_bars
    max_hist = config.max_history_entries

    # 1. Trim chart_data arrays
    chart_data = data.get("chart_data", {})
    timestamps = chart_data.get("timestamps", [])

    if len(timestamps) > max_bars:
        trim_start = len(timestamps) - max_bars
        # Trim top-level OHLCV arrays
        for key in ("timestamps", "open", "high", "low", "close", "volume"):
            if key in chart_data and isinstance(chart_data[key], list):
                chart_data[key] = chart_data[key][trim_start:]

        # Trim indicator groups (overlays, rsi, macd, dual_macd, oscillators, etc.)
        for group_key, group_val in chart_data.items():
            if isinstance(group_val, dict):
                for ind_key, ind_arr in group_val.items():
                    if isinstance(ind_arr, list) and len(ind_arr) > max_bars:
                        group_val[ind_key] = ind_arr[trim_start:]

        timestamps = chart_data.get("timestamps", [])

    # 2. Filter signals by timestamp window
    if timestamps:
        cutoff_ts = timestamps[0]  # Earliest timestamp in trimmed window
        signals = data.get("signals", [])
        if signals:
            data["signals"] = [s for s in signals if s.get("timestamp", "") >= cutoff_ts]

    # 3. Cap strategy histories (already newest-first from rows.reverse())
    history_keys = [
        "dual_macd_history",
        "trend_pulse_history",
        "regime_flex_history",
        "sector_pulse_history",
    ]
    for key in history_keys:
        hist = data.get(key)
        if isinstance(hist, list) and len(hist) > max_hist:
            data[key] = hist[:max_hist]

    # 4. Update bar_count
    data["bar_count"] = len(chart_data.get("timestamps", []))

    return data


def _merge_screeners(config: TransformConfig) -> dict:
    """Merge momentum + PEAD screener data into a single dict.

    Returns {momentum: <content or null>, pead: <content or null>}
    """
    result: dict = {"momentum": None, "pead": None}

    for key, path in [
        ("momentum", config.momentum_path),
        ("pead", config.pead_path),
    ]:
        if path.exists():
            try:
                result[key] = json.loads(path.read_text(encoding="utf-8"))
            except (json.JSONDecodeError, OSError) as e:
                logger.warning("Failed to read screener %s: %s", key, e)

    return result


class DataTransformer:
    """Transform pipeline output into dashboard-ready data."""

    def __init__(self, config: TransformConfig) -> None:
        self.config = config

    def transform(self, output_dir: Path) -> TransformResult:
        """Transform source data → output_dir/data/.

        Returns TransformResult with stats.
        """
        data_dir = output_dir / "data"
        data_dir.mkdir(parents=True, exist_ok=True)

        source_data = self.config.source_dir / "data"
        symbols: list[str] = []
        file_count = 0
        total_size = 0

        # 1. Copy pass-through files
        for name in (
            "summary.json",
            "score_history.json",
            "indicators.json",
        ):
            src = source_data / name
            if _copy_json_if_exists(src, data_dir / name):
                file_count += 1
                total_size += (data_dir / name).stat().st_size

        # Copy manifest from source root
        manifest_src = self.config.source_dir / "manifest.json"
        if _copy_json_if_exists(manifest_src, data_dir / "manifest.json"):
            file_count += 1
            total_size += (data_dir / "manifest.json").stat().st_size

        # 2. Trim per-symbol JSON files
        if source_data.exists():
            for src_file in sorted(source_data.glob("*_*.json")):
                # Skip non-symbol files
                name = src_file.stem
                if name in ("summary", "score_history", "indicators"):
                    continue

                try:
                    raw = json.loads(src_file.read_text(encoding="utf-8"))
                    trimmed = _trim_symbol_data(raw, self.config)
                    dst = data_dir / src_file.name
                    dst.write_text(
                        json.dumps(trimmed, separators=(",", ":")),
                        encoding="utf-8",
                    )

                    sym = trimmed.get("symbol", name.split("_")[0])
                    if sym not in symbols:
                        symbols.append(sym)
                    file_count += 1
                    total_size += dst.stat().st_size
                except (json.JSONDecodeError, OSError) as e:
                    logger.warning("Failed to transform %s: %s", src_file.name, e)

        # 3. Copy regime pre-rendered HTML files
        regime_src = source_data / "regime"
        if regime_src.is_dir():
            regime_dst = data_dir / "regime"
            regime_dst.mkdir(exist_ok=True)
            for html_file in regime_src.glob("*.html"):
                shutil.copy2(html_file, regime_dst / html_file.name)
                file_count += 1
                total_size += (regime_dst / html_file.name).stat().st_size

        # 4. Merge screeners
        screeners = _merge_screeners(self.config)
        screeners_path = data_dir / "screeners.json"
        screeners_path.write_text(
            json.dumps(screeners, separators=(",", ":")),
            encoding="utf-8",
        )
        file_count += 1
        total_size += screeners_path.stat().st_size

        # 5. Copy strategies.json if exists
        strategies_src = source_data / "strategies.json"
        if _copy_json_if_exists(strategies_src, data_dir / "strategies.json"):
            file_count += 1
            total_size += (data_dir / "strategies.json").stat().st_size

        # 6. Copy market_caps.json cache if exists (for treemap sizing)
        if _copy_json_if_exists(self.config.market_caps_path, data_dir / "market_caps.json"):
            file_count += 1
            total_size += (data_dir / "market_caps.json").stat().st_size

        return TransformResult(
            symbols=symbols,
            file_count=file_count,
            total_size_bytes=total_size,
        )
