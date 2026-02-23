"""Tests for scripts/r2_universe_builder.py — R2 Universe Builder (Job 0)."""

from __future__ import annotations

import importlib.util
import sys
import types
from pathlib import Path
from typing import Any

import yaml

# Add project root
sys.path.insert(0, str(Path(__file__).resolve().parent.parent.parent.parent))

# Import the script module dynamically (it's not a package)
_spec = importlib.util.spec_from_file_location(
    "r2_universe_builder",
    str(
        Path(__file__).resolve().parent.parent.parent.parent / "scripts" / "r2_universe_builder.py"
    ),
)
builder = importlib.util.module_from_spec(_spec)

# Prevent module-level imports from failing in test environment
_fake_fmp = types.ModuleType("src.infrastructure.adapters.fmp.index_constituents")
_fake_fmp.FMPIndexConstituentsAdapter = type("FMPIndexConstituentsAdapter", (), {})  # type: ignore[attr-defined]
sys.modules.setdefault("src.infrastructure.adapters.fmp.index_constituents", _fake_fmp)

_fake_yahoo = types.ModuleType("src.infrastructure.adapters.yahoo.fundamentals_adapter")
_fake_yahoo.YahooFundamentalsAdapter = type("YahooFundamentalsAdapter", (), {})  # type: ignore[attr-defined]
sys.modules.setdefault("src.infrastructure.adapters.yahoo.fundamentals_adapter", _fake_yahoo)

_spec.loader.exec_module(builder)

# Clean up mocks so they don't pollute other test files in the same process
for _key in list(sys.modules):
    if sys.modules.get(_key) in (_fake_fmp, _fake_yahoo):
        del sys.modules[_key]


# ── Helpers ──────────────────────────────────────────────────────────────────


def _make_stock(
    symbol: str,
    price: float = 100.0,
    volume: float = 100_000,
    market_cap: float = 1_000_000_000,
    sector: str = "technology",
) -> dict[str, Any]:
    """Create a minimal raw stock dict matching FMP screener output."""
    return {
        "symbol": symbol,
        "name": f"{symbol} Inc.",
        "price": price,
        "volume": volume,
        "marketCap": market_cap,
        "sector": sector,
    }


def _default_screening(**overrides: Any) -> dict[str, Any]:
    """Return default screening config, with optional overrides."""
    base = {
        "min_market_cap": 500_000_000,
        "min_daily_dollar_volume": 5_000_000,
        "min_turnover_rate": 0.01,
        "sort_by": "dollar_volume",
        "float_cache_days": 7,
        "turnover_miss_threshold": 0.5,
    }
    base.update(overrides)
    return base


# ── TestTurnoverLabel ────────────────────────────────────────────────────────


class TestTurnoverLabel:
    """Tests for _turnover_label() classification."""

    def test_no_data(self) -> None:
        assert builder._turnover_label(0.0) == "no_data"

    def test_inactive(self) -> None:
        assert builder._turnover_label(0.001) == "inactive"
        assert builder._turnover_label(0.005) == "inactive"
        assert builder._turnover_label(0.0099) == "inactive"

    def test_moderate(self) -> None:
        assert builder._turnover_label(0.01) == "moderate"
        assert builder._turnover_label(0.02) == "moderate"
        assert builder._turnover_label(0.0299) == "moderate"

    def test_active(self) -> None:
        assert builder._turnover_label(0.03) == "active"
        assert builder._turnover_label(0.04) == "active"

    def test_highly_active(self) -> None:
        assert builder._turnover_label(0.05) == "highly_active"
        assert builder._turnover_label(0.08) == "highly_active"

    def test_extremely_active(self) -> None:
        assert builder._turnover_label(0.10) == "extremely_active"
        assert builder._turnover_label(0.25) == "extremely_active"


# ── TestComputeAndFilter ─────────────────────────────────────────────────────


class TestComputeAndFilter:
    """Tests for compute_and_filter()."""

    def test_basic_filtering(self) -> None:
        """Stocks below min_market_cap, min_daily_dollar_volume, or
        min_turnover_rate are excluded."""
        raw = [
            # Good: cap=1B, dv=100*100k=10M, turnover=100k/1M=0.1
            _make_stock("GOOD", price=100, volume=100_000, market_cap=1_000_000_000),
            # Bad cap: 100M < 500M threshold
            _make_stock("LOWCAP", price=50, volume=200_000, market_cap=100_000_000),
            # Bad dollar volume: 1*100=100 < 5M threshold
            _make_stock("LOWDV", price=1, volume=100, market_cap=2_000_000_000),
            # Bad turnover: volume=100k, float=100M => 0.001 < 0.01
            _make_stock("LOWTR", price=200, volume=100_000, market_cap=5_000_000_000),
        ]

        float_shares = {
            "GOOD": 1_000_000.0,  # turnover = 100k/1M = 0.1
            "LOWCAP": 500_000.0,
            "LOWDV": 500_000.0,
            "LOWTR": 100_000_000.0,  # turnover = 100k/100M = 0.001
        }

        screening = _default_screening()
        filtered, stats = builder.compute_and_filter(raw, float_shares, screening)

        symbols = [s["symbol"] for s in filtered]
        assert "GOOD" in symbols
        assert "LOWCAP" not in symbols
        assert "LOWDV" not in symbols
        assert "LOWTR" not in symbols
        assert stats["turnover_filter_active"] is True

    def test_auto_disable_turnover(self) -> None:
        """When >50% of symbols lack float data, turnover filter is
        auto-disabled and stocks with zero turnover pass through."""
        # 3 stocks, only 1 has float data => miss ratio = 2/3 = 66% > 50%
        raw = [
            _make_stock("A", price=100, volume=100_000, market_cap=1_000_000_000),
            _make_stock("B", price=200, volume=50_000, market_cap=2_000_000_000),
            _make_stock("C", price=150, volume=80_000, market_cap=3_000_000_000),
        ]
        # Only A has float data
        float_shares = {"A": 500_000.0}

        screening = _default_screening(turnover_miss_threshold=0.5)
        filtered, stats = builder.compute_and_filter(raw, float_shares, screening)

        assert stats["turnover_filter_active"] is False
        assert stats["float_misses"] == 2
        assert stats["float_hits"] == 1
        # B and C pass despite zero turnover because filter is disabled
        symbols = [s["symbol"] for s in filtered]
        assert "B" in symbols
        assert "C" in symbols

    def test_sort_by_dollar_volume(self) -> None:
        """Output should be sorted by dollar_volume descending."""
        raw = [
            _make_stock("LOW", price=10, volume=600_000, market_cap=1_000_000_000),
            _make_stock("HIGH", price=300, volume=500_000, market_cap=5_000_000_000),
            _make_stock("MID", price=100, volume=200_000, market_cap=2_000_000_000),
        ]
        # All have float data so turnover filter stays active
        float_shares = {
            "LOW": 1_000_000.0,  # turnover=0.6
            "HIGH": 1_000_000.0,  # turnover=0.5
            "MID": 1_000_000.0,  # turnover=0.2
        }

        screening = _default_screening()
        filtered, _ = builder.compute_and_filter(raw, float_shares, screening)

        # Expected dollar volumes: HIGH=150M, MID=20M, LOW=6M
        symbols = [s["symbol"] for s in filtered]
        assert symbols == ["HIGH", "MID", "LOW"]


# ── TestMergeCurated ─────────────────────────────────────────────────────────


class TestMergeCurated:
    """Tests for merge_curated() — no cap, all filtered stocks included."""

    def _make_curated(self, symbol: str, tier_group: str = "sector") -> dict[str, Any]:
        return {
            "symbol": symbol,
            "name": f"{symbol} Inc.",
            "tier": "tech",
            "tier_group": tier_group,
            "sectors": ["tech"],
            "timeframes": ["1d", "1w"],
        }

    def _make_screened(self, symbol: str, dollar_volume: float = 50_000_000) -> dict[str, Any]:
        return {
            "symbol": symbol,
            "name": f"{symbol} Corp.",
            "marketCap": 2_000_000_000,
            "dollar_volume": dollar_volume,
            "turnover_rate": 0.05,
            "sector": "Technology",
        }

    def test_curated_always_kept(self) -> None:
        """All curated symbols are always included."""
        curated = [self._make_curated(f"C{i}") for i in range(5)]
        screened = [self._make_screened(f"S{i}") for i in range(10)]

        tickers, stats = builder.merge_curated(screened, curated, screener_metadata={})

        curated_syms = {f"C{i}" for i in range(5)}
        result_syms = {t["symbol"] for t in tickers}
        assert curated_syms.issubset(result_syms)
        assert stats["curated_count"] == 5

    def test_all_screened_included(self) -> None:
        """All screened stocks are included (no cap)."""
        curated = [self._make_curated(f"C{i}") for i in range(3)]
        screened = [self._make_screened(f"S{i}") for i in range(20)]

        tickers, stats = builder.merge_curated(screened, curated, screener_metadata={})

        assert stats["curated_count"] == 3
        assert stats["screened_added"] == 20
        assert stats["final_count"] == 23
        assert len(tickers) == 23

    def test_dedup(self) -> None:
        """A screened symbol that is already in curated should not appear twice."""
        curated = [self._make_curated("AAPL")]
        screened = [
            self._make_screened("AAPL", dollar_volume=999_000_000),
            self._make_screened("MSFT", dollar_volume=800_000_000),
        ]

        tickers, stats = builder.merge_curated(screened, curated, screener_metadata={})

        symbols = [t["symbol"] for t in tickers]
        assert symbols.count("AAPL") == 1
        assert "MSFT" in symbols
        assert stats["screened_added"] == 1  # only MSFT added


# ── TestLoadCuratedFromYaml ──────────────────────────────────────────────────


class TestLoadCuratedFromYaml:
    """Tests for load_curated_from_yaml()."""

    def test_loads_all_sections(self, tmp_path: Path) -> None:
        """Market, sectors, speculative, and holdout symbols all loaded
        with correct tier_group."""
        yaml_content = {
            "market": [
                {"symbol": "SPY", "name": "S&P 500"},
                {"symbol": "QQQ", "name": "Nasdaq 100"},
            ],
            "sectors": {
                "tech": {
                    "etf": "XLK",
                    "stocks": ["AAPL", "MSFT"],
                },
                "energy": {
                    "etf": "XLE",
                    "stocks": ["XOM"],
                },
            },
            "speculative": {
                "stocks": ["PLTR", "RIVN"],
            },
            "validation": {
                "holdout": ["TEST1", "TEST2"],
            },
        }

        yaml_path = tmp_path / "universe.yaml"
        with open(yaml_path, "w") as f:
            yaml.dump(yaml_content, f)

        result = builder.load_curated_from_yaml(yaml_path)
        by_sym = {r["symbol"]: r for r in result}

        # Market
        assert by_sym["SPY"]["tier_group"] == "market"
        assert by_sym["QQQ"]["tier_group"] == "market"

        # Sector ETFs
        assert by_sym["XLK"]["tier_group"] == "sector"
        assert by_sym["XLE"]["tier_group"] == "sector"

        # Sector stocks
        assert by_sym["AAPL"]["tier_group"] == "sector"
        assert by_sym["MSFT"]["tier_group"] == "sector"
        assert by_sym["XOM"]["tier_group"] == "sector"

        # Speculative
        assert by_sym["PLTR"]["tier_group"] == "speculative"
        assert by_sym["RIVN"]["tier_group"] == "speculative"

        # Holdout
        assert by_sym["TEST1"]["tier_group"] == "holdout"
        assert by_sym["TEST2"]["tier_group"] == "holdout"

        # Total count: 2 market + 2 ETF + 3 stocks + 2 speculative + 2 holdout = 11
        assert len(result) == 11

    def test_multi_sector_symbol(self, tmp_path: Path) -> None:
        """A symbol appearing in two sectors should have both in its
        sectors list."""
        yaml_content = {
            "sectors": {
                "tech": {
                    "etf": "XLK",
                    "stocks": ["AMZN"],
                },
                "consumer": {
                    "etf": "XLY",
                    "stocks": ["AMZN"],  # AMZN in both tech and consumer
                },
            },
        }

        yaml_path = tmp_path / "universe.yaml"
        with open(yaml_path, "w") as f:
            yaml.dump(yaml_content, f)

        result = builder.load_curated_from_yaml(yaml_path)
        by_sym = {r["symbol"]: r for r in result}

        # AMZN should appear once with both sectors
        assert result.count(by_sym["AMZN"]) == 1  # no duplicates
        assert "tech" in by_sym["AMZN"]["sectors"]
        assert "consumer" in by_sym["AMZN"]["sectors"]


# ── TestGetScreeningConfig ───────────────────────────────────────────────────


class TestGetScreeningConfig:
    """Tests for get_screening_config()."""

    def test_loads_from_yaml(self, tmp_path: Path) -> None:
        """r2_screening section values override defaults."""
        yaml_content = {
            "r2_screening": {
                "min_market_cap": 1_000_000_000,
                "min_daily_dollar_volume": 10_000_000,
                "min_turnover_rate": 0.005,
            },
        }

        yaml_path = tmp_path / "universe.yaml"
        with open(yaml_path, "w") as f:
            yaml.dump(yaml_content, f)

        config = builder.get_screening_config(yaml_path)

        assert config["min_market_cap"] == 1_000_000_000
        assert config["min_daily_dollar_volume"] == 10_000_000
        assert config["min_turnover_rate"] == 0.005
        # Defaults still present for unspecified keys
        assert config["sort_by"] == "dollar_volume"
        assert config["float_cache_days"] == 7
        assert config["turnover_miss_threshold"] == 0.5

    def test_defaults_when_missing(self, tmp_path: Path) -> None:
        """YAML without r2_screening section uses all defaults."""
        yaml_content = {
            "market": [{"symbol": "SPY"}],
        }

        yaml_path = tmp_path / "universe.yaml"
        with open(yaml_path, "w") as f:
            yaml.dump(yaml_content, f)

        config = builder.get_screening_config(yaml_path)

        assert config["min_market_cap"] == 500_000_000
        assert config["min_daily_dollar_volume"] == 5_000_000
        assert config["min_turnover_rate"] == 0.01
        assert config["sort_by"] == "dollar_volume"
        assert config["float_cache_days"] == 7
        assert config["turnover_miss_threshold"] == 0.5
