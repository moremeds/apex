"""Tests for generate_validation_universe script."""

import tempfile
from pathlib import Path

import pytest
import yaml

# Import from scripts directory
import sys
sys.path.insert(0, str(Path(__file__).parent.parent.parent.parent / "scripts"))
from generate_validation_universe import (
    UniverseGenerationConfig,
    generate_universe,
    LARGE_CAP_SYMBOLS,
    MID_CAP_SYMBOLS,
    SMALL_CAP_SYMBOLS,
    GICS_SECTORS,
    main,
)


class TestUniverseGenerationConfig:
    """Tests for UniverseGenerationConfig."""

    def test_default_config(self):
        """Test default configuration values."""
        config = UniverseGenerationConfig()

        assert config.seed == 42
        assert config.version == "v1.0"
        assert config.total_symbols == 200
        assert config.holdout_pct == 0.30
        assert config.large_cap_pct == 0.40
        assert config.mid_cap_pct == 0.40
        assert config.small_cap_pct == 0.20
        assert config.min_per_sector == 5

    def test_custom_config(self):
        """Test custom configuration."""
        config = UniverseGenerationConfig(
            seed=123,
            total_symbols=100,
            holdout_pct=0.20,
        )

        assert config.seed == 123
        assert config.total_symbols == 100
        assert config.holdout_pct == 0.20


class TestSymbolPools:
    """Tests for symbol pool constants."""

    def test_large_cap_not_empty(self):
        """Test large cap pool has symbols."""
        assert len(LARGE_CAP_SYMBOLS) > 50

    def test_mid_cap_not_empty(self):
        """Test mid cap pool has symbols."""
        assert len(MID_CAP_SYMBOLS) > 50

    def test_small_cap_not_empty(self):
        """Test small cap pool has symbols."""
        assert len(SMALL_CAP_SYMBOLS) > 50

    def test_gics_sectors_coverage(self):
        """Test all 11 GICS sectors are covered."""
        expected_sectors = {
            "Technology", "Healthcare", "Financials",
            "Consumer Discretionary", "Communication Services",
            "Consumer Staples", "Industrials", "Energy",
            "Materials", "Utilities", "Real Estate",
        }
        assert set(GICS_SECTORS.keys()) == expected_sectors

    def test_each_sector_has_symbols(self):
        """Test each sector has at least some symbols."""
        for sector, symbols in GICS_SECTORS.items():
            assert len(symbols) >= 5, f"{sector} has fewer than 5 symbols"


class TestGenerateUniverse:
    """Tests for generate_universe function."""

    def test_generates_correct_structure(self):
        """Test output has required keys."""
        config = UniverseGenerationConfig(total_symbols=50)
        result = generate_universe(config)

        assert "training_universe" in result
        assert "holdout_universe" in result
        assert "generation_config" in result

    def test_generates_correct_counts(self):
        """Test approximate symbol counts."""
        config = UniverseGenerationConfig(total_symbols=100, holdout_pct=0.30)
        result = generate_universe(config)

        # Allow some flexibility due to deduplication
        total = len(result["training_universe"]) + len(result["holdout_universe"])
        assert 80 <= total <= 100

        # Holdout should be ~30%
        holdout_ratio = len(result["holdout_universe"]) / total
        assert 0.25 <= holdout_ratio <= 0.35

    def test_deterministic_with_same_seed(self):
        """Test same seed produces same universe."""
        config = UniverseGenerationConfig(seed=42, total_symbols=50)

        result1 = generate_universe(config)
        result2 = generate_universe(config)

        assert result1["training_universe"] == result2["training_universe"]
        assert result1["holdout_universe"] == result2["holdout_universe"]

    def test_different_seed_produces_different_universe(self):
        """Test different seed produces different universe."""
        config1 = UniverseGenerationConfig(seed=42, total_symbols=50)
        config2 = UniverseGenerationConfig(seed=123, total_symbols=50)

        result1 = generate_universe(config1)
        result2 = generate_universe(config2)

        # At least one should be different
        assert (result1["training_universe"] != result2["training_universe"] or
                result1["holdout_universe"] != result2["holdout_universe"])

    def test_no_overlap_between_train_and_holdout(self):
        """Test training and holdout have no overlap."""
        config = UniverseGenerationConfig(total_symbols=100)
        result = generate_universe(config)

        train_set = set(result["training_universe"])
        holdout_set = set(result["holdout_universe"])

        assert train_set.isdisjoint(holdout_set)

    def test_symbols_are_sorted(self):
        """Test output symbols are alphabetically sorted."""
        config = UniverseGenerationConfig(total_symbols=50)
        result = generate_universe(config)

        assert result["training_universe"] == sorted(result["training_universe"])
        assert result["holdout_universe"] == sorted(result["holdout_universe"])

    def test_generation_config_metadata(self):
        """Test generation config has required metadata."""
        config = UniverseGenerationConfig(seed=42, total_symbols=100)
        result = generate_universe(config)

        gen_config = result["generation_config"]
        assert gen_config["seed"] == 42
        assert gen_config["version"] == "v1.0"
        assert "generated_at" in gen_config
        assert "total_symbols" in gen_config
        assert "training_count" in gen_config
        assert "holdout_count" in gen_config


class TestMain:
    """Tests for main entry point."""

    def test_main_creates_output_file(self):
        """Test main function creates YAML output."""
        with tempfile.TemporaryDirectory() as tmpdir:
            exit_code = main([
                "--seed", "42",
                "--total", "50",
                "--output", tmpdir,
            ])

            assert exit_code == 0
            output_path = Path(tmpdir) / "regime_universe.yaml"
            assert output_path.exists()

    def test_main_output_is_valid_yaml(self):
        """Test output is valid YAML."""
        with tempfile.TemporaryDirectory() as tmpdir:
            main(["--seed", "42", "--total", "50", "--output", tmpdir])

            output_path = Path(tmpdir) / "regime_universe.yaml"
            with open(output_path) as f:
                data = yaml.safe_load(f)

            assert "training_universe" in data
            assert "holdout_universe" in data

    def test_main_respects_seed(self):
        """Test main respects seed for reproducibility."""
        with tempfile.TemporaryDirectory() as tmpdir1, \
             tempfile.TemporaryDirectory() as tmpdir2:

            main(["--seed", "42", "--total", "50", "--output", tmpdir1])
            main(["--seed", "42", "--total", "50", "--output", tmpdir2])

            with open(Path(tmpdir1) / "regime_universe.yaml") as f1, \
                 open(Path(tmpdir2) / "regime_universe.yaml") as f2:
                data1 = yaml.safe_load(f1)
                data2 = yaml.safe_load(f2)

            assert data1["training_universe"] == data2["training_universe"]
            assert data1["holdout_universe"] == data2["holdout_universe"]
