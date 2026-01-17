"""Tests for deterministic hashing utilities."""

from __future__ import annotations

from unittest.mock import patch

from src.backtest import (
    canonical_json,
    generate_experiment_id,
    generate_run_id,
    generate_trial_id,
    get_git_sha,
    quantize_float,
)


class TestQuantizeFloat:
    """Tests for float quantization."""

    def test_basic_quantization(self) -> None:
        """Test basic float quantization - returns float, not string."""
        assert quantize_float(1.23456789, precision=4) == 1.2346
        assert quantize_float(0.001234, precision=4) == 0.0012

    def test_negative_values(self) -> None:
        """Test negative value quantization."""
        assert quantize_float(-1.23456, precision=4) == -1.2346
        assert quantize_float(-0.0001, precision=4) == -0.0001

    def test_zero(self) -> None:
        """Test zero quantization."""
        result = quantize_float(0.0, precision=4)
        assert result == 0.0
        assert isinstance(result, float)

    def test_nan(self) -> None:
        """Test NaN handling - uses sentinel string."""
        assert quantize_float(float("nan"), precision=4) == "__NaN__"

    def test_infinity(self) -> None:
        """Test infinity handling - uses sentinel strings."""
        assert quantize_float(float("inf"), precision=4) == "__Inf__"
        assert quantize_float(float("-inf"), precision=4) == "__-Inf__"

    def test_precision_levels(self) -> None:
        """Test different precision levels."""
        value = 1.23456789012345
        assert quantize_float(value, precision=2) == 1.23
        assert quantize_float(value, precision=4) == 1.2346
        assert quantize_float(value, precision=8) == 1.23456789

    def test_determinism(self) -> None:
        """Test that quantization is deterministic."""
        value = 3.14159265358979
        results = [quantize_float(value, precision=6) for _ in range(100)]
        assert len(set(results)) == 1  # All results should be identical

    def test_returns_float_type(self) -> None:
        """Test that normal values return float type for JSON consistency."""
        result = quantize_float(3.14159, precision=4)
        assert isinstance(result, float)
        assert result == 3.1416


class TestCanonicalJson:
    """Tests for canonical JSON serialization."""

    def test_sorted_keys(self) -> None:
        """Test that keys are sorted."""
        obj = {"z": 1, "a": 2, "m": 3}
        result = canonical_json(obj)
        # Implementation uses compact format (no spaces after separators)
        assert '"a":2' in result or '"a": 2' in result
        # Verify order: a comes before m comes before z
        assert result.index('"a"') < result.index('"m"') < result.index('"z"')

    def test_nested_sorting(self) -> None:
        """Test that nested keys are sorted."""
        obj = {"b": {"z": 1, "a": 2}, "a": 1}
        result = canonical_json(obj)
        # Key 'a' at top level should come before 'b'
        assert result.index('"a":') < result.index('"b":')

    def test_float_quantization(self) -> None:
        """Test that floats are quantized in JSON."""
        obj = {"value": 1.23456789012345}
        result = canonical_json(obj, precision=4)  # Use correct param name
        assert "1.2346" in result

    def test_list_handling(self) -> None:
        """Test list serialization."""
        obj = {"items": [3, 1, 2]}
        result = canonical_json(obj)
        # List order is preserved; check for [3,1,2] format (compact)
        assert "[3,1,2]" in result or "[3, 1, 2]" in result

    def test_determinism(self) -> None:
        """Test that serialization is deterministic."""
        obj = {
            "params": {"fast": 10.123456, "slow": 50.654321},
            "symbols": ["AAPL", "MSFT"],
            "train_days": 252,
        }
        results = [canonical_json(obj) for _ in range(100)]
        assert len(set(results)) == 1


class TestIdGeneration:
    """Tests for ID generation functions."""

    def test_experiment_id_format(self) -> None:
        """Test experiment ID format."""
        exp_id = generate_experiment_id(
            name="test",
            strategy="ma_cross",
            parameters={"fast": 10},
            universe={"symbols": ["AAPL"]},
            temporal={"folds": 5},
            data_version="v1",
        )
        assert exp_id.startswith("exp_")
        assert len(exp_id) == 16  # exp_ + 12 hex chars

    def test_trial_id_format(self) -> None:
        """Test trial ID format."""
        trial_id = generate_trial_id(
            experiment_id="exp_abc123",
            params={"fast": 10, "slow": 50},
        )
        assert trial_id.startswith("trial_")
        assert len(trial_id) == 18  # trial_ + 12 hex chars

    def test_run_id_format(self) -> None:
        """Test run ID format."""
        run_id = generate_run_id(
            trial_id="trial_abc123",
            symbol="AAPL",
            window_id="window_1",
            profile_version="v1",
            data_version="v1",
        )
        assert run_id.startswith("run_")
        assert len(run_id) == 16  # run_ + 12 hex chars

    def test_deterministic_experiment_id(self) -> None:
        """Test that same inputs produce same experiment ID."""
        kwargs = {
            "name": "test",
            "strategy": "ma_cross",
            "parameters": {"fast": 10.5, "slow": 50.5},
            "universe": {"symbols": ["AAPL", "MSFT"]},
            "temporal": {"folds": 5, "train_days": 252},
            "data_version": "v1",
        }
        ids = [generate_experiment_id(**kwargs) for _ in range(10)]
        assert len(set(ids)) == 1

    def test_different_inputs_different_ids(self) -> None:
        """Test that different inputs produce different IDs."""
        base = {
            "name": "test",
            "strategy": "ma_cross",
            "parameters": {"fast": 10},
            "universe": {"symbols": ["AAPL"]},
            "temporal": {"folds": 5},
            "data_version": "v1",
        }

        id1 = generate_experiment_id(**base)

        modified = base.copy()
        modified["parameters"] = {"fast": 20}
        id2 = generate_experiment_id(**modified)

        assert id1 != id2

    def test_float_precision_affects_id(self) -> None:
        """Test that very small float differences don't affect ID."""
        # Due to quantization at 8 decimal places, tiny differences beyond
        # that precision should not affect ID
        id1 = generate_trial_id(
            experiment_id="exp_123",
            params={"value": 10.000000001},  # 9 decimal places
        )
        id2 = generate_trial_id(
            experiment_id="exp_123",
            params={"value": 10.000000002},  # 9 decimal places
        )
        # These should be the same after 8-digit quantization
        assert id1 == id2

        # But differences at or above 8 decimal precision should differ
        id3 = generate_trial_id(
            experiment_id="exp_123",
            params={"value": 10.00000010},  # Differs at 8th decimal
        )
        assert id1 != id3


class TestGetGitSha:
    """Tests for git SHA retrieval."""

    def test_returns_string_or_none(self) -> None:
        """Test that get_git_sha returns a string or None."""
        # Clear cache to test fresh
        get_git_sha.cache_clear()
        result = get_git_sha()
        assert result is None or isinstance(result, str)

    def test_short_sha_format(self) -> None:
        """Test that short SHA has expected format (8 hex chars)."""
        get_git_sha.cache_clear()
        result = get_git_sha(short=True)
        if result is not None:
            # Should be 8 hex characters
            assert len(result) == 8
            assert all(c in "0123456789abcdef" for c in result)

    def test_caching(self) -> None:
        """Test that result is cached."""
        get_git_sha.cache_clear()
        result1 = get_git_sha()
        result2 = get_git_sha()
        # Cache info should show hit
        assert get_git_sha.cache_info().hits > 0
        assert result1 == result2

    def test_handles_git_unavailable(self) -> None:
        """Test graceful handling when git is unavailable."""
        get_git_sha.cache_clear()
        with patch("src.backtest.core.hashing.subprocess.run") as mock_run:
            mock_run.side_effect = FileNotFoundError("git not found")
            result = get_git_sha()
            assert result is None


class TestCodeVersionInExperimentId:
    """Tests for code version in experiment IDs."""

    def test_explicit_code_version_affects_id(self) -> None:
        """Test that different code versions produce different IDs."""
        base = {
            "name": "test",
            "strategy": "ma_cross",
            "parameters": {"fast": 10},
            "universe": {"symbols": ["AAPL"]},
            "temporal": {"folds": 5},
            "data_version": "v1",
        }

        id_v1 = generate_experiment_id(**base, code_version="abc123")
        id_v2 = generate_experiment_id(**base, code_version="def456")

        assert id_v1 != id_v2

    def test_empty_code_version_excludes_from_hash(self) -> None:
        """Test that empty string code_version excludes it from hash."""
        base = {
            "name": "test",
            "strategy": "ma_cross",
            "parameters": {"fast": 10},
            "universe": {"symbols": ["AAPL"]},
            "temporal": {"folds": 5},
            "data_version": "v1",
        }

        # Empty string should not include code_version in hash
        id_empty = generate_experiment_id(**base, code_version="")

        # Same as not providing it and having no git repo
        get_git_sha.cache_clear()
        with patch("src.backtest.core.hashing.subprocess.run") as mock_run:
            mock_run.side_effect = FileNotFoundError("git not found")
            id_no_git = generate_experiment_id(**base, code_version=None)

        # Both should produce same ID (no code_version in hash)
        assert id_empty == id_no_git

    def test_auto_detect_code_version_with_git(self) -> None:
        """Test that code_version is auto-detected from git when None."""
        get_git_sha.cache_clear()
        base = {
            "name": "test",
            "strategy": "ma_cross",
            "parameters": {"fast": 10},
            "universe": {"symbols": ["AAPL"]},
            "temporal": {"folds": 5},
            "data_version": "v1",
        }

        # Mock git to return a known SHA
        with patch("src.backtest.core.hashing.subprocess.run") as mock_run:
            mock_run.return_value.returncode = 0
            mock_run.return_value.stdout = "abc12345\n"
            get_git_sha.cache_clear()
            id_auto = generate_experiment_id(**base, code_version=None)

        # Explicit version should match auto-detected
        get_git_sha.cache_clear()
        id_explicit = generate_experiment_id(**base, code_version="abc12345")

        assert id_auto == id_explicit

    def test_same_code_version_deterministic(self) -> None:
        """Test that same code version produces deterministic IDs."""
        base = {
            "name": "test",
            "strategy": "ma_cross",
            "parameters": {"fast": 10},
            "universe": {"symbols": ["AAPL"]},
            "temporal": {"folds": 5},
            "data_version": "v1",
            "code_version": "v1.2.3",
        }

        ids = [generate_experiment_id(**base) for _ in range(10)]
        assert len(set(ids)) == 1
