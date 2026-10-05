"""Unit tests for signal schema version validation."""

import pytest

from src.domain.signals.schemas import (
    SCHEMA_VERSION_STR,
    SchemaVersionError,
    validate_schema_version,
)


class TestSchemaVersion:
    """Tests for schema version validation."""

    def test_validate_schema_version_valid(self) -> None:
        """Test validating a correct schema version."""
        major, minor = validate_schema_version(SCHEMA_VERSION_STR)
        assert major == 1
        assert minor == 0

    def test_validate_schema_version_invalid_prefix(self) -> None:
        """Test validation fails for wrong prefix."""
        with pytest.raises(SchemaVersionError) as exc_info:
            validate_schema_version("wrong_prefix@1.0")
        assert "mismatch" in str(exc_info.value)

    def test_validate_schema_version_incompatible_major(self) -> None:
        """Test validation fails for incompatible major version."""
        with pytest.raises(SchemaVersionError):
            validate_schema_version("signal_v2@99.0")
