"""Tests for R2Client — Cloudflare R2 storage adapter."""

from __future__ import annotations

import json
from datetime import datetime, timezone
from unittest.mock import MagicMock, patch

import pandas as pd
import pyarrow as pa
import pyarrow.parquet as pq
import pytest

from src.infrastructure.adapters.r2.client import R2Client, _load_r2_credentials

# ── Credential loading ──────────────────────────────────────────


class TestCredentialLoading:
    def test_loads_from_env_vars(self, monkeypatch):
        """Credentials loaded from environment variables."""
        monkeypatch.setenv("R2_ACCESS_KEY_ID", "test-key")
        monkeypatch.setenv("R2_SECRET_ACCESS_KEY", "test-secret")
        monkeypatch.setenv("R2_ENDPOINT", "https://test.r2.cloudflarestorage.com")
        monkeypatch.setenv("R2_BUCKET", "test-bucket")

        creds = _load_r2_credentials()
        assert creds["access_key_id"] == "test-key"
        assert creds["secret_access_key"] == "test-secret"
        assert creds["endpoint"] == "https://test.r2.cloudflarestorage.com"
        assert creds["bucket"] == "test-bucket"

    def test_raises_when_missing(self, monkeypatch):
        """ValueError raised when no credentials available."""
        monkeypatch.delenv("R2_ACCESS_KEY_ID", raising=False)
        monkeypatch.delenv("R2_SECRET_ACCESS_KEY", raising=False)
        monkeypatch.delenv("R2_ENDPOINT", raising=False)
        monkeypatch.delenv("R2_BUCKET", raising=False)

        with patch("src.infrastructure.adapters.r2.client._SECRETS_PATH", "/nonexistent"):
            with pytest.raises(ValueError, match="R2 credentials missing"):
                _load_r2_credentials()


# ── Fixtures ────────────────────────────────────────────────────


@pytest.fixture
def mock_s3_client():
    """Create a mock boto3 S3 client."""
    client = MagicMock()
    # Set up NoSuchKey exception class
    client.exceptions.NoSuchKey = type("NoSuchKey", (Exception,), {})
    return client


@pytest.fixture
def r2(mock_s3_client):
    """R2Client with mocked boto3 client."""
    with patch("src.infrastructure.adapters.r2.client.R2Client._create_client") as mock_create:
        mock_create.return_value = mock_s3_client
        client = R2Client(
            access_key_id="test",
            secret_access_key="test",
            endpoint="https://test.r2.dev",
            bucket="test-bucket",
        )
        return client


@pytest.fixture
def sample_df():
    """Small DataFrame for Parquet round-trip tests."""
    return pd.DataFrame(
        {
            "open": [100.0, 101.0, 102.0],
            "high": [105.0, 106.0, 107.0],
            "low": [95.0, 96.0, 97.0],
            "close": [102.0, 103.0, 104.0],
            "volume": [1_000_000, 1_100_000, 1_200_000],
        },
        index=pd.date_range("2024-01-01", periods=3, freq="D"),
    )


# ── Parquet operations ──────────────────────────────────────────


class TestParquetOps:
    def test_put_parquet(self, r2, mock_s3_client, sample_df):
        """put_parquet uploads Parquet bytes to R2."""
        size = r2.put_parquet("parquet/test.parquet", sample_df)
        assert size > 0
        mock_s3_client.put_object.assert_called_once()
        call_kwargs = mock_s3_client.put_object.call_args[1]
        assert call_kwargs["Bucket"] == "test-bucket"
        assert call_kwargs["Key"] == "parquet/test.parquet"

    def test_get_parquet_exists(self, r2, mock_s3_client, sample_df):
        """get_parquet returns DataFrame when key exists."""
        # Prepare parquet bytes
        table = pa.Table.from_pandas(sample_df, preserve_index=True)
        import io

        buf = io.BytesIO()
        pq.write_table(table, buf, compression="snappy")
        parquet_bytes = buf.getvalue()

        body_mock = MagicMock()
        body_mock.read.return_value = parquet_bytes
        mock_s3_client.get_object.return_value = {"Body": body_mock}

        result = r2.get_parquet("parquet/test.parquet")
        assert result is not None
        assert len(result) == 3
        assert list(result.columns) == ["open", "high", "low", "close", "volume"]

    def test_get_parquet_missing(self, r2, mock_s3_client):
        """get_parquet returns None for missing key."""
        mock_s3_client.get_object.side_effect = mock_s3_client.exceptions.NoSuchKey()
        result = r2.get_parquet("parquet/nonexistent.parquet")
        assert result is None


# ── JSON operations ─────────────────────────────────────────────


class TestJsonOps:
    def test_put_json(self, r2, mock_s3_client):
        """put_json uploads JSON bytes to R2."""
        data = {"symbols": ["AAPL", "MSFT"], "count": 2}
        size = r2.put_json("meta/test.json", data)
        assert size > 0
        call_kwargs = mock_s3_client.put_object.call_args[1]
        assert call_kwargs["ContentType"] == "application/json"
        body = json.loads(call_kwargs["Body"])
        assert body["symbols"] == ["AAPL", "MSFT"]

    def test_get_json_exists(self, r2, mock_s3_client):
        """get_json returns parsed dict when key exists."""
        data = {"symbols": ["AAPL"], "count": 1}
        body_mock = MagicMock()
        body_mock.read.return_value = json.dumps(data).encode()
        mock_s3_client.get_object.return_value = {"Body": body_mock}

        result = r2.get_json("meta/test.json")
        assert result == data

    def test_get_json_missing(self, r2, mock_s3_client):
        """get_json returns None for missing key."""
        mock_s3_client.get_object.side_effect = mock_s3_client.exceptions.NoSuchKey()
        result = r2.get_json("meta/nonexistent.json")
        assert result is None


# ── Batch operations ────────────────────────────────────────────


class TestBatchOps:
    def test_put_parquet_batch_all_succeed(self, r2, sample_df):
        """Batch upload returns empty list when all succeed."""
        items = [
            ("parquet/a.parquet", sample_df),
            ("parquet/b.parquet", sample_df),
        ]
        failed = r2.put_parquet_batch(items, workers=2)
        assert failed == []

    def test_put_parquet_batch_with_failure(self, r2, mock_s3_client, sample_df):
        """Batch upload returns failed keys."""
        call_count = 0

        def _side_effect(**kwargs):
            nonlocal call_count
            call_count += 1
            if kwargs.get("Key") == "parquet/bad.parquet":
                raise RuntimeError("upload failed")

        mock_s3_client.put_object.side_effect = _side_effect

        items = [
            ("parquet/good.parquet", sample_df),
            ("parquet/bad.parquet", sample_df),
        ]
        failed = r2.put_parquet_batch(items, workers=2)
        assert "parquet/bad.parquet" in failed
        assert "parquet/good.parquet" not in failed

    def test_put_json_batch_all_succeed(self, r2):
        """JSON batch upload returns empty list when all succeed."""
        items = [
            ("meta/a.json", {"a": 1}),
            ("meta/b.json", {"b": 2}),
        ]
        failed = r2.put_json_batch(items, workers=2)
        assert failed == []


# ── Metadata operations ─────────────────────────────────────────


class TestMetadataOps:
    def test_list_keys(self, r2, mock_s3_client):
        """list_keys returns all keys under prefix."""
        paginator = MagicMock()
        paginator.paginate.return_value = [
            {
                "Contents": [
                    {"Key": "parquet/1d/AAPL.parquet"},
                    {"Key": "parquet/1d/MSFT.parquet"},
                ]
            }
        ]
        mock_s3_client.get_paginator.return_value = paginator

        keys = r2.list_keys("parquet/1d/")
        assert keys == ["parquet/1d/AAPL.parquet", "parquet/1d/MSFT.parquet"]

    def test_key_exists_true(self, r2, mock_s3_client):
        """key_exists returns True for existing key."""
        mock_s3_client.head_object.return_value = {}
        assert r2.key_exists("meta/test.json") is True

    def test_key_exists_false(self, r2, mock_s3_client):
        """key_exists returns False for missing key."""
        mock_s3_client.head_object.side_effect = Exception("404")
        assert r2.key_exists("meta/nonexistent.json") is False

    def test_get_last_modified(self, r2, mock_s3_client):
        """get_last_modified returns datetime for existing key."""
        dt = datetime(2024, 6, 15, 10, 30, 0, tzinfo=timezone.utc)
        mock_s3_client.head_object.return_value = {"LastModified": dt}
        result = r2.get_last_modified("meta/test.json")
        assert result == dt

    def test_get_last_modified_missing(self, r2, mock_s3_client):
        """get_last_modified returns None for missing key."""
        mock_s3_client.head_object.side_effect = Exception("404")
        assert r2.get_last_modified("meta/nonexistent.json") is None

    def test_delete_key(self, r2, mock_s3_client):
        """delete_key returns True on success."""
        assert r2.delete_key("meta/old.json") is True
        mock_s3_client.delete_object.assert_called_once()

    def test_delete_key_failure(self, r2, mock_s3_client):
        """delete_key returns False on error."""
        mock_s3_client.delete_object.side_effect = Exception("fail")
        assert r2.delete_key("meta/old.json") is False
