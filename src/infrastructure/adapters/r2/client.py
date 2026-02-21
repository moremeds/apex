"""R2Client — Cloudflare R2 storage adapter using boto3 S3-compatible API.

Provides Parquet and JSON round-trip operations for the APEX data pipeline.
Credentials loaded from env vars (CI) or config/secrets.yaml (local dev).
"""

from __future__ import annotations

import io
import json
import logging
import os
from concurrent.futures import ThreadPoolExecutor, as_completed
from datetime import datetime, timezone
from pathlib import Path
from typing import Any

import pandas as pd
import pyarrow as pa
import pyarrow.parquet as pq
import yaml
from tenacity import retry, stop_after_attempt, wait_exponential

logger = logging.getLogger(__name__)

_PROJECT_ROOT = Path(__file__).resolve().parent.parent.parent.parent.parent
_SECRETS_PATH = _PROJECT_ROOT / "config" / "secrets.yaml"


def _load_r2_credentials() -> dict[str, str]:
    """Load R2 credentials from env vars or config/secrets.yaml.

    Env vars take priority: R2_ACCESS_KEY_ID, R2_SECRET_ACCESS_KEY,
    R2_ENDPOINT, R2_BUCKET.

    Raises ValueError if credentials are incomplete.
    """
    creds: dict[str, str] = {}

    # Try env vars first
    env_map = {
        "access_key_id": "R2_ACCESS_KEY_ID",
        "secret_access_key": "R2_SECRET_ACCESS_KEY",
        "endpoint": "R2_ENDPOINT",
        "bucket": "R2_BUCKET",
    }
    for key, env_var in env_map.items():
        val = os.environ.get(env_var, "")
        if val:
            creds[key] = val

    # Fallback to secrets.yaml
    if not creds.get("access_key_id"):
        try:
            with open(_SECRETS_PATH) as f:
                data = yaml.safe_load(f) or {}
            r2_cfg = data.get("r2", {})
            for key in ("access_key_id", "secret_access_key", "endpoint", "bucket"):
                val = r2_cfg.get(key, "")
                if val and key not in creds:
                    creds[key] = val
        except FileNotFoundError:
            pass

    required = ("access_key_id", "secret_access_key", "endpoint", "bucket")
    missing = [k for k in required if not creds.get(k)]
    if missing:
        raise ValueError(
            f"R2 credentials missing: {missing}. "
            "Set R2_ACCESS_KEY_ID, R2_SECRET_ACCESS_KEY, R2_ENDPOINT, R2_BUCKET "
            "env vars or add to config/secrets.yaml under r2:"
        )

    return creds


class R2Client:
    """Cloudflare R2 storage client (S3-compatible via boto3).

    Usage:
        r2 = R2Client()
        r2.put_parquet("parquet/historical/1d/AAPL.parquet", df)
        df = r2.get_parquet("parquet/historical/1d/AAPL.parquet")
    """

    def __init__(
        self,
        access_key_id: str | None = None,
        secret_access_key: str | None = None,
        endpoint: str | None = None,
        bucket: str | None = None,
    ) -> None:
        if access_key_id and secret_access_key and endpoint and bucket:
            self._access_key_id = access_key_id
            self._secret_access_key = secret_access_key
            self._endpoint = endpoint
            self._bucket = bucket
        else:
            creds = _load_r2_credentials()
            self._access_key_id = access_key_id or creds["access_key_id"]
            self._secret_access_key = secret_access_key or creds["secret_access_key"]
            self._endpoint = endpoint or creds["endpoint"]
            self._bucket = bucket or creds["bucket"]

        self._client = self._create_client()

    def _create_client(self) -> Any:
        """Create boto3 S3 client configured for R2."""
        import boto3

        return boto3.client(
            "s3",
            endpoint_url=self._endpoint,
            aws_access_key_id=self._access_key_id,
            aws_secret_access_key=self._secret_access_key,
            region_name="auto",
        )

    # ── Parquet operations ──────────────────────────────────────

    @retry(stop=stop_after_attempt(3), wait=wait_exponential(multiplier=1, max=10))
    def put_parquet(self, key: str, df: pd.DataFrame) -> int:
        """Upload DataFrame as Parquet to R2.

        Returns the size in bytes of the uploaded object.
        """
        table = pa.Table.from_pandas(df, preserve_index=True)
        buf = io.BytesIO()
        pq.write_table(table, buf, compression="snappy")
        body = buf.getvalue()

        self._client.put_object(Bucket=self._bucket, Key=key, Body=body)
        logger.debug("put_parquet %s (%d bytes)", key, len(body))
        return len(body)

    @retry(stop=stop_after_attempt(3), wait=wait_exponential(multiplier=1, max=10))
    def get_parquet(self, key: str) -> pd.DataFrame | None:
        """Download Parquet from R2 and return as DataFrame.

        Returns None if key does not exist.
        """
        try:
            resp = self._client.get_object(Bucket=self._bucket, Key=key)
            body = resp["Body"].read()
            table = pq.read_table(io.BytesIO(body))
            return table.to_pandas()
        except self._client.exceptions.NoSuchKey:
            return None
        except Exception as e:
            if "NoSuchKey" in str(e) or "404" in str(e):
                return None
            raise

    # ── JSON operations ─────────────────────────────────────────

    @retry(stop=stop_after_attempt(3), wait=wait_exponential(multiplier=1, max=10))
    def put_json(self, key: str, data: Any) -> int:
        """Upload dict/list as JSON to R2.

        Returns the size in bytes of the uploaded object.
        """
        body = json.dumps(data, separators=(",", ":"), default=str).encode("utf-8")
        self._client.put_object(
            Bucket=self._bucket,
            Key=key,
            Body=body,
            ContentType="application/json",
        )
        logger.debug("put_json %s (%d bytes)", key, len(body))
        return len(body)

    @retry(stop=stop_after_attempt(3), wait=wait_exponential(multiplier=1, max=10))
    def get_json(self, key: str) -> Any | None:
        """Download JSON from R2 and parse.

        Returns None if key does not exist.
        """
        try:
            resp = self._client.get_object(Bucket=self._bucket, Key=key)
            body = resp["Body"].read()
            return json.loads(body)
        except self._client.exceptions.NoSuchKey:
            return None
        except Exception as e:
            if "NoSuchKey" in str(e) or "404" in str(e):
                return None
            raise

    # ── File operations ─────────────────────────────────────────

    @retry(stop=stop_after_attempt(3), wait=wait_exponential(multiplier=1, max=10))
    def put_file(self, key: str, path: str | Path) -> int:
        """Upload a local file to R2.

        Returns the size in bytes of the uploaded file.
        """
        path = Path(path)
        with open(path, "rb") as f:
            body = f.read()
        self._client.put_object(Bucket=self._bucket, Key=key, Body=body)
        logger.debug("put_file %s (%d bytes)", key, len(body))
        return len(body)

    # ── Batch operations ────────────────────────────────────────

    def put_parquet_batch(
        self,
        items: list[tuple[str, pd.DataFrame]],
        workers: int = 20,
    ) -> list[str]:
        """Upload multiple DataFrames as Parquet in parallel.

        Args:
            items: List of (key, DataFrame) tuples.
            workers: Max concurrent uploads.

        Returns:
            List of keys that failed to upload.
        """
        failed: list[str] = []

        def _upload(key: str, df: pd.DataFrame) -> str | None:
            try:
                self.put_parquet(key, df)
                return None
            except Exception as e:
                logger.error("Failed to upload %s: %s", key, e)
                return key

        with ThreadPoolExecutor(max_workers=workers) as pool:
            futures = {pool.submit(_upload, k, df): k for k, df in items}
            for future in as_completed(futures):
                result = future.result()
                if result is not None:
                    failed.append(result)

        if failed:
            logger.warning("Batch upload: %d/%d failed", len(failed), len(items))
        return failed

    def put_json_batch(
        self,
        items: list[tuple[str, Any]],
        workers: int = 20,
    ) -> list[str]:
        """Upload multiple JSON objects in parallel.

        Args:
            items: List of (key, data) tuples.
            workers: Max concurrent uploads.

        Returns:
            List of keys that failed to upload.
        """
        failed: list[str] = []

        def _upload(key: str, data: Any) -> str | None:
            try:
                self.put_json(key, data)
                return None
            except Exception as e:
                logger.error("Failed to upload %s: %s", key, e)
                return key

        with ThreadPoolExecutor(max_workers=workers) as pool:
            futures = {pool.submit(_upload, k, d): k for k, d in items}
            for future in as_completed(futures):
                result = future.result()
                if result is not None:
                    failed.append(result)

        if failed:
            logger.warning("Batch upload: %d/%d failed", len(failed), len(items))
        return failed

    # ── Metadata / utility operations ───────────────────────────

    def list_keys(self, prefix: str) -> list[str]:
        """List R2 object keys under prefix."""
        keys: list[str] = []
        paginator = self._client.get_paginator("list_objects_v2")
        for page in paginator.paginate(Bucket=self._bucket, Prefix=prefix):
            for obj in page.get("Contents", []):
                keys.append(obj["Key"])
        return keys

    def key_exists(self, key: str) -> bool:
        """Check if a key exists in R2 via HEAD request."""
        try:
            self._client.head_object(Bucket=self._bucket, Key=key)
            return True
        except Exception:
            return False

    def get_last_modified(self, key: str) -> datetime | None:
        """Get last modified timestamp for a key."""
        try:
            resp = self._client.head_object(Bucket=self._bucket, Key=key)
            lm = resp.get("LastModified")
            if lm and isinstance(lm, datetime):
                result: datetime = lm.astimezone(timezone.utc)
                return result
            return None
        except Exception:
            return None

    def delete_key(self, key: str) -> bool:
        """Delete an object from R2. Returns True on success."""
        try:
            self._client.delete_object(Bucket=self._bucket, Key=key)
            return True
        except Exception as e:
            logger.error("Failed to delete %s: %s", key, e)
            return False
