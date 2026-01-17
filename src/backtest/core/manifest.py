"""
Run Manifest - Immutable artifact manifest with SHA-256 verification.

Phase 6: Execution Evidence

The RunManifest captures all metadata needed to verify and reproduce a backtest run:
- Git commit and dirty state
- Data and parameter fingerprints
- Artifact checksums
- Timing information
- Runner environment details
"""

from __future__ import annotations

import hashlib
import json
import os
import subprocess
from dataclasses import dataclass, field
from datetime import datetime
from pathlib import Path
from typing import Any, Dict, Optional

from src.utils.logging_setup import get_logger

logger = get_logger(__name__)

MANIFEST_VERSION = "manifest@1.0"


@dataclass
class RunManifest:
    """
    Immutable manifest for a backtest run with SHA-256 verification.

    Contains all metadata needed to:
    1. Verify artifact integrity (checksums)
    2. Reproduce the run (git commit, params)
    3. Audit the execution (timing, runner info)
    """

    manifest_version: str = MANIFEST_VERSION
    run_id: str = ""

    # Git state
    git_commit: str = ""
    git_dirty: bool = False
    git_branch: str = ""

    # Fingerprints (content-addressed)
    data_fingerprint: str = ""  # SHA-256 of input data
    params_fingerprint: str = ""  # SHA-256 of parameters

    # Artifact checksums: {"filename": "sha256:abc..."}
    artifact_checksums: Dict[str, str] = field(default_factory=dict)

    # Timing
    started_at: Optional[datetime] = None
    finished_at: Optional[datetime] = None
    duration_sec: float = 0.0

    # Runner info
    runner_info: Dict[str, str] = field(default_factory=dict)

    # Metrics summary (for quick verification)
    metrics_summary: Dict[str, float] = field(default_factory=dict)

    def to_dict(self) -> Dict[str, Any]:
        """Convert to dictionary for JSON serialization."""
        return {
            "manifest_version": self.manifest_version,
            "run_id": self.run_id,
            "git_commit": self.git_commit,
            "git_dirty": self.git_dirty,
            "git_branch": self.git_branch,
            "data_fingerprint": self.data_fingerprint,
            "params_fingerprint": self.params_fingerprint,
            "artifact_checksums": self.artifact_checksums,
            "started_at": self.started_at.isoformat() if self.started_at else None,
            "finished_at": self.finished_at.isoformat() if self.finished_at else None,
            "duration_sec": self.duration_sec,
            "runner_info": self.runner_info,
            "metrics_summary": self.metrics_summary,
        }

    @classmethod
    def from_dict(cls, data: Dict[str, Any]) -> "RunManifest":
        """Create RunManifest from dictionary."""
        started_at = None
        if data.get("started_at"):
            started_at = datetime.fromisoformat(data["started_at"])

        finished_at = None
        if data.get("finished_at"):
            finished_at = datetime.fromisoformat(data["finished_at"])

        return cls(
            manifest_version=data.get("manifest_version", MANIFEST_VERSION),
            run_id=data.get("run_id", ""),
            git_commit=data.get("git_commit", ""),
            git_dirty=data.get("git_dirty", False),
            git_branch=data.get("git_branch", ""),
            data_fingerprint=data.get("data_fingerprint", ""),
            params_fingerprint=data.get("params_fingerprint", ""),
            artifact_checksums=data.get("artifact_checksums", {}),
            started_at=started_at,
            finished_at=finished_at,
            duration_sec=data.get("duration_sec", 0.0),
            runner_info=data.get("runner_info", {}),
            metrics_summary=data.get("metrics_summary", {}),
        )

    def save(self, path: Path) -> None:
        """Save manifest to JSON file."""
        path = Path(path)
        path.parent.mkdir(parents=True, exist_ok=True)
        with open(path, "w") as f:
            json.dump(self.to_dict(), f, indent=2)
        logger.info(f"Manifest saved: {path}")

    @classmethod
    def load(cls, path: Path) -> "RunManifest":
        """Load manifest from JSON file."""
        with open(path) as f:
            data = json.load(f)
        return cls.from_dict(data)


def compute_sha256(path: Path) -> str:
    """Compute SHA-256 hash of a file."""
    sha256 = hashlib.sha256()
    with open(path, "rb") as f:
        for chunk in iter(lambda: f.read(8192), b""):
            sha256.update(chunk)
    return f"sha256:{sha256.hexdigest()}"


def compute_data_fingerprint(data: Any) -> str:
    """Compute SHA-256 fingerprint of data (JSON-serializable)."""
    json_str = json.dumps(data, sort_keys=True, default=str)
    return f"sha256:{hashlib.sha256(json_str.encode()).hexdigest()}"


def get_git_info() -> Dict[str, Any]:
    """Get current git commit and dirty state."""
    result = {
        "commit": "",
        "dirty": False,
        "branch": "",
    }

    try:
        # Get commit hash
        commit = subprocess.run(
            ["git", "rev-parse", "HEAD"],
            capture_output=True,
            text=True,
            timeout=5,
        )
        if commit.returncode == 0:
            result["commit"] = commit.stdout.strip()

        # Check for dirty state
        status = subprocess.run(
            ["git", "status", "--porcelain"],
            capture_output=True,
            text=True,
            timeout=5,
        )
        if status.returncode == 0:
            result["dirty"] = len(status.stdout.strip()) > 0

        # Get branch name
        branch = subprocess.run(
            ["git", "rev-parse", "--abbrev-ref", "HEAD"],
            capture_output=True,
            text=True,
            timeout=5,
        )
        if branch.returncode == 0:
            result["branch"] = branch.stdout.strip()

    except (subprocess.TimeoutExpired, FileNotFoundError) as e:
        logger.warning(f"Could not get git info: {e}")

    return result


def get_runner_info() -> Dict[str, str]:
    """Get runner environment information."""
    import platform
    import sys

    return {
        "hostname": platform.node(),
        "platform": platform.platform(),
        "python_version": sys.version.split()[0],
        "pid": str(os.getpid()),
        "ci_job_id": os.environ.get("CI_JOB_ID", ""),
        "github_run_id": os.environ.get("GITHUB_RUN_ID", ""),
    }


def create_manifest(
    run_id: str,
    artifacts_dir: Path,
    params: Dict[str, Any],
    data_hash: str = "",
    metrics_summary: Optional[Dict[str, float]] = None,
) -> RunManifest:
    """
    Create a complete manifest for a backtest run.

    Args:
        run_id: Unique run identifier
        artifacts_dir: Directory containing artifacts to checksum
        params: Parameters used for the run
        data_hash: Optional pre-computed data hash
        metrics_summary: Optional summary metrics for quick verification

    Returns:
        RunManifest with all fields populated
    """
    git_info = get_git_info()
    runner_info = get_runner_info()

    # Compute artifact checksums
    artifact_checksums = {}
    artifacts_dir = Path(artifacts_dir)
    if artifacts_dir.exists():
        for artifact_path in artifacts_dir.iterdir():
            if artifact_path.is_file() and artifact_path.name != "manifest.json":
                artifact_checksums[artifact_path.name] = compute_sha256(artifact_path)

    return RunManifest(
        run_id=run_id,
        git_commit=git_info["commit"],
        git_dirty=git_info["dirty"],
        git_branch=git_info["branch"],
        data_fingerprint=data_hash or "",
        params_fingerprint=compute_data_fingerprint(params),
        artifact_checksums=artifact_checksums,
        started_at=datetime.now(),
        runner_info=runner_info,
        metrics_summary=metrics_summary or {},
    )


def generate_sha256sums(artifacts_dir: Path, output_path: Optional[Path] = None) -> str:
    """
    Generate sha256sums.txt file for all artifacts.

    Args:
        artifacts_dir: Directory containing artifacts
        output_path: Optional output path (defaults to artifacts_dir/sha256sums.txt)

    Returns:
        Contents of sha256sums.txt
    """
    artifacts_dir = Path(artifacts_dir)
    lines = []

    for artifact_path in sorted(artifacts_dir.iterdir()):
        if artifact_path.is_file() and artifact_path.name != "sha256sums.txt":
            checksum = compute_sha256(artifact_path)
            # Extract just the hash (remove "sha256:" prefix)
            hash_only = checksum.replace("sha256:", "")
            lines.append(f"{hash_only}  {artifact_path.name}")

    content = "\n".join(lines) + "\n"

    if output_path is None:
        output_path = artifacts_dir / "sha256sums.txt"

    with open(output_path, "w") as f:
        f.write(content)

    logger.info(f"sha256sums.txt written: {output_path}")
    return content
