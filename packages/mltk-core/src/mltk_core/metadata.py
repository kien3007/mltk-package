"""Artifact metadata recording and reproducibility tracking."""

from __future__ import annotations

import hashlib
import json
import os
import subprocess
from dataclasses import asdict, dataclass, field
from datetime import datetime, timezone
from importlib import metadata
from typing import Any

import pandas as pd

from mltk_core.logging import get_run_id


def compute_data_hash(df: pd.DataFrame) -> str:
    """Compute a deterministic SHA-256 hash representing DataFrame columns and contents."""
    h = hashlib.sha256()
    h.update(str(list(df.columns)).encode("utf-8"))
    h.update(str(df.shape).encode("utf-8"))
    # Hash pandas internal values deterministically
    try:
        sample_bytes = pd.util.hash_pandas_object(df, index=True).values.tobytes()
        h.update(sample_bytes)
    except Exception:
        # Fallback to string sample representation if hashing object columns fails
        h.update(df.head(100).to_csv().encode("utf-8"))
    return h.hexdigest()[:16]


def get_git_sha() -> str:
    """Retrieve the current Git SHA if available."""
    try:
        res = subprocess.run(
            ["git", "rev-parse", "HEAD"],
            capture_output=True,
            text=True,
            check=True,
            timeout=2,
        )
        return res.stdout.strip()
    except Exception:
        return "unknown"


def get_installed_package_versions(packages: list[str] | None = None) -> dict[str, str]:
    """Get version numbers of key packages."""
    target_packages = packages or [
        "pandas",
        "numpy",
        "scikit-learn",
        "pandera",
        "pydantic",
        "mltk-core",
    ]
    versions: dict[str, str] = {}
    for pkg in target_packages:
        try:
            versions[pkg] = metadata.version(pkg)
        except Exception:
            versions[pkg] = "not-installed"
    return versions


@dataclass
class ArtifactMetadata:
    """Metadata recorded with every MLTK artifact bundle."""

    created_at: str = field(default_factory=lambda: datetime.now(timezone.utc).isoformat())
    mltk_core_version: str = "0.1.0"
    package_versions: dict[str, str] = field(default_factory=get_installed_package_versions)
    git_sha: str = field(default_factory=get_git_sha)
    data_hash: str = ""
    random_state: int | None = None
    run_id: str = field(default_factory=get_run_id)
    extra: dict[str, Any] = field(default_factory=dict)

    def to_dict(self) -> dict[str, Any]:
        return asdict(self)

    def to_json(self, indent: int = 2) -> str:
        return json.dumps(self.to_dict(), indent=indent, default=str)

    def save(self, filepath: str) -> None:
        os.makedirs(os.path.dirname(os.path.abspath(filepath)), exist_ok=True)
        with open(filepath, "w", encoding="utf-8") as f:
            f.write(self.to_json())

    @classmethod
    def load(cls, filepath: str) -> ArtifactMetadata:
        with open(filepath, "r", encoding="utf-8") as f:
            data = json.load(f)
        return cls(**data)
