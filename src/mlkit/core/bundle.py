"""Production bundle serialization, integrity checks, version tracking, and loading.

Bundle layout::

    bundle/
      pipeline.joblib    fitted Pipeline (joblib)
      config.yaml        TaskConfig
      schema.json        input feature name -> dtype
      baseline.json      drift baseline (DriftMonitor.to_dict())
      metrics.json       evaluation metrics (CV + holdout)
      report.json        aggregated pipeline report (optional)
      versions.json      runtime library versions
      manifest.json      SHA256 of every file above (integrity check)

Security note (ADR-003): ``joblib``/pickle can execute arbitrary code on load.
Only load bundles from sources you trust. ``load_bundle`` verifies the SHA256 manifest
before unpickling, which detects accidental corruption or tampering of the files
*after* the manifest was written, but it is not a substitute for trusting the source.
"""

from __future__ import annotations

import contextlib
import hashlib
import importlib.metadata
import json
import platform
import sys
import warnings
from pathlib import Path
from typing import Any

import joblib
import pandas as pd
import yaml

from mlkit.core.config import TaskConfig
from mlkit.core.errors import ValidationError

BUNDLE_FORMAT_VERSION = 2
_TRACKED_PACKAGES = ["mlkit", "pandas", "numpy", "scikit-learn", "lightgbm", "xgboost", "joblib"]


def _mlkit_version() -> str:
    try:
        return importlib.metadata.version("mlkit")
    except importlib.metadata.PackageNotFoundError:
        from mlkit import __version__

        return __version__


def get_environment_versions() -> dict[str, str]:
    """Capture current runtime environment versions for auditability."""
    versions: dict[str, str] = {
        "python": sys.version.split()[0],
        "platform": platform.platform(),
        "mlkit": _mlkit_version(),
    }
    for pkg in _TRACKED_PACKAGES[1:]:
        with contextlib.suppress(importlib.metadata.PackageNotFoundError):
            versions[pkg] = importlib.metadata.version(pkg)
    return versions


def _sha256(path: Path) -> str:
    h = hashlib.sha256()
    with open(path, "rb") as f:
        for chunk in iter(lambda: f.read(1 << 20), b""):
            h.update(chunk)
    return h.hexdigest()


def _write_json(path: Path, data: Any) -> None:
    with open(path, "w", encoding="utf-8") as f:
        json.dump(data, f, indent=2, default=str, ensure_ascii=False)


def save_bundle(
    pipeline: Any,
    path: str | Path,
    cfg: TaskConfig,
    X_sample: pd.DataFrame | None = None,
    baseline: dict[str, Any] | None = None,
    metrics: dict[str, Any] | None = None,
    report: Any | None = None,
) -> Path:
    """Save a complete production bundle (see module docstring for layout)."""
    bundle_dir = Path(path)
    bundle_dir.mkdir(parents=True, exist_ok=True)

    joblib.dump(pipeline, bundle_dir / "pipeline.joblib")

    with open(bundle_dir / "config.yaml", "w", encoding="utf-8") as f:
        yaml.safe_dump(cfg.model_dump(mode="json"), f, default_flow_style=False, sort_keys=False)

    schema = {str(c): str(t) for c, t in X_sample.dtypes.items()} if X_sample is not None else {}
    _write_json(bundle_dir / "schema.json", schema)
    _write_json(bundle_dir / "baseline.json", baseline or {})
    _write_json(bundle_dir / "metrics.json", metrics or {})
    files = ["pipeline.joblib", "config.yaml", "schema.json", "baseline.json", "metrics.json"]

    if report is not None:
        _write_json(bundle_dir / "report.json", report.to_dict() if hasattr(report, "to_dict") else report)
        files.append("report.json")

    _write_json(bundle_dir / "versions.json", get_environment_versions())
    files.append("versions.json")

    manifest = {
        "format_version": BUNDLE_FORMAT_VERSION,
        "files": {name: _sha256(bundle_dir / name) for name in files},
    }
    _write_json(bundle_dir / "manifest.json", manifest)
    return bundle_dir


def verify_bundle(path: str | Path) -> dict[str, str]:
    """Verify file hashes against ``manifest.json``. Raises ValidationError on mismatch."""
    bundle_dir = Path(path)
    manifest_file = bundle_dir / "manifest.json"
    if not manifest_file.exists():
        warnings.warn(
            f"Bundle at {bundle_dir} has no manifest.json (legacy bundle); integrity not verified.",
            UserWarning,
            stacklevel=2,
        )
        return {}
    with open(manifest_file, encoding="utf-8") as f:
        manifest = json.load(f)
    expected: dict[str, str] = manifest.get("files", {})
    problems = []
    for name, digest in expected.items():
        p = bundle_dir / name
        if not p.exists():
            problems.append(f"{name}: missing")
        elif _sha256(p) != digest:
            problems.append(f"{name}: checksum mismatch")
    if problems:
        raise ValidationError(f"Bundle integrity check failed: {'; '.join(problems)}")
    return expected


def load_bundle(path: str | Path, verify: bool = True) -> tuple[Any, TaskConfig, dict[str, Any]]:
    """Load a bundle after verifying integrity and runtime compatibility.

    Returns ``(pipeline, cfg, metadata)`` where ``metadata`` holds the parsed
    ``schema``, ``baseline``, ``metrics``, ``report`` and ``versions`` JSON files.
    """
    bundle_dir = Path(path)
    if not bundle_dir.exists():
        raise FileNotFoundError(f"Bundle directory does not exist: {bundle_dir}")

    if verify:
        verify_bundle(bundle_dir)

    v_file = bundle_dir / "versions.json"
    if v_file.exists():
        with open(v_file, encoding="utf-8") as f:
            saved_versions = json.load(f)
        current_versions = get_environment_versions()
        mismatches = [
            f"{pkg}: saved={saved_v}, current={current_versions[pkg]}"
            for pkg, saved_v in saved_versions.items()
            if pkg in _TRACKED_PACKAGES and current_versions.get(pkg) not in (None, saved_v)
        ]
        if mismatches:
            warnings.warn(
                f"Bundle was saved with different library versions: {', '.join(mismatches)}",
                UserWarning,
                stacklevel=2,
            )

    pipeline = joblib.load(bundle_dir / "pipeline.joblib")
    cfg = TaskConfig.from_yaml(bundle_dir / "config.yaml")

    metadata: dict[str, Any] = {}
    for meta_file in ["schema.json", "baseline.json", "metrics.json", "report.json", "versions.json"]:
        p = bundle_dir / meta_file
        if p.exists():
            with open(p, encoding="utf-8") as f:
                metadata[p.stem] = json.load(f)

    return pipeline, cfg, metadata
