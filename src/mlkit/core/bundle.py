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

Security (ADR-012): ``joblib``/pickle can execute arbitrary code on load. ``load_bundle``
verifies the SHA256 manifest *before* unpickling (detects corruption / edits after saving).
To also prove *who* wrote a bundle, set a signing key (``MLKIT_BUNDLE_KEY`` environment
variable or ``signing_key=``): ``save_bundle`` then adds an HMAC-SHA256 signature of the
manifest, and ``load_bundle`` with a key refuses unsigned or wrongly signed bundles before
unpickling anything.
"""

from __future__ import annotations

import contextlib
import hashlib
import hmac
import importlib.metadata
import json
import os
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
    # the import package is ``mlkit``; the PyPI distribution is ``mltk-package``
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


SIGNING_KEY_ENV = "MLKIT_BUNDLE_KEY"


def _signing_key(key: str | bytes | None) -> bytes | None:
    key = key if key is not None else os.environ.get(SIGNING_KEY_ENV)
    if not key:
        return None
    return key.encode("utf-8") if isinstance(key, str) else key


def _signature(files: dict[str, str], key: bytes) -> str:
    payload = json.dumps({"format_version": BUNDLE_FORMAT_VERSION, "files": files}, sort_keys=True)
    return hmac.new(key, payload.encode("utf-8"), hashlib.sha256).hexdigest()


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
    signing_key: str | bytes | None = None,
) -> Path:
    """Save a complete production bundle (see module docstring for layout).

    With a ``signing_key`` (or ``MLKIT_BUNDLE_KEY``) the manifest carries an HMAC signature.
    """
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
        _write_json(
            bundle_dir / "report.json", report.to_dict() if hasattr(report, "to_dict") else report
        )
        files.append("report.json")

    _write_json(bundle_dir / "versions.json", get_environment_versions())
    files.append("versions.json")

    manifest: dict[str, Any] = {
        "format_version": BUNDLE_FORMAT_VERSION,
        "files": {name: _sha256(bundle_dir / name) for name in files},
    }
    key = _signing_key(signing_key)
    if key is not None:
        manifest["signature"] = {
            "algorithm": "hmac-sha256",
            "value": _signature(manifest["files"], key),
        }
    _write_json(bundle_dir / "manifest.json", manifest)
    return bundle_dir


def verify_bundle(path: str | Path, signing_key: str | bytes | None = None) -> dict[str, str]:
    """Verify file hashes (and, given a key, the HMAC signature) of ``manifest.json``.

    Raises ValidationError on any mismatch; with a key, an unsigned bundle is rejected too.
    """
    bundle_dir = Path(path)
    manifest_file = bundle_dir / "manifest.json"
    key = _signing_key(signing_key)
    if not manifest_file.exists():
        if key is not None:
            raise ValidationError(
                "Bundle has no manifest.json and cannot be verified against the signing key."
            )
        warnings.warn(
            f"Bundle at {bundle_dir} has no manifest.json (legacy bundle); integrity not verified.",
            UserWarning,
            stacklevel=2,
        )
        return {}
    with open(manifest_file, encoding="utf-8") as f:
        manifest = json.load(f)
    expected: dict[str, str] = manifest.get("files", {})
    if key is not None:
        sig = (manifest.get("signature") or {}).get("value")
        if not sig:
            raise ValidationError(
                "Bundle is not signed but a signing key is configured; refusing to load."
            )
        if not hmac.compare_digest(sig, _signature(expected, key)):
            raise ValidationError(
                "Bundle signature does not match the signing key; refusing to load."
            )
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


def load_bundle(
    path: str | Path, verify: bool = True, signing_key: str | bytes | None = None
) -> tuple[Any, TaskConfig, dict[str, Any]]:
    """Load a bundle after verifying integrity and runtime compatibility.

    Returns ``(pipeline, cfg, metadata)`` where ``metadata`` holds the parsed
    ``schema``, ``baseline``, ``metrics``, ``report`` and ``versions`` JSON files.
    """
    bundle_dir = Path(path)
    if not bundle_dir.exists():
        raise FileNotFoundError(f"Bundle directory does not exist: {bundle_dir}")

    if verify:
        verify_bundle(bundle_dir, signing_key=signing_key)

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
    for meta_file in [
        "schema.json",
        "baseline.json",
        "metrics.json",
        "report.json",
        "versions.json",
    ]:
        p = bundle_dir / meta_file
        if p.exists():
            with open(p, encoding="utf-8") as f:
                metadata[p.stem] = json.load(f)

    return pipeline, cfg, metadata
