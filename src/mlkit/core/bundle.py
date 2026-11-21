"""Production bundle serialization, version tracking, and deserialization."""

from __future__ import annotations

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


def get_environment_versions() -> dict[str, str]:
    """Capture current runtime environment versions for auditability."""
    versions: dict[str, str] = {
        "python": sys.version.split()[0],
        "platform": platform.platform(),
        "mlkit": "0.1.0",
    }
    for pkg in ["pandas", "numpy", "scikit-learn", "lightgbm", "xgboost", "joblib"]:
        try:
            versions[pkg] = importlib.metadata.version(pkg)
        except Exception:
            pass
    return versions


def save_bundle(
    pipeline: Any,
    path: str | Path,
    cfg: TaskConfig,
    X_sample: pd.DataFrame | None = None,
    baseline: dict[str, Any] | None = None,
    metrics: dict[str, Any] | None = None,
) -> Path:
    """
    Save complete production bundle: pipeline, config, schema, baseline, metrics, versions.
    """
    bundle_dir = Path(path)
    bundle_dir.mkdir(parents=True, exist_ok=True)

    # 1. Pipeline model
    joblib.dump(pipeline, bundle_dir / "pipeline.joblib")

    # 2. Config
    with open(bundle_dir / "config.yaml", "w", encoding="utf-8") as f:
        yaml.dump(cfg.model_dump(), f, default_flow_style=False)

    # 3. Schema
    schema = {}
    if X_sample is not None:
        schema = {str(c): str(t) for c, t in X_sample.dtypes.items()}
    with open(bundle_dir / "schema.json", "w", encoding="utf-8") as f:
        json.dump(schema, f, indent=2)

    # 4. Baseline stats
    with open(bundle_dir / "baseline.json", "w", encoding="utf-8") as f:
        json.dump(baseline or {}, f, indent=2, default=str)

    # 5. Metrics
    with open(bundle_dir / "metrics.json", "w", encoding="utf-8") as f:
        json.dump(metrics or {}, f, indent=2, default=str)

    # 6. Versions
    with open(bundle_dir / "versions.json", "w", encoding="utf-8") as f:
        json.dump(get_environment_versions(), f, indent=2)

    return bundle_dir


def load_bundle(path: str | Path) -> tuple[Any, TaskConfig, dict[str, Any]]:
    """
    Load bundle and verify runtime compatibility.
    """
    bundle_dir = Path(path)
    if not bundle_dir.exists():
        raise FileNotFoundError(f"Bundle directory does not exist: {bundle_dir}")

    # Check versions
    v_file = bundle_dir / "versions.json"
    if v_file.exists():
        with open(v_file, encoding="utf-8") as f:
            saved_versions = json.load(f)
        current_versions = get_environment_versions()
        mismatches = []
        for pkg, saved_v in saved_versions.items():
            curr_v = current_versions.get(pkg)
            if curr_v and curr_v != saved_v:
                mismatches.append(f"{pkg}: saved={saved_v}, current={curr_v}")
        if mismatches:
            warnings.warn(
                f"Bundle was saved with different library versions: {', '.join(mismatches)}",
                UserWarning,
                stacklevel=2,
            )

    # Load pipeline
    pipeline = joblib.load(bundle_dir / "pipeline.joblib")

    # Load config
    cfg = TaskConfig.from_yaml(bundle_dir / "config.yaml")

    # Load metadata
    metadata = {}
    for meta_file in ["schema.json", "baseline.json", "metrics.json"]:
        p = bundle_dir / meta_file
        if p.exists():
            with open(p, encoding="utf-8") as f:
                metadata[p.stem] = json.load(f)

    return pipeline, cfg, metadata
