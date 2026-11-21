"""Artifact packaging, serialization (skops/joblib), and MLflow registration."""

from __future__ import annotations

import json
from pathlib import Path
from typing import Any

import numpy as np
import pandas as pd
from mltk_core.base import ColumnType, infer_column_types
from mltk_core.metadata import ArtifactMetadata, compute_data_hash

from mltk_modelling.model_card import generate_model_card


def compute_reference_statistics(df: pd.DataFrame) -> dict[str, Any]:
    """Calculate baseline reference statistics for downstream data drift monitoring."""
    stats: dict[str, Any] = {}
    col_types = infer_column_types(df)

    for col, ctype in col_types.items():
        s = df[col].dropna()
        if ctype == ColumnType.NUMERIC:
            stats[col] = {
                "type": "numeric",
                "count": int(len(s)),
                "mean": float(s.mean()) if not s.empty else 0.0,
                "std": float(s.std()) if not s.empty else 0.0,
                "min": float(s.min()) if not s.empty else 0.0,
                "max": float(s.max()) if not s.empty else 0.0,
                "quantiles": [float(q) for q in np.percentile(s, np.linspace(0, 100, 11))]
                if not s.empty
                else [],
            }
        elif ctype == ColumnType.CATEGORICAL:
            counts = s.astype(str).value_counts(normalize=True).head(50).to_dict()
            stats[col] = {
                "type": "categorical",
                "count": int(len(s)),
                "proportions": {str(k): float(v) for k, v in counts.items()},
            }
    return stats


def save_artifact(
    pipeline: Any,
    X_train: pd.DataFrame,
    y_train: Any | None = None,
    X_test: pd.DataFrame | None = None,
    y_test: Any | None = None,
    metrics: dict[str, Any] | None = None,
    output_dir: str = "./artifacts/run",
    model_name: str = "mltk_model",
    random_state: int | None = 42,
    log_to_mlflow: bool = False,
    register_model: str | None = None,
) -> str:
    """
    Save complete model artifact bundle: pipeline, schema, reference stats, metrics, and metadata.
    """
    out_path = Path(output_dir)
    out_path.mkdir(parents=True, exist_ok=True)

    # 1. Save pipeline using skops if possible, fallback to joblib
    pipeline_file = out_path / "pipeline.skops"
    try:
        import skops.io as sio

        sio.dump(pipeline, str(pipeline_file))
    except Exception:
        # Fallback to joblib
        import joblib

        pipeline_file = out_path / "pipeline.joblib"
        joblib.dump(pipeline, str(pipeline_file))

    # 2. Extract feature names and input schema
    input_schema = {str(c): str(t.value) for c, t in infer_column_types(X_train).items()}
    with open(out_path / "input_schema.json", "w", encoding="utf-8") as f:
        json.dump(input_schema, f, indent=2)

    # Selected / final features
    if hasattr(pipeline, "named_steps"):
        final_features = list(X_train.columns)
    elif hasattr(pipeline, "feature_names_in_"):
        final_features = list(pipeline.feature_names_in_)
    else:
        final_features = list(X_train.columns)

    with open(out_path / "selected_features.json", "w", encoding="utf-8") as f:
        json.dump(final_features, f, indent=2)

    # 3. Compute baseline reference statistics for monitoring
    ref_stats = compute_reference_statistics(X_train)
    with open(out_path / "reference_stats.json", "w", encoding="utf-8") as f:
        json.dump(ref_stats, f, indent=2)

    # 4. Metrics
    final_metrics = metrics or {}
    with open(out_path / "metrics.json", "w", encoding="utf-8") as f:
        json.dump(final_metrics, f, indent=2)

    # 5. Metadata
    metadata = ArtifactMetadata(
        data_hash=compute_data_hash(X_train),
        random_state=random_state,
        extra={"model_name": model_name},
    )
    metadata.save(str(out_path / "artifact_metadata.json"))

    # 6. Model card
    generate_model_card(
        model_name=model_name,
        task_type="classification"
        if y_train is not None and len(np.unique(y_train)) <= 2
        else "regression",
        metrics=final_metrics,
        features=final_features,
        metadata=metadata,
        output_filepath=str(out_path / "model_card.yaml"),
    )

    # 7. Optional MLflow integration
    if log_to_mlflow:
        try:
            import mlflow

            with mlflow.start_run(run_name=model_name):
                mlflow.log_params(
                    {
                        "model_name": model_name,
                        "random_state": random_state,
                        "git_sha": metadata.git_sha,
                    }
                )
                for k, v in final_metrics.items():
                    if isinstance(v, (int, float)):
                        mlflow.log_metric(k, float(v))
                mlflow.log_artifacts(str(out_path))

                if register_model:
                    mlflow.sklearn.log_model(
                        pipeline,
                        artifact_path="model",
                        registered_model_name=register_model,
                    )
        except Exception as exc:
            print(f"Warning: MLflow logging skipped: {exc}")

    return str(out_path)


def load_artifact(artifact_dir: str) -> tuple[Any, ArtifactMetadata]:
    """Load trained pipeline and metadata from artifact directory."""
    path = Path(artifact_dir)
    meta = ArtifactMetadata.load(str(path / "artifact_metadata.json"))

    skops_file = path / "pipeline.skops"
    joblib_file = path / "pipeline.joblib"

    if skops_file.exists():
        import skops.io as sio

        # Allow trusted loading for locally created artifacts by discovering types
        untrusted = sio.get_untrusted_types(file=str(skops_file))
        pipeline = sio.load(str(skops_file), trusted=untrusted)
    elif joblib_file.exists():
        import joblib

        pipeline = joblib.load(str(joblib_file))
    else:
        raise FileNotFoundError(f"No pipeline artifact found in {artifact_dir}")

    return pipeline, meta
