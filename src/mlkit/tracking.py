"""Optional experiment tracking with MLflow (``pip install 'mlkit[tracking]'``).

Enable from config::

    tracking:
      mlflow: true
      experiment: adult-income      # optional
      tracking_uri: file:./mlruns   # optional
"""

from __future__ import annotations

from typing import TYPE_CHECKING, Any

from mlkit.core.errors import MLKitError

if TYPE_CHECKING:
    from mlkit.runner import TrainResult


def _flatten(prefix: str, data: dict[str, Any]) -> dict[str, float]:
    out: dict[str, float] = {}
    for k, v in data.items():
        if isinstance(v, (int, float)) and not isinstance(v, bool):
            out[f"{prefix}{k}"] = float(v)
    return out


def log_to_mlflow(result: TrainResult, run_name: str | None = None) -> str:
    """Log params, metrics and the bundle directory of a training run. Returns the MLflow run id."""
    try:
        import mlflow
    except ImportError as exc:
        raise MLKitError("MLflow tracking requires: pip install 'mlkit[tracking]'") from exc

    opts = result.cfg.tracking
    if opts.get("tracking_uri"):
        mlflow.set_tracking_uri(opts["tracking_uri"])
    if opts.get("experiment"):
        mlflow.set_experiment(opts["experiment"])

    cfg = result.cfg
    with mlflow.start_run(run_name=run_name) as run:
        mlflow.log_params(
            {
                "task_type": cfg.task_type,
                "target": cfg.target,
                "seed": cfg.seed,
                "cv_folds": cfg.cv_folds,
                "test_size": cfg.test_size,
                "split": result.metrics.get("split"),
                "champion": result.metrics.get("champion"),
            }
        )
        metrics = _flatten("holdout_", result.metrics.get("holdout", {}))
        metrics.update(_flatten("cv_", result.metrics.get("cv", {})))
        mlflow.log_metrics(metrics)
        if result.bundle_dir is not None:
            mlflow.log_artifacts(str(result.bundle_dir), artifact_path="bundle")
        return run.info.run_id
