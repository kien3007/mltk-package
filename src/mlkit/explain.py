"""Model explanations: per-prediction reason codes and permutation importance.

``explain_frame`` answers *"why this score for this row?"*: the top-k input columns that pushed
the prediction up or down. Contributions are computed on the model's feature space and summed
back onto the **original input columns** (one-hot levels, date parts, missing flags and text
statistics are attributed to the column they came from).

Methods (``method="auto"`` picks the first that applies):

- ``linear``: exact for linear models, ``coef * (x - median)``;
- ``shap``: SHAP TreeExplainer for tree models when the ``shap`` package is installed;
- ``ablation``: model-agnostic, ``f(x) - f(x with the feature set to its training median)``.

Explained output: P(positive class) for binary, P(predicted class) for multiclass, the
prediction for regression.
"""

from __future__ import annotations

from typing import Any

import numpy as np
import pandas as pd

from mlkit.modelling.evaluation import compute_metrics

_TREE_TRAINERS = (
    "RandomForestTrainer",
    "ExtraTreesTrainer",
    "LGBMTrainer",
    "XGBTrainer",
    "HistGBTrainer",
)


def _final_model(pipeline: Any) -> tuple[Any, dict[str, float]]:
    """(fitted trainer, feature medians) behind a Pipeline / ModelArena / trainer."""
    final = pipeline.steps[-1][1] if hasattr(pipeline, "steps") else pipeline
    medians = dict(getattr(final, "feature_medians_", {}) or {})
    model = getattr(final, "best_estimator_", final)
    calibrator = getattr(model, "_calibrator", None)  # CalibratedModelWrapper -> first base model
    if calibrator is not None and getattr(calibrator, "calibrated_classifiers_", None):
        model = calibrator.calibrated_classifiers_[0].estimator
    return model, medians


def _features(pipeline: Any, X: pd.DataFrame) -> pd.DataFrame:
    return pipeline.transform(X) if hasattr(pipeline, "steps") else X


def _origin(columns: list[str], inputs: list[str]) -> dict[str, str]:
    """Map each model feature to the input column it was derived from (longest prefix match)."""
    by_len = sorted(inputs, key=len, reverse=True)
    out = {}
    for c in columns:
        out[c] = next((name for name in by_len if c == name or c.startswith(name + "_")), c)
    return out


def _output(model: Any, Z: pd.DataFrame, task_type: str, cls: np.ndarray | None) -> np.ndarray:
    if task_type != "classification":
        return np.asarray(model.predict(Z), dtype=float)
    proba = np.asarray(model.predict_proba(Z))
    if proba.shape[1] == 2:
        return proba[:, 1]
    assert cls is not None
    return proba[np.arange(len(proba)), cls]


def _linear_contrib(
    model: Any, Z: pd.DataFrame, med: np.ndarray, cls: np.ndarray | None
) -> np.ndarray | None:
    est = getattr(model, "_estimator", model)
    coef = getattr(est, "coef_", None)
    if coef is None:
        return None
    coef = np.atleast_2d(np.asarray(coef, dtype=float))
    rows = (
        coef[cls]
        if (cls is not None and coef.shape[0] > 1)
        else np.repeat(coef[:1], len(Z), axis=0)
    )
    return rows * (Z.to_numpy(dtype=float) - med)


def _shap_contrib(model: Any, Z: pd.DataFrame, cls: np.ndarray | None) -> np.ndarray | None:
    if type(model).__name__ not in _TREE_TRAINERS:
        return None
    import shap  # optional dependency (extra "explain")

    est = getattr(model, "_estimator", model)
    Zp = model._prep(Z) if hasattr(model, "_prep") else Z
    values = shap.TreeExplainer(est).shap_values(Zp)
    if isinstance(values, list):  # older shap: one array per class
        values = np.stack(values, axis=-1)
    values = np.asarray(values, dtype=float)
    if values.ndim == 3:  # (n, p, n_classes)
        if values.shape[2] == 2:
            return values[:, :, 1]
        assert cls is not None
        return values[np.arange(len(values)), :, cls]
    return values


def _ablation_contrib(
    model: Any, Z: pd.DataFrame, med: np.ndarray, task_type: str, cls: np.ndarray | None
) -> np.ndarray:
    base = _output(model, Z, task_type, cls)
    contrib = np.zeros(Z.shape, dtype=float)
    for j, col in enumerate(Z.columns):
        Zj = Z.copy()
        Zj[col] = med[j]
        contrib[:, j] = base - _output(model, Zj, task_type, cls)
    return contrib


def feature_contributions(
    pipeline: Any, X: pd.DataFrame, task_type: str = "classification", method: str = "auto"
) -> tuple[pd.DataFrame, str]:
    """Per-row contributions summed onto the original input columns, and the method used."""
    if method not in ("auto", "linear", "shap", "ablation"):
        raise ValueError("method must be 'auto', 'linear', 'shap' or 'ablation'")
    model, medians = _final_model(pipeline)
    Z = _features(pipeline, X).reset_index(drop=True)
    Z = Z.apply(pd.to_numeric, errors="coerce").astype(float)
    med = np.array([medians.get(c, Z[c].median()) for c in Z.columns], dtype=float)
    med = np.where(np.isfinite(med), med, 0.0)
    cls = None
    if task_type == "classification":
        proba = np.asarray(model.predict_proba(Z))
        cls = proba.argmax(axis=1) if proba.shape[1] > 2 else None

    contrib, used = None, method
    if method in ("auto", "linear"):
        contrib, used = _linear_contrib(model, Z, med, cls), "linear"
    if contrib is None and method in ("auto", "shap"):
        try:
            contrib, used = _shap_contrib(model, Z, cls), "shap"
        except Exception:  # noqa: BLE001 - shap missing or unsupported model: fall back
            contrib = None
    if contrib is None:
        contrib, used = _ablation_contrib(model, Z, med, task_type, cls), "ablation"

    frame = pd.DataFrame(contrib, columns=Z.columns)
    inputs = [str(c) for c in getattr(pipeline, "feature_names_in_", list(X.columns))]
    origin = _origin(list(Z.columns), inputs)
    grouped = frame.T.groupby(lambda c: origin[c], sort=False).sum().T
    return grouped, used


def explain_frame(
    pipeline: Any,
    X: pd.DataFrame,
    task_type: str = "classification",
    top_k: int = 3,
    method: str = "auto",
) -> pd.DataFrame:
    """Top-k reasons per row: ``reason_i`` (input column) and ``reason_i_impact`` (signed)."""
    grouped, used = feature_contributions(pipeline, X, task_type=task_type, method=method)
    values = grouped.to_numpy()
    names = np.asarray(grouped.columns)
    order = np.argsort(-np.abs(values), axis=1)[:, :top_k]
    out = pd.DataFrame(index=range(len(grouped)))
    for i in range(min(top_k, values.shape[1])):
        idx = order[:, i]
        out[f"reason_{i + 1}"] = names[idx]
        out[f"reason_{i + 1}_impact"] = values[np.arange(len(values)), idx]
    out.attrs["method"] = used
    return out


def permutation_importance(
    pipeline: Any,
    X: pd.DataFrame,
    y: Any,
    task_type: str = "classification",
    n_repeats: int = 5,
    random_state: int = 0,
) -> pd.DataFrame:
    """Drop in score when one *input* column is shuffled (the whole pipeline is re-run).

    Score: ROC-AUC (classification, one-vs-rest for multiclass) or R^2 (regression).
    ``y`` must be encoded like at training time (see ``mlkit.runner.prepare_target``).
    """
    rng = np.random.default_rng(random_state)
    y_arr = np.asarray(y)

    def score(frame: pd.DataFrame) -> float:
        if task_type == "classification":
            proba = np.asarray(pipeline.predict_proba(frame))
            return float(compute_metrics(y_arr, proba.argmax(axis=1), proba)["roc_auc"])
        return float(
            compute_metrics(y_arr, np.asarray(pipeline.predict(frame)), task_type="regression")[
                "r2"
            ]
        )

    base = score(X)
    rows = []
    for col in X.columns:
        drops = []
        for _ in range(n_repeats):
            shuffled = X.copy()
            shuffled[col] = rng.permutation(shuffled[col].to_numpy())
            drops.append(base - score(shuffled))
        rows.append(
            {
                "feature": col,
                "importance_mean": float(np.mean(drops)),
                "importance_std": float(np.std(drops)),
            }
        )
    return pd.DataFrame(rows).sort_values("importance_mean", ascending=False).reset_index(drop=True)
