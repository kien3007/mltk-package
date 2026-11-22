"""Evaluation metrics and cross-validation runners."""

from __future__ import annotations

import contextlib
from typing import Any, Literal

import numpy as np
import pandas as pd
from sklearn.base import BaseEstimator, clone
from sklearn.calibration import CalibratedClassifierCV
from sklearn.metrics import (
    accuracy_score,
    brier_score_loss,
    f1_score,
    mean_absolute_error,
    mean_squared_error,
    precision_score,
    r2_score,
    recall_score,
    roc_auc_score,
)
from sklearn.model_selection import GroupKFold, KFold, StratifiedKFold, TimeSeriesSplit

from mlkit.core.config import TaskConfig
from mlkit.core.errors import NotFittedError

#: Metrics reported per task type (usable as keys of ``TaskConfig.metric_thresholds``).
AVAILABLE_METRICS: dict[str, frozenset[str]] = {
    "classification": frozenset(
        {"accuracy", "precision", "recall", "f1", "roc_auc", "brier", "ece"}
    ),
    "regression": frozenset({"rmse", "mae", "r2"}),
}

#: Metrics where a smaller value is better (a threshold is then an upper bound).
LOWER_IS_BETTER: frozenset[str] = frozenset({"rmse", "mae", "brier", "ece"})


def expected_calibration_error(y_true: np.ndarray, y_prob: np.ndarray, n_bins: int = 10) -> float:
    """Expected Calibration Error for binary probabilities (equal-width bins).

    ECE = sum_b (n_b / n) * |mean(y in bin b) - mean(p in bin b)|; 0 means perfectly calibrated.
    """
    y = np.asarray(y_true, dtype=float).ravel()
    p = np.asarray(y_prob, dtype=float).ravel()
    if len(y) == 0:
        return 0.0
    idx = np.minimum((np.clip(p, 0.0, 1.0) * n_bins).astype(int), n_bins - 1)
    ece = 0.0
    for b in range(n_bins):
        mask = idx == b
        if mask.any():
            ece += float(mask.mean()) * abs(float(y[mask].mean()) - float(p[mask].mean()))
    return float(ece)


def compute_metrics(
    y_true: np.ndarray,
    y_pred: np.ndarray,
    y_prob: np.ndarray | None = None,
    task_type: Literal["classification", "regression"] = "classification",
) -> dict[str, float]:
    """Compute task-specific performance metrics."""
    metrics: dict[str, float] = {}

    if task_type == "classification":
        metrics["accuracy"] = float(accuracy_score(y_true, y_pred))
        metrics["precision"] = float(precision_score(y_true, y_pred, zero_division=0))
        metrics["recall"] = float(recall_score(y_true, y_pred, zero_division=0))
        metrics["f1"] = float(f1_score(y_true, y_pred, zero_division=0))

        if y_prob is not None:
            probs = y_prob[:, 1] if y_prob.ndim == 2 and y_prob.shape[1] > 1 else y_prob.ravel()
            try:
                metrics["roc_auc"] = float(roc_auc_score(y_true, probs))
            except Exception:
                metrics["roc_auc"] = 0.5
            # Probability quality (binary 0/1 targets only)
            if set(np.unique(np.asarray(y_true))) <= {0, 1}:
                metrics["brier"] = float(brier_score_loss(y_true, probs))
                metrics["ece"] = expected_calibration_error(np.asarray(y_true), probs)
    else:
        metrics["rmse"] = float(np.sqrt(mean_squared_error(y_true, y_pred)))
        metrics["mae"] = float(mean_absolute_error(y_true, y_pred))
        metrics["r2"] = float(r2_score(y_true, y_pred))

    return metrics


class CVEvaluator:
    """Cross-validation evaluation engine supporting Stratified, KFold, Group, and TimeSeries splits."""

    def __init__(
        self,
        cv: int = 5,
        task_type: Literal["classification", "regression"] = "classification",
        random_state: int = 42,
    ) -> None:
        self.cv = cv
        self.task_type = task_type
        self.random_state = random_state

    def evaluate(
        self,
        estimator: Any,
        X: pd.DataFrame,
        y: Any,
        cfg: TaskConfig | None = None,
        groups: Any = None,
        time_index: Any = None,
    ) -> dict[str, Any]:
        """Run k-fold cross validation and return aggregated metrics.

        Split selection (first match wins):
        - ``groups`` given (or ``cfg.group_col`` present in X) -> GroupKFold
        - ``time_index`` given (or ``cfg.time_col`` set) -> TimeSeriesSplit on rows
          ordered chronologically (rows are sorted by ``time_index`` / ``cfg.time_col``
          when available, otherwise assumed to be already ordered)
        - classification -> StratifiedKFold, regression -> KFold
        """
        y_arr = np.asarray(y)
        if groups is None and cfg and cfg.group_col and cfg.group_col in X.columns:
            groups = X[cfg.group_col]
        if time_index is None and cfg and cfg.time_col and cfg.time_col in X.columns:
            time_index = X[cfg.time_col]

        # Determine splitter
        if groups is not None:
            groups_arr = np.asarray(groups)
            n_groups = len(np.unique(groups_arr))
            cv_splitter = GroupKFold(n_splits=min(self.cv, n_groups)).split(
                X, y_arr, groups=groups_arr
            )
        elif time_index is not None or (cfg and cfg.time_col):
            if time_index is not None:
                order = np.argsort(
                    pd.to_datetime(pd.Series(np.asarray(time_index)), errors="coerce").to_numpy(),
                    kind="stable",
                )
                X = X.iloc[order]
                y_arr = y_arr[order]
            cv_splitter = TimeSeriesSplit(n_splits=self.cv).split(X)
        elif self.task_type == "classification":
            cv_splitter = StratifiedKFold(
                n_splits=self.cv, shuffle=True, random_state=self.random_state
            ).split(X, y_arr)
        else:
            cv_splitter = KFold(
                n_splits=self.cv, shuffle=True, random_state=self.random_state
            ).split(X)

        fold_metrics = []
        for train_idx, val_idx in cv_splitter:
            X_tr, X_val = X.iloc[train_idx], X.iloc[val_idx]
            y_tr, y_val = y_arr[train_idx], y_arr[val_idx]

            model = clone(estimator)
            model.fit(X_tr, y_tr)
            preds = model.predict(X_val)

            probs = None
            if self.task_type == "classification" and hasattr(model, "predict_proba"):
                with contextlib.suppress(Exception):
                    probs = model.predict_proba(X_val)

            m = compute_metrics(y_val, preds, probs, task_type=self.task_type)
            fold_metrics.append(m)

        summary: dict[str, Any] = {"folds": fold_metrics}
        primary_metric = "roc_auc" if self.task_type == "classification" else "rmse"
        for k in fold_metrics[0]:
            vals = [f[k] for f in fold_metrics]
            summary[f"{k}_mean"] = float(np.mean(vals))
            summary[f"{k}_std"] = float(np.std(vals))

        summary["primary_metric"] = primary_metric
        summary["primary_score"] = summary.get(f"{primary_metric}_mean", 0.0)
        return summary


def compute_bootstrap_ci(
    y_true: np.ndarray,
    y_pred: np.ndarray,
    y_prob: np.ndarray | None = None,
    task_type: Literal["classification", "regression"] = "classification",
    n_bootstraps: int = 200,
    ci: float = 0.95,
    random_state: int = 42,
) -> dict[str, dict[str, float]]:
    """
    Compute non-parametric bootstrap confidence intervals for performance metrics.

    Parameters
    ----------
    y_true : np.ndarray
        True targets.
    y_pred : np.ndarray
        Predicted values.
    y_prob : np.ndarray, optional
        Predicted probabilities.
    task_type : "classification" or "regression"
        Task type.
    n_bootstraps : int
        Number of bootstrap resamples (default: 200).
    ci : float
        Confidence interval level (default: 0.95).
    random_state : int
        Seed for reproducibility.

    Returns
    -------
    dict[str, dict[str, float]]
        Mapping of metric names to mean, std, ci_lower, and ci_upper.
    """
    rng = np.random.default_rng(random_state)
    n = len(y_true)
    y_t = np.asarray(y_true)
    y_p = np.asarray(y_pred)
    y_pr = np.asarray(y_prob) if y_prob is not None else None

    alpha_low = (1.0 - ci) / 2.0
    alpha_high = 1.0 - alpha_low

    samples: dict[str, list[float]] = {}
    for _ in range(n_bootstraps):
        idx = rng.choice(n, size=n, replace=True)
        sample_true = y_t[idx]
        sample_pred = y_p[idx]
        sample_prob = y_pr[idx] if y_pr is not None else None

        if task_type == "classification" and len(np.unique(sample_true)) < 2:
            continue

        res = compute_metrics(sample_true, sample_pred, sample_prob, task_type=task_type)
        for k, v in res.items():
            samples.setdefault(k, []).append(v)

    out: dict[str, dict[str, float]] = {}
    for k, vals in samples.items():
        if vals:
            arr = np.asarray(vals)
            out[k] = {
                "mean": float(np.mean(arr)),
                "std": float(np.std(arr)),
                "ci_lower": float(np.percentile(arr, alpha_low * 100)),
                "ci_upper": float(np.percentile(arr, alpha_high * 100)),
            }
    return out


class CalibratedModelWrapper(BaseEstimator):
    """
    Wraps an estimator with CalibratedClassifierCV to output calibrated class probabilities.
    """

    def __init__(
        self,
        base_estimator: Any,
        method: Literal["sigmoid", "isotonic"] = "sigmoid",
        cv: int = 3,
    ) -> None:
        self.base_estimator = base_estimator
        self.method = method
        self.cv = cv
        self._calibrator: Any = None
        self._is_fitted: bool = False

    def fit(self, X: pd.DataFrame | np.ndarray, y: Any) -> CalibratedModelWrapper:
        self._calibrator = CalibratedClassifierCV(
            estimator=self.base_estimator,
            method=self.method,
            cv=self.cv,
        )
        self._calibrator.fit(X, y)
        self._is_fitted = True
        return self

    def predict(self, X: pd.DataFrame | np.ndarray) -> np.ndarray:
        if not self._is_fitted:
            raise NotFittedError("CalibratedModelWrapper is not fitted.")
        return self._calibrator.predict(X)

    def predict_proba(self, X: pd.DataFrame | np.ndarray) -> np.ndarray:
        if not self._is_fitted:
            raise NotFittedError("CalibratedModelWrapper is not fitted.")
        return self._calibrator.predict_proba(X)
