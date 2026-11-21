"""Evaluation metrics and cross-validation runners."""

from __future__ import annotations

from typing import Any, Literal

import numpy as np
import pandas as pd
from sklearn.base import clone
from sklearn.metrics import (
    accuracy_score,
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
    ) -> dict[str, Any]:
        """Run k-fold cross validation and return aggregated metrics."""
        y_arr = np.asarray(y)

        # Determine splitter
        if cfg and cfg.group_col and cfg.group_col in X.columns:
            groups = X[cfg.group_col]
            cv_splitter = GroupKFold(n_splits=self.cv).split(X, y_arr, groups=groups)
        elif cfg and cfg.time_col:
            cv_splitter = TimeSeriesSplit(n_splits=self.cv).split(X)
        elif self.task_type == "classification":
            cv_splitter = StratifiedKFold(n_splits=self.cv, shuffle=True, random_state=self.random_state).split(
                X, y_arr
            )
        else:
            cv_splitter = KFold(n_splits=self.cv, shuffle=True, random_state=self.random_state).split(X)

        fold_metrics = []
        for train_idx, val_idx in cv_splitter:
            X_tr, X_val = X.iloc[train_idx], X.iloc[val_idx]
            y_tr, y_val = y_arr[train_idx], y_arr[val_idx]

            model = clone(estimator)
            model.fit(X_tr, y_tr)
            preds = model.predict(X_val)

            probs = None
            if self.task_type == "classification" and hasattr(model, "predict_proba"):
                try:
                    probs = model.predict_proba(X_val)
                except Exception:
                    pass

            m = compute_metrics(y_val, preds, probs, task_type=self.task_type)
            fold_metrics.append(m)

        summary: dict[str, Any] = {"folds": fold_metrics}
        primary_metric = "roc_auc" if self.task_type == "classification" else "rmse"
        for k in fold_metrics[0].keys():
            vals = [f[k] for f in fold_metrics]
            summary[f"{k}_mean"] = float(np.mean(vals))
            summary[f"{k}_std"] = float(np.std(vals))

        summary["primary_metric"] = primary_metric
        summary["primary_score"] = summary.get(f"{primary_metric}_mean", 0.0)
        return summary
