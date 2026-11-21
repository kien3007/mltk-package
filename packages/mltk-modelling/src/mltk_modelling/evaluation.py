"""Model evaluation and cross-validation metrics computation."""

from __future__ import annotations

from typing import Any, Literal

import numpy as np
import pandas as pd
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
from sklearn.model_selection import KFold, StratifiedKFold


def compute_metrics(
    y_true: np.ndarray,
    y_pred: np.ndarray,
    y_prob: np.ndarray | None = None,
    task_type: Literal["classification", "regression"] = "classification",
) -> dict[str, float]:
    """Compute comprehensive performance metrics."""
    metrics: dict[str, float] = {}

    if task_type == "classification":
        metrics["accuracy"] = float(accuracy_score(y_true, y_pred))
        metrics["precision"] = float(precision_score(y_true, y_pred, zero_division=0))
        metrics["recall"] = float(recall_score(y_true, y_pred, zero_division=0))
        metrics["f1"] = float(f1_score(y_true, y_pred, zero_division=0))

        if y_prob is not None:
            # If 2D proba array, take positive class probability
            probs = y_prob[:, 1] if y_prob.ndim == 2 and y_prob.shape[1] > 1 else y_prob.ravel()
            try:
                metrics["roc_auc"] = float(roc_auc_score(y_true, probs))
            except Exception:
                metrics["roc_auc"] = 0.5
            try:
                metrics["brier_score"] = float(brier_score_loss(y_true, probs))
            except Exception:
                pass
    else:
        metrics["rmse"] = float(np.sqrt(mean_squared_error(y_true, y_pred)))
        metrics["mae"] = float(mean_absolute_error(y_true, y_pred))
        metrics["r2"] = float(r2_score(y_true, y_pred))

    return metrics


class CVEvaluator:
    """
    Evaluator performing K-Fold / StratifiedKFold cross-validation.
    """

    def __init__(
        self,
        cv: int = 5,
        task_type: Literal["auto", "classification", "regression"] = "auto",
        random_state: int | None = 42,
    ) -> None:
        self.cv = cv
        self.task_type = task_type
        self.random_state = random_state

    def evaluate(
        self,
        estimator: Any,
        X: pd.DataFrame,
        y: Any,
    ) -> dict[str, Any]:
        """Run cross-validation and return metrics aggregated across all folds."""
        from sklearn.base import clone

        y_arr = np.asarray(y)
        unique_targets = np.unique(y_arr[~pd.isna(y_arr)])
        is_classification = (
            len(unique_targets) <= 2
            if self.task_type == "auto"
            else self.task_type == "classification"
        )
        task_label = "classification" if is_classification else "regression"

        if is_classification:
            kf = StratifiedKFold(n_splits=self.cv, shuffle=True, random_state=self.random_state)
            splits = kf.split(X, y_arr)
        else:
            kf = KFold(n_splits=self.cv, shuffle=True, random_state=self.random_state)
            splits = kf.split(X)

        fold_metrics: list[dict[str, float]] = []
        oof_preds = np.zeros(len(X))
        oof_probs = np.zeros(len(X)) if is_classification else None

        for train_idx, val_idx in splits:
            X_train, X_val = X.iloc[train_idx], X.iloc[val_idx]
            y_train, y_val = y_arr[train_idx], y_arr[val_idx]

            fold_est = clone(estimator)
            fold_est.fit(X_train, y_train)

            preds = fold_est.predict(X_val)
            oof_preds[val_idx] = preds

            probs = None
            if is_classification and hasattr(fold_est, "predict_proba"):
                prob_arr = fold_est.predict_proba(X_val)
                probs = (
                    prob_arr[:, 1]
                    if prob_arr.ndim == 2 and prob_arr.shape[1] > 1
                    else prob_arr.ravel()
                )
                if oof_probs is not None:
                    oof_probs[val_idx] = probs

            m = compute_metrics(y_val, preds, probs, task_type=task_label)
            fold_metrics.append(m)

        # Aggregate mean and std
        summary: dict[str, Any] = {"folds": fold_metrics}
        all_metric_keys = fold_metrics[0].keys()
        for k in all_metric_keys:
            vals = [fm[k] for fm in fold_metrics if k in fm]
            summary[f"{k}_mean"] = float(np.mean(vals))
            summary[f"{k}_std"] = float(np.std(vals))

        # Overall OOF metrics
        summary["oof_metrics"] = compute_metrics(y_arr, oof_preds, oof_probs, task_type=task_label)
        return summary
