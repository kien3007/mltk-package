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
    log_loss,
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
        {"accuracy", "precision", "recall", "f1", "roc_auc", "pr_auc", "log_loss", "brier", "ece"}
    ),
    "regression": frozenset({"rmse", "mae", "r2"}),
}

#: Metrics where a smaller value is better (a threshold is then an upper bound).
LOWER_IS_BETTER: frozenset[str] = frozenset({"rmse", "mae", "brier", "ece", "log_loss"})


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


def optimal_threshold(
    y_true: np.ndarray, y_prob: np.ndarray, criterion: Literal["f1", "youden"] = "f1"
) -> float:
    """Decision threshold on P(class=1) maximising F1 or Youden's J (TPR - FPR)."""
    from sklearn.metrics import precision_recall_curve, roc_curve

    y = np.asarray(y_true).astype(int)
    p = np.asarray(y_prob, dtype=float)
    if len(np.unique(y)) < 2:
        return 0.5
    if criterion == "youden":
        fpr, tpr, thr = roc_curve(y, p)
        finite = np.isfinite(thr)
        return float(thr[finite][np.argmax((tpr - fpr)[finite])])
    if criterion == "f1":
        precision, recall, thr = precision_recall_curve(y, p)
        f1 = 2 * precision[:-1] * recall[:-1] / np.clip(precision[:-1] + recall[:-1], 1e-12, None)
        return float(thr[np.argmax(f1)])
    raise ValueError(f"Unknown threshold criterion {criterion!r}; use 'f1' or 'youden'.")


def _binary_average_precision(y: np.ndarray, p: np.ndarray) -> float:
    """Average precision (area under the precision-recall curve), sklearn-equivalent."""
    order = np.argsort(p, kind="mergesort")[::-1]
    ps, ys = p[order], y[order]
    idx = np.r_[np.where(np.diff(ps))[0], len(ys) - 1]
    tps = np.cumsum(ys)[idx]
    fps = idx + 1 - tps
    if tps[-1] == 0:
        return 0.0
    precision = tps / (tps + fps)
    recall = tps / tps[-1]
    return float(np.sum(np.diff(np.r_[0.0, recall]) * precision))


def _binary_log_loss(y: np.ndarray, p: np.ndarray) -> float:
    eps = np.finfo(float).eps
    p = np.clip(p, eps, 1 - eps)
    return float(-np.mean(y * np.log(p) + (1 - y) * np.log(1 - p)))


def compute_metrics(
    y_true: np.ndarray,
    y_pred: np.ndarray,
    y_prob: np.ndarray | None = None,
    task_type: Literal["classification", "regression"] = "classification",
) -> dict[str, float]:
    """Compute task-specific performance metrics.

    Binary classification: accuracy, precision, recall, f1, and with probabilities roc_auc,
    pr_auc (average precision), log_loss, brier, ece.
    Multiclass (labels 0..K-1, ``y_prob`` with K columns): accuracy, macro precision/recall/f1,
    and with probabilities one-vs-rest macro roc_auc, log_loss, multiclass brier (sum over
    classes, 0..2) and top-label ece.
    Regression: rmse, mae, r2.
    """
    metrics: dict[str, float] = {}
    if task_type != "classification":
        metrics["rmse"] = float(np.sqrt(mean_squared_error(y_true, y_pred)))
        metrics["mae"] = float(mean_absolute_error(y_true, y_pred))
        metrics["r2"] = float(r2_score(y_true, y_pred))
        return metrics

    y_arr = np.asarray(y_true)
    prob = None if y_prob is None else np.asarray(y_prob, dtype=float)
    n_cols = prob.shape[1] if prob is not None and prob.ndim == 2 else 0
    multiclass = n_cols > 2 or len(np.unique(y_arr)) > 2
    average = "macro" if multiclass else "binary"
    metrics["accuracy"] = float(accuracy_score(y_arr, y_pred))
    metrics["precision"] = float(precision_score(y_arr, y_pred, average=average, zero_division=0))
    metrics["recall"] = float(recall_score(y_arr, y_pred, average=average, zero_division=0))
    metrics["f1"] = float(f1_score(y_arr, y_pred, average=average, zero_division=0))
    if prob is None:
        return metrics

    if multiclass:
        k = max(n_cols, int(y_arr.max()) + 1)
        labels = list(range(k))
        sums = prob.sum(axis=1, keepdims=True)
        prob = np.divide(prob, sums, out=np.full_like(prob, 1.0 / prob.shape[1]), where=sums > 0)
        try:
            metrics["roc_auc"] = float(
                roc_auc_score(y_arr, prob, multi_class="ovr", average="macro", labels=labels)
            )
        except ValueError:
            metrics["roc_auc"] = float("nan")  # e.g. a class absent from this sample
        try:
            metrics["log_loss"] = float(log_loss(y_arr, prob, labels=labels))
        except ValueError:
            metrics["log_loss"] = float("nan")
        onehot = np.eye(k)[y_arr.astype(int)][:, : prob.shape[1]]
        metrics["brier"] = float(np.mean(np.sum((prob - onehot) ** 2, axis=1)))
        correct = (prob.argmax(axis=1) == y_arr).astype(int)
        metrics["ece"] = expected_calibration_error(correct, prob.max(axis=1))
        return metrics

    probs = prob[:, 1] if prob.ndim == 2 and prob.shape[1] > 1 else prob.ravel()
    try:
        metrics["roc_auc"] = float(roc_auc_score(y_arr, probs))
    except ValueError:
        metrics["roc_auc"] = 0.5
    if set(np.unique(y_arr)) <= {0, 1}:  # probability-quality metrics need 0/1 labels
        y01 = y_arr.astype(int)
        metrics["pr_auc"] = _binary_average_precision(y01, probs)
        metrics["log_loss"] = _binary_log_loss(y01, probs)
        metrics["brier"] = float(brier_score_loss(y01, probs))
        metrics["ece"] = expected_calibration_error(y01, probs)
    return metrics


#: One CV fold: (X_train, y_train, X_valid, y_valid).
FoldData = tuple[pd.DataFrame, np.ndarray, pd.DataFrame, np.ndarray]


def _time_order(time_index: Any) -> np.ndarray:
    """Stable chronological order of rows; numeric indices are used as-is, others parsed as dates."""
    s = pd.Series(np.asarray(time_index))
    key = s if pd.api.types.is_numeric_dtype(s) else pd.to_datetime(s, errors="coerce")
    return np.argsort(key.to_numpy(), kind="stable")


class CVEvaluator:
    """Cross-validation engine supporting Stratified, KFold, Group, and TimeSeries splits.

    ``split`` produces the folds, ``evaluate_folds`` scores an estimator on prepared folds
    and ``evaluate`` combines both. Keeping them separate lets :class:`~mlkit.core.Pipeline`
    re-fit upstream preprocessing inside every fold (pipeline-level CV) while using exactly
    the same fold definition.
    """

    def __init__(
        self,
        cv: int = 5,
        task_type: Literal["classification", "regression"] = "classification",
        random_state: int = 42,
    ) -> None:
        self.cv = cv
        self.task_type = task_type
        self.random_state = random_state

    def split(
        self,
        X: pd.DataFrame,
        y: Any,
        cfg: TaskConfig | None = None,
        groups: Any = None,
        time_index: Any = None,
    ) -> list[tuple[np.ndarray, np.ndarray]]:
        """Return ``(train_idx, valid_idx)`` positional indices into ``X``.

        Split selection (first match wins):
        - ``groups`` given (or ``cfg.group_col`` present in X) -> GroupKFold, except for
          forecasting tasks where the group column is a series id and time decides
        - ``time_index`` given (or ``cfg.time_col`` set) -> TimeSeriesSplit in chronological
          order (sorted by ``time_index`` / ``cfg.time_col`` when available, otherwise rows are
          assumed to be ordered already); validation rows are always later than training rows
        - classification -> StratifiedKFold, regression -> KFold
        """
        y_arr = np.asarray(y)
        if groups is None and cfg and cfg.group_col and cfg.group_col in X.columns:
            groups = X[cfg.group_col]
        if time_index is None and cfg and cfg.time_col and cfg.time_col in X.columns:
            time_index = X[cfg.time_col]

        if groups is not None and getattr(cfg, "forecasting", None) and time_index is not None:
            groups = None  # forecasting: series ids are not split keys, time is (ADR-010)
        if groups is not None:
            groups_arr = np.asarray(groups)
            n_splits = min(self.cv, len(np.unique(groups_arr)))
            return list(GroupKFold(n_splits=n_splits).split(X, y_arr, groups=groups_arr))
        if time_index is not None or (cfg and cfg.time_col):
            order = _time_order(time_index) if time_index is not None else np.arange(len(X))
            return [
                (order[tr], order[va]) for tr, va in TimeSeriesSplit(n_splits=self.cv).split(order)
            ]
        if self.task_type == "classification":
            splitter = StratifiedKFold(
                n_splits=self.cv, shuffle=True, random_state=self.random_state
            )
            return list(splitter.split(X, y_arr))
        splitter_kf = KFold(n_splits=self.cv, shuffle=True, random_state=self.random_state)
        return list(splitter_kf.split(X))

    def evaluate_folds(
        self,
        estimator: Any,
        folds: list[FoldData],
        oof: list[tuple[np.ndarray, np.ndarray]] | None = None,
    ) -> dict[str, Any]:
        """Fit a fresh clone of ``estimator`` on every fold and aggregate the metrics.

        If ``oof`` is a list, the out-of-fold ``(y_valid, p_positive)`` pairs are appended to it
        (binary classification with probabilities only).
        """
        if not folds:
            raise ValueError("No CV folds to evaluate.")
        fold_metrics = []
        for X_tr, y_tr, X_val, y_val in folds:
            model = clone(estimator)
            model.fit(X_tr, y_tr)
            preds = model.predict(X_val)

            probs = None
            if self.task_type == "classification" and hasattr(model, "predict_proba"):
                with contextlib.suppress(Exception):
                    probs = model.predict_proba(X_val)

            fold_metrics.append(compute_metrics(y_val, preds, probs, task_type=self.task_type))
            if (
                oof is not None
                and probs is not None
                and np.ndim(probs) == 2
                and probs.shape[1] == 2
            ):
                oof.append((np.asarray(y_val), np.asarray(probs)[:, 1]))

        summary: dict[str, Any] = {"folds": fold_metrics}
        primary_metric = "roc_auc" if self.task_type == "classification" else "rmse"
        for k in fold_metrics[0]:
            vals = [f[k] for f in fold_metrics if k in f]
            summary[f"{k}_mean"] = float(np.mean(vals))
            summary[f"{k}_std"] = float(np.std(vals))

        summary["primary_metric"] = primary_metric
        summary["primary_score"] = summary.get(f"{primary_metric}_mean", 0.0)
        return summary

    def evaluate(
        self,
        estimator: Any,
        X: pd.DataFrame,
        y: Any,
        cfg: TaskConfig | None = None,
        groups: Any = None,
        time_index: Any = None,
    ) -> dict[str, Any]:
        """Cross-validate ``estimator`` on already-prepared features ``X`` (model-level CV)."""
        y_arr = np.asarray(y)
        folds: list[FoldData] = [
            (X.iloc[tr], y_arr[tr], X.iloc[va], y_arr[va])
            for tr, va in self.split(X, y_arr, cfg=cfg, groups=groups, time_index=time_index)
        ]
        return self.evaluate_folds(estimator, folds)


def _fast_metrics(
    y: np.ndarray, pred: np.ndarray, prob: np.ndarray | None, task_type: str
) -> dict[str, float] | None:
    """NumPy re-implementation of :func:`compute_metrics` for bootstrap loops.

    sklearn's metric functions spend most of their time validating inputs (~40 ms per
    call), which dominates a 200-resample bootstrap. Returns ``None`` when the fast path
    does not apply (non 0/1 labels), in which case the caller falls back to sklearn.
    Equality with ``compute_metrics`` is covered by tests.
    """
    if task_type != "classification":
        err = y.astype(float) - pred.astype(float)
        ss_tot = float(np.sum((y - y.mean()) ** 2))
        return {
            "rmse": float(np.sqrt(np.mean(err**2))),
            "mae": float(np.mean(np.abs(err))),
            "r2": 1.0 - float(np.sum(err**2)) / ss_tot if ss_tot > 0 else 0.0,
        }
    if not (set(np.unique(y)) <= {0, 1} and set(np.unique(pred)) <= {0, 1}):
        return None
    y_i, p_i = y.astype(int), pred.astype(int)
    tp = int(np.sum((y_i == 1) & (p_i == 1)))
    fp = int(np.sum((y_i == 0) & (p_i == 1)))
    fn = int(np.sum((y_i == 1) & (p_i == 0)))
    precision = tp / (tp + fp) if tp + fp else 0.0
    recall = tp / (tp + fn) if tp + fn else 0.0
    m = {
        "accuracy": float(np.mean(y_i == p_i)),
        "precision": float(precision),
        "recall": float(recall),
        "f1": float(2 * precision * recall / (precision + recall)) if precision + recall else 0.0,
    }
    if prob is not None:
        from scipy.stats import rankdata

        p1 = prob[:, 1] if prob.ndim == 2 and prob.shape[1] > 1 else prob.ravel()
        n_pos, n_neg = int(y_i.sum()), int(len(y_i) - y_i.sum())
        if n_pos and n_neg:
            ranks = rankdata(p1)  # average ranks for ties == sklearn's ROC AUC
            m["roc_auc"] = float(
                (ranks[y_i == 1].sum() - n_pos * (n_pos + 1) / 2) / (n_pos * n_neg)
            )
        else:
            m["roc_auc"] = 0.5
        m["pr_auc"] = _binary_average_precision(y_i, p1)
        m["log_loss"] = _binary_log_loss(y_i, p1)
        m["brier"] = float(np.mean((p1 - y_i) ** 2))
        m["ece"] = expected_calibration_error(y_i, p1)
    return m


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

        res = _fast_metrics(sample_true, sample_pred, sample_prob, task_type)
        if res is None:
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

    @property
    def classes_(self) -> np.ndarray:
        return self._calibrator.classes_

    @property
    def feature_importances_(self) -> np.ndarray | None:
        """Mean importances of the per-fold base estimators (None if unavailable)."""
        if not self._is_fitted:
            return None
        imps = [
            np.asarray(c.estimator.feature_importances_)
            for c in self._calibrator.calibrated_classifiers_
            if getattr(c.estimator, "feature_importances_", None) is not None
        ]
        return np.mean(imps, axis=0) if imps else None
