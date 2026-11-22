"""Data and prediction drift monitoring using PSI, KS-test, and Chi-Square.

The baseline captured by :class:`DriftMonitor` is fully JSON-serialisable
(:meth:`DriftMonitor.to_dict` / :meth:`DriftMonitor.from_dict`) so it can be stored
inside a production bundle and re-used by ``mlkit monitor`` on future batches.
"""

from __future__ import annotations

from typing import Any

import numpy as np
import pandas as pd
from scipy.stats import chi2_contingency, chisquare, ks_2samp

from mlkit.core.config import TaskConfig
from mlkit.core.registry import register
from mlkit.core.report import Report, Severity
from mlkit.core.step import BaseStep

BASELINE_SCHEMA_VERSION = 2
_OTHER = "__unseen__"


def _psi_from_proportions(exp_pct: np.ndarray, act_pct: np.ndarray, eps: float = 1e-4) -> float:
    exp_pct = np.where(exp_pct <= 0, eps, exp_pct)
    act_pct = np.where(act_pct <= 0, eps, act_pct)
    return float(max(0.0, np.sum((act_pct - exp_pct) * np.log(act_pct / exp_pct))))


def quantile_bin_edges(values: np.ndarray, num_buckets: int = 10) -> np.ndarray:
    """Quantile bin edges of ``values`` with open outer edges (-inf, +inf)."""
    clean = values[~np.isnan(values)]
    if len(clean) == 0:
        return np.array([-np.inf, np.inf])
    edges = np.percentile(clean, np.linspace(0, 100, num_buckets + 1))
    edges[0], edges[-1] = -np.inf, np.inf
    return np.unique(edges)


def psi_from_bins(edges: np.ndarray, expected_pct: np.ndarray, actual: np.ndarray) -> float:
    """PSI of ``actual`` against stored baseline bin edges and proportions."""
    act_clean = actual[~np.isnan(actual)]
    if len(act_clean) == 0 or len(edges) < 2:
        return 0.0
    act_counts, _ = np.histogram(act_clean, bins=edges)
    return _psi_from_proportions(np.asarray(expected_pct, dtype=float), act_counts / len(act_clean))


def calculate_psi(
    expected: np.ndarray,
    actual: np.ndarray,
    num_buckets: int = 10,
    eps: float = 1e-4,
) -> float:
    """
    Compute Population Stability Index (PSI) between baseline and current data.

    Returns exactly 0.0 when distributions are identical.
    """
    expected = np.asarray(expected, dtype=float)
    actual = np.asarray(actual, dtype=float)
    exp_clean = expected[~np.isnan(expected)]
    act_clean = actual[~np.isnan(actual)]

    if len(exp_clean) == 0 or len(act_clean) == 0:
        return 0.0

    if len(exp_clean) == len(act_clean) and np.allclose(np.sort(exp_clean), np.sort(act_clean)):
        return 0.0

    edges = quantile_bin_edges(exp_clean, num_buckets)
    if len(edges) < 2:
        return 0.0
    exp_counts, _ = np.histogram(exp_clean, bins=edges)
    act_counts, _ = np.histogram(act_clean, bins=edges)
    return _psi_from_proportions(exp_counts / len(exp_clean), act_counts / len(act_clean), eps)


def _status_for_psi(psi: float, warn: float, error: float) -> Severity:
    if psi >= error:
        return "error"
    if psi >= warn:
        return "warn"
    return "ok"


@register("monitoring.drift")
class DriftMonitor(BaseStep):
    """
    Monitor data and prediction drift against a serialisable baseline snapshot.

    Numeric features: PSI on baseline quantile bins + two-sample KS test against a
    bounded reference sample. Categorical features: PSI over category shares plus a
    Chi-square goodness-of-fit test; categories unseen in the baseline are reported.
    Predictions: call :meth:`fit_predictions` on reference predictions and pass
    ``y_pred`` to :meth:`monitor`.
    """

    def __init__(
        self,
        psi_threshold_warn: float = 0.1,
        psi_threshold_error: float = 0.2,
        ks_pvalue_threshold: float = 0.05,
        num_buckets: int = 10,
        sample_size: int = 5000,
        unseen_threshold_warn: float = 0.05,
        random_state: int = 42,
    ) -> None:
        super().__init__()
        self.psi_threshold_warn = psi_threshold_warn
        self.psi_threshold_error = psi_threshold_error
        self.ks_pvalue_threshold = ks_pvalue_threshold
        self.num_buckets = num_buckets
        self.sample_size = sample_size
        self.unseen_threshold_warn = unseen_threshold_warn
        self.random_state = random_state

        self.numeric_baselines_: dict[str, dict[str, Any]] = {}
        self.categorical_baselines_: dict[str, dict[str, float]] = {}
        self.categorical_counts_: dict[str, int] = {}
        self.prediction_baseline_: dict[str, Any] | None = None

    # ------------------------------------------------------------------ fit
    def _numeric_snapshot(self, values: np.ndarray) -> dict[str, Any]:
        values = values[~np.isnan(values)]
        edges = quantile_bin_edges(values, self.num_buckets)
        counts, _ = np.histogram(values, bins=edges) if len(values) else (np.zeros(1), None)
        rng = np.random.default_rng(self.random_state)
        sample = values if len(values) <= self.sample_size else rng.choice(values, self.sample_size, replace=False)
        return {
            "edges": [float(e) for e in edges],
            "proportions": [float(c) for c in (counts / max(len(values), 1))],
            "sample": [float(v) for v in sample],
            "mean": float(np.mean(values)) if len(values) else None,
            "std": float(np.std(values)) if len(values) else None,
            "n": int(len(values)),
        }

    def _fit(self, X: pd.DataFrame, y: Any = None, cfg: TaskConfig | None = None) -> DriftMonitor:
        self.numeric_baselines_ = {}
        self.categorical_baselines_ = {}
        self.categorical_counts_ = {}

        num_cols = [c for c in X.columns if pd.api.types.is_numeric_dtype(X[c]) and not pd.api.types.is_bool_dtype(X[c])]
        for c in num_cols:
            self.numeric_baselines_[str(c)] = self._numeric_snapshot(X[c].to_numpy(dtype=float))

        cat_cols = X.select_dtypes(include=["object", "category", "string", "bool"]).columns
        for c in cat_cols:
            proportions = X[c].dropna().astype(str).value_counts(normalize=True).to_dict()
            self.categorical_baselines_[str(c)] = {str(k): float(v) for k, v in proportions.items()}
            self.categorical_counts_[str(c)] = int(X[c].notna().sum())

        self.report_.add(
            "baseline_snapshot",
            "ok",
            f"Captured baseline distribution for {len(num_cols)} numeric and {len(cat_cols)} categorical features.",
        )
        return self

    def fit_predictions(self, y_pred: np.ndarray) -> DriftMonitor:
        """Capture the reference distribution of model outputs (scores or predictions)."""
        arr = np.asarray(y_pred, dtype=float).ravel()
        self.prediction_baseline_ = self._numeric_snapshot(arr)
        return self

    @property
    def baseline_distributions_(self) -> dict[str, np.ndarray]:
        """Backward-compatible view: reference sample per numeric feature."""
        return {c: np.asarray(b["sample"]) for c, b in self.numeric_baselines_.items()}

    # --------------------------------------------------------- serialisation
    def to_dict(self) -> dict[str, Any]:
        return {
            "schema_version": BASELINE_SCHEMA_VERSION,
            "params": {
                "psi_threshold_warn": self.psi_threshold_warn,
                "psi_threshold_error": self.psi_threshold_error,
                "ks_pvalue_threshold": self.ks_pvalue_threshold,
                "num_buckets": self.num_buckets,
                "sample_size": self.sample_size,
                "unseen_threshold_warn": self.unseen_threshold_warn,
            },
            "numeric": self.numeric_baselines_,
            "categorical": self.categorical_baselines_,
            "categorical_counts": self.categorical_counts_,
            "prediction": self.prediction_baseline_,
        }

    @classmethod
    def from_dict(cls, data: dict[str, Any]) -> DriftMonitor:
        if data.get("schema_version") != BASELINE_SCHEMA_VERSION:
            raise ValueError(
                "Unsupported drift baseline format "
                f"(schema_version={data.get('schema_version')!r}); re-run training to regenerate it."
            )
        monitor = cls(**data.get("params", {}))
        monitor.numeric_baselines_ = data.get("numeric", {})
        monitor.categorical_baselines_ = data.get("categorical", {})
        monitor.categorical_counts_ = data.get("categorical_counts", {})
        monitor.prediction_baseline_ = data.get("prediction")
        monitor._is_fitted = True
        return monitor

    # -------------------------------------------------------------- monitor
    def _numeric_check(self, report: Report, name: str, base: dict[str, Any], values: np.ndarray) -> None:
        values = values[~np.isnan(values)]
        if len(values) == 0:
            report.add(name, "warn", "No non-null values in current batch.")
            return
        psi = psi_from_bins(np.asarray(base["edges"]), np.asarray(base["proportions"]), values)
        sample = np.asarray(base["sample"], dtype=float)
        ks_pval = float(ks_2samp(sample, values).pvalue) if len(sample) else 1.0
        status = _status_for_psi(psi, self.psi_threshold_warn, self.psi_threshold_error)
        label = {"error": "Severe Drift", "warn": "Moderate Drift", "ok": "Stable"}[status]
        msg = f"{label}: PSI={psi:.4f} (KS p-val={ks_pval:.4e})"
        report.add(name, status, msg, {"psi": psi, "ks_pvalue": ks_pval})

    @staticmethod
    def _chi2_two_sample(
        base_pct: np.ndarray, curr_pct: np.ndarray, n_base: int | None, n_curr: int
    ) -> float:
        """Two-sample chi-square test of homogeneity (baseline vs. batch).

        Unlike a goodness-of-fit test this accounts for sampling noise in the baseline.
        Rare categories (expected count < 5 in either sample) are pooled with unseen ones.
        Falls back to goodness-of-fit when the baseline size is unknown (legacy baselines).
        """
        curr_counts = curr_pct * n_curr
        if not n_base:
            exp = base_pct * n_curr
            keep = exp >= 5
            exp_cells = np.append(exp[keep], exp[~keep].sum())
            obs_cells = np.append(curr_counts[keep], curr_counts[~keep].sum())
            mask = exp_cells > 0
            if mask.sum() < 2:
                return 1.0
            exp_cells = exp_cells[mask] / exp_cells[mask].sum() * obs_cells[mask].sum()
            return float(chisquare(f_obs=obs_cells[mask], f_exp=exp_cells).pvalue)

        base_counts = base_pct * n_base
        total = base_counts + curr_counts
        expected_min = np.minimum(total * n_base / (n_base + n_curr), total * n_curr / (n_base + n_curr))
        keep = expected_min >= 5
        table = np.vstack(
            [
                np.append(base_counts[keep], base_counts[~keep].sum()),
                np.append(curr_counts[keep], curr_counts[~keep].sum()),
            ]
        )
        table = table[:, table.sum(axis=0) > 0]
        if table.shape[1] < 2:
            return 1.0
        return float(chi2_contingency(table, correction=False).pvalue)

    def monitor(self, X_current: pd.DataFrame, y_pred: np.ndarray | None = None) -> Report:
        """Evaluate drift of a new batch (and optionally its predictions) against the baseline."""
        report = Report(title="Data & Prediction Drift Report")

        # 0. Schema: features missing from the batch
        expected = set(self.numeric_baselines_) | set(self.categorical_baselines_)
        missing = sorted(c for c in expected if c not in X_current.columns)
        if missing:
            report.add("schema_missing_columns", "error", f"Columns missing from batch: {missing}")

        # 1. Numeric feature drift
        for col, base in self.numeric_baselines_.items():
            if col in X_current.columns:
                vals = pd.to_numeric(X_current[col], errors="coerce").to_numpy(dtype=float)
                self._numeric_check(report, f"drift_{col}", base, vals)

        # 2. Categorical feature drift
        for col, base_props in self.categorical_baselines_.items():
            if col not in X_current.columns:
                continue
            curr = X_current[col].dropna().astype(str)
            if curr.empty:
                continue
            counts = curr.value_counts()
            unseen_share = float(counts[[c for c in counts.index if c not in base_props]].sum() / len(curr))
            cats = list(base_props)
            exp_pct = np.array([base_props[c] for c in cats] + [0.0])
            act_pct = np.array([counts.get(c, 0) for c in cats] + [len(curr) * unseen_share]) / len(curr)
            psi = _psi_from_proportions(exp_pct, act_pct)

            p_val = self._chi2_two_sample(exp_pct, act_pct, self.categorical_counts_.get(col), len(curr))

            status = _status_for_psi(psi, self.psi_threshold_warn, self.psi_threshold_error)
            if status == "ok" and (p_val < self.ks_pvalue_threshold or unseen_share > self.unseen_threshold_warn):
                status = "warn"
            report.add(
                f"cat_drift_{col}",
                status,
                f"PSI={psi:.4f}, Chi-square p-value={p_val:.4e}, unseen share={unseen_share:.2%}",
                {"psi": psi, "p_value": p_val, "unseen_share": unseen_share},
            )

        # 3. Prediction drift
        if y_pred is not None and self.prediction_baseline_ is not None:
            self._numeric_check(
                report,
                "prediction_drift",
                self.prediction_baseline_,
                np.asarray(y_pred, dtype=float).ravel(),
            )

        return report
