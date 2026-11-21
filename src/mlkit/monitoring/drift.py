"""Data and prediction drift monitoring using exact PSI, KS-test, and Chi-Square."""

from __future__ import annotations

from typing import Any

import numpy as np
import pandas as pd
from scipy.stats import chisquare, ks_2samp

from mlkit.core.config import TaskConfig
from mlkit.core.registry import register
from mlkit.core.report import Report
from mlkit.core.step import BaseStep


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
    exp_clean = expected[~np.isnan(expected)]
    act_clean = actual[~np.isnan(actual)]

    if len(exp_clean) == 0 or len(act_clean) == 0:
        return 0.0

    # If identical data, return 0.0 immediately
    if len(exp_clean) == len(act_clean) and np.allclose(np.sort(exp_clean), np.sort(act_clean)):
        return 0.0

    # Quantile bin edges from expected distribution
    percentiles = np.linspace(0, 100, num_buckets + 1)
    bin_edges = np.percentile(exp_clean, percentiles)
    bin_edges[0] = -np.inf
    bin_edges[-1] = np.inf
    bin_edges = np.unique(bin_edges)

    if len(bin_edges) < 2:
        return 0.0

    exp_counts, _ = np.histogram(exp_clean, bins=bin_edges)
    act_counts, _ = np.histogram(act_clean, bins=bin_edges)

    exp_pct = exp_counts / len(exp_clean)
    act_pct = act_counts / len(act_clean)

    # Smooth zero bins
    exp_pct = np.where(exp_pct == 0, eps, exp_pct)
    act_pct = np.where(act_pct == 0, eps, act_pct)

    psi_val = np.sum((act_pct - exp_pct) * np.log(act_pct / exp_pct))
    return float(max(0.0, psi_val))


@register("monitoring.drift")
class DriftMonitor(BaseStep):
    """
    Monitor data and prediction drift against baseline reference snapshots.
    """

    def __init__(
        self,
        psi_threshold_warn: float = 0.1,
        psi_threshold_error: float = 0.2,
        ks_pvalue_threshold: float = 0.05,
    ) -> None:
        super().__init__()
        self.psi_threshold_warn = psi_threshold_warn
        self.psi_threshold_error = psi_threshold_error
        self.ks_pvalue_threshold = ks_pvalue_threshold

        self.baseline_distributions_: dict[str, np.ndarray] = {}
        self.categorical_baselines_: dict[str, dict[str, float]] = {}

    def _fit(self, X: pd.DataFrame, y: Any = None, cfg: TaskConfig | None = None) -> DriftMonitor:
        self.baseline_distributions_ = {}
        self.categorical_baselines_ = {}

        num_cols = X.select_dtypes(include=[np.number]).columns
        for c in num_cols:
            self.baseline_distributions_[c] = X[c].dropna().to_numpy()

        cat_cols = X.select_dtypes(include=["object", "category"]).columns
        for c in cat_cols:
            proportions = X[c].dropna().astype(str).value_counts(normalize=True).to_dict()
            self.categorical_baselines_[c] = {str(k): float(v) for k, v in proportions.items()}

        self.report_.add(
            "baseline_snapshot",
            "ok",
            f"Captured baseline distribution for {len(num_cols)} numeric and {len(cat_cols)} categorical features.",
        )
        return self

    def monitor(self, X_current: pd.DataFrame, y_pred: np.ndarray | None = None) -> Report:
        """Evaluate drift on a new batch of data."""
        report = Report(title="Data & Prediction Drift Report")

        # 1. Numeric Feature Drift (PSI & KS Test)
        for col, base_vals in self.baseline_distributions_.items():
            if col in X_current.columns:
                curr_vals = X_current[col].dropna().to_numpy()
                if len(curr_vals) > 0:
                    psi = calculate_psi(base_vals, curr_vals)
                    ks_stat, ks_pval = ks_2samp(base_vals, curr_vals)

                    if psi >= self.psi_threshold_error:
                        status = "error"
                        msg = f"Severe Drift: PSI={psi:.4f} >= {self.psi_threshold_error} (KS p-val={ks_pval:.4e})"
                    elif psi >= self.psi_threshold_warn:
                        status = "warn"
                        msg = f"Moderate Drift: PSI={psi:.4f} >= {self.psi_threshold_warn} (KS p-val={ks_pval:.4e})"
                    else:
                        status = "ok"
                        msg = f"Stable: PSI={psi:.4f} < {self.psi_threshold_warn}"

                    report.add(f"drift_{col}", status, msg, {"psi": psi, "ks_pvalue": float(ks_pval)})

        # 2. Categorical Feature Drift (Chi-Square)
        for col, base_props in self.categorical_baselines_.items():
            if col in X_current.columns:
                curr_counts = X_current[col].dropna().astype(str).value_counts()
                common_cats = [c for c in base_props if c in curr_counts]
                if len(common_cats) > 1:
                    exp_freqs = np.array([base_props[c] for c in common_cats], dtype=float)
                    obs_freqs = np.array([curr_counts[c] for c in common_cats], dtype=float)
                    # Normalize expected to match sum of observed exactly
                    if exp_freqs.sum() > 0 and obs_freqs.sum() > 0:
                        exp_freqs = exp_freqs * (obs_freqs.sum() / exp_freqs.sum())
                        chi2_stat, p_val = chisquare(f_obs=obs_freqs, f_exp=exp_freqs)
                        status = "warn" if p_val < self.ks_pvalue_threshold else "ok"
                        report.add(
                            f"cat_drift_{col}",
                            status,
                            f"Chi-square p-value: {p_val:.4e}",
                            {"p_value": float(p_val)},
                        )

        return report
