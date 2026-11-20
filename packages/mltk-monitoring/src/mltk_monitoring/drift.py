"""Data drift detection and distribution monitoring."""

from __future__ import annotations

import json
from pathlib import Path
from typing import Any
import numpy as np
import pandas as pd
from scipy import stats

from mltk_core.base import BaseMonitor, ColumnType, infer_column_types
from mltk_core.check import CheckResult, Report


def calculate_psi(expected: np.ndarray, actual: np.ndarray, num_buckets: int = 10) -> float:
    """Calculate Population Stability Index (PSI)."""
    expected = expected[~np.isnan(expected)]
    actual = actual[~np.isnan(actual)]
    if len(expected) == 0 or len(actual) == 0:
        return 0.0

    percentiles = np.linspace(0, 100, num_buckets + 1)
    breakpoints = np.percentile(expected, percentiles)
    breakpoints = np.unique(breakpoints)
    if len(breakpoints) <= 1:
        return 0.0

    expected_counts, _ = np.histogram(expected, bins=breakpoints)
    actual_counts, _ = np.histogram(actual, bins=breakpoints)

    expected_pct = (expected_counts + 1e-4) / (len(expected) + 1e-4 * len(expected_counts))
    actual_pct = (actual_counts + 1e-4) / (len(actual) + 1e-4 * len(actual_counts))

    psi = np.sum((actual_pct - expected_pct) * np.log(actual_pct / expected_pct))
    return float(psi)


def calculate_categorical_psi(
    expected_props: dict[str, float], actual_series: pd.Series
) -> float:
    """Calculate PSI between baseline categorical proportions and current series."""
    actual_counts = actual_series.astype(str).value_counts(normalize=True).to_dict()
    all_keys = set(expected_props.keys()).union(set(actual_counts.keys()))

    psi = 0.0
    for k in all_keys:
        e = expected_props.get(k, 1e-4)
        a = actual_counts.get(k, 1e-4)
        psi += (a - e) * np.log(a / e)
    return float(psi)


class DataDriftMonitor(BaseMonitor):
    """
    Monitor feature distribution changes and data drift compared to baseline.
    """

    def __init__(
        self,
        psi_warning: float = 0.1,
        psi_critical: float = 0.25,
    ) -> None:
        super().__init__()
        self.psi_warning = psi_warning
        self.psi_critical = psi_critical
        self.reference_df_: pd.DataFrame | None = None

    def _fit_reference(self, reference: pd.DataFrame) -> None:
        self.reference_df_ = reference.copy()
        col_types = infer_column_types(reference)
        self.reference_stats_ = {}

        for col, ctype in col_types.items():
            s = reference[col].dropna()
            if ctype == ColumnType.NUMERIC:
                self.reference_stats_[col] = {
                    "type": "numeric",
                    "values": s.values.astype(float),
                }
            else:
                props = s.astype(str).value_counts(normalize=True).to_dict()
                self.reference_stats_[col] = {
                    "type": "categorical",
                    "proportions": {str(k): float(v) for k, v in props.items()},
                }

    @classmethod
    def from_artifact(cls, artifact_dir: str) -> DataDriftMonitor:
        """Initialize monitor loaded directly from reference statistics in an artifact directory."""
        path = Path(artifact_dir)
        stats_file = path / "reference_stats.json"
        if not stats_file.exists():
            raise FileNotFoundError(f"reference_stats.json not found in {artifact_dir}")

        with open(stats_file, "r", encoding="utf-8") as f:
            raw_stats = json.load(f)

        monitor = cls()
        monitor.reference_stats_ = raw_stats
        monitor._is_fitted = True
        return monitor

    def evaluate(self, current: pd.DataFrame) -> Report:
        """Compare current dataset against reference baseline and produce drift Report."""
        results: list[CheckResult] = []

        for col, ref in self.reference_stats_.items():
            if col not in current.columns:
                results.append(
                    CheckResult(
                        name=f"drift_{col}",
                        level="ERROR",
                        passed=False,
                        message=f"Feature '{col}' is missing from current dataset.",
                    )
                )
                continue

            curr_s = current[col].dropna()
            if curr_s.empty:
                results.append(
                    CheckResult(
                        name=f"drift_{col}",
                        level="WARNING",
                        passed=False,
                        message=f"Feature '{col}' has 100% missing values in current dataset.",
                    )
                )
                continue

            if ref.get("type") == "numeric":
                ref_vals = ref.get("values")
                if ref_vals is None and "quantiles" in ref:
                    # Synthetic reconstruction from quantiles if loaded from JSON
                    ref_vals = np.array(ref["quantiles"])

                curr_vals = curr_s.values.astype(float)
                psi_val = calculate_psi(ref_vals, curr_vals)

                # Kolmogorov-Smirnov 2-sample test
                ks_stat, ks_p = stats.ks_2samp(ref_vals, curr_vals)

                details = {"psi": round(psi_val, 4), "ks_statistic": round(ks_stat, 4), "ks_pvalue": round(ks_p, 4)}

                if psi_val >= self.psi_critical:
                    results.append(
                        CheckResult(
                            name=f"drift_{col}",
                            level="ERROR",
                            passed=False,
                            message=f"CRITICAL DRIFT: PSI={psi_val:.4f} >= {self.psi_critical}",
                            details=details,
                        )
                    )
                elif psi_val >= self.psi_warning:
                    results.append(
                        CheckResult(
                            name=f"drift_{col}",
                            level="WARNING",
                            passed=False,
                            message=f"MODERATE DRIFT: PSI={psi_val:.4f} >= {self.psi_warning}",
                            details=details,
                        )
                    )
                else:
                    results.append(
                        CheckResult(
                            name=f"drift_{col}",
                            level="INFO",
                            passed=True,
                            message=f"Stable: PSI={psi_val:.4f}",
                            details=details,
                        )
                    )
            else:
                props = ref.get("proportions", {})
                psi_val = calculate_categorical_psi(props, curr_s)
                details = {"psi": round(psi_val, 4)}

                if psi_val >= self.psi_critical:
                    results.append(
                        CheckResult(
                            name=f"drift_{col}",
                            level="ERROR",
                            passed=False,
                            message=f"CRITICAL DRIFT: Categorical PSI={psi_val:.4f}",
                            details=details,
                        )
                    )
                elif psi_val >= self.psi_warning:
                    results.append(
                        CheckResult(
                            name=f"drift_{col}",
                            level="WARNING",
                            passed=False,
                            message=f"MODERATE DRIFT: Categorical PSI={psi_val:.4f}",
                            details=details,
                        )
                    )
                else:
                    results.append(
                        CheckResult(
                            name=f"drift_{col}",
                            level="INFO",
                            passed=True,
                            message=f"Stable: Categorical PSI={psi_val:.4f}",
                            details=details,
                        )
                    )

        return Report(results=results, title="MLTK Data Drift & Statistical Stability Report")
