"""Prediction drift and performance degradation monitoring."""

from __future__ import annotations

from typing import Any
import numpy as np
import pandas as pd

from mltk_core.check import CheckResult, Report
from mltk_monitoring.drift import calculate_psi


class PredictionDriftMonitor:
    """
    Monitor drift on model prediction scores or probabilities.
    """

    def __init__(self, reference_predictions: np.ndarray, psi_threshold: float = 0.2) -> None:
        self.ref_preds = np.asarray(reference_predictions)
        self.psi_threshold = psi_threshold

    def evaluate(self, current_predictions: np.ndarray) -> Report:
        curr_preds = np.asarray(current_predictions)
        psi_val = calculate_psi(self.ref_preds, curr_preds)

        details = {
            "psi": round(psi_val, 4),
            "ref_mean": float(np.mean(self.ref_preds)),
            "curr_mean": float(np.mean(curr_preds)),
        }

        if psi_val >= self.psi_threshold:
            res = CheckResult(
                name="prediction_drift",
                level="WARNING",
                passed=False,
                message=f"Prediction distribution shifted: PSI={psi_val:.4f} >= {self.psi_threshold}",
                details=details,
            )
        else:
            res = CheckResult(
                name="prediction_drift",
                level="INFO",
                passed=True,
                message=f"Prediction distribution stable: PSI={psi_val:.4f}",
                details=details,
            )

        return Report(results=[res], title="MLTK Prediction Drift Report")


class PerformanceMonitor:
    """
    Monitor production metrics degradation compared to training baseline.
    """

    def __init__(self, baseline_metrics: dict[str, float], degradation_threshold: float = 0.1) -> None:
        self.baseline_metrics = baseline_metrics
        self.degradation_threshold = degradation_threshold

    def evaluate(self, current_metrics: dict[str, float]) -> Report:
        results: list[CheckResult] = []

        for metric_name, base_val in self.baseline_metrics.items():
            if metric_name not in current_metrics:
                continue

            curr_val = current_metrics[metric_name]
            diff = curr_val - base_val
            pct_change = (diff / abs(base_val)) if base_val != 0 else 0.0

            details = {
                "baseline": base_val,
                "current": curr_val,
                "absolute_diff": round(diff, 4),
                "relative_change_pct": round(pct_change * 100, 2),
            }

            # If metric is higher-is-better (like roc_auc, accuracy, f1)
            higher_is_better = any(m in metric_name for m in ["auc", "acc", "f1", "r2", "prec", "rec"])
            degraded = (pct_change < -self.degradation_threshold) if higher_is_better else (pct_change > self.degradation_threshold)

            if degraded:
                results.append(
                    CheckResult(
                        name=f"perf_{metric_name}",
                        level="WARNING",
                        passed=False,
                        message=f"Performance degradation on {metric_name}: baseline={base_val:.4f}, current={curr_val:.4f} ({pct_change*100:+.1f}%)",
                        details=details,
                    )
                )
            else:
                results.append(
                    CheckResult(
                        name=f"perf_{metric_name}",
                        level="INFO",
                        passed=True,
                        message=f"Performance on {metric_name} healthy: {curr_val:.4f} vs baseline {base_val:.4f}",
                        details=details,
                    )
                )

        return Report(results=results, title="MLTK Performance Monitoring Report")
