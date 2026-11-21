"""Evidently adapter isolating external API changes from MLTK."""

from __future__ import annotations

import pandas as pd
from mltk_core.check import CheckResult, Report


class EvidentlyDriftAdapter:
    """
    Adapter wrapping Evidently's DataDriftPreset.
    Pins dependency to avoid breaking user workflows if Evidently changes API.
    """

    def __init__(self) -> None:
        self._is_available: bool = False
        try:
            import evidently  # noqa: F401

            self._is_available = True
        except ImportError:
            self._is_available = False

    def evaluate(self, reference: pd.DataFrame, current: pd.DataFrame) -> Report:
        """Run Evidently drift check if installed, otherwise return informative report."""
        if not self._is_available:
            return Report(
                results=[
                    CheckResult(
                        name="evidently_drift",
                        level="INFO",
                        passed=True,
                        message="Evidently is not installed. Using native MLTK DataDriftMonitor is recommended.",
                    )
                ],
                title="Evidently Drift Report",
            )

        try:
            from evidently.metric_preset import DataDriftPreset
            from evidently.report import Report as EvReport

            ev_report = EvReport(metrics=[DataDriftPreset()])
            ev_report.run(reference_data=reference, current_data=current)
            summary = ev_report.as_dict()

            drift_share = summary["metrics"][0]["result"]["drift_share"]
            dataset_drift = summary["metrics"][0]["result"]["dataset_drift"]

            level = "WARNING" if dataset_drift else "INFO"
            res = CheckResult(
                name="evidently_drift",
                level=level,
                passed=not dataset_drift,
                message=f"Evidently dataset drift: {dataset_drift} (drift share: {drift_share:.1%})",
                details=summary["metrics"][0]["result"],
            )
            return Report(results=[res], title="Evidently Drift Report")
        except Exception as exc:
            return Report(
                results=[
                    CheckResult(
                        name="evidently_drift",
                        level="WARNING",
                        passed=False,
                        message=f"Evidently execution failed: {exc}",
                    )
                ]
            )
