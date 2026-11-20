"""Runner and high-level interface for QC execution."""

from __future__ import annotations

from typing import Any
import pandas as pd

from mltk_core.check import Check, Report
from mltk_qc.checks import (
    ConstantColumnCheck,
    DuplicateCheck,
    HighCardinalityCheck,
    LeakageCheck,
    MissingValueCheck,
    TargetDistributionCheck,
)


class QCReport(Report):
    """Specialized report for QC checks."""

    def __init__(self, results: list[Any] | None = None) -> None:
        super().__init__(results=results, title="MLTK Dataset Quality & EDA Report")


def run_qc(
    df: pd.DataFrame,
    target: pd.Series | str | None = None,
    checks: list[Check] | None = None,
) -> QCReport:
    """
    Execute standard quality control and EDA checks on a dataset.

    Parameters
    ----------
    df : pd.DataFrame
        The feature DataFrame (or dataset containing target if target is column name).
    target : pd.Series or str, optional
        Target variable or name of target column. If provided as column name, it is extracted from df.
    checks : list[Check], optional
        Custom list of checks to run. If None, runs default suite.
    """
    if isinstance(target, str):
        if target in df.columns:
            y = df[target]
            X = df.drop(columns=[target])
        else:
            raise KeyError(f"Target column '{target}' not found in DataFrame.")
    else:
        y = target
        X = df

    if checks is None:
        active_checks: list[Check] = [
            MissingValueCheck(),
            DuplicateCheck(),
            ConstantColumnCheck(),
            HighCardinalityCheck(),
            TargetDistributionCheck(target=y),
            LeakageCheck(target=y),
        ]
    else:
        active_checks = checks

    results = []
    for check in active_checks:
        res = check.run(X)
        results.append(res)

    return QCReport(results)
