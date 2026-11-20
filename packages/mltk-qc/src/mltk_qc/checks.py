"""Quality and validation checks for tabular datasets."""

from __future__ import annotations

from typing import Any
import numpy as np
import pandas as pd

from mltk_core.base import ColumnType, infer_column_types
from mltk_core.check import Check, CheckResult


class MissingValueCheck(Check):
    """Check missing value ratios across columns and rows."""

    def __init__(self, col_threshold: float = 0.5, row_threshold: float = 0.7) -> None:
        self.col_threshold = col_threshold
        self.row_threshold = row_threshold

    @property
    def name(self) -> str:
        return "missing_values"

    def run(self, df: pd.DataFrame) -> CheckResult:
        col_missing = df.isnull().mean()
        high_missing_cols = col_missing[col_missing > self.col_threshold].to_dict()

        row_missing = df.isnull().mean(axis=1)
        high_missing_rows_pct = float((row_missing > self.row_threshold).mean() * 100)

        details = {
            "columns_exceeding_threshold": high_missing_cols,
            "overall_missing_ratio": float(df.isnull().values.mean()),
            "rows_with_high_missing_pct": high_missing_rows_pct,
        }

        if high_missing_cols:
            return CheckResult(
                name=self.name,
                level="WARNING",
                passed=False,
                message=f"{len(high_missing_cols)} columns have > {int(self.col_threshold*100)}% missing values: {list(high_missing_cols.keys())}",
                details=details,
            )
        return CheckResult(
            name=self.name,
            level="INFO",
            passed=True,
            message="Missing value rates within acceptable bounds.",
            details=details,
        )


class DuplicateCheck(Check):
    """Check for duplicate rows."""

    def __init__(self, subset: list[str] | None = None) -> None:
        self.subset = subset

    @property
    def name(self) -> str:
        return "duplicates"

    def run(self, df: pd.DataFrame) -> CheckResult:
        dup_count = int(df.duplicated(subset=self.subset).sum())
        dup_pct = float((dup_count / len(df)) * 100) if len(df) > 0 else 0.0

        details = {"duplicate_rows_count": dup_count, "duplicate_pct": dup_pct}
        if dup_count > 0:
            return CheckResult(
                name=self.name,
                level="WARNING",
                passed=False,
                message=f"Found {dup_count} duplicate rows ({dup_pct:.1f}%).",
                details=details,
            )
        return CheckResult(
            name=self.name,
            level="INFO",
            passed=True,
            message="No duplicate rows detected.",
            details=details,
        )


class ConstantColumnCheck(Check):
    """Check for constant or zero-variance columns."""

    @property
    def name(self) -> str:
        return "constant_columns"

    def run(self, df: pd.DataFrame) -> CheckResult:
        constant_cols = [c for c in df.columns if df[c].nunique(dropna=False) <= 1]
        details = {"constant_columns": constant_cols}

        if constant_cols:
            return CheckResult(
                name=self.name,
                level="WARNING",
                passed=False,
                message=f"Found {len(constant_cols)} constant column(s) with no variance: {constant_cols}",
                details=details,
            )
        return CheckResult(
            name=self.name,
            level="INFO",
            passed=True,
            message="No constant columns detected.",
            details=details,
        )


class HighCardinalityCheck(Check):
    """Check for categorical columns with high cardinality."""

    def __init__(self, threshold: int = 50) -> None:
        self.threshold = threshold

    @property
    def name(self) -> str:
        return "high_cardinality"

    def run(self, df: pd.DataFrame) -> CheckResult:
        col_types = infer_column_types(df)
        high_card_cols: dict[str, int] = {}

        for col, ctype in col_types.items():
            if ctype == ColumnType.CATEGORICAL:
                n_unique = int(df[col].nunique(dropna=True))
                if n_unique > self.threshold:
                    high_card_cols[col] = n_unique

        details = {"high_cardinality_columns": high_card_cols, "threshold": self.threshold}
        if high_card_cols:
            return CheckResult(
                name=self.name,
                level="WARNING",
                passed=False,
                message=f"{len(high_card_cols)} categorical column(s) have cardinality > {self.threshold}: {list(high_card_cols.keys())}",
                details=details,
            )
        return CheckResult(
            name=self.name,
            level="INFO",
            passed=True,
            message=f"All categorical features have cardinality <= {self.threshold}.",
            details=details,
        )


class TargetDistributionCheck(Check):
    """Analyze target variable distribution and flag severe class imbalance."""

    def __init__(self, target: pd.Series | None = None, min_class_pct: float = 0.05) -> None:
        self.target = target
        self.min_class_pct = min_class_pct

    @property
    def name(self) -> str:
        return "target_distribution"

    def run(self, df: pd.DataFrame) -> CheckResult:
        if self.target is None:
            return CheckResult(
                name=self.name,
                level="INFO",
                passed=True,
                message="Target was not provided for QC check.",
            )

        y = self.target.dropna()
        n_unique = y.nunique()

        # Binary or Multiclass
        if n_unique <= 20:
            val_counts = y.value_counts(normalize=True).to_dict()
            min_pct = min(val_counts.values()) if val_counts else 0.0
            details = {"class_proportions": {str(k): float(v) for k, v in val_counts.items()}}

            if min_pct < self.min_class_pct:
                return CheckResult(
                    name=self.name,
                    level="WARNING",
                    passed=False,
                    message=f"Severe class imbalance: minority class is only {min_pct*100:.1f}% (threshold {self.min_class_pct*100}%).",
                    details=details,
                )
            return CheckResult(
                name=self.name,
                level="INFO",
                passed=True,
                message="Target class distribution is well balanced.",
                details=details,
            )
        else:
            # Continuous regression target
            details = {
                "mean": float(y.mean()),
                "std": float(y.std()),
                "min": float(y.min()),
                "max": float(y.max()),
                "skew": float(y.skew()),
            }
            return CheckResult(
                name=self.name,
                level="INFO",
                passed=True,
                message="Target regression distribution summarized.",
                details=details,
            )


class LeakageCheck(Check):
    """Detect potential data leakage: features near-perfectly correlated with target."""

    def __init__(self, target: pd.Series | None = None, threshold: float = 0.99) -> None:
        self.target = target
        self.threshold = threshold

    @property
    def name(self) -> str:
        return "data_leakage"

    def run(self, df: pd.DataFrame) -> CheckResult:
        if self.target is None:
            return CheckResult(name=self.name, level="INFO", passed=True, message="No target provided.")

        y = pd.to_numeric(self.target, errors="coerce")
        if y.isnull().all():
            return CheckResult(name=self.name, level="INFO", passed=True, message="Target is non-numeric.")

        col_types = infer_column_types(df)
        leaked_cols: dict[str, float] = {}

        for col, ctype in col_types.items():
            if ctype == ColumnType.NUMERIC:
                s = pd.to_numeric(df[col], errors="coerce")
                corr = abs(s.corr(y, method="spearman"))
                if not np.isnan(corr) and corr >= self.threshold:
                    leaked_cols[col] = float(corr)

        details = {"leaked_columns": leaked_cols, "threshold": self.threshold}
        if leaked_cols:
            return CheckResult(
                name=self.name,
                level="ERROR",
                passed=False,
                message=f"CRITICAL: Potential target leakage detected in {list(leaked_cols.keys())} with Spearman correlation >= {self.threshold}.",
                details=details,
            )
        return CheckResult(
            name=self.name,
            level="INFO",
            passed=True,
            message="No obvious target leakage detected among numeric features.",
            details=details,
        )
