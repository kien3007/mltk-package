"""Quality Control and EDA checks with gatekeeper enforcement."""

from __future__ import annotations

from typing import Any

import numpy as np
import pandas as pd
from scipy.stats import spearmanr

from mlkit.core.config import TaskConfig
from mlkit.core.registry import register
from mlkit.core.report import Severity
from mlkit.core.step import BaseStep


@register("qc.quality_check")
class QualityCheckStep(BaseStep):
    """
    Automated data quality inspection running Pandera-style and statistical checks.
    """

    def __init__(
        self,
        max_missing_rate: float = 0.5,
        max_cardinality: int = 100,
        leakage_threshold: float = 0.99,
        enforce_gate: bool = True,
    ) -> None:
        super().__init__()
        self.max_missing_rate = max_missing_rate
        self.max_cardinality = max_cardinality
        self.leakage_threshold = leakage_threshold
        self.enforce_gate = enforce_gate
        self.data_quality_score_: float = 100.0

    def _fit(
        self,
        X: pd.DataFrame,
        y: Any = None,
        cfg: TaskConfig | None = None,
    ) -> QualityCheckStep:
        self.report_.title = "Quality Control & EDA Report"

        # 1. Empty DataFrame Check
        if X.empty:
            self.report_.add("empty_data", "error", "Input DataFrame is completely empty.")
            if self.enforce_gate and (cfg is None or cfg.quality_gate):
                self.report_.raise_if_failed()
            return self
        else:
            self.report_.add(
                "data_shape",
                "ok",
                f"Dataset contains {len(X)} rows and {len(X.columns)} columns.",
                {"rows": len(X), "cols": len(X.columns)},
            )

        # 2. Duplicate rows check
        dup_count = int(X.duplicated().sum())
        dup_pct = dup_count / len(X)
        if dup_pct > 0.05:
            self.report_.add(
                "duplicates",
                "warn",
                f"{dup_count} duplicate rows ({dup_pct * 100:.1f}%) detected.",
            )
        else:
            self.report_.add(
                "duplicates", "ok", f"Duplicate rows check passed ({dup_count} duplicates)."
            )

        # 3. Missing rate check
        missing_s = X.isna().mean()
        high_missing = missing_s[missing_s > self.max_missing_rate]
        if not high_missing.empty:
            cols = list(high_missing.index)
            self.report_.add(
                "missing_values",
                "warn",
                f"{len(cols)} columns exceed missing threshold ({self.max_missing_rate * 100:.0f}%): {cols[:5]}",
                {"columns": cols},
            )
        else:
            self.report_.add("missing_values", "ok", "Missing values within acceptable thresholds.")

        # 4. Constant columns check
        constant_cols = [c for c in X.columns if X[c].nunique(dropna=False) <= 1]
        if constant_cols:
            self.report_.add(
                "constant_columns",
                "warn",
                f"{len(constant_cols)} constant columns detected: {constant_cols}",
                {"columns": constant_cols},
            )
        else:
            self.report_.add("constant_columns", "ok", "No constant columns detected.")

        # 5. High cardinality check
        cat_cols = X.select_dtypes(include=["object", "category"]).columns
        high_card_cols = [c for c in cat_cols if X[c].nunique() > self.max_cardinality]
        if high_card_cols:
            self.report_.add(
                "high_cardinality",
                "warn",
                f"{len(high_card_cols)} categorical columns have cardinality > {self.max_cardinality}: {high_card_cols}",
            )
        else:
            self.report_.add(
                "high_cardinality", "ok", "All categorical features have healthy cardinality."
            )

        # 6. Target Checks (if y provided)
        if y is not None:
            y_s = pd.Series(y).dropna()
            if y_s.empty:
                self.report_.add(
                    "target_distribution", "error", "Target variable is completely empty or null."
                )
            else:
                n_unique_y = y_s.nunique()
                if n_unique_y <= 1:
                    self.report_.add(
                        "target_distribution",
                        "error",
                        f"Target contains only {n_unique_y} unique class. Cannot train a supervised model.",
                    )
                else:
                    self.report_.add(
                        "target_distribution",
                        "ok",
                        f"Target variable has {n_unique_y} unique values.",
                    )

            # 7. Data Leakage Check
            num_cols = X.select_dtypes(include=[np.number]).columns
            leakage_cols = []
            if len(y_s) == len(X) and pd.api.types.is_numeric_dtype(y_s):
                for c in num_cols:
                    valid_idx = ~(X[c].isna() | y_s.isna())
                    if valid_idx.sum() > 20:
                        corr, _ = spearmanr(X.loc[valid_idx, c], y_s.loc[valid_idx])
                        if not np.isnan(corr) and abs(corr) >= self.leakage_threshold:
                            leakage_cols.append((c, float(corr)))

            if leakage_cols:
                self.report_.add(
                    "data_leakage",
                    "error",
                    f"Suspicious target leakage detected (correlation >= {self.leakage_threshold}): {leakage_cols}",
                    {"leakage_columns": leakage_cols},
                )
            else:
                self.report_.add("data_leakage", "ok", "No obvious target leakage detected.")

        # 8. Compute Overall Data Quality Score (0 - 100)
        score = 100.0
        mean_missing = float(X.isna().mean().mean())
        score -= min(35.0, mean_missing * 70.0)
        score -= min(20.0, dup_pct * 100.0)
        if len(X.columns) > 0:
            score -= min(20.0, (len(constant_cols) / len(X.columns)) * 50.0)
        if self.report_.has_errors:
            score -= 25.0
        self.data_quality_score_ = float(max(0.0, min(100.0, round(score, 1))))
        dqs_status: Severity = (
            "ok"
            if self.data_quality_score_ >= 75
            else ("warn" if self.data_quality_score_ >= 50 else "error")
        )
        self.report_.add(
            "data_quality_score",
            dqs_status,
            f"Overall Data Quality Score: {self.data_quality_score_} / 100",
            {"score": self.data_quality_score_},
        )

        # Gate enforcement
        if self.enforce_gate and (cfg is None or cfg.quality_gate) and self.report_.has_errors:
            self.report_.raise_if_failed(gate_name="QC Quality Gate")

        return self

    def _transform(self, X: pd.DataFrame) -> pd.DataFrame:
        # QC step is an inspection pass; data passes through unchanged
        return X


def run_qc(df: pd.DataFrame, y: Any = None, cfg: TaskConfig | None = None) -> QualityCheckStep:
    """Convenience runner for QC inspection."""
    step = QualityCheckStep()
    step.fit(df, y=y, cfg=cfg)
    return step
