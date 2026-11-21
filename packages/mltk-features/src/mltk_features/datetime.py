"""Datetime feature extraction transformer."""

from __future__ import annotations

from typing import Any

import pandas as pd
from mltk_core.base import BaseFrameTransformer, ColumnType, infer_column_types


class DatetimeFeatures(BaseFrameTransformer):
    """
    Extract calendar and cyclical features from datetime columns.

    Skips gracefully if no datetime columns are detected.
    """

    def __init__(
        self,
        columns: list[str] | str = "auto",
        features: list[str] | None = None,
        drop_original: bool = False,
    ) -> None:
        super().__init__()
        self.columns = columns
        self.features = features
        self.drop_original = drop_original
        self.detected_cols_: list[str] = []

    def _fit(self, X: pd.DataFrame, y: Any = None) -> BaseFrameTransformer:
        if isinstance(self.columns, list):
            self.detected_cols_ = [c for c in self.columns if c in X.columns]
        else:
            col_types = infer_column_types(X)
            self.detected_cols_ = [c for c, t in col_types.items() if t == ColumnType.DATETIME]

            # Attempt auto-detection for string columns that parse cleanly as datetime
            if not self.detected_cols_:
                for col in X.select_dtypes(include=["object"]).columns:
                    sample = X[col].dropna().head(20)
                    if not sample.empty:
                        try:
                            pd.to_datetime(sample)
                            self.detected_cols_.append(col)
                        except Exception:
                            pass

        return self

    def _transform(self, X: pd.DataFrame) -> pd.DataFrame:
        if not self.detected_cols_:
            return X.copy()

        active_features = self.features or ["year", "month", "day", "dayofweek", "is_weekend"]
        out = X.copy()
        for col in self.detected_cols_:
            if col not in out.columns:
                continue

            dt_series = pd.to_datetime(out[col], errors="coerce")

            for feat in active_features:
                col_name = f"{col}_{feat}"
                if feat == "year":
                    out[col_name] = dt_series.dt.year.fillna(-1).astype(int)
                elif feat == "month":
                    out[col_name] = dt_series.dt.month.fillna(-1).astype(int)
                elif feat == "day":
                    out[col_name] = dt_series.dt.day.fillna(-1).astype(int)
                elif feat == "dayofweek":
                    out[col_name] = dt_series.dt.dayofweek.fillna(-1).astype(int)
                elif feat == "hour":
                    out[col_name] = dt_series.dt.hour.fillna(-1).astype(int)
                elif feat == "is_weekend":
                    out[col_name] = (dt_series.dt.dayofweek >= 5).astype(int)

            if self.drop_original:
                out.drop(columns=[col], inplace=True)

        return out
