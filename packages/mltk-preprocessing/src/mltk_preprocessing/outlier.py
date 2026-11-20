"""Outlier clipping transformer."""

from __future__ import annotations

from typing import Any
import pandas as pd

from mltk_core.base import BaseFrameTransformer, ColumnType, infer_column_types


class OutlierClipper(BaseFrameTransformer):
    """
    Clip extreme numerical outliers based on training set quantiles.
    """

    def __init__(
        self,
        lower_quantile: float = 0.01,
        upper_quantile: float = 0.99,
        column_overrides: dict[str, tuple[float, float]] | None = None,
    ) -> None:
        super().__init__()
        self.lower_quantile = lower_quantile
        self.upper_quantile = upper_quantile
        self.column_overrides = column_overrides or {}
        self.bounds_: dict[str, tuple[float, float]] = {}

    def _fit(self, X: pd.DataFrame, y: Any = None) -> BaseFrameTransformer:
        col_types = infer_column_types(X)
        self.bounds_ = {}

        for col, ctype in col_types.items():
            if ctype == ColumnType.NUMERIC:
                if col in self.column_overrides:
                    l_q, u_q = self.column_overrides[col]
                else:
                    l_q, u_q = self.lower_quantile, self.upper_quantile

                low_b = float(X[col].quantile(l_q))
                high_b = float(X[col].quantile(u_q))
                self.bounds_[col] = (low_b, high_b)

        return self

    def _transform(self, X: pd.DataFrame) -> pd.DataFrame:
        out = X.copy()
        for col, (low_b, high_b) in self.bounds_.items():
            if col in out.columns:
                out[col] = out[col].clip(lower=low_b, upper=high_b)
        return out
