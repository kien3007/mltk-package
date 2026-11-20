"""Infrequent category grouping transformer."""

from __future__ import annotations

from typing import Any
import pandas as pd

from mltk_core.base import BaseFrameTransformer, ColumnType, infer_column_types


class RareCategoryGrouper(BaseFrameTransformer):
    """
    Group rare categories with relative frequency below threshold into a rare label.
    """

    def __init__(self, threshold: float = 0.02, rare_label: str = "__rare__") -> None:
        super().__init__()
        self.threshold = threshold
        self.rare_label = rare_label
        self.frequent_categories_: dict[str, set[Any]] = {}

    def _fit(self, X: pd.DataFrame, y: Any = None) -> BaseFrameTransformer:
        col_types = infer_column_types(X)
        self.frequent_categories_ = {}

        for col, ctype in col_types.items():
            if ctype == ColumnType.CATEGORICAL:
                freq = X[col].value_counts(normalize=True, dropna=True)
                freq_cats = set(freq[freq >= self.threshold].index)
                self.frequent_categories_[col] = freq_cats

        return self

    def _transform(self, X: pd.DataFrame) -> pd.DataFrame:
        out = X.copy()
        for col, freq_cats in self.frequent_categories_.items():
            if col in out.columns:
                # Keep frequent categories; replace everything else with rare_label
                out[col] = out[col].apply(
                    lambda v: v if pd.isna(v) or v in freq_cats else self.rare_label
                )
        return out
