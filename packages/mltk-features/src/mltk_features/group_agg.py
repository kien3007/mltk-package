"""Group aggregation feature transformer."""

from __future__ import annotations

from typing import Any
import pandas as pd

from mltk_core.base import BaseFrameTransformer


class GroupAggFeatures(BaseFrameTransformer):
    """
    Compute grouped aggregations (mean, std, min, max, count) on training set
    and join them onto current datasets.
    """

    def __init__(
        self,
        groupby: list[str] | str | None = None,
        agg_cols: list[str] | None = None,
        aggs: list[str] | None = None,
    ) -> None:
        super().__init__()
        self.groupby = groupby
        self.agg_cols = agg_cols
        self.aggs = aggs
        self.mappings_: dict[str, pd.DataFrame] = {}

    def _fit(self, X: pd.DataFrame, y: Any = None) -> BaseFrameTransformer:
        if self.groupby is None:
            group_list = []
        else:
            group_list = [self.groupby] if isinstance(self.groupby, str) else self.groupby

        agg_list = self.agg_cols or []
        aggs_list = self.aggs or ["mean", "std"]

        valid_group = [c for c in group_list if c in X.columns]
        valid_agg = [c for c in agg_list if c in X.columns]

        if not valid_group or not valid_agg:
            return self

        for agg_col in valid_agg:
            for agg_func in aggs_list:
                agg_df = X.groupby(valid_group)[agg_col].agg(agg_func).reset_index()
                feat_name = f"{'_'.join(valid_group)}_grp_{agg_col}_{agg_func}"
                agg_df.rename(columns={agg_col: feat_name}, inplace=True)
                key = f"{agg_col}_{agg_func}"
                self.mappings_[key] = agg_df

        return self

    def _transform(self, X: pd.DataFrame) -> pd.DataFrame:
        out = X.copy()
        if self.groupby is None:
            group_list = []
        else:
            group_list = [self.groupby] if isinstance(self.groupby, str) else self.groupby

        valid_group = [c for c in group_list if c in out.columns]
        if not valid_group or not self.mappings_:
            return out

        for key, agg_df in self.mappings_.items():
            out = out.merge(agg_df, on=valid_group, how="left")

        return out
