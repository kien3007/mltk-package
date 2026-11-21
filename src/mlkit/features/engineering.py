"""Feature engineering transformers for mlkit."""

from __future__ import annotations

from typing import Any

import numpy as np
import pandas as pd

from mlkit.core.config import TaskConfig
from mlkit.core.registry import register
from mlkit.core.step import BaseStep


@register("features.datetime")
class DatetimeFeatures(BaseStep):
    """Extract date and time components from datetime columns."""

    def __init__(
        self,
        columns: list[str] | None = None,
        extract_parts: list[str] | None = None,
        drop_original: bool = True,
    ) -> None:
        super().__init__()
        self.columns = columns
        self.extract_parts = extract_parts or ["year", "month", "day", "dayofweek", "is_weekend"]
        self.drop_original = drop_original
        self.detected_cols_: list[str] = []

    def _fit(self, X: pd.DataFrame, y: Any = None, cfg: TaskConfig | None = None) -> DatetimeFeatures:
        if self.columns:
            self.detected_cols_ = [c for c in self.columns if c in X.columns]
        else:
            self.detected_cols_ = list(X.select_dtypes(include=["datetime64"]).columns)
            # Check string/object candidates
            for c in X.select_dtypes(include=["object"]).columns:
                if "date" in c.lower() or "time" in c.lower():
                    try:
                        pd.to_datetime(X[c].dropna().head(10))
                        self.detected_cols_.append(c)
                    except Exception:
                        pass

        self.report_.add(
            "datetime_features",
            "ok",
            f"Detected {len(self.detected_cols_)} datetime columns: {self.detected_cols_}",
        )
        return self

    def _transform(self, X: pd.DataFrame) -> pd.DataFrame:
        out = X.copy()
        for col in self.detected_cols_:
            if col not in out.columns:
                continue
            s = pd.to_datetime(out[col], errors="coerce")
            if "year" in self.extract_parts:
                out[f"{col}_year"] = s.dt.year
            if "month" in self.extract_parts:
                out[f"{col}_month"] = s.dt.month
            if "day" in self.extract_parts:
                out[f"{col}_day"] = s.dt.day
            if "dayofweek" in self.extract_parts:
                out[f"{col}_dayofweek"] = s.dt.dayofweek
            if "hour" in self.extract_parts:
                out[f"{col}_hour"] = s.dt.hour
            if "is_weekend" in self.extract_parts:
                out[f"{col}_is_weekend"] = s.dt.dayofweek.isin([5, 6]).astype(int)

            if self.drop_original:
                out.drop(columns=[col], inplace=True)
        return out


@register("features.ratio")
class RatioFeatures(BaseStep):
    """Generate ratio and difference features between pairs of numeric columns."""

    def __init__(
        self,
        pairs: list[tuple[str, str]] | None = None,
        create_diff: bool = True,
        eps: float = 1e-6,
    ) -> None:
        super().__init__()
        self.pairs = pairs or []
        self.create_diff = create_diff
        self.eps = eps

    def _fit(self, X: pd.DataFrame, y: Any = None, cfg: TaskConfig | None = None) -> RatioFeatures:
        valid_pairs = [(a, b) for a, b in self.pairs if a in X.columns and b in X.columns]
        self.report_.add(
            "ratio_features",
            "ok",
            f"Configured {len(valid_pairs)} feature pairs for ratio/diff computation.",
        )
        return self

    def _transform(self, X: pd.DataFrame) -> pd.DataFrame:
        out = X.copy()
        for col_a, col_b in self.pairs:
            if col_a in out.columns and col_b in out.columns:
                a_vals = pd.to_numeric(out[col_a], errors="coerce")
                b_vals = pd.to_numeric(out[col_b], errors="coerce")
                out[f"{col_a}_div_{col_b}"] = a_vals / (b_vals + self.eps)
                if self.create_diff:
                    out[f"{col_a}_sub_{col_b}"] = a_vals - b_vals
        return out


@register("features.group_agg")
class GroupAggFeatures(BaseStep):
    """Aggregate numeric features grouped by categorical keys."""

    def __init__(
        self,
        group_cols: list[str] | None = None,
        target_cols: list[str] | None = None,
        aggregations: list[str] | None = None,
    ) -> None:
        super().__init__()
        self.group_cols = group_cols or []
        self.target_cols = target_cols or []
        self.aggregations = aggregations or ["mean", "std"]
        self.lookup_tables_: dict[str, pd.DataFrame] = {}

    def _fit(self, X: pd.DataFrame, y: Any = None, cfg: TaskConfig | None = None) -> GroupAggFeatures:
        self.lookup_tables_ = {}
        for g in self.group_cols:
            if g in X.columns:
                num_targets = [t for t in self.target_cols if t in X.columns and t != g]
                if num_targets:
                    agg_df = X.groupby(g, observed=True)[num_targets].agg(self.aggregations)
                    # Flatten multi-index
                    agg_df.columns = [f"{g}_{col}_{stat}" for col, stat in agg_df.columns]
                    self.lookup_tables_[g] = agg_df

        self.report_.add(
            "group_agg",
            "ok",
            f"Built group aggregation lookups for groups: {list(self.lookup_tables_.keys())}",
        )
        return self

    def _transform(self, X: pd.DataFrame) -> pd.DataFrame:
        out = X.copy()
        for g, agg_df in self.lookup_tables_.items():
            if g in out.columns:
                out = out.merge(agg_df, on=g, how="left")
        return out


@register("features.binning")
class BinningFeatures(BaseStep):
    """Discretize continuous numeric features into quantile bins."""

    def __init__(self, columns: list[str] | None = None, n_bins: int = 5) -> None:
        super().__init__()
        self.columns = columns
        self.n_bins = n_bins
        self.bin_edges_: dict[str, np.ndarray] = {}

    def _fit(self, X: pd.DataFrame, y: Any = None, cfg: TaskConfig | None = None) -> BinningFeatures:
        target_cols = self.columns or list(X.select_dtypes(include=[np.number]).columns)
        self.bin_edges_ = {}
        for c in target_cols:
            if c in X.columns:
                s = X[c].dropna()
                if not s.empty and s.nunique() > self.n_bins:
                    quantiles = np.linspace(0, 1, self.n_bins + 1)
                    edges = np.percentile(s, quantiles * 100)
                    edges[0] = -np.inf
                    edges[-1] = np.inf
                    self.bin_edges_[c] = np.unique(edges)

        self.report_.add(
            "binning_features",
            "ok",
            f"Computed bin edges for {len(self.bin_edges_)} continuous columns.",
        )
        return self

    def _transform(self, X: pd.DataFrame) -> pd.DataFrame:
        out = X.copy()
        for c, edges in self.bin_edges_.items():
            if c in out.columns and len(edges) > 1:
                out[f"{c}_bin"] = pd.cut(out[c], bins=edges, labels=False, include_lowest=True).fillna(-1).astype(int)
        return out
