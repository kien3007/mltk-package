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

    def _fit(
        self, X: pd.DataFrame, y: Any = None, cfg: TaskConfig | None = None
    ) -> DatetimeFeatures:
        if self.columns:
            self.detected_cols_ = [c for c in self.columns if c in X.columns]
        else:
            self.detected_cols_ = list(
                X.select_dtypes(include=["datetime64", "datetimetz"]).columns
            )
            time_col = cfg.time_col if cfg else None
            # Check string/object/category candidates (by name hint or configured time_col)
            for c in X.select_dtypes(include=["object", "category", "string"]).columns:
                name = str(c).lower()
                if c == time_col or "date" in name or "time" in name:
                    sample = X[c].dropna().astype(str).head(50)
                    if sample.empty:
                        continue
                    parsed = pd.to_datetime(sample, errors="coerce", format="mixed")
                    if parsed.notna().mean() >= 0.9:
                        self.detected_cols_.append(c)

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
        self.pairs = pairs  # stored as given (sklearn clone contract); defaults resolved at use
        self.create_diff = create_diff
        self.eps = eps

    def _fit(self, X: pd.DataFrame, y: Any = None, cfg: TaskConfig | None = None) -> RatioFeatures:
        valid_pairs = [(a, b) for a, b in (self.pairs or []) if a in X.columns and b in X.columns]
        self.report_.add(
            "ratio_features",
            "ok",
            f"Configured {len(valid_pairs)} feature pairs for ratio/diff computation.",
        )
        return self

    def _transform(self, X: pd.DataFrame) -> pd.DataFrame:
        out = X.copy()
        for col_a, col_b in self.pairs or []:
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
        self.group_cols = group_cols
        self.target_cols = target_cols
        self.aggregations = aggregations
        self.lookup_tables_: dict[str, pd.DataFrame] = {}

    def _fit(
        self, X: pd.DataFrame, y: Any = None, cfg: TaskConfig | None = None
    ) -> GroupAggFeatures:
        self.lookup_tables_ = {}
        self.lookup_tables_ = {}
        aggregations = self.aggregations or ["mean", "std"]
        for g in self.group_cols or []:
            if g in X.columns:
                num_targets = [t for t in self.target_cols or [] if t in X.columns and t != g]
                if num_targets:
                    agg_df = X.groupby(g, observed=True)[num_targets].agg(aggregations)
                    # Flatten multi-index
                    agg_df.columns = [
                        f"{g}_{col}_{stat}" for col, stat in agg_df.columns.to_flat_index()
                    ]
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

    def _fit(
        self, X: pd.DataFrame, y: Any = None, cfg: TaskConfig | None = None
    ) -> BinningFeatures:
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
                out[f"{c}_bin"] = (
                    pd.cut(out[c], bins=list(edges), labels=False, include_lowest=True)
                    .fillna(-1)
                    .astype(int)
                )
        return out


@register("features.multi_table")
class MultiTableAggregator(BaseStep):
    """
    Automated feature aggregation across relational tables (1-to-many joins).

    Useful for datasets where events/transactions/logs belong to a primary entity
    (e.g., account_id, customer_id, user_id).
    """

    def __init__(
        self,
        secondary_df: pd.DataFrame,
        join_key: str,
        numeric_aggs: list[str] | None = None,
        categorical_aggs: list[str] | None = None,
        prefix: str = "rel_",
        time_col: str | None = None,
        window_days: list[int] | None = None,
    ) -> None:
        super().__init__()
        self.secondary_df = secondary_df  # never mutated; not copied so clone() works
        self.join_key = join_key
        self.numeric_aggs = numeric_aggs
        self.categorical_aggs = categorical_aggs
        self.prefix = prefix
        self.time_col = time_col
        self.window_days = window_days
        self.agg_table_: pd.DataFrame = pd.DataFrame()

    def _fit(
        self, X: pd.DataFrame, y: Any = None, cfg: TaskConfig | None = None
    ) -> MultiTableAggregator:
        sec = self.secondary_df
        if self.join_key not in sec.columns:
            self.report_.add(
                "multi_table_agg",
                "warn",
                f"Join key '{self.join_key}' not found in secondary DataFrame.",
            )
            return self

        exclude_cols = {self.join_key}
        if self.time_col:
            exclude_cols.add(self.time_col)

        num_cols = [
            c for c in sec.select_dtypes(include=[np.number]).columns if c not in exclude_cols
        ]
        cat_cols = [
            c
            for c in sec.select_dtypes(include=["object", "category"]).columns
            if c not in exclude_cols
        ]

        agg_dict: dict[str, list[str]] = {}
        for c in num_cols:
            agg_dict[c] = self.numeric_aggs or ["mean", "sum", "std", "min", "max", "count"]
        for c in cat_cols:
            agg_dict[c] = self.categorical_aggs or ["nunique"]

        if not agg_dict:
            self.report_.add(
                "multi_table_agg", "warn", "No valid columns to aggregate in secondary table."
            )
            return self

        grouped = sec.groupby(self.join_key, observed=True).agg(agg_dict)
        grouped.columns = [
            f"{self.prefix}{col}_{stat}" for col, stat in grouped.columns.to_flat_index()
        ]

        if self.time_col and self.time_col in sec.columns and self.window_days:
            sec_time = pd.to_datetime(sec[self.time_col], errors="coerce")
            max_date = sec_time.max()
            if pd.notna(max_date):
                for w in self.window_days:
                    cutoff = max_date - pd.Timedelta(days=w)
                    mask = sec_time >= cutoff
                    sub = sec[mask]
                    if not sub.empty:
                        w_grouped = sub.groupby(self.join_key, observed=True)[num_cols].agg(
                            ["sum", "mean", "count"]
                        )
                        w_grouped.columns = [
                            f"{self.prefix}{col}_{stat}_last_{w}d"
                            for col, stat in w_grouped.columns
                        ]
                        grouped = grouped.join(w_grouped, how="left")

        self.agg_table_ = grouped.reset_index()
        self.report_.add(
            "multi_table_agg",
            "ok",
            f"Built {len(grouped.columns)} aggregated features from secondary table keyed on '{self.join_key}'.",
            {"feature_count": len(grouped.columns)},
        )
        return self

    def _transform(self, X: pd.DataFrame) -> pd.DataFrame:
        if self.agg_table_.empty or self.join_key not in X.columns:
            return X
        out = X.merge(self.agg_table_, on=self.join_key, how="left")
        fill_zero_cols = [
            c for c in out.columns if any(c.endswith(s) for s in ["_count", "_sum", "_nunique"])
        ]
        if fill_zero_cols:
            out[fill_zero_cols] = out[fill_zero_cols].fillna(0)
        return out
