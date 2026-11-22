"""Preprocessing transformers with pandas DataFrame preservation and leakage protection."""

from __future__ import annotations

from typing import Any, Literal

import numpy as np
import pandas as pd
from sklearn.preprocessing import MinMaxScaler, RobustScaler, StandardScaler

from mlkit.core.config import TaskConfig
from mlkit.core.registry import register
from mlkit.core.step import BaseStep


@register("preprocessing.type_caster")
class TypeCaster(BaseStep):
    """Cast column types, drop ID column, and normalize strings."""

    def __init__(
        self,
        dtypes: dict[str, str] | None = None,
        drop_id: bool = True,
        drop_cols: list[str] | None = None,
        drop_group_col: bool = True,
        drop_time_col: bool = True,
    ) -> None:
        super().__init__()
        self.dtypes = dtypes
        self.drop_id = drop_id
        self.drop_cols = drop_cols
        self.drop_group_col = drop_group_col
        self.drop_time_col = drop_time_col
        self.inferred_dtypes_: dict[str, str] = {}
        self.dropped_cols_: list[str] = []

    def _fit(self, X: pd.DataFrame, y: Any = None, cfg: TaskConfig | None = None) -> TypeCaster:
        self.inferred_dtypes_ = {}
        self.dropped_cols_ = [c for c in (self.drop_cols or []) if c in X.columns]
        id_col = cfg.id_col if cfg else None
        if self.drop_id and id_col and id_col in X.columns and id_col not in self.dropped_cols_:
            self.dropped_cols_.append(id_col)
        group_col = cfg.group_col if cfg else None
        if (
            self.drop_group_col
            and group_col
            and group_col in X.columns
            and group_col not in self.dropped_cols_
        ):
            # Group keys are split keys, not features: keeping them leaks identity across folds.
            self.dropped_cols_.append(group_col)
        time_col = cfg.time_col if cfg else None
        if (
            self.drop_time_col
            and time_col
            and time_col in X.columns
            and time_col not in self.dropped_cols_
        ):
            # An absolute time index only encodes "when": it cannot extrapolate to future
            # periods and lets the model memorise period effects. Derive parts upstream
            # (DatetimeFeatures) or set drop_time_col=False to keep it deliberately.
            self.dropped_cols_.append(time_col)

        # Detect prospective ID columns by name or uniqueness
        for c in X.columns:
            looks_like_id = str(c).lower() == "id" or (
                (X[c].dtype == object or pd.api.types.is_string_dtype(X[c]))
                and X[c].nunique() == len(X)
                and "id" in str(c).lower()
            )
            if self.drop_id and c not in self.dropped_cols_ and looks_like_id:
                self.dropped_cols_.append(c)

        for col in X.columns:
            if col in self.dropped_cols_:
                continue
            if self.dtypes and col in self.dtypes:
                self.inferred_dtypes_[col] = self.dtypes[col]
            elif pd.api.types.is_numeric_dtype(X[col]):
                self.inferred_dtypes_[col] = "float64"
            elif pd.api.types.is_datetime64_any_dtype(X[col]):
                self.inferred_dtypes_[col] = "datetime64[ns]"
            else:
                self.inferred_dtypes_[col] = "category"

        self.report_.add(
            "type_caster",
            "ok",
            f"Configured dtypes for {len(self.inferred_dtypes_)} columns, dropped ID cols: {self.dropped_cols_}",
        )
        return self

    def _transform(self, X: pd.DataFrame) -> pd.DataFrame:
        out = X.drop(columns=[c for c in self.dropped_cols_ if c in X.columns], errors="ignore")
        for col, target_dtype in self.inferred_dtypes_.items():
            if col in out.columns:
                try:
                    if target_dtype == "category":
                        # keep missing values as NaN (not the string "nan") for the imputer
                        col_s = out[col]
                        out[col] = col_s.where(col_s.isna(), col_s.astype(str)).astype("category")
                    else:
                        out[col] = out[col].astype(target_dtype)
                except Exception:
                    pass
        return out


@register("preprocessing.imputer")
class MissingImputer(BaseStep):
    """
    Impute numeric missing with median, categorical with mode, with missing indicators.
    """

    def __init__(
        self,
        numeric_strategy: Literal["median", "mean", "zero"] = "median",
        categorical_strategy: Literal["mode", "missing_value"] = "mode",
        add_indicator: bool = False,
    ) -> None:
        super().__init__()
        self.numeric_strategy = numeric_strategy
        self.categorical_strategy = categorical_strategy
        self.add_indicator = add_indicator

        self.fill_values_: dict[str, Any] = {}
        self.numeric_cols_: list[str] = []
        self.categorical_cols_: list[str] = []
        self.indicator_cols_: list[str] = []

    def _fit(self, X: pd.DataFrame, y: Any = None, cfg: TaskConfig | None = None) -> MissingImputer:
        self.fill_values_ = {}
        self.indicator_cols_ = []
        self.numeric_cols_ = list(X.select_dtypes(include=[np.number]).columns)
        self.categorical_cols_ = [c for c in X.columns if c not in self.numeric_cols_]

        for col in self.numeric_cols_:
            s = X[col].dropna()
            if self.add_indicator and X[col].isna().any():
                self.indicator_cols_.append(col)

            if s.empty:
                self.fill_values_[col] = 0.0
            elif self.numeric_strategy == "median":
                self.fill_values_[col] = float(s.median())
            elif self.numeric_strategy == "mean":
                self.fill_values_[col] = float(s.mean())
            else:
                self.fill_values_[col] = 0.0

        for col in self.categorical_cols_:
            s = X[col].dropna()
            if self.add_indicator and X[col].isna().any():
                self.indicator_cols_.append(col)

            if s.empty:
                self.fill_values_[col] = "__missing__"
            elif self.categorical_strategy == "mode":
                mode_val = s.mode()
                self.fill_values_[col] = mode_val.iloc[0] if not mode_val.empty else "__missing__"
            else:
                self.fill_values_[col] = "__missing__"

        self.report_.add(
            "missing_imputer",
            "ok",
            f"Computed impute statistics for {len(self.fill_values_)} columns (indicators: {len(self.indicator_cols_)}).",
        )
        return self

    def _transform(self, X: pd.DataFrame) -> pd.DataFrame:
        out = X.copy()
        if self.add_indicator:
            for col in self.indicator_cols_:
                if col in out.columns:
                    out[f"{col}_is_missing"] = out[col].isna().astype(int)

        for col, fill_val in self.fill_values_.items():
            if col in out.columns:
                if isinstance(out[col].dtype, pd.CategoricalDtype):
                    # the fill value may not be a category of this batch (e.g. all-unseen values)
                    out[col] = out[col].astype(object).fillna(fill_val).astype("category")
                else:
                    out[col] = out[col].fillna(fill_val)
        return out


@register("preprocessing.rare_grouper")
class RareCategoryGrouper(BaseStep):
    """Group rare categories (frequency < min_freq) into '__rare__'."""

    def __init__(self, min_freq: float = 0.01, rare_label: str = "__rare__") -> None:
        super().__init__()
        self.min_freq = min_freq
        self.rare_label = rare_label
        self.frequent_categories_: dict[str, set[str]] = {}

    def _fit(
        self, X: pd.DataFrame, y: Any = None, cfg: TaskConfig | None = None
    ) -> RareCategoryGrouper:
        cat_cols = X.select_dtypes(include=["object", "category"]).columns
        self.frequent_categories_ = {}

        for col in cat_cols:
            s = X[col].dropna().astype(str)
            freqs = s.value_counts(normalize=True)
            frequent = set(freqs[freqs >= self.min_freq].index)
            self.frequent_categories_[col] = frequent

        self.report_.add(
            "rare_grouper",
            "ok",
            f"Fitted rare grouping on {len(self.frequent_categories_)} categorical columns.",
        )
        return self

    def _transform(self, X: pd.DataFrame) -> pd.DataFrame:
        out = X.copy()
        for col, frequent in self.frequent_categories_.items():
            if col in out.columns:
                values = out[col].astype(str)
                out[col] = values.where(values.isin(list(frequent)), self.rare_label)
        return out


@register("preprocessing.categorical_encoder")
class CategoricalEncoder(BaseStep):
    """
    Adaptive categorical encoding:
    - Low cardinality (<= ``low_card_threshold``): One-Hot
    - Mid cardinality (<= ``high_card_threshold``): Ordinal
    - High cardinality: smoothed Target Encoding. During ``fit_transform`` the training
      rows receive **out-of-fold** encodings (K-fold), so a row never sees its own target.
      ``transform`` on new data uses the mapping fitted on the full training set.

    Target encoding is only used for regression and binary targets (ADR-004). For
    multiclass targets, high-cardinality columns fall back to ordinal encoding.
    """

    def __init__(
        self,
        low_card_threshold: int = 10,
        high_card_threshold: int = 50,
        cv: int = 5,
        smoothing: float = 10.0,
        random_state: int | None = 42,
    ) -> None:
        super().__init__()
        self.low_card_threshold = low_card_threshold
        self.high_card_threshold = high_card_threshold
        self.cv = cv
        self.smoothing = smoothing
        self.random_state = random_state

        self.encoding_plan_: dict[str, str] = {}
        self.onehot_categories_: dict[str, list[str]] = {}
        self.ordinal_maps_: dict[str, dict[str, int]] = {}  # category -> code
        self.target_maps_: dict[str, dict[str, float]] = {}
        self.global_target_mean_: float = 0.0
        self._y_numeric: np.ndarray | None = None

    @staticmethod
    def _target_to_numeric(y: Any) -> np.ndarray | None:
        """Map target to floats; returns None when target encoding is not meaningful."""
        y_arr = np.asarray(y)
        if pd.api.types.is_numeric_dtype(y_arr) and not pd.api.types.is_bool_dtype(y_arr):
            y_float = y_arr.astype(float)
            uniques = np.unique(y_float[~np.isnan(y_float)])
            is_integer_labels = np.all(np.mod(uniques, 1) == 0)
            if is_integer_labels and 2 < len(uniques) <= 20 and uniques.min() >= 0:
                # Looks like multiclass labels: a mean over class indices is meaningless.
                return None
            return y_float
        mask = ~pd.isna(y_arr)
        uniques = np.unique(y_arr[mask].astype(str))
        if len(uniques) != 2:
            return None
        out = np.full(len(y_arr), np.nan)
        out[mask] = (y_arr[mask].astype(str) == uniques[1]).astype(float)
        return out

    def _smoothed_map(self, keys: pd.Series, y_num: np.ndarray, prior: float) -> dict[str, float]:
        df = pd.DataFrame({"k": keys.to_numpy(), "_y": y_num}).dropna(subset=["_y"])
        stats = df.groupby("k")["_y"].agg(["count", "mean"])
        smoothed = (stats["count"] * stats["mean"] + self.smoothing * prior) / (
            stats["count"] + self.smoothing
        )
        return {str(k): float(v) for k, v in smoothed.items()}

    def _fit(
        self, X: pd.DataFrame, y: Any = None, cfg: TaskConfig | None = None
    ) -> CategoricalEncoder:
        cat_cols = list(X.select_dtypes(include=["object", "category", "string"]).columns)
        self.encoding_plan_ = {}
        self.onehot_categories_ = {}
        self.ordinal_maps_ = {}
        self.target_maps_ = {}

        y_num = self._target_to_numeric(y) if y is not None else None
        if cfg is not None and cfg.task_type == "regression" and y is not None:
            y_num = np.asarray(y, dtype=float)
        if y is not None and y_num is None:
            self.report_.add(
                "categorical_encoder_target",
                "warn",
                "Target is multiclass/non-numeric; target encoding disabled (ordinal fallback).",
            )
        self._y_numeric = y_num
        self.global_target_mean_ = float(np.nanmean(y_num)) if y_num is not None else 0.0

        for col in cat_cols:
            n_unique = X[col].nunique(dropna=False)
            if n_unique <= self.low_card_threshold:
                strategy = "onehot"
                self.onehot_categories_[col] = [str(v) for v in X[col].dropna().unique()]
            elif n_unique <= self.high_card_threshold or y_num is None:
                strategy = "ordinal"
                cats = [str(v) for v in X[col].dropna().unique()]
                self.ordinal_maps_[col] = {val: idx for idx, val in enumerate(cats)}
            else:
                strategy = "target"
                keys = X[col].astype(str)
                self.target_maps_[col] = self._smoothed_map(keys, y_num, self.global_target_mean_)
            self.encoding_plan_[col] = strategy

        self.report_.add(
            "categorical_encoder",
            "ok",
            f"Encoded {len(self.encoding_plan_)} columns: {self.encoding_plan_}",
        )
        return self

    def _fit_transform_train(self, X: pd.DataFrame, y: Any) -> pd.DataFrame | None:
        target_cols = [c for c, s in self.encoding_plan_.items() if s == "target"]
        y_num = self._y_numeric
        self._y_numeric = None  # never persist the training target inside a bundle
        if not target_cols or y_num is None:
            return None

        from sklearn.model_selection import KFold

        out = self.transform(X)
        n_splits = max(2, min(self.cv, len(X)))
        kf = KFold(n_splits=n_splits, shuffle=True, random_state=self.random_state)
        for col in target_cols:
            keys = X[col].astype(str).reset_index(drop=True)
            oof = np.full(len(X), np.nan)
            for tr_idx, val_idx in kf.split(keys):
                prior = float(np.nanmean(y_num[tr_idx]))
                fold_map = self._smoothed_map(keys.iloc[tr_idx], y_num[tr_idx], prior)
                oof[val_idx] = (
                    keys.iloc[val_idx].map(fold_map).astype(float).fillna(prior).to_numpy()
                )
            out[col] = oof
        self.report_.add(
            "target_encoding_oof",
            "ok",
            f"Applied {n_splits}-fold out-of-fold target encoding to training rows: {target_cols}",
        )
        return out

    def _transform(self, X: pd.DataFrame) -> pd.DataFrame:
        out = X.copy()
        for col, strategy in self.encoding_plan_.items():
            if col not in out.columns:
                continue

            if strategy == "onehot":
                col_str = out[col].astype(str)
                for cat in self.onehot_categories_.get(col, []):
                    out[f"{col}_{cat}"] = (col_str == cat).astype(int)
                out.drop(columns=[col], inplace=True)

            elif strategy == "ordinal":
                col_map = self.ordinal_maps_.get(col, {})
                out[col] = out[col].astype(str).map(col_map).fillna(-1).astype(int)

            elif strategy == "target":
                te_map = self.target_maps_.get(col, {})
                out[col] = (
                    out[col].astype(str).map(te_map).astype(float).fillna(self.global_target_mean_)
                )

        return out


@register("preprocessing.scaler")
class FeatureScaler(BaseStep):
    """Scale numeric features with DataFrame column preservation."""

    def __init__(self, method: Literal["standard", "robust", "minmax"] = "standard") -> None:
        super().__init__()
        self.method = method
        self.scaler_: Any = None
        self.numeric_cols_: list[str] = []

    def _fit(self, X: pd.DataFrame, y: Any = None, cfg: TaskConfig | None = None) -> FeatureScaler:
        self.numeric_cols_ = list(X.select_dtypes(include=[np.number]).columns)
        if self.method == "standard":
            self.scaler_ = StandardScaler()
        elif self.method == "robust":
            self.scaler_ = RobustScaler()
        else:
            self.scaler_ = MinMaxScaler()

        if self.numeric_cols_:
            self.scaler_.fit(X[self.numeric_cols_])

        self.report_.add(
            "feature_scaler",
            "ok",
            f"Fitted {self.method} scaler on {len(self.numeric_cols_)} columns.",
        )
        return self

    def _transform(self, X: pd.DataFrame) -> pd.DataFrame:
        out = X.copy()
        if self.scaler_ is not None and self.numeric_cols_:
            usable = [c for c in self.numeric_cols_ if c in out.columns]
            if usable:
                out[usable] = self.scaler_.transform(out[usable])
        return out


@register("preprocessing.outlier_clipper")
class OutlierClipper(BaseStep):
    """Clip numeric feature outliers using IQR boundaries."""

    def __init__(self, factor: float = 1.5) -> None:
        super().__init__()
        self.factor = factor
        self.bounds_: dict[str, tuple[float, float]] = {}

    def _fit(self, X: pd.DataFrame, y: Any = None, cfg: TaskConfig | None = None) -> OutlierClipper:
        num_cols = X.select_dtypes(include=[np.number]).columns
        self.bounds_ = {}
        for c in num_cols:
            s = X[c].dropna()
            if not s.empty:
                q25, q75 = float(np.percentile(s, 25)), float(np.percentile(s, 75))
                iqr = q75 - q25
                self.bounds_[c] = (q25 - self.factor * iqr, q75 + self.factor * iqr)

        self.report_.add(
            "outlier_clipper",
            "ok",
            f"Computed IQR bounds for {len(self.bounds_)} numeric columns.",
        )
        return self

    def _transform(self, X: pd.DataFrame) -> pd.DataFrame:
        out = X.copy()
        for c, (lower, upper) in self.bounds_.items():
            if c in out.columns:
                out[c] = out[c].clip(lower=lower, upper=upper)
        return out
