"""Preprocessing transformers with pandas DataFrame preservation and leakage protection."""

from __future__ import annotations

import re
from typing import Any, Literal

import numpy as np
import pandas as pd
from sklearn.preprocessing import MinMaxScaler, RobustScaler, StandardScaler

from mlkit.core.config import TaskConfig
from mlkit.core.registry import register
from mlkit.core.step import BaseStep
from mlkit.qc_eda.profile import (
    ProfileOptions,
    apply_coercion,
    coerce_boolean,
    coerce_datetime,
    deduplicate_columns,
    profile_dataframe,
)


@register("preprocessing.type_caster")
class TypeCaster(BaseStep):
    """Schema-driven type casting based on the shared column profile.

    At ``fit`` every column is profiled (:func:`mlkit.qc_eda.profile.infer_column`) and a cast
    plan is frozen, so ``transform`` applies the *same* conversions to new data through
    :func:`mlkit.qc_eda.profile.apply_coercion`:

    - numbers stored as text ("1,234", "$5", "12%", "3,5") and durations -> float
    - yes/no, true/false, y/n -> 0/1
    - date strings, timezone-aware dates and numeric date encodings -> naive datetime64
      (expanded later by ``DatetimeFeatures``)
    - disguised missing tokens ("n/a", "null", "-") and +/-inf -> NaN
    - categorical / codes with leading zeros -> pandas ``category``; text keeps ``object``
    - dropped: identifiers, empty, constant, ignored and unsupported columns, duplicated
      columns and duplicated column *names*, ``cfg.id_col``, ``cfg.group_col``, a numeric
      ``cfg.time_col`` (see the ``drop_*`` switches) and ``drop_cols``

    ``dtypes`` forces a pandas dtype per column ({"zip": "category"}); forced columns are never
    dropped. Semantic overrides go through ``profile_options=ProfileOptions(overrides=...)``.
    """

    def __init__(
        self,
        dtypes: dict[str, str] | None = None,
        drop_id: bool = True,
        drop_cols: list[str] | None = None,
        drop_group_col: bool = True,
        drop_time_col: bool = True,
        drop_constant: bool = True,
        drop_duplicate_columns: bool = True,
        profile_options: ProfileOptions | None = None,
    ) -> None:
        super().__init__()
        self.dtypes = dtypes
        self.drop_id = drop_id
        self.drop_cols = drop_cols
        self.drop_group_col = drop_group_col
        self.drop_time_col = drop_time_col
        self.drop_constant = drop_constant
        self.drop_duplicate_columns = drop_duplicate_columns
        self.profile_options = profile_options
        self.inferred_dtypes_: dict[str, str] = {}
        self.coercions_: dict[Any, str | None] = {}
        self.dropped_cols_: list[str] = []
        self.drop_reasons_: dict[str, str] = {}
        self.semantic_types_: dict[str, str] = {}

    def _drop(self, col: Any, reason: str) -> None:
        if col not in self.dropped_cols_:
            self.dropped_cols_.append(col)
            self.drop_reasons_[str(col)] = reason

    def _fit(self, X: pd.DataFrame, y: Any = None, cfg: TaskConfig | None = None) -> TypeCaster:
        self.inferred_dtypes_ = {}
        self.coercions_ = {}
        self.dropped_cols_ = []
        self.drop_reasons_ = {}
        self.semantic_types_ = {}
        forced = dict(self.dtypes or {})
        X, dup_names = deduplicate_columns(X)
        self.duplicate_names_ = dup_names

        profile = profile_dataframe(X, options=self.profile_options)
        for c in self.drop_cols or []:
            if c in X.columns:
                self._drop(c, "drop_cols")
        if cfg is not None:
            for flag, col, reason in (
                (self.drop_id, cfg.id_col, "id_col"),
                # group keys are split keys, not features: keeping them leaks identity across folds
                (self.drop_group_col, cfg.group_col, "group_col"),
            ):
                if flag and col and col in X.columns:
                    self._drop(col, reason)
            tc = cfg.time_col
            # A numeric time index only encodes "when" and cannot extrapolate to the future:
            # drop it. A real date is kept so DatetimeFeatures can derive calendar parts
            # (month, weekday, ...) and then drop the raw timestamp.
            if (
                self.drop_time_col
                and tc
                and tc in X.columns
                and profile.columns[str(tc)].semantic_type != "datetime"
            ):
                self._drop(tc, "time_col")
        if self.drop_duplicate_columns:
            for group in profile.duplicate_columns:
                for c in group[1:]:
                    if c not in forced:
                        self._drop(
                            next(k for k in X.columns if str(k) == c), f"duplicate of '{group[0]}'"
                        )

        for col in X.columns:
            p = profile.columns[str(col)]
            self.semantic_types_[str(col)] = p.semantic_type
            if col in self.dropped_cols_:
                continue
            if col in forced:
                self.inferred_dtypes_[col] = forced[col]
                continue
            kind = p.semantic_type
            if kind == "identifier" and self.drop_id:
                self._drop(col, "identifier")
            elif kind in ("empty", "ignored", "unsupported"):
                self._drop(
                    col, kind if kind != "unsupported" else "; ".join(p.notes) or "unsupported"
                )
            elif kind == "constant" and self.drop_constant:
                self._drop(col, "constant")
            elif kind == "datetime":
                self.inferred_dtypes_[col] = "datetime64[ns]"
            elif kind == "boolean":
                self.inferred_dtypes_[col] = "boolean"
            elif kind == "text":
                self.inferred_dtypes_[col] = "text"
            elif p.coercion in ("numeric", "timedelta") or (
                kind in ("numeric", "binary") and pd.api.types.is_numeric_dtype(X[col])
            ):
                self.inferred_dtypes_[col] = "float64"
            elif pd.api.types.is_numeric_dtype(X[col]) and p.coercion != "category":
                self.inferred_dtypes_[col] = "float64"  # e.g. kept identifiers / constants
            else:
                self.inferred_dtypes_[col] = "category"
            self.coercions_[col] = p.coercion
        self.plan_semantics_ = {c: self.semantic_types_[str(c)] for c in self.inferred_dtypes_}

        self.report_.add(
            "type_caster",
            "ok",
            f"Configured dtypes for {len(self.inferred_dtypes_)} columns, dropped ID cols: {self.dropped_cols_}",
            {
                "drop_reasons": self.drop_reasons_,
                "semantic_types": self.semantic_types_,
                "duplicate_names": dup_names,
            },
        )
        return self

    def _transform(self, X: pd.DataFrame) -> pd.DataFrame:
        X, _ = deduplicate_columns(X)
        out = X.drop(columns=[c for c in self.dropped_cols_ if c in X.columns], errors="ignore")
        semantics = getattr(self, "plan_semantics_", {})
        coercions = getattr(self, "coercions_", {})
        # fast path: plain numeric columns are cast as one block
        block = [
            c
            for c, t in self.inferred_dtypes_.items()
            if t == "float64"
            and c in out.columns
            and pd.api.types.is_numeric_dtype(out[c])
            and not coercions.get(c)
        ]
        if block:
            out[block] = out[block].astype(float).replace([np.inf, -np.inf], np.nan)
        done = set(block)
        for col, target in self.inferred_dtypes_.items():
            if col not in out.columns or col in done:
                continue
            try:
                converted = apply_coercion(out[col], semantics.get(col, ""), coercions.get(col))
                if target == "float64":
                    out[col] = (
                        pd.to_numeric(converted, errors="coerce")
                        .astype(float)
                        .replace([np.inf, -np.inf], np.nan)
                    )
                elif target == "boolean":
                    out[col] = (
                        coerce_boolean(converted)
                        if not pd.api.types.is_numeric_dtype(converted)
                        else converted.astype(float)
                    )
                elif target == "datetime64[ns]":
                    out[col] = coerce_datetime(converted, coercions.get(col))
                elif target == "text":
                    out[col] = converted.where(converted.isna(), converted.astype(str)).astype(
                        object
                    )
                elif target == "category":
                    # keep missing values as NaN (not the string "nan") for the imputer
                    out[col] = converted.where(converted.isna(), converted.astype(str)).astype(
                        "category"
                    )
                else:
                    out[col] = converted.astype(target)  # type: ignore[call-overload]  # user-forced dtype
            except (TypeError, ValueError, OverflowError) as exc:
                # never crash on unexpected values in new data: leave the column as-is and report
                self.report_.add(
                    "type_caster_transform", "warn", f"Could not convert '{col}': {exc}"
                )
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
        add_indicator: bool | str = False,
        indicator_min_rate: float = 0.01,
    ) -> None:
        """``add_indicator``: True (every column with missing values), False, or "auto"
        (only columns whose training missing rate is >= ``indicator_min_rate``; keeps the
        often-informative "was missing" signal without flooding the data with flags)."""
        super().__init__()
        self.numeric_strategy = numeric_strategy
        self.categorical_strategy = categorical_strategy
        self.add_indicator = add_indicator
        self.indicator_min_rate = indicator_min_rate

        self.fill_values_: dict[str, Any] = {}
        self.numeric_cols_: list[str] = []
        self.categorical_cols_: list[str] = []
        self.indicator_cols_: list[str] = []

    def _fit(self, X: pd.DataFrame, y: Any = None, cfg: TaskConfig | None = None) -> MissingImputer:
        self.fill_values_ = {}
        self.indicator_cols_ = []
        self.numeric_cols_ = list(X.select_dtypes(include=[np.number]).columns)
        self.categorical_cols_ = [c for c in X.columns if c not in self.numeric_cols_]

        def wants_indicator(col: Any) -> bool:
            rate = float(X[col].isna().mean())
            if self.add_indicator == "auto":
                return rate >= self.indicator_min_rate
            return bool(self.add_indicator) and rate > 0

        for col in self.numeric_cols_:
            s = X[col].dropna()
            if wants_indicator(col):
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
            if wants_indicator(col):
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
        if self.indicator_cols_:
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

    def __init__(
        self,
        min_freq: float | None = 0.01,
        rare_label: str = "__rare__",
        min_count: int | None = None,
        max_cardinality: int | None = None,
    ) -> None:
        """A category is rare if its share < ``min_freq`` or its count < ``min_count``.

        Columns with more than ``max_cardinality`` distinct values are left untouched: for
        high-cardinality columns a relative threshold can erase *every* level (e.g. 400 cities
        each holding 0.25% of rows) - smoothing target/frequency encoders handle rarity there.
        """
        super().__init__()
        self.min_freq = min_freq
        self.rare_label = rare_label
        self.min_count = min_count
        self.max_cardinality = max_cardinality
        self.frequent_categories_: dict[str, set[str]] = {}

    def _fit(
        self, X: pd.DataFrame, y: Any = None, cfg: TaskConfig | None = None
    ) -> RareCategoryGrouper:
        cat_cols = X.select_dtypes(include=["object", "category"]).columns
        self.frequent_categories_ = {}

        for col in cat_cols:
            s = X[col].dropna().astype(str)
            counts = s.value_counts()
            if self.max_cardinality is not None and len(counts) > self.max_cardinality:
                continue
            keep = pd.Series(True, index=counts.index)
            if self.min_freq is not None:
                keep &= counts / max(len(s), 1) >= self.min_freq
            if self.min_count is not None:
                keep &= counts >= self.min_count
            self.frequent_categories_[col] = set(counts.index[keep])

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
                grouped = values.where(values.isin(list(frequent)), self.rare_label)
                out[col] = grouped.where(out[col].notna(), np.nan)  # keep missing as missing
        return out


@register("preprocessing.categorical_encoder")
class CategoricalEncoder(BaseStep):
    """
    Adaptive categorical encoding:
    - Low cardinality (<= ``low_card_threshold``): One-Hot
    - Mid cardinality (<= ``high_card_threshold``): ``mid_card_strategy`` (default "target")
    - High cardinality: smoothed Target Encoding. During ``fit_transform`` the training
      rows receive **out-of-fold** encodings (K-fold), so a row never sees its own target.
      ``transform`` on new data uses the mapping fitted on the full training set.
    - Without a usable target (no ``y``, multiclass): frequency encoding (share of rows).
      Ordinal codes are only used when requested, because the arbitrary order they impose
      is meaningless for nominal categories.

    ``strategies`` overrides the choice per column: {"grade": "ordinal", "city": "frequency"}.
    Target encoding is only used for regression and binary targets (ADR-004).
    """

    def __init__(
        self,
        low_card_threshold: int = 10,
        high_card_threshold: int = 50,
        cv: int = 5,
        smoothing: float = 10.0,
        random_state: int | None = 42,
        mid_card_strategy: str = "target",
        strategies: dict[str, str] | None = None,
    ) -> None:
        super().__init__()
        self.low_card_threshold = low_card_threshold
        self.high_card_threshold = high_card_threshold
        self.cv = cv
        self.smoothing = smoothing
        self.random_state = random_state
        self.mid_card_strategy = mid_card_strategy
        self.strategies = strategies
        self.frequency_maps_: dict[str, dict[str, float]] = {}

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
        self.frequency_maps_ = {}
        valid = {"onehot", "ordinal", "target", "frequency"}
        overrides = dict(self.strategies or {})
        bad = {k: v for k, v in overrides.items() if v not in valid}
        if bad or self.mid_card_strategy not in valid - {"onehot"}:
            raise ValueError(
                f"Unknown encoding strategy {bad or self.mid_card_strategy}; use {sorted(valid)}"
            )

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
            if col in overrides:
                strategy = overrides[col]
            elif n_unique <= self.low_card_threshold:
                strategy = "onehot"
            elif n_unique <= self.high_card_threshold:
                strategy = self.mid_card_strategy
            else:
                strategy = "target"
            if strategy == "target" and y_num is None:
                strategy = "frequency"

            keys = X[col].astype(str)
            if strategy == "onehot":
                self.onehot_categories_[col] = [str(v) for v in X[col].dropna().unique()]
            elif strategy == "ordinal":
                cats = sorted(str(v) for v in X[col].dropna().unique())
                self.ordinal_maps_[col] = {val: idx for idx, val in enumerate(cats)}
            elif strategy == "frequency":
                self.frequency_maps_[col] = {
                    str(k): float(v) for k, v in keys.value_counts(normalize=True).items()
                }
            else:
                assert y_num is not None  # target strategy falls back to frequency without a target
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

    @staticmethod
    def _onehot_name(col: Any, cat: str, existing: Any) -> str:
        """``<col>_<category>`` without characters model libraries reject, kept unique."""
        base = re.sub(r'[\[\]{}",:<>]', "_", f"{col}_{cat}")
        name, k = base, 1
        while name in existing:
            k += 1
            name = f"{base}__{k}"
        return name

    def _transform(self, X: pd.DataFrame) -> pd.DataFrame:
        out = X.copy()
        for col, strategy in self.encoding_plan_.items():
            if col not in out.columns:
                continue

            if strategy == "onehot":
                col_str = out[col].astype(str)
                for cat in self.onehot_categories_.get(col, []):
                    out[self._onehot_name(col, cat, out.columns)] = (col_str == cat).astype(int)
                out.drop(columns=[col], inplace=True)

            elif strategy == "ordinal":
                col_map = self.ordinal_maps_.get(col, {})
                out[col] = out[col].astype(str).map(col_map).fillna(-1).astype(int)

            elif strategy == "frequency":
                out[col] = (
                    out[col]
                    .astype(str)
                    .map(self.frequency_maps_.get(col, {}))
                    .astype(float)
                    .fillna(0.0)
                )

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
