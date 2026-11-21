"""Model trainers and estimators compatible with pandas DataFrame and sklearn."""

from __future__ import annotations

from typing import Any, Literal

import numpy as np
import pandas as pd
from mltk_core.base import ColumnType, infer_column_types
from mltk_core.exceptions import NotFittedError
from sklearn.base import BaseEstimator


class LGBMTrainer(BaseEstimator):
    """
    LightGBM estimator wrapper supporting pandas DataFrame natively.

    Automatically configures binary classification or regression depending on the target.
    """

    def __init__(
        self,
        n_estimators: int = 200,
        learning_rate: float = 0.05,
        num_leaves: int = 31,
        max_depth: int = -1,
        task_type: Literal["auto", "classification", "regression"] = "auto",
        handle_imbalance: bool = True,
        random_state: int | None = 42,
        **kwargs: Any,
    ) -> None:
        self.n_estimators = n_estimators
        self.learning_rate = learning_rate
        self.num_leaves = num_leaves
        self.max_depth = max_depth
        self.task_type = task_type
        self.handle_imbalance = handle_imbalance
        self.random_state = random_state
        self.kwargs = kwargs

        self._estimator: Any = None
        self._is_fitted: bool = False
        self.feature_names_in_: list[str] = []
        self.effective_task_: str = "classification"

    def fit(self, X: pd.DataFrame | np.ndarray, y: Any) -> LGBMTrainer:
        import lightgbm as lgb

        df = X if isinstance(X, pd.DataFrame) else pd.DataFrame(X)
        self.feature_names_in_ = [str(c) for c in df.columns]

        y_arr = np.asarray(y)
        unique_targets = np.unique(y_arr[~pd.isna(y_arr)])

        if self.task_type == "auto":
            self.effective_task_ = "classification" if len(unique_targets) <= 2 else "regression"
        else:
            self.effective_task_ = self.task_type

        # Configure model params
        params: dict[str, Any] = {
            "n_estimators": self.n_estimators,
            "learning_rate": self.learning_rate,
            "num_leaves": self.num_leaves,
            "max_depth": self.max_depth,
            "random_state": self.random_state,
            "verbosity": -1,
            **self.kwargs,
        }

        if self.effective_task_ == "classification":
            if self.handle_imbalance and len(unique_targets) == 2:
                neg_count = np.sum(y_arr == unique_targets[0])
                pos_count = np.sum(y_arr == unique_targets[1])
                if pos_count > 0:
                    params["scale_pos_weight"] = float(neg_count / pos_count)
            self._estimator = lgb.LGBMClassifier(**params)
        else:
            self._estimator = lgb.LGBMRegressor(**params)

        # Identify categorical columns for LightGBM
        col_types = infer_column_types(df)
        cat_cols = [c for c, t in col_types.items() if t == ColumnType.CATEGORICAL]
        if cat_cols:
            df_fit = df.copy()
            for c in cat_cols:
                df_fit[c] = df_fit[c].astype("category")
            self._estimator.fit(df_fit, y_arr)
        else:
            self._estimator.fit(df, y_arr)

        self._is_fitted = True
        return self

    def predict(self, X: pd.DataFrame | np.ndarray) -> np.ndarray:
        if not self._is_fitted or self._estimator is None:
            raise NotFittedError("Trainer is not fitted.")
        df = X if isinstance(X, pd.DataFrame) else pd.DataFrame(X, columns=self.feature_names_in_)
        col_types = infer_column_types(df)
        cat_cols = [c for c, t in col_types.items() if t == ColumnType.CATEGORICAL]
        if cat_cols:
            df_pred = df.copy()
            for c in cat_cols:
                df_pred[c] = df_pred[c].astype("category")
            return self._estimator.predict(df_pred)
        return self._estimator.predict(df)

    def predict_proba(self, X: pd.DataFrame | np.ndarray) -> np.ndarray:
        if not self._is_fitted or self._estimator is None:
            raise NotFittedError("Trainer is not fitted.")
        if self.effective_task_ != "classification":
            raise AttributeError("predict_proba is only available for classification.")

        df = X if isinstance(X, pd.DataFrame) else pd.DataFrame(X, columns=self.feature_names_in_)
        col_types = infer_column_types(df)
        cat_cols = [c for c, t in col_types.items() if t == ColumnType.CATEGORICAL]
        if cat_cols:
            df_pred = df.copy()
            for c in cat_cols:
                df_pred[c] = df_pred[c].astype("category")
            return self._estimator.predict_proba(df_pred)
        return self._estimator.predict_proba(df)

    @property
    def feature_importances_(self) -> np.ndarray:
        if not self._is_fitted or self._estimator is None:
            raise NotFittedError("Trainer is not fitted.")
        return getattr(self._estimator, "feature_importances_", np.array([]))


class XGBTrainer(BaseEstimator):
    """
    XGBoost estimator wrapper supporting pandas DataFrame.
    """

    def __init__(
        self,
        n_estimators: int = 200,
        learning_rate: float = 0.05,
        max_depth: int = 6,
        task_type: Literal["auto", "classification", "regression"] = "auto",
        random_state: int | None = 42,
        **kwargs: Any,
    ) -> None:
        self.n_estimators = n_estimators
        self.learning_rate = learning_rate
        self.max_depth = max_depth
        self.task_type = task_type
        self.random_state = random_state
        self.kwargs = kwargs

        self._estimator: Any = None
        self._is_fitted: bool = False
        self.feature_names_in_: list[str] = []
        self.effective_task_: str = "classification"

    def fit(self, X: pd.DataFrame | np.ndarray, y: Any) -> XGBTrainer:
        import xgboost as xgb

        df = X if isinstance(X, pd.DataFrame) else pd.DataFrame(X)
        self.feature_names_in_ = [str(c) for c in df.columns]

        y_arr = np.asarray(y)
        unique_targets = np.unique(y_arr[~pd.isna(y_arr)])

        if self.task_type == "auto":
            self.effective_task_ = "classification" if len(unique_targets) <= 2 else "regression"
        else:
            self.effective_task_ = self.task_type

        params = {
            "n_estimators": self.n_estimators,
            "learning_rate": self.learning_rate,
            "max_depth": self.max_depth,
            "random_state": self.random_state,
            "enable_categorical": True,
            **self.kwargs,
        }

        if self.effective_task_ == "classification":
            self._estimator = xgb.XGBClassifier(**params)
        else:
            self._estimator = xgb.XGBRegressor(**params)

        col_types = infer_column_types(df)
        cat_cols = [c for c, t in col_types.items() if t == ColumnType.CATEGORICAL]
        if cat_cols:
            df_fit = df.copy()
            for c in cat_cols:
                df_fit[c] = df_fit[c].astype("category")
            self._estimator.fit(df_fit, y_arr)
        else:
            self._estimator.fit(df, y_arr)

        self._is_fitted = True
        return self

    def predict(self, X: pd.DataFrame | np.ndarray) -> np.ndarray:
        if not self._is_fitted or self._estimator is None:
            raise NotFittedError("Trainer is not fitted.")
        df = X if isinstance(X, pd.DataFrame) else pd.DataFrame(X, columns=self.feature_names_in_)
        return self._estimator.predict(df)

    def predict_proba(self, X: pd.DataFrame | np.ndarray) -> np.ndarray:
        if not self._is_fitted or self._estimator is None:
            raise NotFittedError("Trainer is not fitted.")
        df = X if isinstance(X, pd.DataFrame) else pd.DataFrame(X, columns=self.feature_names_in_)
        return self._estimator.predict_proba(df)

    @property
    def feature_importances_(self) -> np.ndarray:
        if not self._is_fitted or self._estimator is None:
            raise NotFittedError("Trainer is not fitted.")
        return getattr(self._estimator, "feature_importances_", np.array([]))


class SklearnModelWrapper(BaseEstimator):
    """
    Generic wrapper for any Scikit-Learn estimator to integrate seamlessly with MLTK.

    Provides:
    - Preservation of pandas DataFrame columns (feature_names_in_)
    - Unified feature_importances_ property (extracting from feature_importances_ or abs(coef_))
    - Unified predict_proba with automatic fallback
    - Automatic task detection
    """

    def __init__(
        self,
        estimator: Any = None,
        task_type: Literal["auto", "classification", "regression"] = "auto",
    ) -> None:
        self.estimator = estimator
        self.task_type = task_type

        self._estimator: Any = None
        self._is_fitted: bool = False
        self.feature_names_in_: list[str] = []
        self.effective_task_: str = "classification"

    def fit(self, X: pd.DataFrame | np.ndarray, y: Any) -> SklearnModelWrapper:
        from sklearn.base import clone

        if self.estimator is None:
            raise ValueError("An estimator instance must be provided to SklearnModelWrapper.")

        df = X if isinstance(X, pd.DataFrame) else pd.DataFrame(X)
        self.feature_names_in_ = [str(c) for c in df.columns]

        y_arr = np.asarray(y)
        unique_targets = np.unique(y_arr[~pd.isna(y_arr)])
        if self.task_type == "auto":
            self.effective_task_ = "classification" if len(unique_targets) <= 2 else "regression"
        else:
            self.effective_task_ = self.task_type

        self._estimator = clone(self.estimator)
        self._estimator.fit(df, y_arr)
        self._is_fitted = True
        return self

    def predict(self, X: pd.DataFrame | np.ndarray) -> np.ndarray:
        if not self._is_fitted or self._estimator is None:
            raise NotFittedError("Trainer is not fitted.")
        df = X if isinstance(X, pd.DataFrame) else pd.DataFrame(X, columns=self.feature_names_in_)
        return self._estimator.predict(df)

    def predict_proba(self, X: pd.DataFrame | np.ndarray) -> np.ndarray:
        if not self._is_fitted or self._estimator is None:
            raise NotFittedError("Trainer is not fitted.")
        df = X if isinstance(X, pd.DataFrame) else pd.DataFrame(X, columns=self.feature_names_in_)
        if hasattr(self._estimator, "predict_proba"):
            return self._estimator.predict_proba(df)
        elif hasattr(self._estimator, "decision_function"):
            dfn = np.asarray(self._estimator.decision_function(df))
            if dfn.ndim == 1:
                p1 = 1.0 / (1.0 + np.exp(-np.clip(dfn, -500, 500)))
                return np.column_stack([1.0 - p1, p1])
            else:
                exp_dfn = np.exp(dfn - np.max(dfn, axis=1, keepdims=True))
                return exp_dfn / np.sum(exp_dfn, axis=1, keepdims=True)
        else:
            raise AttributeError(
                f"{type(self._estimator).__name__} does not support probability predictions."
            )

    @property
    def feature_importances_(self) -> np.ndarray:
        if not self._is_fitted or self._estimator is None:
            raise NotFittedError("Trainer is not fitted.")
        if hasattr(self._estimator, "feature_importances_"):
            return np.asarray(self._estimator.feature_importances_)
        elif hasattr(self._estimator, "coef_"):
            coef = np.asarray(self._estimator.coef_)
            if coef.ndim == 1:
                return np.abs(coef)
            return np.mean(np.abs(coef), axis=0)
        return np.array([])


class RandomForestTrainer(BaseEstimator):
    """
    Random Forest estimator natively supporting pandas DataFrame and automatic class balancing.
    """

    def __init__(
        self,
        n_estimators: int = 100,
        max_depth: int | None = None,
        min_samples_split: int = 2,
        min_samples_leaf: int = 1,
        handle_imbalance: bool = True,
        task_type: Literal["auto", "classification", "regression"] = "auto",
        random_state: int | None = 42,
        **kwargs: Any,
    ) -> None:
        self.n_estimators = n_estimators
        self.max_depth = max_depth
        self.min_samples_split = min_samples_split
        self.min_samples_leaf = min_samples_leaf
        self.handle_imbalance = handle_imbalance
        self.task_type = task_type
        self.random_state = random_state
        self.kwargs = kwargs

        self._estimator: Any = None
        self._is_fitted: bool = False
        self.feature_names_in_: list[str] = []
        self.effective_task_: str = "classification"

    def fit(self, X: pd.DataFrame | np.ndarray, y: Any) -> RandomForestTrainer:
        from sklearn.ensemble import RandomForestClassifier, RandomForestRegressor

        df = X if isinstance(X, pd.DataFrame) else pd.DataFrame(X)
        self.feature_names_in_ = [str(c) for c in df.columns]

        y_arr = np.asarray(y)
        unique_targets = np.unique(y_arr[~pd.isna(y_arr)])

        if self.task_type == "auto":
            self.effective_task_ = "classification" if len(unique_targets) <= 2 else "regression"
        else:
            self.effective_task_ = self.task_type

        params: dict[str, Any] = {
            "n_estimators": self.n_estimators,
            "max_depth": self.max_depth,
            "min_samples_split": self.min_samples_split,
            "min_samples_leaf": self.min_samples_leaf,
            "random_state": self.random_state,
            **self.kwargs,
        }

        if self.effective_task_ == "classification":
            if self.handle_imbalance:
                params["class_weight"] = "balanced"
            self._estimator = RandomForestClassifier(**params)
        else:
            self._estimator = RandomForestRegressor(**params)

        self._estimator.fit(df, y_arr)
        self._is_fitted = True
        return self

    def predict(self, X: pd.DataFrame | np.ndarray) -> np.ndarray:
        if not self._is_fitted or self._estimator is None:
            raise NotFittedError("Trainer is not fitted.")
        df = X if isinstance(X, pd.DataFrame) else pd.DataFrame(X, columns=self.feature_names_in_)
        return self._estimator.predict(df)

    def predict_proba(self, X: pd.DataFrame | np.ndarray) -> np.ndarray:
        if not self._is_fitted or self._estimator is None:
            raise NotFittedError("Trainer is not fitted.")
        if self.effective_task_ != "classification":
            raise AttributeError("predict_proba is only available for classification.")
        df = X if isinstance(X, pd.DataFrame) else pd.DataFrame(X, columns=self.feature_names_in_)
        return self._estimator.predict_proba(df)

    @property
    def feature_importances_(self) -> np.ndarray:
        if not self._is_fitted or self._estimator is None:
            raise NotFittedError("Trainer is not fitted.")
        return getattr(self._estimator, "feature_importances_", np.array([]))


class LinearTrainer(BaseEstimator):
    """
    Linear / Logistic model trainer with automatic regularization, balancing, and coefficient feature importances.
    """

    def __init__(
        self,
        C: float = 1.0,
        max_iter: int = 1000,
        handle_imbalance: bool = True,
        task_type: Literal["auto", "classification", "regression"] = "auto",
        random_state: int | None = 42,
        **kwargs: Any,
    ) -> None:
        self.C = C
        self.max_iter = max_iter
        self.handle_imbalance = handle_imbalance
        self.task_type = task_type
        self.random_state = random_state
        self.kwargs = kwargs

        self._estimator: Any = None
        self._is_fitted: bool = False
        self.feature_names_in_: list[str] = []
        self.effective_task_: str = "classification"

    def fit(self, X: pd.DataFrame | np.ndarray, y: Any) -> LinearTrainer:
        from sklearn.linear_model import LogisticRegression, Ridge

        df = X if isinstance(X, pd.DataFrame) else pd.DataFrame(X)
        self.feature_names_in_ = [str(c) for c in df.columns]

        y_arr = np.asarray(y)
        unique_targets = np.unique(y_arr[~pd.isna(y_arr)])

        if self.task_type == "auto":
            self.effective_task_ = "classification" if len(unique_targets) <= 2 else "regression"
        else:
            self.effective_task_ = self.task_type

        if self.effective_task_ == "classification":
            params: dict[str, Any] = {
                "C": self.C,
                "max_iter": self.max_iter,
                "random_state": self.random_state,
                **self.kwargs,
            }
            if self.handle_imbalance:
                params["class_weight"] = "balanced"
            self._estimator = LogisticRegression(**params)
        else:
            alpha = 1.0 / max(self.C, 1e-5)
            self._estimator = Ridge(alpha=alpha, random_state=self.random_state, **self.kwargs)

        self._estimator.fit(df, y_arr)
        self._is_fitted = True
        return self

    def predict(self, X: pd.DataFrame | np.ndarray) -> np.ndarray:
        if not self._is_fitted or self._estimator is None:
            raise NotFittedError("Trainer is not fitted.")
        df = X if isinstance(X, pd.DataFrame) else pd.DataFrame(X, columns=self.feature_names_in_)
        return self._estimator.predict(df)

    def predict_proba(self, X: pd.DataFrame | np.ndarray) -> np.ndarray:
        if not self._is_fitted or self._estimator is None:
            raise NotFittedError("Trainer is not fitted.")
        if self.effective_task_ != "classification":
            raise AttributeError("predict_proba is only available for classification.")
        df = X if isinstance(X, pd.DataFrame) else pd.DataFrame(X, columns=self.feature_names_in_)
        return self._estimator.predict_proba(df)

    @property
    def feature_importances_(self) -> np.ndarray:
        if not self._is_fitted or self._estimator is None:
            raise NotFittedError("Trainer is not fitted.")
        coef = getattr(self._estimator, "coef_", None)
        if coef is None:
            return np.array([])
        coef_arr = np.asarray(coef)
        if coef_arr.ndim == 1:
            return np.abs(coef_arr)
        return np.mean(np.abs(coef_arr), axis=0)


class CatBoostTrainer(BaseEstimator):
    """
    CatBoost estimator wrapper supporting pandas DataFrame and native categorical features.
    """

    def __init__(
        self,
        iterations: int = 300,
        learning_rate: float = 0.05,
        depth: int = 6,
        task_type: Literal["auto", "classification", "regression"] = "auto",
        random_state: int | None = 42,
        **kwargs: Any,
    ) -> None:
        self.iterations = iterations
        self.learning_rate = learning_rate
        self.depth = depth
        self.task_type = task_type
        self.random_state = random_state
        self.kwargs = kwargs

        self._estimator: Any = None
        self._is_fitted: bool = False
        self.feature_names_in_: list[str] = []
        self.effective_task_: str = "classification"

    def fit(self, X: pd.DataFrame | np.ndarray, y: Any) -> CatBoostTrainer:
        try:
            import catboost as cb
        except ImportError as err:
            raise ImportError(
                "CatBoost is not installed. Please install it with `pip install catboost`."
            ) from err

        df = X if isinstance(X, pd.DataFrame) else pd.DataFrame(X)
        self.feature_names_in_ = [str(c) for c in df.columns]

        y_arr = np.asarray(y)
        unique_targets = np.unique(y_arr[~pd.isna(y_arr)])

        if self.task_type == "auto":
            self.effective_task_ = "classification" if len(unique_targets) <= 2 else "regression"
        else:
            self.effective_task_ = self.task_type

        col_types = infer_column_types(df)
        cat_cols = [c for c, t in col_types.items() if t == ColumnType.CATEGORICAL]

        params = {
            "iterations": self.iterations,
            "learning_rate": self.learning_rate,
            "depth": self.depth,
            "random_seed": self.random_state,
            "verbose": False,
            **self.kwargs,
        }

        if self.effective_task_ == "classification":
            self._estimator = cb.CatBoostClassifier(**params)
        else:
            self._estimator = cb.CatBoostRegressor(**params)

        df_fit = df.copy()
        if cat_cols:
            for c in cat_cols:
                df_fit[c] = df_fit[c].astype(str).fillna("__missing__")
            self._estimator.fit(df_fit, y_arr, cat_features=cat_cols)
        else:
            self._estimator.fit(df_fit, y_arr)

        self._is_fitted = True
        return self

    def predict(self, X: pd.DataFrame | np.ndarray) -> np.ndarray:
        if not self._is_fitted or self._estimator is None:
            raise NotFittedError("Trainer is not fitted.")
        df = X if isinstance(X, pd.DataFrame) else pd.DataFrame(X, columns=self.feature_names_in_)
        col_types = infer_column_types(df)
        cat_cols = [c for c, t in col_types.items() if t == ColumnType.CATEGORICAL]
        df_pred = df.copy()
        for c in cat_cols:
            df_pred[c] = df_pred[c].astype(str).fillna("__missing__")
        return self._estimator.predict(df_pred)

    def predict_proba(self, X: pd.DataFrame | np.ndarray) -> np.ndarray:
        if not self._is_fitted or self._estimator is None:
            raise NotFittedError("Trainer is not fitted.")
        if self.effective_task_ != "classification":
            raise AttributeError("predict_proba is only available for classification.")
        df = X if isinstance(X, pd.DataFrame) else pd.DataFrame(X, columns=self.feature_names_in_)
        col_types = infer_column_types(df)
        cat_cols = [c for c, t in col_types.items() if t == ColumnType.CATEGORICAL]
        df_pred = df.copy()
        for c in cat_cols:
            df_pred[c] = df_pred[c].astype(str).fillna("__missing__")
        return self._estimator.predict_proba(df_pred)

    @property
    def feature_importances_(self) -> np.ndarray:
        if not self._is_fitted or self._estimator is None:
            raise NotFittedError("Trainer is not fitted.")
        return getattr(self._estimator, "feature_importances_", np.array([]))
