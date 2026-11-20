"""Model trainers and estimators compatible with pandas DataFrame and sklearn."""

from __future__ import annotations

from typing import Any, Literal
import numpy as np
import pandas as pd
from sklearn.base import BaseEstimator, ClassifierMixin, RegressorMixin

from mltk_core.base import ColumnType, infer_column_types
from mltk_core.exceptions import NotFittedError


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
