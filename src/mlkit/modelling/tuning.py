"""
mlkit.modelling.tuning
~~~~~~~~~~~~~~~~~~~~~~
Hyperparameter search using Optuna (optional dependency).

Usage::

    from mlkit.modelling.tuning import OptunaSearch, LGBM_SPACE, RF_SPACE

    searcher = OptunaSearch(estimator_cls=LGBMTrainer, space=LGBM_SPACE, n_trials=50)
    searcher.fit(X_train, y_train, cfg=cfg)
    print(searcher.best_params_)
    print(searcher.best_score_)
    best_model = searcher.best_estimator_

Requires the ``optuna`` package::

    pip install optuna
    # or
    uv add optuna
"""

from __future__ import annotations

import logging
from collections.abc import Callable
from typing import TYPE_CHECKING, Any, Literal

import numpy as np
import pandas as pd

from mlkit.core.errors import MLKitError
from mlkit.core.step import BaseStep

if TYPE_CHECKING:
    from mlkit.core.config import TaskConfig

logger = logging.getLogger(__name__)

# ---------------------------------------------------------------------------
# Pre-defined search spaces
# ---------------------------------------------------------------------------
ParamSpace = dict[str, Callable[[Any], Any]]  # trial arg is typed as Any to avoid optuna import


def LGBM_SPACE(trial: Any) -> dict[str, Any]:
    """Default Optuna search space for LightGBM."""
    return {
        "n_estimators": trial.suggest_int("n_estimators", 100, 600, step=50),
        "learning_rate": trial.suggest_float("learning_rate", 0.01, 0.3, log=True),
        "num_leaves": trial.suggest_int("num_leaves", 15, 127),
        "max_depth": trial.suggest_int("max_depth", 3, 10),
        "min_child_samples": trial.suggest_int("min_child_samples", 10, 100),
        "subsample": trial.suggest_float("subsample", 0.5, 1.0),
        "subsample_freq": 1,  # bagging is a no-op in LightGBM unless subsample_freq > 0
        "colsample_bytree": trial.suggest_float("colsample_bytree", 0.5, 1.0),
        "reg_alpha": trial.suggest_float("reg_alpha", 1e-4, 10.0, log=True),
        "reg_lambda": trial.suggest_float("reg_lambda", 1e-4, 10.0, log=True),
    }


def RF_SPACE(trial: Any) -> dict[str, Any]:
    """Default Optuna search space for RandomForest."""
    return {
        "n_estimators": trial.suggest_int("n_estimators", 50, 400, step=50),
        "max_depth": trial.suggest_int("max_depth", 3, 20),
        "min_samples_split": trial.suggest_int("min_samples_split", 2, 20),
        "min_samples_leaf": trial.suggest_int("min_samples_leaf", 1, 10),
        "max_features": trial.suggest_categorical("max_features", ["sqrt", "log2"]),
    }


def LINEAR_SPACE(trial: Any) -> dict[str, Any]:
    """Default Optuna search space for Logistic/Ridge."""
    return {
        "C": trial.suggest_float("C", 1e-4, 100.0, log=True),
    }


# ---------------------------------------------------------------------------
# OptunaSearch
# ---------------------------------------------------------------------------
class OptunaSearch(BaseStep):
    """
    Hyperparameter optimiser wrapping Optuna.

    Parameters
    ----------
    estimator_cls : type
        An mlkit trainer class (e.g. ``LGBMTrainer``, ``RandomForestTrainer``).
        It is instantiated fresh inside each trial.
    space : callable
        A function ``(trial) -> dict[str, Any]`` that samples hyper-params.
        Use one of the built-in spaces: ``LGBM_SPACE``, ``RF_SPACE``, ``LINEAR_SPACE``.
    n_trials : int
        Number of Optuna trials to run (default 50).
    cv : int
        Cross-validation folds (default 5).
    random_state : int
        Global seed for reproducibility.
    direction : str
        ``'maximize'`` (AUC, R²) or ``'minimize'`` (RMSE).  Auto-detected from
        ``cfg.task_type`` when ``None``.
    timeout : float | None
        Optuna timeout in seconds.
    """

    def __init__(
        self,
        estimator_cls: type,
        space: Callable[[Any], dict[str, Any]],
        n_trials: int = 50,
        cv: int = 5,
        random_state: int = 42,
        direction: str | None = None,
        timeout: float | None = None,
    ) -> None:
        super().__init__()
        self.estimator_cls = estimator_cls
        self.space = space
        self.n_trials = n_trials
        self.cv = cv
        self.random_state = random_state
        self.direction = direction
        self.timeout = timeout

        self.best_params_: dict[str, Any] = {}
        self.best_score_: float = float("nan")
        self.best_estimator_: Any = None
        self.study_: Any = None  # optuna.Study

    # ------------------------------------------------------------------
    def _fit(
        self,
        X: pd.DataFrame,
        y: Any = None,
        cfg: TaskConfig | None = None,
    ) -> OptunaSearch:
        try:
            import optuna
        except ImportError as e:
            raise MLKitError(
                "optuna is required for OptunaSearch. Install it with: pip install optuna"
            ) from e

        optuna.logging.set_verbosity(optuna.logging.WARNING)

        task_type = cfg.task_type if cfg else "classification"
        direction: Literal["maximize", "minimize"]
        if self.direction is None:
            direction = "maximize" if task_type == "classification" else "minimize"
        elif self.direction in ("maximize", "minimize"):
            direction = "maximize" if self.direction == "maximize" else "minimize"
        else:
            raise ValueError(f"direction must be 'maximize' or 'minimize', got {self.direction!r}")

        y_arr = np.asarray(y)

        from mlkit.modelling.evaluation import CVEvaluator

        evaluator = CVEvaluator(cv=self.cv, task_type=task_type, random_state=self.random_state)

        def objective(trial: Any) -> float:
            params = self.space(trial)
            # keep the *full* dict (incl. constants such as subsample_freq) for the final refit
            trial.set_user_attr("params", params)
            estimator = self.estimator_cls(task_type=task_type, random_state=self.random_state, **params)
            try:
                res = evaluator.evaluate(estimator, X, y_arr, cfg=cfg)
                return float(res["primary_score"])
            except Exception as exc:
                logger.debug("Trial failed with exception: %s", exc)
                raise optuna.exceptions.TrialPruned() from exc

        sampler = optuna.samplers.TPESampler(seed=self.random_state)
        study = optuna.create_study(direction=direction, sampler=sampler)
        study.optimize(objective, n_trials=self.n_trials, timeout=self.timeout, show_progress_bar=False)

        self.study_ = study
        # study.best_params only holds suggested values; constants from the space live in user_attrs
        self.best_params_ = dict(study.best_trial.user_attrs.get("params", study.best_params))
        self.best_score_ = study.best_value

        # Refit best model on full data
        self.best_estimator_ = self.estimator_cls(
            task_type=task_type, random_state=self.random_state, **self.best_params_
        )
        self.best_estimator_.fit(X, y_arr)

        self.report_.add(
            "optuna_tuning",
            "ok",
            f"Optuna finished {len(study.trials)} trials. Best {study.best_trial.value:.4f} with params: {self.best_params_}",
        )
        return self

    # ------------------------------------------------------------------
    def predict(self, X: pd.DataFrame) -> Any:
        return self.best_estimator_.predict(X)

    def predict_proba(self, X: pd.DataFrame) -> Any:
        return self.best_estimator_.predict_proba(X)

    @property
    def feature_importances_(self) -> Any:
        return getattr(self.best_estimator_, "feature_importances_", None)

    # ------------------------------------------------------------------
    def trials_dataframe(self) -> pd.DataFrame:
        """Return Optuna trials as a tidy DataFrame."""
        if self.study_ is None:
            raise MLKitError("OptunaSearch has not been fitted yet.")
        return self.study_.trials_dataframe()
