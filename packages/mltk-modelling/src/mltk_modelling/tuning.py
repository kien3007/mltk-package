"""Hyperparameter optimization using Optuna."""

from __future__ import annotations

from collections.abc import Callable
from typing import Any

import pandas as pd
from sklearn.model_selection import StratifiedKFold, cross_val_score


class OptunaSearch:
    """
    Optuna hyperparameter tuning runner.
    """

    def __init__(
        self,
        estimator_factory: Callable[[dict[str, Any]], Any],
        param_distributions: Callable[[Any], dict[str, Any]],
        n_trials: int = 20,
        timeout: float | None = None,
        cv: int = 3,
        scoring: str = "roc_auc",
        direction: str = "maximize",
        random_state: int | None = 42,
    ) -> None:
        self.estimator_factory = estimator_factory
        self.param_distributions = param_distributions
        self.n_trials = n_trials
        self.timeout = timeout
        self.cv = cv
        self.scoring = scoring
        self.direction = direction
        self.random_state = random_state

        self.best_params_: dict[str, Any] = {}
        self.best_score_: float = 0.0
        self.best_estimator_: Any = None

    def fit(self, X: pd.DataFrame, y: Any) -> OptunaSearch:
        import optuna

        optuna.logging.set_verbosity(optuna.logging.WARNING)

        def objective(trial: optuna.Trial) -> float:
            params = self.param_distributions(trial)
            est = self.estimator_factory(params)
            cv_strat = StratifiedKFold(
                n_splits=self.cv, shuffle=True, random_state=self.random_state
            )
            scores = cross_val_score(est, X, y, cv=cv_strat, scoring=self.scoring)
            return float(scores.mean())

        study = optuna.create_study(direction=self.direction)
        study.optimize(objective, n_trials=self.n_trials, timeout=self.timeout)

        self.best_params_ = study.best_params
        self.best_score_ = study.best_value
        self.best_estimator_ = self.estimator_factory(self.best_params_)
        self.best_estimator_.fit(X, y)

        return self
