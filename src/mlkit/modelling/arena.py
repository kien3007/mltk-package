"""Model Arena: multi-model zoo benchmarking, baseline comparison, and selection."""

from __future__ import annotations

from typing import Any

import pandas as pd

from mlkit.core.config import TaskConfig, TaskType
from mlkit.core.errors import BaselineComparisonError
from mlkit.core.registry import register
from mlkit.core.step import BaseStep
from mlkit.modelling.evaluation import CVEvaluator
from mlkit.modelling.trainers import (
    DummyTrainer,
    LGBMTrainer,
    LinearTrainer,
    RandomForestTrainer,
    XGBTrainer,
)

DEFAULT_CANDIDATES = ("linear", "random_forest", "lightgbm")


@register("modelling.arena")
class ModelArena(BaseStep):
    """
    Automated model benchmark comparing candidate algorithms against a naive baseline.
    """

    def __init__(
        self,
        cv: int = 5,
        enforce_baseline_gate: bool = True,
        models: dict[str, Any] | None = None,
        random_state: int = 42,
        candidates: list[str] | tuple[str, ...] | None = None,
        handle_imbalance: bool = True,
    ) -> None:
        super().__init__()
        self.cv = cv
        self.enforce_baseline_gate = enforce_baseline_gate
        self.models = models
        self.random_state = random_state
        self.candidates = candidates
        self.handle_imbalance = handle_imbalance

        self.best_model_name_: str = ""
        self.best_estimator_: Any = None
        self.comparison_table_: pd.DataFrame = pd.DataFrame()

    @property
    def custom_models(self) -> dict[str, Any] | None:
        """Backward-compatible alias of ``models``."""
        return self.models

    def _build_zoo(self, task_type: TaskType) -> dict[str, Any]:
        rs = self.random_state
        imb = (
            self.handle_imbalance
        )  # False keeps raw class priors => better calibrated probabilities
        factories = {
            "linear": (
                "Linear Model",
                lambda: LinearTrainer(task_type=task_type, random_state=rs, handle_imbalance=imb),
            ),
            "random_forest": (
                "Random Forest",
                lambda: RandomForestTrainer(
                    task_type=task_type, random_state=rs, handle_imbalance=imb
                ),
            ),
            "lightgbm": (
                "LightGBM",
                lambda: LGBMTrainer(task_type=task_type, random_state=rs, handle_imbalance=imb),
            ),
            "xgboost": ("XGBoost", lambda: XGBTrainer(task_type=task_type, random_state=rs)),
        }
        zoo: dict[str, Any] = {"Baseline (Dummy)": DummyTrainer(task_type=task_type)}
        for key in self.candidates if self.candidates is not None else DEFAULT_CANDIDATES:
            if key not in factories:
                raise ValueError(f"Unknown arena candidate '{key}'. Options: {sorted(factories)}")
            if key == "xgboost":
                import importlib.util

                if importlib.util.find_spec("xgboost") is None:
                    self.report_.add("arena_candidates", "warn", "xgboost not installed; skipped.")
                    continue
            label, factory = factories[key]
            zoo[label] = factory()
        if self.models:
            zoo.update(self.models)
        if len(zoo) < 2:
            raise ValueError("ModelArena needs at least one candidate besides the baseline.")
        return zoo

    def _fit(
        self,
        X: pd.DataFrame,
        y: Any = None,
        cfg: TaskConfig | None = None,
        groups: Any = None,
        time_index: Any = None,
    ) -> ModelArena:
        task_type: TaskType = cfg.task_type if cfg else "classification"
        evaluator = CVEvaluator(cv=self.cv, task_type=task_type, random_state=self.random_state)

        # 1. Establish Model Zoo
        zoo = self._build_zoo(task_type)
        self.cv_results_: dict[str, dict[str, Any]] = {}

        records = []
        eval_scores = {}

        for name, model in zoo.items():
            res = evaluator.evaluate(model, X, y, cfg=cfg, groups=groups, time_index=time_index)
            self.cv_results_[name] = res
            score = res["primary_score"]
            eval_scores[name] = (score, model)

            rec = {
                "Model": name,
                "Score": round(score, 4),
            }
            if task_type == "classification":
                rec["ROC-AUC"] = (
                    f"{res.get('roc_auc_mean', 0):.4f} ± {res.get('roc_auc_std', 0):.4f}"
                )
                rec["Accuracy"] = f"{res.get('accuracy_mean', 0) * 100:.2f}%"
                rec["F1"] = f"{res.get('f1_mean', 0):.4f}"
            else:
                rec["RMSE"] = f"{res.get('rmse_mean', 0):.4f} ± {res.get('rmse_std', 0):.4f}"
                rec["R2"] = f"{res.get('r2_mean', 0):.4f}"
                rec["MAE"] = f"{res.get('mae_mean', 0):.4f}"

            records.append(rec)

        self.comparison_table_ = pd.DataFrame(records)

        # 2. Select winner (highest AUC for classification, lowest RMSE for regression)
        candidates = {k: v for k, v in eval_scores.items() if "Baseline" not in k}
        baseline_score = eval_scores["Baseline (Dummy)"][0]

        if task_type == "classification":
            winner_name = max(candidates, key=lambda k: candidates[k][0])
            winner_score = candidates[winner_name][0]
            passed_baseline = winner_score > baseline_score
        else:
            winner_name = min(candidates, key=lambda k: candidates[k][0])
            winner_score = candidates[winner_name][0]
            passed_baseline = winner_score < baseline_score

        self.best_model_name_ = winner_name
        winning_trainer = candidates[winner_name][1]

        # 3. Gate enforcement
        if (
            self.enforce_baseline_gate
            and (cfg is None or cfg.baseline_gate)
            and not passed_baseline
        ):
            msg = f"Best candidate '{winner_name}' ({winner_score:.4f}) failed to beat naive baseline ({baseline_score:.4f})."
            self.report_.add("baseline_gate", "error", msg)
            raise BaselineComparisonError(msg)
        else:
            self.report_.add(
                "baseline_gate",
                "ok",
                f"Winner '{winner_name}' ({winner_score:.4f}) successfully beat naive baseline ({baseline_score:.4f}).",
            )

        # 4. Fit winning model on full dataset
        self.best_estimator_ = winning_trainer
        self.best_estimator_.fit(X, y)
        self.report_.add(
            "model_selection",
            "ok",
            f"Selected best model: {self.best_model_name_}",
        )
        return self

    def predict(self, X: pd.DataFrame) -> Any:
        return self.best_estimator_.predict(X)

    def predict_proba(self, X: pd.DataFrame) -> Any:
        return self.best_estimator_.predict_proba(X)

    @property
    def feature_importances_(self) -> Any:
        return getattr(self.best_estimator_, "feature_importances_", None)
