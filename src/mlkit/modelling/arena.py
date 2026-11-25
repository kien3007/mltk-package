"""Model Arena: multi-model zoo benchmarking, baseline comparison, and selection."""

from __future__ import annotations

from typing import Any, Literal

import numpy as np
import pandas as pd
from sklearn.base import clone

from mlkit.core.config import TaskConfig, TaskType
from mlkit.core.errors import BaselineComparisonError
from mlkit.core.registry import register
from mlkit.core.step import BaseStep
from mlkit.modelling.evaluation import (
    CalibratedModelWrapper,
    CVEvaluator,
    FoldData,
    optimal_threshold,
)
from mlkit.modelling.trainers import (
    DummyTrainer,
    ExtraTreesTrainer,
    HistGBTrainer,
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
        calibration: str | None = None,
        calibration_cv: int = 3,
        threshold: str | float = 0.5,
        tuning: dict[str, Any] | None = None,
    ) -> None:
        """
        Parameters (selection of)
        -------------------------
        calibration : None | "sigmoid" | "isotonic" | "auto"
            Post-hoc probability calibration of the winner (classification). The calibrated
            model is cross-validated on the *same folds*; it is kept only if it lowers the CV
            Brier score ("auto" tries both methods). Calibration folds are stratified, not
            time-aware.
        threshold : float | "f1" | "youden"
            Decision threshold on P(class=1) used by ``predict`` (binary classification).
            "f1"/"youden" tune it on out-of-fold probabilities of the final model.
        """
        super().__init__()
        self.cv = cv
        self.enforce_baseline_gate = enforce_baseline_gate
        self.models = models
        self.random_state = random_state
        self.candidates = candidates
        self.handle_imbalance = handle_imbalance
        self.calibration = calibration
        self.calibration_cv = calibration_cv
        self.threshold = threshold
        self.tuning = tuning

        self.best_model_name_: str = ""
        self.best_estimator_: Any = None
        self.comparison_table_: pd.DataFrame = pd.DataFrame()

    #: Pipeline may pass pre-built per-fold data (pipeline-level CV).
    supports_fold_data = True

    def make_splits(
        self,
        X: pd.DataFrame,
        y: Any,
        cfg: TaskConfig | None = None,
        groups: Any = None,
        time_index: Any = None,
    ) -> list[tuple[Any, Any]]:
        """Fold definition used by this arena (positional indices into ``X``)."""
        task_type: TaskType = cfg.task_type if cfg else "classification"
        evaluator = CVEvaluator(cv=self.cv, task_type=task_type, random_state=self.random_state)
        return evaluator.split(X, y, cfg=cfg, groups=groups, time_index=time_index)

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
            "hist_gb": (
                "HistGradientBoosting",
                lambda: HistGBTrainer(task_type=task_type, random_state=rs, handle_imbalance=imb),
            ),
            "extra_trees": (
                "Extra Trees",
                lambda: ExtraTreesTrainer(
                    task_type=task_type, random_state=rs, handle_imbalance=imb
                ),
            ),
        }
        self._label_to_key = {label: key for key, (label, _) in factories.items()}
        self._key_to_label = {key: label for key, (label, _) in factories.items()}
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
        fold_data: list[FoldData] | None = None,
    ) -> ModelArena:
        """Benchmark the zoo and fit the winner on ``X``.

        ``fold_data`` (supplied by :class:`~mlkit.core.Pipeline` with ``cv_scope="pipeline"``)
        holds per-fold features produced by upstream steps that were re-fitted inside each
        fold, so CV scores are free of preprocessing/selection leakage. Without it the arena
        cross-validates on the already-transformed ``X`` (model-level CV).
        """
        task_type: TaskType = cfg.task_type if cfg else "classification"
        evaluator = CVEvaluator(cv=self.cv, task_type=task_type, random_state=self.random_state)
        self.cv_scope_ = "pipeline" if fold_data is not None else "model"

        # 1. Establish Model Zoo
        zoo = self._build_zoo(task_type)
        if self.calibration not in (None, "sigmoid", "isotonic", "auto"):
            raise ValueError("calibration must be None, 'sigmoid', 'isotonic' or 'auto'")
        y_arr = np.asarray(y)
        folds: list[FoldData] = (
            fold_data
            if fold_data is not None
            else [
                (X.iloc[tr], y_arr[tr], X.iloc[va], y_arr[va])
                for tr, va in evaluator.split(
                    X, y_arr, cfg=cfg, groups=groups, time_index=time_index
                )
            ]
        )

        self.cv_results_: dict[str, dict[str, Any]] = {}

        records = []
        eval_scores = {}

        for name, model in zoo.items():
            res = evaluator.evaluate_folds(model, folds)
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

        self.tuned_params_: dict[str, dict[str, Any]] = {}
        if self.tuning:
            for name, model, res in self._tune(evaluator, folds, task_type, eval_scores):
                self.cv_results_[name] = res
                eval_scores[name] = (res["primary_score"], model)
                records.append(self._record(name, res, task_type))
        self.comparison_table_ = pd.DataFrame(records)
        # medians of the model's input features: background for explanations (mlkit.explain)
        self.feature_medians_ = X.median(numeric_only=True).to_dict()

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

        # 4. Optional calibration, validated on the same folds
        final_model: Any = winning_trainer
        self.calibration_: str | None = None
        if task_type == "classification" and self.calibration:
            final_model = self._select_calibration(evaluator, winning_trainer, folds)

        # 5. Decision threshold from out-of-fold probabilities of the final model
        self.decision_threshold_ = self._select_threshold(evaluator, final_model, folds, task_type)

        # 6. Fit the final model on the full dataset
        self.best_estimator_ = final_model
        self.best_estimator_.fit(X, y)
        self.report_.add(
            "model_selection",
            "ok",
            f"Selected best model: {self.best_model_name_}"
            + (f" + {self.calibration_} calibration" if self.calibration_ else "")
            + f"; decision threshold={self.decision_threshold_:.4f}",
        )
        return self

    @staticmethod
    def _record(name: str, res: dict[str, Any], task_type: str) -> dict[str, Any]:
        rec: dict[str, Any] = {"Model": name, "Score": round(res["primary_score"], 4)}
        if task_type == "classification":
            rec["ROC-AUC"] = f"{res.get('roc_auc_mean', 0):.4f} ± {res.get('roc_auc_std', 0):.4f}"
            rec["Accuracy"] = f"{res.get('accuracy_mean', 0) * 100:.2f}%"
            rec["F1"] = f"{res.get('f1_mean', 0):.4f}"
        else:
            rec["RMSE"] = f"{res.get('rmse_mean', 0):.4f} ± {res.get('rmse_std', 0):.4f}"
            rec["R2"] = f"{res.get('r2_mean', 0):.4f}"
            rec["MAE"] = f"{res.get('mae_mean', 0):.4f}"
        return rec

    def _tune(
        self,
        evaluator: CVEvaluator,
        folds: list[FoldData],
        task_type: TaskType,
        eval_scores: dict[str, tuple[float, Any]],
    ) -> list[tuple[str, Any, dict[str, Any]]]:
        """Optuna search (TPE) on the arena's own CV folds; tuned models join the comparison.

        ``tuning``: ``{"n_trials": 20, "timeout": None, "models": [candidate keys]}``. Without
        ``models`` the best non-baseline candidate family is tuned.
        """
        import inspect

        try:
            import optuna
        except ImportError:
            self.report_.add(
                "tuning",
                "warn",
                "Tuning requested but optuna is not installed (pip install 'mlkit[tuning]').",
            )
            return []
        from mlkit.modelling.tuning import tunable_models

        opts = dict(self.tuning or {})
        n_trials = int(opts.get("n_trials", 20))
        timeout = opts.get("timeout")
        spaces = tunable_models()
        keys = list(opts.get("models") or [])
        if not keys:
            higher = task_type == "classification"
            ranked = sorted(
                (k for k in eval_scores if "Baseline" not in k and k in self._label_to_key),
                key=lambda k: eval_scores[k][0],
                reverse=higher,
            )
            keys = [self._label_to_key[ranked[0]]] if ranked else []
        unknown = [k for k in keys if k not in spaces]
        if unknown:
            raise ValueError(f"Cannot tune {unknown}; tunable: {sorted(spaces)}")
        optuna.logging.set_verbosity(optuna.logging.WARNING)
        out = []
        for key in keys:
            cls, space = spaces[key]
            base: dict[str, Any] = {"task_type": task_type, "random_state": self.random_state}
            if "handle_imbalance" in inspect.signature(cls).parameters:
                base["handle_imbalance"] = self.handle_imbalance

            def objective(
                trial: Any, cls: Any = cls, space: Any = space, base: Any = base
            ) -> float:
                params = space(trial)
                trial.set_user_attr("params", params)
                return float(
                    evaluator.evaluate_folds(cls(**base, **params), folds)["primary_score"]
                )

            study = optuna.create_study(
                direction="maximize" if task_type == "classification" else "minimize",
                sampler=optuna.samplers.TPESampler(seed=self.random_state),
            )
            study.optimize(objective, n_trials=n_trials, timeout=timeout, catch=(Exception,))
            completed = [t for t in study.trials if t.state == optuna.trial.TrialState.COMPLETE]
            if not completed:
                self.report_.add("tuning", "warn", f"No successful trial for '{key}'.")
                continue
            params = dict(study.best_trial.user_attrs["params"])
            model = cls(**base, **params)
            res = evaluator.evaluate_folds(model, folds)
            name = f"{self._key_to_label[key]} (tuned)"
            self.tuned_params_[name] = params
            self.report_.add(
                "tuning",
                "ok",
                f"{name}: {len(completed)} trials, CV {res['primary_score']:.4f}, params {params}",
            )
            out.append((name, model, res))
        return out

    def _select_calibration(
        self, evaluator: CVEvaluator, winner: Any, folds: list[FoldData]
    ) -> Any:
        base_brier = self.cv_results_[self.best_model_name_].get("brier_mean")
        if base_brier is None:
            self.report_.add(
                "calibration", "warn", "Winner has no probabilities; calibration skipped."
            )
            return winner
        methods: list[Literal["sigmoid", "isotonic"]] = (
            ["sigmoid", "isotonic"] if self.calibration == "auto" else [self.calibration]  # type: ignore[list-item]
        )
        best_model, best_brier = winner, float(base_brier)
        for method in methods:
            wrapped = CalibratedModelWrapper(
                base_estimator=clone(winner), method=method, cv=self.calibration_cv
            )
            res = evaluator.evaluate_folds(wrapped, folds)
            self.cv_results_[f"{self.best_model_name_} + {method}"] = res
            if res.get("brier_mean", np.inf) < best_brier:
                best_model, best_brier, self.calibration_ = (
                    wrapped,
                    float(res["brier_mean"]),
                    method,
                )
        if self.calibration_:
            self.report_.add(
                "calibration",
                "ok",
                f"{self.calibration_} calibration kept: CV Brier {base_brier:.4f} -> {best_brier:.4f}",
            )
        else:
            self.report_.add(
                "calibration",
                "ok",
                f"Calibration did not lower CV Brier ({base_brier:.4f}); kept raw model.",
            )
        return best_model

    def _select_threshold(
        self, evaluator: CVEvaluator, model: Any, folds: list[FoldData], task_type: str
    ) -> float:
        if task_type != "classification":
            return 0.5
        n_classes = (
            len(np.unique(np.concatenate([np.asarray(f[1]) for f in folds]))) if folds else 2
        )
        if n_classes > 2:
            if self.threshold != 0.5:
                self.report_.add(
                    "decision_threshold",
                    "warn",
                    "Decision thresholds apply to binary targets only; "
                    "multiclass predictions use the most probable class.",
                )
            return 0.5
        if isinstance(self.threshold, (int, float)) and not isinstance(self.threshold, bool):
            if not 0.0 < float(self.threshold) < 1.0:
                raise ValueError("threshold must be in (0, 1)")
            return float(self.threshold)
        if self.threshold not in ("f1", "youden"):
            raise ValueError("threshold must be a float in (0, 1), 'f1' or 'youden'")
        oof: list[tuple[np.ndarray, np.ndarray]] = []
        evaluator.evaluate_folds(model, folds, oof=oof)
        if not oof:
            self.report_.add(
                "decision_threshold", "warn", "No out-of-fold probabilities; using 0.5."
            )
            return 0.5
        y_oof = np.concatenate([o[0] for o in oof])
        p_oof = np.concatenate([o[1] for o in oof])
        thr = optimal_threshold(y_oof, p_oof, criterion=self.threshold)  # type: ignore[arg-type]
        self.report_.add(
            "decision_threshold",
            "ok",
            f"Tuned threshold ({self.threshold}) on {len(y_oof)} OOF rows: {thr:.4f}",
        )
        return thr

    def predict(self, X: pd.DataFrame) -> Any:
        """Class labels using ``decision_threshold_`` for binary classifiers."""
        thr = getattr(self, "decision_threshold_", 0.5)
        classes = getattr(self.best_estimator_, "classes_", None)
        if (
            thr != 0.5
            and classes is not None
            and len(classes) == 2
            and hasattr(self.best_estimator_, "predict_proba")
        ):
            proba = np.asarray(self.best_estimator_.predict_proba(X))[:, 1]
            return np.asarray(classes)[(proba >= thr).astype(int)]
        return self.best_estimator_.predict(X)

    def predict_proba(self, X: pd.DataFrame) -> Any:
        return self.best_estimator_.predict_proba(X)

    @property
    def feature_importances_(self) -> Any:
        return getattr(self.best_estimator_, "feature_importances_", None)
