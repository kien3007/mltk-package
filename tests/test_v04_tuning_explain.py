"""v0.4: tuning inside ModelArena, per-prediction explanations, permutation importance."""

from __future__ import annotations

import json

import numpy as np
import pandas as pd
import pytest

from mlkit.core import Pipeline, TaskConfig
from mlkit.explain import explain_frame, feature_contributions, permutation_importance
from mlkit.modelling import ModelArena
from mlkit.preprocessing import CategoricalEncoder, MissingImputer, TypeCaster

rng = np.random.default_rng(4)


def _frame(n: int = 1200) -> tuple[pd.DataFrame, np.ndarray]:
    X = pd.DataFrame(
        {
            "strong": rng.normal(size=n),
            "weak": rng.normal(size=n),
            "noise": rng.normal(size=n),
            "city": rng.choice(["hn", "hcm", "dn"], n),
        }
    )
    logit = 3 * X["strong"] + 0.5 * X["weak"] + (X["city"] == "hn") * 0.8
    y = (rng.uniform(size=n) < 1 / (1 + np.exp(-logit))).astype(int).to_numpy()
    return X, y


def _pipe(candidates, **arena) -> Pipeline:
    return Pipeline(
        [
            ("types", TypeCaster()),
            ("imp", MissingImputer()),
            ("enc", CategoricalEncoder()),
            (
                "arena",
                ModelArena(cv=3, candidates=candidates, enforce_baseline_gate=False, **arena),
            ),
        ]
    )


class TestTuning:
    def test_tuned_model_joins_the_comparison(self):
        pytest.importorskip("optuna")
        X, y = _frame()
        pipe = _pipe(["linear", "lightgbm"], tuning={"n_trials": 4, "models": ["lightgbm"]})
        pipe.fit(X, y, cfg=TaskConfig(target="y"))
        arena = pipe.named_steps["arena"]
        assert "LightGBM (tuned)" in arena.cv_results_ and "LightGBM (tuned)" in set(
            arena.comparison_table_["Model"]
        )
        assert set(arena.tuned_params_["LightGBM (tuned)"]) >= {"learning_rate", "num_leaves"}

    def test_default_tunes_best_family_and_rejects_unknown(self):
        pytest.importorskip("optuna")
        X, y = _frame(600)
        arena = ModelArena(
            cv=3, candidates=["linear"], tuning={"n_trials": 2}, enforce_baseline_gate=False
        )
        arena.fit(X[["strong", "weak", "noise"]], y, cfg=TaskConfig(target="y"))
        assert list(arena.tuned_params_) == ["Linear Model (tuned)"]
        with pytest.raises(ValueError, match="Cannot tune"):
            ModelArena(
                cv=3, candidates=["linear"], tuning={"n_trials": 1, "models": ["magic"]}
            ).fit(X[["strong", "weak"]], y, cfg=TaskConfig(target="y"))

    def test_runner_records_tuned_params(self):
        pytest.importorskip("optuna")
        from mlkit.runner import train

        X, y = _frame(800)
        cfg = TaskConfig(
            target="y", cv_folds=3, modelling={"candidates": ["linear"], "tuning": {"n_trials": 2}}
        )
        res = train(cfg, X.assign(y=y))
        assert "Linear Model (tuned)" in res.metrics["tuned_params"]


class TestExplanations:
    @pytest.mark.parametrize(
        "candidates, method",
        [(["linear"], "linear"), (["lightgbm"], "ablation"), (["random_forest"], "shap")],
    )
    def test_strong_feature_is_top_reason(self, candidates, method):
        if method == "shap":
            pytest.importorskip("shap")
        X, y = _frame()
        pipe = _pipe(candidates)
        pipe.fit(X, y, cfg=TaskConfig(target="y"))
        reasons = explain_frame(pipe, X.head(300), top_k=2, method=method)
        assert reasons.attrs["method"] == method
        assert (reasons["reason_1"] == "strong").mean() > 0.6
        assert {"reason_1", "reason_1_impact", "reason_2", "reason_2_impact"} == set(
            reasons.columns
        )
        # sign: rows with large 'strong' push the probability up
        top = X.head(300)["strong"].to_numpy() > 1.5
        sel = reasons["reason_1"].to_numpy() == "strong"
        assert (reasons["reason_1_impact"].to_numpy()[top & sel] > 0).all()

    def test_onehot_levels_are_attributed_to_their_column(self):
        X, y = _frame()
        pipe = _pipe(["linear"])
        pipe.fit(X, y, cfg=TaskConfig(target="y"))
        contrib, _ = feature_contributions(pipe, X.head(50))
        assert set(contrib.columns) == {
            "strong",
            "weak",
            "noise",
            "city",
        }  # city_hn/... summed back

    def test_multiclass_and_regression(self):
        n = 900
        X = pd.DataFrame({"a": rng.normal(size=n), "b": rng.normal(size=n)})
        y3 = np.digitize(X["a"], [-0.5, 0.5])
        pipe = _pipe(["lightgbm"]).fit(X, y3, cfg=TaskConfig(target="y"))
        e = explain_frame(pipe, X.head(100), top_k=1)
        assert (e["reason_1"] == "a").mean() > 0.8
        yr = 3 * X["a"].to_numpy() + rng.normal(scale=0.1, size=n)
        pr = _pipe(["linear"]).fit(X, yr, cfg=TaskConfig(task_type="regression", target="y"))
        er = explain_frame(pr, X.head(100), task_type="regression", top_k=1)
        assert (er["reason_1"] == "a").all()

    def test_permutation_importance(self):
        X, y = _frame()
        pipe = _pipe(["linear"]).fit(X, y, cfg=TaskConfig(target="y"))
        pi = permutation_importance(pipe, X, y, n_repeats=2)
        assert (
            pi.iloc[0]["feature"] == "strong"
            and pi.set_index("feature").loc["noise", "importance_mean"] < 0.02
        )

    def test_invalid_method(self):
        X, y = _frame(300)
        pipe = _pipe(["linear"]).fit(X, y, cfg=TaskConfig(target="y"))
        with pytest.raises(ValueError, match="method"):
            explain_frame(pipe, X, method="lime")


def test_predict_cli_and_api_explain(tmp_path):
    from mlkit.cli import cmd_predict
    from mlkit.runner import predict_bundle, train

    X, y = _frame(800)
    df = X.assign(y=y)
    train(
        TaskConfig(target="y", cv_folds=3, modelling={"candidates": ["linear"]}),
        df,
        output_dir=tmp_path / "b",
    )
    out = predict_bundle(tmp_path / "b", X.head(10), explain=3)
    assert {"reason_1", "reason_3_impact", "score"} <= set(out.columns) and len(out) == 10
    data = tmp_path / "d.csv"
    X.head(20).to_csv(data, index=False)
    assert (
        cmd_predict(str(tmp_path / "b"), str(data), out_path=str(tmp_path / "p.csv"), explain=2)
        == 0
    )
    assert "reason_2" in pd.read_csv(tmp_path / "p.csv").columns

    pytest.importorskip("fastapi")
    from fastapi.testclient import TestClient

    from mlkit.serving import create_app

    client = TestClient(create_app(tmp_path / "b"))
    records = json.loads(X.head(2).to_json(orient="records"))
    preds = client.post("/predict", json={"records": records, "explain": 2}).json()["predictions"]
    assert "reason_1" in preds[0]
