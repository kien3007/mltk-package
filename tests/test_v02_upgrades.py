"""Regression tests for the v0.2 upgrade: leakage fixes, drift baselines, bundles, runner, CLI."""

from __future__ import annotations

import json
import logging

import numpy as np
import pandas as pd
import pytest

from mlkit.core.config import TaskConfig
from mlkit.core.errors import ValidationError

rng = np.random.default_rng(7)


def _high_card_frame(n: int = 1500, n_cats: int = 300) -> pd.DataFrame:
    return pd.DataFrame(
        {
            "hc": rng.integers(0, n_cats, n).astype(str),
            "noise": rng.normal(size=n),
        }
    )


# --------------------------------------------------------------------- encoder
class TestTargetEncodingLeakage:
    def test_random_target_gives_chance_cv_auc(self):
        """With a random target, OOF encoding must not let CV 'learn' anything."""
        from mlkit.core.pipeline import Pipeline
        from mlkit.modelling import ModelArena
        from mlkit.preprocessing import CategoricalEncoder

        X = _high_card_frame()
        y = pd.Series(rng.integers(0, 2, len(X)))
        arena = ModelArena(cv=3, enforce_baseline_gate=False, candidates=["linear"])
        pipe = Pipeline([("enc", CategoricalEncoder(low_card_threshold=5, high_card_threshold=20)), ("arena", arena)])
        pipe.fit(X, y, cfg=TaskConfig(task_type="classification", target="y"))

        assert pipe.named_steps["enc"].encoding_plan_["hc"] == "target"
        auc = arena.cv_results_["Linear Model"]["roc_auc_mean"]
        assert auc < 0.58, f"CV AUC {auc:.3f} suggests target leakage"

    def test_fit_transform_differs_from_transform(self):
        from mlkit.preprocessing import CategoricalEncoder

        X = _high_card_frame(600, 100)
        y = rng.integers(0, 2, len(X))
        enc = CategoricalEncoder(low_card_threshold=5, high_card_threshold=20)
        train_out = enc.fit_transform(X, y)
        full_out = enc.transform(X)
        assert not np.allclose(train_out["hc"].to_numpy(), full_out["hc"].to_numpy())
        assert enc._y_numeric is None  # training target not kept on the fitted object

    def test_multiclass_falls_back_to_ordinal(self):
        from mlkit.preprocessing import CategoricalEncoder

        X = _high_card_frame(600, 100)
        y = rng.choice(["a", "b", "c"], len(X))
        enc = CategoricalEncoder(low_card_threshold=5, high_card_threshold=20).fit(X, y)
        assert enc.encoding_plan_["hc"] == "ordinal"
        assert enc.report_.has_warnings

    def test_refit_does_not_accumulate_state(self):
        from mlkit.preprocessing import CategoricalEncoder, MissingImputer, TypeCaster

        a = pd.DataFrame({"c": ["x", "y", None], "n": [1.0, None, 3.0], "id": ["1", "2", "3"]})
        b = pd.DataFrame({"d": ["p", "q", "r"], "m": [1.0, 2.0, 3.0]})
        enc = CategoricalEncoder().fit(a)
        enc.fit(b)
        assert set(enc.encoding_plan_) == {"d"} and set(enc.onehot_categories_) == {"d"}
        tc = TypeCaster().fit(a)
        tc.fit(b)
        assert tc.dropped_cols_ == []
        imp = MissingImputer(add_indicator=True).fit(a)
        imp.fit(b)
        assert imp.indicator_cols_ == []
        assert len(enc.report_.items) == 1


class TestPreprocessingRobustness:
    def test_type_caster_keeps_nan_for_imputer(self):
        from mlkit.preprocessing import MissingImputer, TypeCaster

        df = pd.DataFrame({"c": ["a", "a", None, "b"]})
        out = TypeCaster().fit_transform(df)
        assert out["c"].isna().sum() == 1
        imputed = MissingImputer().fit_transform(out)
        assert imputed["c"].tolist() == ["a", "a", "a", "b"]

    def test_imputer_handles_unseen_only_categories(self):
        from mlkit.preprocessing import MissingImputer, TypeCaster

        train = pd.DataFrame({"c": ["a", "b", "a"]})
        batch = pd.DataFrame({"c": ["zzz", None, "zzz"]})
        tc = TypeCaster().fit(train)
        imp = MissingImputer().fit(tc.transform(train))
        out = imp.transform(tc.transform(batch))
        assert out["c"].isna().sum() == 0

    def test_type_caster_drops_group_col(self):
        from mlkit.preprocessing import TypeCaster

        df = pd.DataFrame({"cust": ["a", "b"], "x": [1, 2]})
        cfg = TaskConfig(target="y", group_col="cust")
        assert list(TypeCaster().fit_transform(df, cfg=cfg).columns) == ["x"]

    def test_datetime_detected_on_category_dtype_and_time_col(self):
        from mlkit.features import DatetimeFeatures

        df = pd.DataFrame({"ts": pd.Series(["2024-01-01", "2024-06-15", "2024-12-31"], dtype="category")})
        out = DatetimeFeatures().fit_transform(df, cfg=TaskConfig(target="y", time_col="ts"))
        assert "ts_month" in out.columns and "ts" not in out.columns


# -------------------------------------------------------------------- pipeline
class TestPipelineAndConfig:
    def test_fit_param_routing(self):
        from mlkit.core.pipeline import Pipeline
        from mlkit.modelling import ModelArena

        X = pd.DataFrame({"x": rng.normal(size=120)})
        y = (X["x"] + rng.normal(scale=0.3, size=120) > 0).astype(int)
        groups = np.repeat(np.arange(12), 10)
        arena = ModelArena(cv=3, candidates=["linear"])
        Pipeline([("arena", arena)]).fit(X, y, cfg=TaskConfig(target="y"), arena__groups=groups)
        assert arena.best_model_name_ == "Linear Model"

        with pytest.raises(ValueError, match="Invalid fit parameter"):
            Pipeline([("arena", ModelArena())]).fit(X, y, nope=1)

    def test_arena_unknown_candidate(self):
        from mlkit.modelling import ModelArena

        with pytest.raises(ValueError, match="Unknown arena candidate"):
            ModelArena(candidates=["nope"]).fit(pd.DataFrame({"x": [1.0, 2.0]}), [0, 1])

    def test_cv_groups_never_overlap(self):
        from sklearn.base import BaseEstimator

        from mlkit.modelling import CVEvaluator

        seen: list[tuple[set, set]] = []

        class Spy(BaseEstimator):
            def fit(self, X, y):
                self.train_ids_ = set(X["g"])
                return self

            def predict(self, X):
                seen.append((self.train_ids_, set(X["g"])))
                return np.zeros(len(X))

        X = pd.DataFrame({"g": np.repeat(np.arange(10), 5)})
        y = rng.normal(size=50)
        CVEvaluator(cv=5, task_type="regression").evaluate(Spy(), X, y, groups=X["g"])
        assert seen and all(not (tr & va) for tr, va in seen)

    def test_config_validation(self):
        with pytest.raises(ValueError):
            TaskConfig(test_size=1.5)
        with pytest.raises(ValueError):
            TaskConfig(cv_folds=1)
        with pytest.raises(ValueError):
            TaskConfig(target="y", id_col="y")


# ---------------------------------------------------------------------- drift
class TestDriftBaseline:
    def _monitor(self):
        from mlkit.monitoring import DriftMonitor

        base = pd.DataFrame({"num": rng.normal(size=2000), "cat": rng.choice(["A", "B", "C"], 2000)})
        mon = DriftMonitor().fit(base)
        mon.fit_predictions(rng.uniform(size=2000))
        return mon, base

    def test_json_roundtrip_preserves_results(self):
        from mlkit.monitoring import DriftMonitor

        mon, base = self._monitor()
        restored = DriftMonitor.from_dict(json.loads(json.dumps(mon.to_dict())))
        batch = base.sample(500, random_state=1)
        a, b = mon.monitor(batch).to_dict(), restored.monitor(batch).to_dict()
        assert a == b

    def test_stable_batch_passes(self):
        from mlkit.monitoring import DriftMonitor

        mon, _ = self._monitor()
        fresh = pd.DataFrame({"num": rng.normal(size=1000), "cat": rng.choice(["A", "B", "C"], 1000)})
        report = DriftMonitor.from_dict(mon.to_dict()).monitor(fresh, y_pred=rng.uniform(size=1000))
        assert report.is_passed

    def test_detects_numeric_categorical_prediction_and_schema_drift(self):
        mon, _ = self._monitor()
        batch = pd.DataFrame({"num": rng.normal(3, 1, 500), "cat": ["NEW"] * 400 + ["A"] * 100})
        report = mon.monitor(batch, y_pred=rng.uniform(0.8, 1.0, 500))
        status = {i.name: i.status for i in report.items}
        assert status["drift_num"] == "error"
        assert status["cat_drift_cat"] == "error"
        assert status["prediction_drift"] == "error"
        unseen = next(i for i in report.items if i.name == "cat_drift_cat").details["unseen_share"]
        assert unseen == pytest.approx(0.8)

        missing = mon.monitor(pd.DataFrame({"num": rng.normal(size=10)}))
        assert any(i.name == "schema_missing_columns" for i in missing.items)

    def test_rejects_unknown_baseline_format(self):
        from mlkit.monitoring import DriftMonitor

        with pytest.raises(ValueError, match="Unsupported drift baseline"):
            DriftMonitor.from_dict({"some": "legacy"})


# --------------------------------------------------------------------- bundle
class TestBundleIntegrity:
    def test_tampered_bundle_rejected(self, tmp_path):
        from mlkit.core.bundle import load_bundle, save_bundle
        from mlkit.core.pipeline import Pipeline
        from mlkit.modelling import DummyTrainer

        pipe = Pipeline([("m", DummyTrainer(task_type="regression"))])
        X = pd.DataFrame({"x": [1.0, 2.0, 3.0]})
        pipe.fit(X, [1.0, 2.0, 3.0])
        out = save_bundle(pipe, tmp_path / "b", cfg=TaskConfig(task_type="regression", target="y"), X_sample=X)

        manifest = json.loads((out / "manifest.json").read_text())
        assert set(manifest["files"]) >= {"pipeline.joblib", "config.yaml", "versions.json"}
        versions = json.loads((out / "versions.json").read_text())
        assert versions["mlkit"] != ""

        load_bundle(out)  # intact bundle loads
        (out / "metrics.json").write_text('{"rmse": 0}')
        with pytest.raises(ValidationError, match="checksum mismatch"):
            load_bundle(out)
        load_bundle(out, verify=False)


# --------------------------------------------------------------------- runner
def _binary_frame(n: int = 400) -> pd.DataFrame:
    df = pd.DataFrame(
        {
            "when": pd.date_range("2023-01-01", periods=n, freq="D").astype(str),
            "cust": rng.integers(0, 40, n).astype(str),
            "x1": rng.normal(size=n),
            "x2": rng.normal(size=n),
            "city": rng.choice(["HN", "HCM", "DN"], n),
        }
    )
    df["label"] = np.where(df["x1"] + rng.normal(scale=0.5, size=n) > 0, ">50K", "<=50K.")
    return df


class TestRunner:
    def test_prepare_target(self):
        from mlkit.runner import prepare_target

        cfg = TaskConfig(target="label")
        y, info = prepare_target(pd.Series(["no", "yes.", "yes", " no"]), cfg)
        assert y.tolist() == [0, 1, 1, 0] and info["positive_label"] == "yes"
        y, info = prepare_target(
            pd.Series(["no", "yes"]), TaskConfig(target="label", modelling={"positive_label": "no"})
        )
        assert y.tolist() == [1, 0]
        with pytest.raises(ValidationError, match="binary"):
            prepare_target(pd.Series(["a", "b", "c"]), cfg)
        with pytest.raises(ValidationError, match="missing"):
            prepare_target(pd.Series(["a", None]), cfg)

    def test_time_split_is_chronological(self):
        from mlkit.runner import split_data

        df = _binary_frame().sample(frac=1.0, random_state=3)
        cfg = TaskConfig(target="label", time_col="when")
        s = split_data(df.drop(columns="label"), df["label"], cfg, arena_step="arena")
        assert s.strategy == "time"
        assert pd.to_datetime(s.X_train["when"]).max() < pd.to_datetime(s.X_test["when"]).min()
        assert len(s.fit_params["arena__time_index"]) == len(s.X_train)

    def test_group_split_is_disjoint(self):
        from mlkit.runner import split_data

        df = _binary_frame()
        cfg = TaskConfig(target="label", group_col="cust")
        s = split_data(df.drop(columns="label"), df["label"], cfg, arena_step="arena")
        assert not set(s.X_train["cust"]) & set(s.X_test["cust"])

    def test_pipeline_from_yaml_steps(self):
        from mlkit.runner import build_pipeline_from_config

        cfg = TaskConfig(
            target="y",
            steps=[
                {"name": "imp", "step": "preprocessing.imputer", "params": {"add_indicator": True}},
                {"name": "arena", "step": "modelling.arena", "params": {"candidates": ["linear"]}},
            ],
        )
        pipe = build_pipeline_from_config(cfg)
        assert [n for n, _ in pipe.steps] == ["imp", "arena"]
        assert pipe.named_steps["imp"].add_indicator is True
        assert pipe.named_steps["arena"].cv == cfg.cv_folds

        with pytest.raises(ValidationError, match="last step"):
            build_pipeline_from_config(TaskConfig(target="y", steps=[{"step": "preprocessing.imputer"}]))
        with pytest.raises(ValidationError, match="Duplicate"):
            build_pipeline_from_config(
                TaskConfig(target="y", steps=[{"name": "a", "step": "preprocessing.imputer"}] * 2)
            )

    def test_validate_input(self):
        from mlkit.runner import validate_input

        df = pd.DataFrame({"b": [1], "a": [2], "extra": [3]})
        assert list(validate_input(df, {"a": "int64", "b": "int64"}).columns) == ["a", "b"]
        with pytest.raises(ValidationError, match="missing"):
            validate_input(df, {"a": "int64", "zzz": "int64"})

    def test_table_io_roundtrip(self, tmp_path):
        from mlkit.runner import read_table, write_table

        df = pd.DataFrame({"a": [1, 2], "b": ["x", "y"]})
        for ext in ("csv", "parquet", "json"):
            p = write_table(df, tmp_path / f"t.{ext}")
            pd.testing.assert_frame_equal(read_table(p), df, check_dtype=False)
        with pytest.raises(ValueError, match="Unsupported"):
            read_table(write_table(df, tmp_path / "t.txt"))


# ------------------------------------------------------------------- CLI e2e
@pytest.fixture(scope="module")
def trained_bundle(tmp_path_factory):
    from mlkit.cli import cmd_run

    d = tmp_path_factory.mktemp("e2e")
    df = _binary_frame()
    data = d / "train.csv"
    df.to_csv(data, index=False)
    cfg = d / "task.yaml"
    cfg.write_text(
        f"""
task_type: classification
target: label
time_col: when
cv_folds: 3
data_path: "{data.as_posix()}"
modelling: {{candidates: [linear, lightgbm]}}
""",
        encoding="utf-8",
    )
    bundle = d / "bundle"
    assert cmd_run(str(cfg), output_dir=str(bundle), html_report=str(d / "report.html")) == 0
    return d, bundle, df


class TestCLIEndToEnd:
    def test_bundle_contents(self, trained_bundle):
        d, bundle, _ = trained_bundle
        for f in ("pipeline.joblib", "baseline.json", "metrics.json", "report.json", "manifest.json"):
            assert (bundle / f).exists(), f
        metrics = json.loads((bundle / "metrics.json").read_text())
        assert metrics["split"] == "time"
        assert metrics["holdout"]["roc_auc"] > 0.75
        assert "roc_auc" in metrics["holdout_ci"]
        assert (d / "report.html").exists()

    def test_predict(self, trained_bundle):
        from mlkit.cli import cmd_predict

        d, bundle, df = trained_bundle
        out = d / "preds.csv"
        assert cmd_predict(str(bundle), str(d / "train.csv"), out_path=str(out)) == 0
        preds = pd.read_csv(out)
        assert len(preds) == len(df)
        assert {"score", "prediction", "label"} <= set(preds.columns)
        assert set(preds["label"]) <= {">50K", "<=50K"}

    def test_predict_missing_column_fails(self, trained_bundle):
        from mlkit.cli import cmd_predict

        d, bundle, df = trained_bundle
        bad = d / "bad.csv"
        df.drop(columns=["x1"]).to_csv(bad, index=False)
        assert cmd_predict(str(bundle), str(bad)) == 1

    def test_monitor_detects_drift(self, trained_bundle):
        from mlkit.cli import cmd_monitor

        d, bundle, df = trained_bundle
        assert cmd_monitor(str(bundle), str(d / "train.csv")) == 0

        drifted = df.copy()
        drifted["x1"] = drifted["x1"] + 5
        p = d / "drifted.csv"
        drifted.to_csv(p, index=False)
        out = d / "drift.json"
        assert cmd_monitor(str(bundle), str(p), out_path=str(out)) == 2
        report = json.loads(out.read_text(encoding="utf-8"))
        assert not report["is_passed"]

    def test_main_steps_and_help(self, capsys):
        from mlkit.cli import main

        with pytest.raises(SystemExit) as exc:
            main(["steps"])
        assert exc.value.code == 0
        assert "modelling.arena" in capsys.readouterr().out

    def test_serving(self, trained_bundle):
        pytest.importorskip("fastapi")
        pytest.importorskip("httpx")
        from fastapi.testclient import TestClient

        from mlkit.serving import create_app

        _, bundle, df = trained_bundle
        client = TestClient(create_app(bundle))
        assert client.get("/health").json()["status"] == "ok"
        records = json.loads(df.drop(columns="label").head(5).to_json(orient="records"))
        resp = client.post("/predict", json={"records": records})
        assert resp.status_code == 200 and len(resp.json()["predictions"]) == 5
        bad = client.post("/predict", json={"records": [{"x1": 1.0}]})
        assert bad.status_code == 422
        mon = client.post("/monitor", json={"records": records})
        assert mon.status_code == 200 and "items" in mon.json()


# -------------------------------------------------------------------- logging
def test_json_logging_carries_run_id():
    from mlkit.core.log import JsonFormatter, run_context

    record = logging.LogRecord("mlkit.test", logging.INFO, __file__, 1, "hello %s", ("x",), None)
    with run_context("abc123"):
        payload = json.loads(JsonFormatter().format(record))
    assert payload["run_id"] == "abc123" and payload["msg"] == "hello x"
