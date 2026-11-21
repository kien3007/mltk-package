"""Unit tests for MLKit core abstractions: TaskConfig, Step, Report, Registry, and Bundle."""

import tempfile
from pathlib import Path

import numpy as np
import pandas as pd
import pytest

from mlkit.core.bundle import load_bundle, save_bundle
from mlkit.core.config import TaskConfig
from mlkit.core.errors import NotFittedError, QualityGateError, RegistryError
from mlkit.core.pipeline import Pipeline
from mlkit.core.registry import get_step, list_steps, register
from mlkit.core.report import Report
from mlkit.core.step import BaseStep


def test_task_config_yaml_io():
    with tempfile.TemporaryDirectory() as tmpdir:
        yaml_file = Path(tmpdir) / "config.yaml"
        cfg = TaskConfig(task_type="classification", target="income", cv_folds=3, seed=123)
        cfg.to_yaml(yaml_file)

        loaded_cfg = TaskConfig.from_yaml(yaml_file)
        assert loaded_cfg.task_type == "classification"
        assert loaded_cfg.target == "income"
        assert loaded_cfg.cv_folds == 3
        assert loaded_cfg.seed == 123


def test_report_gatekeeper():
    rep = Report(title="Test QC")
    rep.add("check_1", "ok", "Passed 1")
    rep.add("check_2", "warn", "Warning 2")
    assert rep.is_passed
    assert not rep.has_errors
    assert rep.has_warnings

    # Add error
    rep.add("check_3", "error", "Fatal failure")
    assert not rep.is_passed
    assert rep.has_errors

    with pytest.raises(QualityGateError, match="Fatal failure"):
        rep.raise_if_failed(gate_name="Test Gate")


def test_base_step_contract():
    class DummyMultiplyStep(BaseStep):
        def __init__(self, factor: float = 2.0):
            super().__init__()
            self.factor = factor

        def _transform(self, X: pd.DataFrame) -> pd.DataFrame:
            return X * self.factor

    df = pd.DataFrame({"a": [1.0, 2.0], "b": [3.0, 4.0]})
    step = DummyMultiplyStep(factor=3.0)

    # Calling transform before fit raises NotFittedError
    with pytest.raises(NotFittedError):
        step.transform(df)

    # fit_transform returns DataFrame and does not mutate input
    out_df, report = step.run(df)
    assert isinstance(out_df, pd.DataFrame)
    assert out_df["a"].tolist() == [3.0, 6.0]
    assert df["a"].tolist() == [1.0, 2.0]  # Input unchanged!


def test_step_registry():
    @register("test.custom_step")
    class CustomTestStep(BaseStep):
        pass

    assert "test.custom_step" in list_steps()
    inst = get_step("test.custom_step")
    assert isinstance(inst, CustomTestStep)

    with pytest.raises(RegistryError):
        get_step("non_existent_step")


class MockEstimator(BaseStep):
    def fit(self, X, y=None, cfg=None):
        self._is_fitted = True
        return self

    def predict(self, X):
        return np.ones(len(X))


def test_bundle_save_and_load():
    pipe = Pipeline([("model", MockEstimator())])
    df = pd.DataFrame({"x": [1, 2, 3]})
    pipe.fit(df)

    cfg = TaskConfig(task_type="regression", target="x")

    with tempfile.TemporaryDirectory() as tmpdir:
        bundle_dir = save_bundle(
            pipeline=pipe,
            path=tmpdir,
            cfg=cfg,
            X_sample=df,
            metrics={"rmse": 0.12},
        )

        loaded_pipe, loaded_cfg, metadata = load_bundle(bundle_dir)
        assert loaded_cfg.task_type == "regression"
        assert metadata["metrics"]["rmse"] == 0.12
        preds = loaded_pipe.predict(df)
        assert len(preds) == 3
