"""Unit tests for mltk-modelling."""

import tempfile

import numpy as np
import pandas as pd
import pytest
from mltk_modelling import (
    CatBoostTrainer,
    CVEvaluator,
    LGBMTrainer,
    LinearTrainer,
    RandomForestTrainer,
    SklearnModelWrapper,
    XGBTrainer,
    load_artifact,
    save_artifact,
)
from sklearn.linear_model import LogisticRegression


def test_lgbm_trainer_classification():
    np.random.seed(42)
    n = 60
    X = pd.DataFrame(
        {
            "num1": np.random.randn(n),
            "num2": np.random.randn(n),
            "cat1": np.random.choice(["A", "B"], size=n),
        }
    )
    y = pd.Series((X["num1"] + (X["cat1"] == "A") > 0).astype(int))

    trainer = LGBMTrainer(n_estimators=10, random_state=42)
    trainer.fit(X, y)

    preds = trainer.predict(X)
    assert len(preds) == n
    probs = trainer.predict_proba(X)
    assert probs.shape == (n, 2)
    assert len(trainer.feature_importances_) == 3


def test_xgb_trainer():
    np.random.seed(42)
    n = 50
    X = pd.DataFrame({"f1": np.random.randn(n), "f2": np.random.randn(n)})
    y = (X["f1"] > 0).astype(int)

    trainer = XGBTrainer(n_estimators=10, random_state=42)
    trainer.fit(X, y)

    preds = trainer.predict(X)
    assert len(preds) == n
    probs = trainer.predict_proba(X)
    assert probs.shape == (n, 2)
    assert len(trainer.feature_importances_) == 2


def test_sklearn_model_wrapper():
    np.random.seed(42)
    n = 60
    X = pd.DataFrame({"f1": np.random.randn(n), "f2": np.random.randn(n)})
    y = (X["f1"] + X["f2"] > 0).astype(int)

    base_lr = LogisticRegression(random_state=42)
    wrapper = SklearnModelWrapper(estimator=base_lr)
    wrapper.fit(X, y)

    preds = wrapper.predict(X)
    assert len(preds) == n
    probs = wrapper.predict_proba(X)
    assert probs.shape == (n, 2)
    assert len(wrapper.feature_importances_) == 2
    assert wrapper.feature_names_in_ == ["f1", "f2"]


def test_random_forest_trainer():
    np.random.seed(42)
    n = 60
    X = pd.DataFrame({"f1": np.random.randn(n), "f2": np.random.randn(n)})
    y = (X["f1"] > 0).astype(int)

    trainer = RandomForestTrainer(n_estimators=15, random_state=42)
    trainer.fit(X, y)

    preds = trainer.predict(X)
    assert len(preds) == n
    probs = trainer.predict_proba(X)
    assert probs.shape == (n, 2)
    assert len(trainer.feature_importances_) == 2


def test_linear_trainer():
    np.random.seed(42)
    n = 60
    X = pd.DataFrame({"f1": np.random.randn(n), "f2": np.random.randn(n)})
    y = (X["f1"] > 0).astype(int)

    trainer = LinearTrainer(C=1.0, random_state=42)
    trainer.fit(X, y)

    preds = trainer.predict(X)
    assert len(preds) == n
    probs = trainer.predict_proba(X)
    assert probs.shape == (n, 2)
    assert len(trainer.feature_importances_) == 2
    assert np.all(trainer.feature_importances_ >= 0)


def test_catboost_trainer_import_handling():
    X = pd.DataFrame({"f1": [1, 2, 3], "f2": [4, 5, 6]})
    y = [0, 1, 0]
    trainer = CatBoostTrainer()
    try:
        import catboost  # noqa: F401

        trainer.fit(X, y)
        assert len(trainer.predict(X)) == 3
    except ImportError:
        with pytest.raises(ImportError, match="CatBoost is not installed"):
            trainer.fit(X, y)


def test_cv_evaluator():
    np.random.seed(42)
    n = 50
    X = pd.DataFrame({"f1": np.random.randn(n), "f2": np.random.randn(n)})
    y = (X["f1"] > 0).astype(int)

    trainer = LGBMTrainer(n_estimators=10, random_state=42)
    evaluator = CVEvaluator(cv=3, random_state=42)
    res = evaluator.evaluate(trainer, X, y)

    assert "roc_auc_mean" in res
    assert "accuracy_mean" in res
    assert len(res["folds"]) == 3


def test_save_and_load_artifact():
    np.random.seed(42)
    X = pd.DataFrame({"feat1": [1.0, 2.0, 3.0, 4.0], "feat2": [10.0, 20.0, 30.0, 40.0]})
    y = pd.Series([0, 1, 0, 1])

    trainer = RandomForestTrainer(n_estimators=10, random_state=42)
    trainer.fit(X, y)

    with tempfile.TemporaryDirectory() as tmpdir:
        art_dir = save_artifact(
            pipeline=trainer,
            X_train=X,
            y_train=y,
            metrics={"auc": 0.95},
            output_dir=tmpdir,
            model_name="test_rf_model",
        )

        loaded_pipe, meta = load_artifact(art_dir)
        assert meta.extra["model_name"] == "test_rf_model"
        assert meta.random_state == 42
        preds = loaded_pipe.predict(X)
        assert len(preds) == len(X)
