"""``mlkit evaluate``: performance on labelled data, overall and per group."""

from __future__ import annotations

import json

import numpy as np
import pandas as pd
import pytest

from mlkit.core import TaskConfig
from mlkit.core.errors import ValidationError

rng = np.random.default_rng(21)


def _frame(n: int = 900) -> pd.DataFrame:
    df = pd.DataFrame(
        {
            "week": np.repeat(np.arange(9), n // 9),
            "x1": rng.normal(size=n),
            "x2": rng.normal(size=n),
        }
    )
    df["label"] = np.where(df["x1"] + rng.normal(scale=0.6, size=n) > 0, "buy", "no")
    return df


@pytest.fixture(scope="module")
def bundle(tmp_path_factory):
    from mlkit.runner import train

    d = tmp_path_factory.mktemp("ev")
    cfg = TaskConfig(
        target="label",
        cv_folds=3,
        metric_thresholds={"roc_auc": 0.7, "ece": 0.2},
        modelling={"candidates": ["linear"], "positive_label": "buy"},
    )
    train(cfg, _frame(), output_dir=d / "b")
    return d, d / "b"


def test_labels_encoded_with_training_map(bundle):
    from mlkit.core.bundle import load_bundle
    from mlkit.runner import encode_target_like_training

    _, b = bundle
    _, cfg, meta = load_bundle(b)
    y = encode_target_like_training(pd.Series(["buy", "no.", " buy"]), cfg, meta)
    assert y.tolist() == [1, 0, 1]
    with pytest.raises(ValidationError, match="not seen"):
        encode_target_like_training(pd.Series(["maybe"]), cfg, meta)


def test_overall_and_by_group(bundle):
    from mlkit.runner import evaluate_bundle

    _, b = bundle
    df = _frame()
    res = evaluate_bundle(b, df, by="week")
    assert res.passed, res.failures
    assert res.overall["n"] == len(df) and res.overall["roc_auc"] > 0.8
    assert res.by_group is not None and res.by_group["n"].sum() == len(df)
    assert list(res.by_group["week"]) == sorted(df["week"].unique())
    assert {"roc_auc", "brier", "ece", "positive_rate", "mean_score"} <= set(res.by_group.columns)
    assert "roc_auc" in res.reference  # training holdout for comparison


def test_degraded_group_reported_and_single_class_auc_undefined(bundle):
    from mlkit.runner import evaluate_bundle

    _, b = bundle
    df = _frame()
    bad = df["week"] == 3
    df.loc[bad, "label"] = rng.choice(["buy", "no"], bad.sum())  # labels unrelated to features
    df.loc[df["week"] == 5, "label"] = "buy"  # single-class batch
    res = evaluate_bundle(b, df, by="week")
    assert not res.passed
    assert any(f.startswith("week=3: roc_auc=") for f in res.failures)
    assert any(f == "week=5: roc_auc could not be computed" for f in res.failures)
    assert np.isnan(res.by_group.set_index("week").loc[5, "roc_auc"])


def test_errors(bundle):
    from mlkit.runner import evaluate_bundle

    _, b = bundle
    with pytest.raises(ValidationError, match="target column"):
        evaluate_bundle(b, _frame().drop(columns="label"))
    with pytest.raises(ValidationError, match="Group column"):
        evaluate_bundle(b, _frame(), by="nope")


def test_cli_exit_codes_and_outputs(bundle, capsys):
    from mlkit.cli import cmd_evaluate

    d, b = bundle
    good = d / "good.csv"
    _frame().to_csv(good, index=False)
    assert cmd_evaluate(str(b), str(good), by="week", out_path=str(d / "ev.json")) == 0
    payload = json.loads((d / "ev.json").read_text(encoding="utf-8"))
    assert payload["passed"] and len(payload["by_group"]) == 9
    assert "training holdout" in capsys.readouterr().out

    df = _frame()
    df["label"] = rng.choice(["buy", "no"], len(df))
    noisy = d / "noisy.csv"
    df.to_csv(noisy, index=False)
    assert cmd_evaluate(str(b), str(noisy), by="week", out_path=str(d / "ev.csv")) == 2
    assert len(pd.read_csv(d / "ev.csv")) == 9

    assert cmd_evaluate(str(b), str(d / "missing.csv")) == 1
