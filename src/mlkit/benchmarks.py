"""Generality benchmark: the default pipeline on many dataset shapes, each with an expectation.

The suite is a guard-rail against tuning the toolkit to one dataset. Several datasets are built
so that the signal is only reachable when the data-foundation layer does its job:

- ``dirty_types``: the signal is in a price stored as "$1,234.50" text with "n/a" tokens
- ``free_text``: the signal is the length of a free-text column (``TextFeatures``)
- ``informative_missing``: the signal is *whether* a value is missing (missing indicators)
- ``weekday_effect``: the signal is the weekday of a date stored as text (``DatetimeFeatures``)
- ``high_cardinality``: the signal is in a 400-level categorical (no rare-level erasure)
- ``leaky``: a feature determines the target; QC must stop the pipeline

Run ``mlkit benchmark`` (exit code 1 if any expectation fails) or :func:`run_suite`.
Every case also runs the full EDA (profile, QC, charts, HTML) to prove it never crashes.
Only offline data is used (synthetic generators and scikit-learn's bundled datasets).
"""

from __future__ import annotations

import time
import warnings
from collections.abc import Callable
from dataclasses import dataclass, field
from typing import Any

import numpy as np
import pandas as pd

from mlkit.core.config import TaskConfig
from mlkit.core.errors import QualityGateError, ValidationError


@dataclass
class Case:
    name: str
    make: Callable[[np.random.Generator], pd.DataFrame]
    task_type: str = "classification"
    config: dict[str, Any] = field(default_factory=dict)
    metric: tuple[str, float] | None = None  # (holdout metric, minimum)
    expect_error: type[Exception] | None = None
    qc_warns: tuple[str, ...] = ()
    description: str = ""


# --------------------------------------------------------------------------- generators
def _logistic(rng: np.random.Generator, logit: np.ndarray) -> np.ndarray:
    return (rng.uniform(size=len(logit)) < 1 / (1 + np.exp(-logit))).astype(int)


def _sk(name: str) -> Callable[[np.random.Generator], pd.DataFrame]:
    def make(rng: np.random.Generator) -> pd.DataFrame:
        from sklearn import datasets

        b = getattr(datasets, name)(as_frame=True)
        return b.frame.rename(columns={b.target.name: "target"})

    return make


def _imbalanced(rng: np.random.Generator) -> pd.DataFrame:
    X = rng.normal(size=(20000, 8))
    df = pd.DataFrame(X, columns=[f"x{i}" for i in range(8)])
    df["target"] = _logistic(rng, 1.5 * X[:, 0] + X[:, 1] - 5.2)
    return df


def _high_cardinality(rng: np.random.Generator) -> pd.DataFrame:
    n = 4000
    city = rng.choice([f"c{i}" for i in range(400)], n)
    effect = {c: rng.normal(scale=1.5) for c in np.unique(city)}
    df = pd.DataFrame(
        {"city": city, "segment": rng.choice(["A", "B", "C"], n), "noise": rng.normal(size=n)}
    )
    df["target"] = _logistic(rng, np.array([effect[c] for c in city]))
    return df


def _dirty_types(rng: np.random.Generator) -> pd.DataFrame:
    n = 3000
    price = rng.lognormal(6, 1, n)
    df = pd.DataFrame(
        {
            "price": [f"${v:,.2f}" for v in price],
            "discount": [f"{v:.0f}%" for v in rng.uniform(0, 50, n)],
            "member": rng.choice(["Yes", "No"], n),
            "signup": pd.Timestamp("2022-01-01")
            + pd.to_timedelta(rng.integers(0, 700, n), unit="D"),
            "region": rng.choice(["north", "south", "east", "west"], n),
        }
    )
    df["signup"] = df["signup"].dt.strftime("%d/%m/%Y")
    df["target"] = _logistic(rng, 1.5 * (np.log(price) - 6))
    df.loc[rng.uniform(size=n) < 0.1, "price"] = rng.choice(["n/a", "NULL", "-"], 1)[0]
    return df


def _free_text(rng: np.random.Generator) -> pd.DataFrame:
    n = 3000
    words = np.array(
        ["good", "bad", "fast", "slow", "price", "quality", "late", "great", "broken", "ok"]
    )
    lengths = rng.integers(3, 40, n)
    df = pd.DataFrame(
        {
            "review": [" ".join(rng.choice(words, k)) for k in lengths],
            "stars": rng.integers(1, 6, n),
        }
    )
    df["target"] = _logistic(rng, (lengths - 20) / 5)
    return df


def _informative_missing(rng: np.random.Generator) -> pd.DataFrame:
    n = 4000
    income = rng.lognormal(10, 0.5, n)
    hidden = rng.uniform(size=n) < 0.3
    df = pd.DataFrame({"income": np.where(hidden, np.nan, income), "age": rng.integers(18, 80, n)})
    df["target"] = _logistic(rng, np.where(hidden, 2.0, -1.0))
    return df


def _weekday_effect(rng: np.random.Generator) -> pd.DataFrame:
    n = 4000
    dates = pd.Timestamp("2023-01-01") + pd.to_timedelta(rng.integers(0, 365, n), unit="D")
    # column name gives no hint: dates must be recognised from the values themselves
    df = pd.DataFrame({"visit": dates.strftime("%Y-%m-%d"), "x": rng.normal(size=n)})
    df["target"] = _logistic(rng, np.where(dates.dayofweek >= 5, 2.0, -1.5))
    return df


def _grouped(rng: np.random.Generator) -> pd.DataFrame:
    g = np.repeat(np.arange(200), 20)
    x = rng.normal(size=len(g))
    return pd.DataFrame(
        {"user": g.astype(str), "x": x, "target": _logistic(rng, 2 * x + rng.normal(size=200)[g])}
    )


def _wide(rng: np.random.Generator) -> pd.DataFrame:
    X = rng.normal(size=(400, 500))
    df = pd.DataFrame(X, columns=[f"f{i}" for i in range(500)])
    df["target"] = _logistic(rng, 1.5 * X[:, :5].sum(axis=1))
    return df


def _skewed_regression(rng: np.random.Generator) -> pd.DataFrame:
    X = rng.normal(size=(3000, 5))
    df = pd.DataFrame(X, columns=list("abcde"))
    df["target"] = np.exp(1 + X[:, 0] + 0.5 * X[:, 1] + rng.normal(scale=0.5, size=3000))
    return df


def _int_columns(rng: np.random.Generator) -> pd.DataFrame:
    X = rng.normal(size=(1500, 4))
    df = pd.DataFrame(X)
    df["target"] = _logistic(rng, 2 * X[:, 0])
    return df


def _all_categorical(rng: np.random.Generator) -> pd.DataFrame:
    n = 3000
    a, b = rng.choice(list("xyz"), n), rng.choice(list("pq"), n)
    y = ((a == "x") ^ (b == "p")).astype(int)
    return pd.DataFrame({"a": a, "b": b, "target": np.where(rng.uniform(size=n) < 0.1, 1 - y, y)})


def _tiny(rng: np.random.Generator) -> pd.DataFrame:
    return _sk("load_breast_cancer")(rng).sample(80, random_state=0)


def _leaky(rng: np.random.Generator) -> pd.DataFrame:
    n = 2000
    x = rng.normal(size=n)
    y = _logistic(rng, x)
    return pd.DataFrame(
        {"x": x, "days_since_cancel": np.where(y == 1, rng.integers(1, 30, n), -1), "target": y}
    )


CASES: list[Case] = [
    Case(
        "breast_cancer",
        _sk("load_breast_cancer"),
        metric=("roc_auc", 0.97),
        description="sklearn, n=569 numeric",
    ),
    Case(
        "diabetes",
        _sk("load_diabetes"),
        "regression",
        metric=("r2", 0.30),
        description="sklearn regression n=442",
    ),
    Case(
        "wine_multiclass",
        _sk("load_wine"),
        expect_error=ValidationError,
        description="3 classes: clear rejection",
    ),
    Case("imbalanced_1pct", _imbalanced, metric=("roc_auc", 0.80), qc_warns=("class_imbalance",)),
    Case(
        "high_cardinality",
        _high_cardinality,
        metric=("roc_auc", 0.65),
        qc_warns=("high_cardinality",),
    ),
    Case("dirty_types", _dirty_types, metric=("roc_auc", 0.75), qc_warns=("disguised_missing",)),
    Case("free_text", _free_text, metric=("roc_auc", 0.80), qc_warns=("text_columns",)),
    Case("informative_missing", _informative_missing, metric=("roc_auc", 0.75)),
    Case("weekday_effect", _weekday_effect, metric=("roc_auc", 0.75)),
    Case("grouped_users", _grouped, config={"group_col": "user"}, metric=("roc_auc", 0.70)),
    Case("wide_p500", _wide, metric=("roc_auc", 0.85)),
    Case("skewed_regression", _skewed_regression, "regression", metric=("r2", 0.45)),
    Case("integer_column_names", _int_columns, metric=("roc_auc", 0.80)),
    Case("all_categorical_xor", _all_categorical, metric=("roc_auc", 0.85)),
    Case("tiny_n80", _tiny, metric=("roc_auc", 0.85)),
    Case(
        "leaky_feature",
        _leaky,
        expect_error=QualityGateError,
        description="QC must stop the pipeline",
    ),
]


# ------------------------------------------------------------------------------ runner
def run_case(case: Case, seed: int = 0, with_eda: bool = True) -> dict[str, Any]:
    """Run one case; returns a result row with ``passed`` and a human-readable ``detail``."""
    from mlkit.runner import train

    rng = np.random.default_rng(seed)
    df = case.make(rng)
    row: dict[str, Any] = {
        "case": case.name,
        "rows": len(df),
        "cols": df.shape[1] - 1,
        "passed": False,
    }
    t0 = time.perf_counter()
    with warnings.catch_warnings():
        warnings.simplefilter("ignore")
        if with_eda:
            import matplotlib.pyplot as plt

            from mlkit.qc_eda import run_eda

            eda = run_eda(df, target="target", task_type=case.task_type)
            for _, fig in eda.figures():
                plt.close(fig)
            eda.to_html()
            warned = (
                {i.name for i in eda.qc.report_.items if i.status != "ok"}
                if eda.qc is not None
                else set()
            )
            missing_warns = [w for w in case.qc_warns if w not in warned]
        else:
            missing_warns = []
        cfg = TaskConfig(task_type=case.task_type, target="target", seed=seed, **case.config)  # type: ignore[arg-type]
        try:
            res = train(cfg, df)
        except Exception as exc:  # noqa: BLE001 - outcome is judged below
            row["seconds"] = round(time.perf_counter() - t0, 1)
            ok = case.expect_error is not None and isinstance(exc, case.expect_error)
            row["passed"] = ok and not missing_warns
            row["detail"] = (
                f"{'expected' if ok else 'UNEXPECTED'} {type(exc).__name__}: {str(exc)[:120]}"
            )
            if missing_warns:
                row["detail"] += f"; QC did not warn {missing_warns}"
            return row
    row["seconds"] = round(time.perf_counter() - t0, 1)
    if case.expect_error is not None:
        row["detail"] = f"expected {case.expect_error.__name__} but training succeeded"
        return row
    if case.metric is not None:
        name, minimum = case.metric
        value = res.metrics["holdout"].get(name, float("nan"))
        row[name] = round(float(value), 4)
        ok = bool(value >= minimum)
        row["detail"] = f"{name}={value:.3f} (min {minimum}) champion={res.metrics.get('champion')}"
    else:
        ok = True
        row["detail"] = "trained"
    if missing_warns:
        row["detail"] += f"; QC did not warn {missing_warns}"
    row["passed"] = ok and not missing_warns
    return row


def run_suite(names: list[str] | None = None, seed: int = 0, with_eda: bool = True) -> pd.DataFrame:
    cases = [c for c in CASES if names is None or c.name in names]
    unknown = sorted(set(names or []) - {c.name for c in CASES})
    if unknown:
        raise ValueError(
            f"Unknown benchmark case(s): {unknown}. Available: {[c.name for c in CASES]}"
        )
    return pd.DataFrame([run_case(c, seed=seed, with_eda=with_eda) for c in cases])
